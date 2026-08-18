"""
rag_estrutural.py — RAG Estrutural (contribuição da tese)
==========================================================
Método domain-specific para legislação portuguesa. Estende a
segmentação estrutural DRE (já usada pelo híbrido) com duas técnicas,
AMBAS SEM LLM:

  1. ENRIQUECIMENTO DA REPRESENTAÇÃO (contextual embedding determinístico)
     Prepend ao texto, antes do embedding, um cabeçalho de frontmatter
     com metadados extraídos por REGEX (tipo, número, entidade, sumário).
     Adaptação da técnica de Contextual Retrieval (Anthropic) usando
     metadados estruturados em vez de descrições geradas por LLM.

  2. EXPANSÃO POR GRAFO
     As citações entre diplomas são explícitas no texto ("nos termos da
     Lei 35/2014"), extraídas por regex. Formam um grafo de citações.
     No retrieval, seguir as citações de um documento recuperado e
     acrescentar os diplomas citados ao contexto.

Formato intermédio: bundles compatíveis com OKF (markdown + frontmatter
YAML). NOTA: OKF é o formato de armazenamento, não o método.

Variantes (lidas contra o hibrido-hibrido já medido, que é a base:
estrutural + texto cru + híbrido, sem enriquecimento, sem grafo):

  enriquecido        → estrutural + frontmatter no embedding + híbrido
  enriquecido_grafo  → o anterior + expansão por grafo de citações

Retrieval: híbrido (denso E5 + BM25, fusão RRF), como as outras.
"""

import os
import re
import time
import hashlib
from collections import defaultdict

from config import (
    embedding_model,
    qdrant_client,
    montar_contexto,
    get_groq_client,
    fingerprint_ingestao,
    SYSTEM_PROMPT,
    GEN_MODEL,
    EMBEDDING_DIM,
)
from runner import FrameworkBase
from dre_loader import carregar_corpus_dre, corpus_fingerprint
from dre_segmentador import segmentar_corpus

from qdrant_client import models as qm
from fastembed import SparseTextEmbedding

COLLECTION = "benchmark_estrutural"
BM25_MODEL = "Qdrant/bm25"
PREFETCH = 50
MAX_GRAFO_EXPANSAO = 3

VARIANTES = {
    "enriquecido":       {"grafo": False},
    "enriquecido_grafo": {"grafo": True},
}

# ── Regex de extracção (sem LLM) ─────────────────────────────
RE_SUMARIO = re.compile(r"Sumário:\s*(.+?)(?:\n|$)", re.IGNORECASE)
RE_ENTIDADE = re.compile(r"entidade adjudicante:\s*(.+)", re.IGNORECASE)
RE_CITACAO = re.compile(
    r"(?:decreto[\s\-]lei|decreto|lei|portaria|despacho|regulamento|aviso|resolução)\s*"
    r"(?:n\.?\s*[º°]?\s*)?(\d+(?:-[A-Za-z]+)?/\d{4})",
    re.IGNORECASE,
)


def _limpar(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extrair_frontmatter(doc: dict) -> dict:
    """Metadados estruturados por regex — o frontmatter OKF."""
    texto = doc["texto"]
    fm = {
        "tipo": doc["tipo"],
        "numero": doc["numero"],
        "data": doc["data_publicacao"],
    }
    sm = RE_SUMARIO.search(texto[:500])
    if sm:
        fm["sumario"] = _limpar(sm.group(1))[:200]
    ent = RE_ENTIDADE.search(texto)
    if ent:
        fm["entidade"] = _limpar(ent.group(1))[:100]
    return fm


def extrair_citacoes(doc: dict) -> list[str]:
    """
    Números de diplomas citados no texto, EXCLUINDO o número próprio
    (senão criaria auto-loop no grafo).
    """
    proprio = doc["numero"]
    cits = set()
    for m in RE_CITACAO.finditer(doc["texto"]):
        num = m.group(1)
        if num != proprio:
            cits.add(num)
    return sorted(cits)


def texto_enriquecido(doc: dict, fm: dict) -> str:
    """Texto com cabeçalho de frontmatter prepended (para embedding)."""
    partes = [f"Tipo: {fm['tipo']}"]
    if fm.get("numero"):
        partes.append(f"Número: {fm['numero']}")
    if fm.get("entidade"):
        partes.append(f"Entidade: {fm['entidade']}")
    if fm.get("sumario"):
        partes.append(f"Assunto: {fm['sumario']}")
    cabecalho = " | ".join(partes)
    return f"{cabecalho}\n\n{doc['texto']}"


def _point_id(chunk_id: str) -> int:
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:15], 16)


class Framework(FrameworkBase):

    nome = "estrutural"
    usa_qdrant = True

    def __init__(self, variante: str = "enriquecido"):
        if variante not in VARIANTES:
            raise ValueError(f"Variante '{variante}' inválida. {list(VARIANTES)}")
        self.variante = variante
        self.usa_grafo = VARIANTES[variante]["grafo"]
        self.collection = COLLECTION
        self.nome_run = f"estrutural-{variante}"
        self._groq = None
        self._sparse = None
        # grafo de citações: numero_diploma → set(doc_id que o citam)
        # e doc_id → lista de números citados
        self._citacoes_por_doc = {}
        self._doc_por_numero = {}

    def _get_sparse(self):
        if self._sparse is None:
            self._sparse = SparseTextEmbedding(model_name=BM25_MODEL)
        return self._sparse

    def _get_groq(self):
        if self._groq is None:
            self._groq = get_groq_client()
        return self._groq

    def config_ingestao(self) -> dict:
        # As duas variantes partilham índice (o enriquecimento é igual;
        # só difere o uso do grafo no retrieval, não na indexação).
        return {"pipeline": "estrutural-enriquecido", "bm25": BM25_MODEL}

    def descricao(self) -> dict:
        return {"variante": self.variante, "grafo": self.usa_grafo,
                "retrieval": "hibrido-rrf"}

    # ─────────────────────────────────────────────────────────
    # Ingestão
    # ─────────────────────────────────────────────────────────
    def ingerir(self) -> dict:
        docs = carregar_corpus_dre()
        segmentos, rel_seg = segmentar_corpus(docs)

        # Mapa numero → doc_id (para resolver citações no grafo)
        for d in docs:
            if d["numero"]:
                self._doc_por_numero.setdefault(d["numero"], d["id"])

        # Frontmatter + citações por documento
        fm_por_doc, cit_por_doc = {}, {}
        for d in docs:
            fm_por_doc[d["id"]] = extrair_frontmatter(d)
            cit_por_doc[d["id"]] = extrair_citacoes(d)
        self._citacoes_por_doc = cit_por_doc

        n_com_citacoes = sum(1 for c in cit_por_doc.values() if c)
        print(f"    [EST] Grafo: {n_com_citacoes}/{len(docs)} docs com citações")

        # Texto enriquecido por segmento (herda frontmatter do doc-pai)
        textos_embed = []
        for s in segmentos:
            fm = fm_por_doc.get(s["doc_id"], {"tipo": s["tipo"], "numero": s["numero"]})
            # enriquecer só o 1.º segmento com sumário/entidade; nos
            # restantes basta tipo+número (evita repetir o cabeçalho longo)
            doc_stub = {"texto": s["texto"], "tipo": s["tipo"], "numero": s["numero"]}
            textos_embed.append(texto_enriquecido(doc_stub, fm))

        print(f"    [EST] Embedding denso (E5) de {len(textos_embed)} segmentos enriquecidos...")
        densos = embedding_model.embed_documents(textos_embed)

        print(f"    [EST] Embedding esparso (BM25)...")
        sparse = self._get_sparse()
        esparsos = list(sparse.embed(textos_embed, batch_size=64))

        if qdrant_client.collection_exists(self.collection):
            qdrant_client.delete_collection(self.collection)
        qdrant_client.create_collection(
            collection_name=self.collection,
            vectors_config={"dense": qm.VectorParams(size=EMBEDDING_DIM, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )

        print(f"    [EST] A indexar {len(segmentos)} pontos...")
        pontos = []
        for i, (seg, dv, sv) in enumerate(zip(segmentos, densos, esparsos)):
            payload = {
                "doc_id": seg["doc_id"],
                "chunk_id": seg["chunk_id"],
                "texto": seg["texto"],                 # texto CRU no payload
                "texto_embebido": textos_embed[i],     # o que foi embebido
                "titulo": seg["titulo"],
                "tipo": seg["tipo"],
                "numero": seg["numero"],
                "citacoes": ",".join(cit_por_doc.get(seg["doc_id"], [])),
                "nivel_segmentacao": seg["nivel_segmentacao"],
            }
            pontos.append(qm.PointStruct(
                id=_point_id(seg["chunk_id"]),
                vector={"dense": dv,
                        "bm25": qm.SparseVector(indices=sv.indices.tolist(),
                                                values=sv.values.tolist())},
                payload=payload,
            ))

        for i in range(0, len(pontos), 100):
            qdrant_client.upsert(collection_name=self.collection,
                                 points=pontos[i:i + 100], wait=True)

        print(f"    [EST] {len(pontos)} pontos → {self.collection}")
        return {
            "n_chunks": len(pontos),
            "n_documentos": len(docs),
            "docs_com_citacoes": n_com_citacoes,
            "segmentacao": rel_seg["distribuicao_nivel_documento"],
            "corpus_fingerprint": corpus_fingerprint(docs),
        }

    # ─────────────────────────────────────────────────────────
    # Índice do grafo (reconstruído se ingestão foi saltada)
    # ─────────────────────────────────────────────────────────
    def _garantir_grafo(self):
        if self._citacoes_por_doc and self._doc_por_numero:
            return
        print("    [EST] A reconstruir grafo a partir do índice...")
        offset = None
        while True:
            batch, offset = qdrant_client.scroll(
                collection_name=self.collection,
                limit=256, offset=offset, with_payload=True, with_vectors=False,
            )
            for p in batch:
                did = p.payload.get("doc_id")
                num = p.payload.get("numero")
                if num:
                    self._doc_por_numero.setdefault(num, did)
                cits = p.payload.get("citacoes", "")
                if cits and did not in self._citacoes_por_doc:
                    self._citacoes_por_doc[did] = cits.split(",")
            if offset is None:
                break

    # ─────────────────────────────────────────────────────────
    # Retrieval híbrido + expansão por grafo
    # ─────────────────────────────────────────────────────────
    def _retrieve_hibrido(self, query: str, top_k: int):
        qv = embedding_model.embed_query(query)
        sparse = self._get_sparse()
        sv = next(iter(sparse.query_embed(query)))
        sparse_vec = qm.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
        return qdrant_client.query_points(
            collection_name=self.collection,
            prefetch=[
                qm.Prefetch(query=qv, using="dense", limit=PREFETCH),
                qm.Prefetch(query=sparse_vec, using="bm25", limit=PREFETCH),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k, with_payload=True,
        ).points

    def _expandir_grafo(self, resultados) -> list[dict]:
        """Segue as citações dos documentos recuperados."""
        if not self.usa_grafo:
            return []
        self._garantir_grafo()

        ja = {r.payload.get("doc_id") for r in resultados}
        extra = []
        for r in resultados:
            if len(extra) >= MAX_GRAFO_EXPANSAO:
                break
            did = r.payload.get("doc_id")
            for num in self._citacoes_por_doc.get(did, []):
                if len(extra) >= MAX_GRAFO_EXPANSAO:
                    break
                alvo_id = self._doc_por_numero.get(num)
                if alvo_id and alvo_id not in ja:
                    # buscar um segmento desse documento
                    pts, _ = qdrant_client.scroll(
                        collection_name=self.collection,
                        scroll_filter=qm.Filter(must=[
                            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=alvo_id))
                        ]),
                        limit=1, with_payload=True,
                    )
                    if pts:
                        extra.append(pts[0].payload)
                        ja.add(alvo_id)
        return extra

    def _montar(self, query, top_k):
        resultados = self._retrieve_hibrido(query, top_k)
        contextos, metadados = [], []
        for r in resultados:
            pl = r.payload
            contextos.append(pl["texto"])
            metadados.append({
                "doc_id": pl.get("doc_id", "?"),
                "titulo": pl.get("titulo", ""),
                "tipo": pl.get("tipo", ""),
                "numero": pl.get("numero", ""),
                "score": float(r.score),
                "method": "estrutural-hibrido",
            })
        # expansão por grafo
        for pl in self._expandir_grafo(resultados):
            contextos.append(pl["texto"])
            metadados.append({
                "doc_id": pl.get("doc_id", "?"),
                "titulo": pl.get("titulo", ""),
                "tipo": pl.get("tipo", ""),
                "numero": pl.get("numero", ""),
                "score": 0.0,
                "method": "grafo_expansao",
            })
        return contextos, metadados

    def recuperar(self, query: str, top_k: int = 5) -> dict:
        t0 = time.time()
        contextos, metadados = self._montar(query, top_k)
        tempo = time.time() - t0
        return {
            "resposta": "",
            "contextos": contextos,
            "metadados": metadados,
            "tempo_retrieval_s": tempo,
            "tempo_geracao_s": 0.0,
            "n_blocos_recuperados": len(contextos),
            "n_blocos_usados": len(contextos),
            "tokens_contexto": 0,
        }

    def responder(self, query: str, top_k: int = 5) -> dict:
        t0 = time.time()
        contextos, metadados = self._montar(query, top_k)
        tempo_retrieval = time.time() - t0

        blocos = [
            f"--- FONTE ({m['tipo']} {m['numero']}) ---\n{c}"
            for c, m in zip(contextos, metadados)
        ]
        contexto_fmt, n_usados, n_tokens = montar_contexto(blocos)

        t0 = time.time()
        resp = self._get_groq().chat.completions.create(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"CONTEXTO LEGAL:\n{contexto_fmt}\n\nPERGUNTA: {query}"},
            ],
            temperature=0, max_tokens=1000,
        )
        resposta = resp.choices[0].message.content
        tempo_geracao = time.time() - t0

        return {
            "resposta": resposta,
            "contextos": contextos[:n_usados],
            "metadados": metadados[:n_usados],
            "tempo_retrieval_s": tempo_retrieval,
            "tempo_geracao_s": tempo_geracao,
            "n_blocos_recuperados": len(blocos),
            "n_blocos_usados": n_usados,
            "tokens_contexto": n_tokens,
        }