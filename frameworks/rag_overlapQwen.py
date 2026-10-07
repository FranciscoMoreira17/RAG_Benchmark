"""
rag_overlapQwe.py - Segmentação cega por janela deslizante  (fixed-size + overlap)
Com modelo de Embedding Qwen3 0.6B
==============================================================================
Condição de baseline: fixed-size chunking com overlap, CEGO à estrutura.
Ao contrário do RAG Estrutural (que deteta artigos/blocos/janelas), esta
condição ignora por completo a estrutura do documento e corta TODOS os
documentos em janelas de tamanho fixo em caracteres, com sobreposição.

  Janela  = 1200 caracteres
  Overlap = 15% (180 caracteres)
  Passo   = 1020 caracteres

  --variante denso     → só similaridade vetorial (E5)
  --variante esparso   → só BM25
  --variante hibrido   → fusão RRF dos dois

"""

import time
import hashlib

from config import (
    embedding_model,
    qdrant_client,
    montar_contexto,
    get_groq_client,
    SYSTEM_PROMPT,
    GEN_MODEL,
    EMBEDDING_DIM,
)
from runner import FrameworkBase
from dre_loader import carregar_corpus_dre, corpus_fingerprint

try:
    from frameworks.rag_UnitLevelSegmentationQwen import _get_embedder
except ImportError:
    from rag_UnitLevelSegmentationQwen import _get_embedder

from qdrant_client import models as qm
from fastembed import SparseTextEmbedding

COLLECTION = "benchmark_overlap_qwen"
QWEN_DIM = 1024
EMBEDDING_MODEL="qwen3-0.6b"
BM25_MODEL = "Qdrant/bm25"
PREFETCH = 50

JANELA_CHARS = 1200
OVERLAP_PCT = 0.15
OVERLAP_CHARS = int(JANELA_CHARS * OVERLAP_PCT)   # 180
PASSO = JANELA_CHARS - OVERLAP_CHARS              # 1020
MIN_SEGMENTO_CHARS = 40

VARIANTES = {
    "denso":   {"modo": "denso"},
    "esparso": {"modo": "esparso"},
    "hibrido": {"modo": "hibrido"},
}


def _point_id(chunk_id: str) -> int:
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:15], 16)


def segmentar_cego(docs: list[dict]) -> tuple[list[dict], dict]:
    """
    Corta TODOS os documentos em janelas de JANELA_CHARS com OVERLAP_CHARS
    de sobreposição, ignorando qualquer estrutura. Produz a mesma estrutura
    de dicionário que dre_segmentador, para compatibilidade com a ingestão.
    """
    todos = []
    n_janelas = []
    for d in docs:
        texto = d["texto"]
        segs_doc = []
        i = 0
        while i < len(texto):
            janela = texto[i:i + JANELA_CHARS].strip()
            if len(janela) >= MIN_SEGMENTO_CHARS:
                segs_doc.append(janela)
            i += PASSO
        # documento demasiado curto para uma janela mínima → um segmento
        if not segs_doc and texto.strip():
            segs_doc = [texto.strip()]
        n_janelas.append(len(segs_doc))
        for j, janela in enumerate(segs_doc):
            todos.append({
                "doc_id": d["id"],
                "chunk_id": f"{d['id']}::{j}",
                "texto": janela,
                "rotulo": f"janela@{j}",
                "nivel_segmentacao": "overlap",
                "titulo": d["titulo"],
                "tipo": d["tipo"],
                "numero": d["numero"],
                "data_publicacao": d["data_publicacao"],
                "url": d["url"],
            })

    from statistics import mean, median
    relatorio = {
        "n_documentos": len(docs),
        "n_segmentos": len(todos),
        "segmentos_por_doc_media": mean(n_janelas) if n_janelas else 0,
        "segmentos_por_doc_mediana": median(n_janelas) if n_janelas else 0,
        "segmentos_por_doc_max": max(n_janelas) if n_janelas else 0,
        "janela_chars": JANELA_CHARS,
        "overlap_chars": OVERLAP_CHARS,
    }
    print(f"  [OVL] {len(docs)} docs → {len(todos)} segmentos "
          f"(janela={JANELA_CHARS}, overlap={OVERLAP_CHARS}, "
          f"media {relatorio['segmentos_por_doc_media']:.1f}/doc, "
          f"max {relatorio['segmentos_por_doc_max']})")
    return todos, relatorio


class Framework(FrameworkBase):

    nome = "overlap_qwen"
    usa_qdrant = True

    def __init__(self, variante: str = "hibrido"):
        if variante not in VARIANTES:
            raise ValueError(f"Variante '{variante}' inválida. {list(VARIANTES)}")
        self.variante = variante
        self.modo = VARIANTES[variante]["modo"]
        self.collection = COLLECTION
        self.nome_run = f"overlap-qwen-{variante}"
        self._groq = None
        self._sparse = None

    def _get_sparse(self):
        if self._sparse is None:
            self._sparse = SparseTextEmbedding(model_name=BM25_MODEL)
        return self._sparse

    def _get_groq(self):
        if self._groq is None:
            self._groq = get_groq_client()
        return self._groq

    def config_ingestao(self) -> dict:
        return {
            "pipeline": "dre-overlap-qwen",
            "embedder": EMBEDDING_MODEL,
            "bm25": BM25_MODEL,
            "janela_chars": JANELA_CHARS,
            "overlap_chars": OVERLAP_CHARS,
        }

    def descricao(self) -> dict:
        return {
            "variante": self.variante,
            "modo": self.modo,
            "janela_chars": JANELA_CHARS,
            "overlap_chars": OVERLAP_CHARS,
            "segmentacao": "cega (fixed-size + overlap)",
        }

    # ─────────────────────────────────────────────────────────
    # Ingestão (uma vez para as três variantes)
    # ─────────────────────────────────────────────────────────
    def ingerir(self) -> dict:
        docs = carregar_corpus_dre()
        segmentos, rel_seg = segmentar_cego(docs)

        textos = [s["texto"] for s in segmentos]

        print(f"    [OVL] Embedding denso (E5) de {len(textos)} segmentos...")
        densos = _get_embedder().embed_documents(textos)

        print(f"    [OVL] Embedding esparso (BM25)...")
        sparse = self._get_sparse()
        esparsos = list(sparse.embed(textos, batch_size=64))

        if qdrant_client.collection_exists(self.collection):
            qdrant_client.delete_collection(self.collection)
        qdrant_client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": qm.VectorParams(size=QWEN_DIM, distance=qm.Distance.COSINE),
            },
            sparse_vectors_config={
                "bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF),
            },
        )

        print(f"    [OVL] A indexar {len(segmentos)} pontos...")
        pontos = []
        for seg, dv, sv in zip(segmentos, densos, esparsos):
            payload = {
                "doc_id": seg["doc_id"],
                "chunk_id": seg["chunk_id"],
                "texto": seg["texto"],
                "titulo": seg["titulo"],
                "tipo": seg["tipo"],
                "numero": seg["numero"],
                "data_publicacao": seg["data_publicacao"],
                "nivel_segmentacao": seg["nivel_segmentacao"],
                "rotulo": seg["rotulo"],
            }
            pontos.append(qm.PointStruct(
                id=_point_id(seg["chunk_id"]),
                vector={
                    "dense": dv,
                    "bm25": qm.SparseVector(indices=sv.indices.tolist(),
                                            values=sv.values.tolist()),
                },
                payload=payload,
            ))

        for i in range(0, len(pontos), 100):
            qdrant_client.upsert(collection_name=self.collection,
                                 points=pontos[i:i + 100], wait=True)

        print(f"    [OVL] {len(pontos)} pontos → {self.collection}")
        return {
            "n_chunks": len(pontos),
            "n_documentos": len(docs),
            "segmentacao": {"overlap": rel_seg["n_segmentos"]},
            "janela_chars": JANELA_CHARS,
            "overlap_chars": OVERLAP_CHARS,
            "corpus_fingerprint": corpus_fingerprint(docs),
        }

    # ─────────────────────────────────────────────────────────
    # Retrieval segundo o modo (idêntico ao rag_hibrido)
    # ─────────────────────────────────────────────────────────
    def _retrieve(self, query: str, top_k: int):
        qv_dense = _get_embedder().embed_query(query)

        if self.modo == "denso":
            return qdrant_client.query_points(
                collection_name=self.collection,
                query=qv_dense, using="dense",
                limit=top_k, with_payload=True,
            ).points

        sparse = self._get_sparse()
        sv = next(iter(sparse.query_embed(query)))
        sparse_vec = qm.SparseVector(indices=sv.indices.tolist(),
                                     values=sv.values.tolist())

        if self.modo == "esparso":
            return qdrant_client.query_points(
                collection_name=self.collection,
                query=sparse_vec, using="bm25",
                limit=top_k, with_payload=True,
            ).points

        return qdrant_client.query_points(
            collection_name=self.collection,
            prefetch=[
                qm.Prefetch(query=qv_dense, using="dense", limit=PREFETCH),
                qm.Prefetch(query=sparse_vec, using="bm25", limit=PREFETCH),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k, with_payload=True,
        ).points

    def recuperar(self, query: str, top_k: int = 5) -> dict:
        t0 = time.time()
        resultados = self._retrieve(query, top_k)
        tempo_retrieval = time.time() - t0

        contextos, metadados = [], []
        for r in resultados:
            pl = r.payload
            contextos.append(pl["texto"])
            metadados.append({
                "doc_id": pl.get("doc_id", "?"),
                "titulo": pl.get("titulo", ""),
                "tipo": pl.get("tipo", ""),
                "numero": pl.get("numero", ""),
                "nivel": pl.get("nivel_segmentacao", ""),
                "score": float(r.score),
                "method": self.modo,
            })

        return {
            "resposta": "",
            "contextos": contextos,
            "metadados": metadados,
            "tempo_retrieval_s": tempo_retrieval,
            "tempo_geracao_s": 0.0,
            "n_blocos_recuperados": len(contextos),
            "n_blocos_usados": len(contextos),
            "tokens_contexto": 0,
        }

    def responder(self, query: str, top_k: int = 5) -> dict:
        t0 = time.time()
        resultados = self._retrieve(query, top_k)
        tempo_retrieval = time.time() - t0

        contextos, metadados = [], []
        for r in resultados:
            pl = r.payload
            contextos.append(pl["texto"])
            metadados.append({
                "doc_id": pl.get("doc_id", "?"),
                "titulo": pl.get("titulo", ""),
                "tipo": pl.get("tipo", ""),
                "numero": pl.get("numero", ""),
                "nivel": pl.get("nivel_segmentacao", ""),
                "score": float(r.score),
                "method": self.modo,
            })

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