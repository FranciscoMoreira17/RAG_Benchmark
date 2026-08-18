"""
rag_hibrido.py — Retrieval denso, esparso (BM25) e híbrido (RRF)
=================================================================
Uma framework, três condições, um índice. Cada ponto no Qdrant tem
DOIS vetores: denso (E5, 1024d, semântico) e esparso (BM25, lexical).
Indexa-se uma vez; o modo de retrieval é escolhido no responder().

  --variante denso     → só similaridade vetorial (E5)
  --variante esparso   → só BM25 (baseline lexical)
  --variante hibrido   → fusão RRF dos dois rankings

Porque isto importa para a tese: o corpus DRE está cheio de
identificadores (números de anúncio, NIPC, entidades). O BM25
resolve-os por correspondência exata; o denso capta paráfrase. A
comparação das três condições responde à pergunta central — o
retrieval semântico acrescenta valor neste corpus, ou o BM25 basta?

Unidade indexada = SEGMENTO. Alvo de retrieval = doc_id (guid) do
documento-pai. Vários segmentos podem partilhar doc_id; na
avaliação conta se o doc_id certo foi recuperado.

Nota: geração usa o mesmo gerador e orçamento das outras condições,
para paridade. O BM25 puro raramente é usado com geração — mas
mantém-se disponível para completude.
"""

import os
import time
import hashlib

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

COLLECTION = "benchmark_hibrido"      # partilhada pelas três variantes
BM25_MODEL = "Qdrant/bm25"
PREFETCH = 50                          # candidatos por ramo antes da fusão

VARIANTES = {
    "denso":   {"modo": "denso"},
    "esparso": {"modo": "esparso"},
    "hibrido": {"modo": "hibrido"},
}


def _point_id(chunk_id: str) -> int:
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:15], 16)


class Framework(FrameworkBase):

    nome = "hibrido"
    usa_qdrant = True

    def __init__(self, variante: str = "hibrido"):
        if variante not in VARIANTES:
            raise ValueError(f"Variante '{variante}' inválida. {list(VARIANTES)}")
        self.variante = variante
        self.modo = VARIANTES[variante]["modo"]
        self.collection = COLLECTION
        self.nome_run = f"hibrido-{variante}"
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
        # Todas as variantes partilham índice → mesmo fingerprint.
        return {"pipeline": "dre-hibrido", "bm25": BM25_MODEL}

    def descricao(self) -> dict:
        return {"variante": self.variante, "modo": self.modo}

    # ─────────────────────────────────────────────────────────
    # Ingestão (uma vez para as três variantes)
    # ─────────────────────────────────────────────────────────
    def ingerir(self) -> dict:
        docs = carregar_corpus_dre()
        segmentos, rel_seg = segmentar_corpus(docs)

        textos = [s["texto"] for s in segmentos]

        print(f"    [HIB] Embedding denso (E5) de {len(textos)} segmentos...")
        densos = embedding_model.embed_documents(textos)

        print(f"    [HIB] Embedding esparso (BM25)...")
        sparse = self._get_sparse()
        esparsos = list(sparse.embed(textos, batch_size=64))

        # Recriar collection com DOIS espaços de vetores
        if qdrant_client.collection_exists(self.collection):
            qdrant_client.delete_collection(self.collection)
        qdrant_client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": qm.VectorParams(size=EMBEDDING_DIM, distance=qm.Distance.COSINE),
            },
            sparse_vectors_config={
                "bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF),
            },
        )

        print(f"    [HIB] A indexar {len(segmentos)} pontos...")
        pontos = []
        for i, (seg, dv, sv) in enumerate(zip(segmentos, densos, esparsos)):
            payload = {
                "doc_id": seg["doc_id"],          # guid — alvo de retrieval
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

        print(f"    [HIB] {len(pontos)} pontos → {self.collection}")
        return {
            "n_chunks": len(pontos),
            "n_documentos": len(docs),
            "segmentacao": rel_seg["distribuicao_nivel_documento"],
            "corpus_fingerprint": corpus_fingerprint(docs),
        }

    # ─────────────────────────────────────────────────────────
    # Retrieval segundo o modo
    # ─────────────────────────────────────────────────────────
    def _retrieve(self, query: str, top_k: int):
        qv_dense = embedding_model.embed_query(query)

        if self.modo == "denso":
            return qdrant_client.query_points(
                collection_name=self.collection,
                query=qv_dense, using="dense",
                limit=top_k, with_payload=True,
            ).points

        # esparso da query
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

        # híbrido: prefetch nos dois ramos + fusão RRF
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
        """Só retrieval — sem geração. Para a fase de avaliação de
        retrieval, que não consome gerador nem quota."""
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