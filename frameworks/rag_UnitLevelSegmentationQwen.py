"""
rag_UnitLevelSegmentationQwen.py — Variante Qwen3 da UnitLevelSegmentation
=========================================================================
"""

import os
import time
import hashlib

from config import (
    qdrant_client,
    montar_contexto,
    get_groq_client,
    SYSTEM_PROMPT,
    GEN_MODEL,
)
from runner import FrameworkBase
from dre_loader import carregar_corpus_dre, corpus_fingerprint
from dre_segmentador_qwen import segmentar_corpus

from qdrant_client import models as qm
from fastembed import SparseTextEmbedding

COLLECTION = "benchmark_hibrido_qwen"     
QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_DIM = 1024                           
TETO_TOKENS = 2048                         
QWEN_BATCH = 4                             
BM25_MODEL = "Qdrant/bm25"
PREFETCH = 50                             

VARIANTES = {
    "denso":   {"modo": "denso"},
    "esparso": {"modo": "esparso"},
    "hibrido": {"modo": "hibrido"},
}


def _point_id(chunk_id: str) -> int:
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:15], 16)


class Qwen3Embedder:

    def __init__(self, model_name: str = QWEN_MODEL, max_seq: int = TETO_TOKENS):
        import torch
        from sentence_transformers import SentenceTransformer

        self._torch = torch
        usa_cuda = torch.cuda.is_available()
        model_kwargs = {"attn_implementation": "eager"}
        if usa_cuda:
            model_kwargs["torch_dtype"] = torch.float16
        device = "cuda" if usa_cuda else "cpu"

        self.model = SentenceTransformer(model_name, device=device,
                                         model_kwargs=model_kwargs)
        self.model.max_seq_length = max_seq
        self.model.tokenizer.model_max_length = max_seq
        self.max_seq = max_seq
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"  [Qwen3] {model_name} | dim {self.dim} | device {device} "
              f"| dtype {next(self.model.parameters()).dtype} | max_seq {max_seq}")

    def contar_tokens(self, texto: str) -> int:
        return len(self.model.tokenizer(texto, add_special_tokens=True,
                                        truncation=False)["input_ids"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._torch.no_grad():
            vecs = self.model.encode(
                texts, batch_size=QWEN_BATCH, normalize_embeddings=True,
                show_progress_bar=True, convert_to_numpy=True,
            )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        with self._torch.no_grad():
            vec = self.model.encode(
                [text], prompt_name="query", normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return vec[0].tolist()


_qwen_embedder = None


def _get_embedder() -> Qwen3Embedder:
    global _qwen_embedder
    if _qwen_embedder is None:
        _qwen_embedder = Qwen3Embedder()
    return _qwen_embedder


class Framework(FrameworkBase):

    nome = "ULS_qwen"
    usa_qdrant = True

    def __init__(self, variante: str = "hibrido"):
        if variante not in VARIANTES:
            raise ValueError(f"Variante '{variante}' inválida. {list(VARIANTES)}")
        self.variante = variante
        self.modo = VARIANTES[variante]["modo"]
        self.collection = COLLECTION
        self.nome_run = f"UnitLevelSegmentationQwen-{variante}"
        self._groq = None
        self._sparse = None
        self._embedder = None

    def _get_emb(self) -> Qwen3Embedder:
        if self._embedder is None:
            self._embedder = _get_embedder()
        return self._embedder

    def _get_sparse(self):
        if self._sparse is None:
            self._sparse = SparseTextEmbedding(model_name=BM25_MODEL)
        return self._sparse

    def _get_groq(self):
        if self._groq is None:
            self._groq = get_groq_client()
        return self._groq

    def config_ingestao(self) -> dict:
        # Identifica a variante Qwen: fingerprint distinto do índice E5.
        return {"pipeline": "dre-hibrido-qwen", "embedder": QWEN_MODEL,
                "dim": QWEN_DIM, "teto_tokens": TETO_TOKENS, "bm25": BM25_MODEL}

    def descricao(self) -> dict:
        return {"variante": self.variante, "modo": self.modo,
                "embedder": "qwen3-0.6b", "dim": QWEN_DIM}


    def ingerir(self) -> dict:
        emb = self._get_emb()
        docs = carregar_corpus_dre()

        # Segmentação COM TETO de tokens, usando o tokenizer do Qwen3.
        segmentos, rel_seg = segmentar_corpus(
            docs, teto_tokens=TETO_TOKENS, contar_tokens=emb.contar_tokens)

        textos = [s["texto"] for s in segmentos]

        print(f"    [QWEN] Embedding denso (Qwen3) de {len(textos)} segmentos...")
        densos = emb.embed_documents(textos)

        print(f"    [QWEN] Embedding esparso (BM25)...")
        sparse = self._get_sparse()
        esparsos = list(sparse.embed(textos, batch_size=64))

        # Recriar collection com DOIS espaços de vetores (denso a 1024)
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

        print(f"    [QWEN] A indexar {len(segmentos)} pontos...")
        pontos = []
        for i, (seg, dv, sv) in enumerate(zip(segmentos, densos, esparsos)):
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

        print(f"    [QWEN] {len(pontos)} pontos → {self.collection}")
        return {
            "n_chunks": len(pontos),
            "n_documentos": len(docs),
            "segmentacao": rel_seg["distribuicao_nivel_documento"],
            "corpus_fingerprint": corpus_fingerprint(docs),
        }

    def _retrieve(self, query: str, top_k: int):
        qv_dense = self._get_emb().embed_query(query)

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