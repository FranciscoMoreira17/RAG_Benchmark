"""
rag_llamaindexQwen.py - LlamaIndex com embedder Qwen3
=====================================================
Cópia da rag_llamaindex.py, trocando SÓ o modelo de embedding:
E5-small (384d) -> Qwen3-Embedding-0.6B (1024d).

"""

import os
import time

from config import (
    qdrant_client,
    montar_contexto,
    get_groq_client,
    SYSTEM_PROMPT,
    GEN_MODEL,
    GEN_TEMPERATURE,
)
from runner import FrameworkBase
from dre_loader import carregar_corpus_dre, corpus_fingerprint

try:
    from frameworks.rag_UnitLevelSegmentationQwen import _get_embedder
except ImportError:
    from rag_UnitLevelSegmentationQwen import _get_embedder

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore


COLLECTION = "benchmark_llamaindex_qwen"     # NAO colide com a collection E5


# =====================================================================
# Adapter: Qwen3Embedder -> interface BaseEmbedding do LlamaIndex
# =====================================================================
class LlamaIndexQwenEmbedding(BaseEmbedding):

    def __init__(self):
        super().__init__(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            embed_batch_size=32,
        )

    def _get_query_embedding(self, query: str) -> list[float]:
        return _get_embedder().embed_query(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return _get_embedder().embed_documents([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return _get_embedder().embed_documents(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


class Framework(FrameworkBase):

    nome = "llamaindex_qwen"

    def __init__(self):
        self.collection = COLLECTION
        self.nome_run = "llamaindex-qwen"
        self.embed_model = LlamaIndexQwenEmbedding()

        # MESMO parser do llamaindex-E5 — sem teto.
        self.parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

        Settings.llm = None
        Settings.embed_model = self.embed_model
        Settings.node_parser = self.parser

        self._index = None

    def config_ingestao(self) -> dict:
        return {"pipeline": "llamaindex-sentence-qwen", "chunk_size": 1024,
                "overlap": 200, "embedder": "qwen3-0.6b", "dim": 1024}

    def descricao(self) -> dict:
        return {"embedder": "qwen3-0.6b", "dim": 1024, "splitter": "sentence-1024-200"}

    def _get_vector_store(self) -> QdrantVectorStore:
        return QdrantVectorStore(client=qdrant_client, collection_name=self.collection)

    def _get_index(self) -> VectorStoreIndex:
        if self._index is None:
            vector_store = self._get_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )
        return self._index

    def ingerir(self) -> dict:
        docs = carregar_corpus_dre()
        print(f"    [LlamaIndex-Qwen] {len(docs)} documentos DRE carregados")

        documents = [
            Document(
                text=d["texto"],
                metadata={
                    "doc_id": d["id"],
                    "titulo": d["titulo"],
                    "tipo": d["tipo"],
                    "numero": d["numero"],
                },
            )
            for d in docs
        ]

        nodes = self.parser.get_nodes_from_documents(documents)
        print(f"    [LlamaIndex-Qwen] {len(nodes)} nodes após sentence splitting")

        # O runner (garantir_collection) pode ter criado a collection com a
        # dimensão do E5 (384). Apaga-se para o LlamaIndex a recriar a 1024.
        if qdrant_client.collection_exists(self.collection):
            qdrant_client.delete_collection(self.collection)

        print(f"    [LlamaIndex-Qwen] A indexar no Qdrant ({self.collection})...")
        vector_store = self._get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )

        return {
            "n_chunks": len(nodes),
            "n_documentos": len(docs),
            "corpus_fingerprint": corpus_fingerprint(docs),
        }

    def _extrair(self, source_nodes):
        contextos, metadados = [], []
        for nws in source_nodes:
            node = nws.node
            contextos.append(node.get_content())
            metadados.append({
                "doc_id": node.metadata.get("doc_id", "?"),
                "titulo": node.metadata.get("titulo", ""),
                "tipo": node.metadata.get("tipo", ""),
                "numero": node.metadata.get("numero", ""),
                "score": float(nws.score) if nws.score else 0.0,
                "method": "llamaindex-qwen-denso",
            })
        return contextos, metadados

    def recuperar(self, query: str, top_k: int = 5) -> dict:
        index = self._get_index()
        t0 = time.time()
        source_nodes = index.as_retriever(similarity_top_k=top_k).retrieve(query)
        tempo_retrieval = time.time() - t0
        contextos, metadados = self._extrair(source_nodes)
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
        index = self._get_index()
        t0 = time.time()
        source_nodes = index.as_retriever(similarity_top_k=top_k).retrieve(query)
        tempo_retrieval = time.time() - t0
        contextos, metadados = self._extrair(source_nodes)

        blocos = [
            f"--- FONTE ({m['tipo']} {m['numero']}) ---\n{c}"
            for c, m in zip(contextos, metadados)
        ]
        contexto_fmt, n_usados, n_tokens = montar_contexto(blocos)

        t0 = time.time()
        resp = get_groq_client().chat.completions.create(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"CONTEXTO LEGAL:\n{contexto_fmt}\n\nPERGUNTA: {query}"},
            ],
            temperature=GEN_TEMPERATURE,
            max_tokens=1000,
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