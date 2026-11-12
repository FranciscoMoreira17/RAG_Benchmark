"""
rag_llamaindexMulti.py - LlamaIndex com o embedder escolhido na CLI
===================================================================
Junta rag_llamaindex.py (E5) e rag_llamaindexQwen.py (Qwen3) num só ficheiro.
O pipeline é o nativo do LlamaIndex e é igual para todos os embedders:
  - SentenceSplitter (chunk_size 1024, overlap 200) — corta por frases
  - QdrantVectorStore + VectorIndexRetriever (retrieval denso)
  - Geração directa via Groq (sem response synthesis "compact", para
    equalizar contexto, latência e custo entre condições)

Uso:
    python runner.py --framework llamaindex_multi --embedder e5
    python runner.py --framework llamaindex_multi --embedder qwen

Compatibilidade: com e5 e qwen, a collection, o nome da condição e o
fingerprint são IGUAIS aos dos ficheiros originais (frameworks/old/), por
isso os índices já construídos e os resultados existentes são reaproveitados.
Um embedder novo (registado em embedders.py) segue a regra do qwen:
    collection  benchmark_llamaindex_<chave>
    condição    llamaindex-<chave>
"""

import time

from config import (
    qdrant_client,
    montar_contexto,
    get_collection_name,
    get_groq_client,
    SYSTEM_PROMPT,
    GEN_MODEL,
    GEN_TEMPERATURE,
    GEN_MAX_TOKENS,
)
from runner import FrameworkBase
from dre_loader import carregar_corpus_dre, corpus_fingerprint
from embedders import get_embedder_spec, carregar_embedder

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200


class LlamaIndexEmbedding(BaseEmbedding):
    """Adapta qualquer embedder do registo à interface do LlamaIndex."""

    chave: str = "e5"

    def __init__(self, chave: str, modelo: str):
        super().__init__(model_name=modelo, embed_batch_size=32, chave=chave)

    def _get_query_embedding(self, query: str) -> list[float]:
        return carregar_embedder(self.chave).embed_query(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return carregar_embedder(self.chave).embed_documents([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return carregar_embedder(self.chave).embed_documents(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


class Framework(FrameworkBase):

    def __init__(self, embedder: str = "e5"):
        self.spec = get_embedder_spec(embedder)
        self.legado_e5 = (self.spec.chave == "e5")

        # Identidade da condição. O E5 mantém os nomes do rag_llamaindex.py
        # original; os restantes seguem a regra do rag_llamaindexQwen.py.
        if self.legado_e5:
            self.nome = "llamaindex"
            self.nome_run = "llamaindex"
            self.collection = get_collection_name("llamaindex")
            self._method = "llamaindex-denso"
        else:
            chave = self.spec.chave
            self.nome = f"llamaindex_{chave}"
            self.nome_run = f"llamaindex-{chave}"
            self.collection = f"benchmark_llamaindex_{chave}"
            self._method = f"llamaindex-{chave}-denso"

        self.embed_model = LlamaIndexEmbedding(self.spec.chave, self.spec.modelo)
        self.parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        # A geração é feita fora do LlamaIndex (cliente Groq), por isso o
        # LlamaIndex não precisa de LLM.
        Settings.llm = None
        Settings.embed_model = self.embed_model
        Settings.node_parser = self.parser

        self._index = None

    # ─────────────────────────────────────────────────────────
    # Identificação (fingerprint e snapshot dos resultados)
    # ─────────────────────────────────────────────────────────
    def config_ingestao(self) -> dict:
        if self.legado_e5:
            # ATENÇÃO: o SentenceSplitter usa chunk_size=1024, mas o fingerprint
            # do E5 regista 512 desde a ingestão original. Mantém-se o 512 para
            # não invalidar o índice existente; corrigir implica re-ingerir.
            return {"pipeline": "llamaindex-sentence", "chunk_size": 512}
        return {"pipeline": f"llamaindex-sentence-{self.spec.chave}",
                "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP,
                "embedder": self.spec.tag, "dim": self.spec.dim}

    def descricao(self) -> dict:
        if self.legado_e5:
            return {}
        return {"embedder": self.spec.tag, "dim": self.spec.dim,
                "splitter": f"sentence-{CHUNK_SIZE}-{CHUNK_OVERLAP}"}

    # ─────────────────────────────────────────────────────────
    # Índice
    # ─────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────
    # Ingestão
    # ─────────────────────────────────────────────────────────
    def ingerir(self) -> dict:
        etiqueta = f"[LlamaIndex-{self.spec.chave}]"
        docs = carregar_corpus_dre()
        print(f"    {etiqueta} {len(docs)} documentos DRE carregados")

        documents = [
            Document(
                text=d["texto"],
                metadata={
                    "doc_id": d["id"],          # guid — alvo de retrieval
                    "titulo": d["titulo"],
                    "tipo": d["tipo"],
                    "numero": d["numero"],
                },
            )
            for d in docs
        ]

        nodes = self.parser.get_nodes_from_documents(documents)
        print(f"    {etiqueta} {len(nodes)} nodes após sentence splitting")

        # A collection é apagada para o LlamaIndex a recriar com a dimensão
        # do embedder escolhido (o runner pode tê-la criado com a do E5) e
        # para uma re-ingestão não acumular pontos da ingestão anterior.
        if qdrant_client.collection_exists(self.collection):
            qdrant_client.delete_collection(self.collection)

        print(f"    {etiqueta} A indexar no Qdrant ({self.collection})...")
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

    # ─────────────────────────────────────────────────────────
    # Retrieval + Geração
    # ─────────────────────────────────────────────────────────
    def _pesquisar(self, query: str, top_k: int):
        index = self._get_index()
        t0 = time.time()
        source_nodes = index.as_retriever(similarity_top_k=top_k).retrieve(query)
        tempo = time.time() - t0

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
                "method": self._method,
            })
        return contextos, metadados, tempo

    def recuperar(self, query: str, top_k: int = 5) -> dict:
        """Só retrieval — sem geração, para a avaliação de retrieval."""
        contextos, metadados, tempo_retrieval = self._pesquisar(query, top_k)
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
        contextos, metadados, tempo_retrieval = self._pesquisar(query, top_k)

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
            max_tokens=GEN_MAX_TOKENS,
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
