"""
rag_llamaindex.py — Implementação LlamaIndex para o Benchmark
================================================================
Usa os componentes nativos do LlamaIndex:
  - SimpleDirectoryReader para ingestão de PDFs
  - SentenceSplitter para chunking (default do LlamaIndex)
  - QdrantVectorStore para armazenamento
  - Google GenAI LLM para geração
  - QueryEngine com response synthesis

Diferenças chave face ao LangChain:
  - Chunking por frases (sentence boundaries) em vez de caracteres
  - Node-based indexing (cada chunk é um "node" com relações)
  - Response synthesis mode "compact" (concatena contexto, comprime)
  - Metadata automática (filename, page, section) extraída nativamente
"""

import os
import time

from config import (
    embedding_model,
    qdrant_client,
    get_collection_name,
    montar_contexto,
    get_groq_client,
    SYSTEM_PROMPT,
    GEN_MODEL,
    GEN_TEMPERATURE,
    QDRANT_URL,
    EMBEDDING_DIM,
    DOCS_DIR,
)
from runner import FrameworkBase
from dre_loader import carregar_corpus_dre, corpus_fingerprint

# ── LlamaIndex imports ───────────────────────────────────────
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.groq import Groq



# =====================================================================
# Adapter: E5Embedder → interface BaseEmbedding do LlamaIndex
# =====================================================================
class LlamaIndexE5Embedding(BaseEmbedding):
    """
    Adapta o E5Embedder partilhado (config.py) à interface
    BaseEmbedding do LlamaIndex. Mesmos vetores que o LangChain.
    """

    def __init__(self):
        super().__init__(
            model_name="intfloat/multilingual-e5-large",
            embed_batch_size=32,
        )

    def _get_query_embedding(self, query: str) -> list[float]:
        return embedding_model.embed_query(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return embedding_model.embed_documents([text])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return embedding_model.embed_documents(texts)


# =====================================================================
# Framework LlamaIndex
# =====================================================================
class Framework(FrameworkBase):

    nome = "llamaindex"

    def __init__(self):
        self.collection = get_collection_name(self.nome)

        # Embedding adapter
        self.embed_model = LlamaIndexE5Embedding()

        # LLM
        #self.llm = GoogleGenAI(
        #    model=LLM_MODEL,
        #    temperature=LLM_TEMPERATURE,
        #)

        self.llm = Groq(model=GEN_MODEL, api_key=os.environ.get("GROQ_API_KEY"))

        # Node parser (decisão nativa do LlamaIndex — sentence splitting)
        self.parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200,
        )

        # Configurar LlamaIndex Settings globais
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.node_parser = self.parser

        # Prompt customizado (mesmo system prompt que as outras frameworks)
        self.qa_prompt = PromptTemplate(
            "SISTEMA: " + SYSTEM_PROMPT + "\n\n"
            "CONTEXTO LEGAL:\n"
            "-----\n"
            "{context_str}\n"
            "-----\n\n"
            "PERGUNTA: {query_str}\n\n"
            "RESPOSTA:"
        )

        self._index = None

    def _get_vector_store(self) -> QdrantVectorStore:
        """Cria a referência ao vector store Qdrant."""
        return QdrantVectorStore(
            client=qdrant_client,
            collection_name=self.collection,
        )

    def _get_index(self) -> VectorStoreIndex:
        """Obtém ou cria o índice sobre o vector store existente."""
        if self._index is None:
            vector_store = self._get_vector_store()
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
            )
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )
        return self._index

    # ─────────────────────────────────────────────────────────
    # Ingestão
    # ─────────────────────────────────────────────────────────
    def config_ingestao(self) -> dict:
        return {"pipeline": "llamaindex-sentence", "chunk_size": 512}

    def ingerir(self) -> dict:
        """
        Pipeline nativo do LlamaIndex sobre o corpus DRE:
          documento inteiro → SentenceSplitter → QdrantVectorStore

        O SentenceSplitter corta por fronteiras de frase (não de
        caracteres como o LangChain) — é essa a diferença de
        estratégia que o benchmark compara. Cada node herda o doc_id
        (guid) do documento-pai, que é o alvo de retrieval.
        """
        from llama_index.core.schema import Document

        docs = carregar_corpus_dre()
        print(f"    [LlamaIndex] {len(docs)} documentos DRE carregados")

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
        print(f"    [LlamaIndex] {len(nodes)} nodes após sentence splitting")

        # Garantir que o doc_id se propaga a cada node
        for node in nodes:
            if hasattr(node, "metadata"):
                node.metadata.setdefault("doc_id", node.metadata.get("doc_id", "?"))

        print(f"    [LlamaIndex] A indexar no Qdrant ({self.collection})...")
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
    def recuperar(self, query: str, top_k: int = 5) -> dict:
        """Só retrieval — sem geração, para avaliação de retrieval."""
        index = self._get_index()
        t0 = time.time()
        retriever = index.as_retriever(similarity_top_k=top_k)
        source_nodes = retriever.retrieve(query)
        tempo_retrieval = time.time() - t0

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
                "method": "llamaindex-denso",
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
        """
        Retrieval nativo do LlamaIndex (VectorIndexRetriever sobre os
        nodes produzidos pelo SentenceSplitter), seguido de geração
        directa via Groq.

        NOTA METODOLÓGICA: a síntese nativa (`response_mode="compact"`)
        é deliberadamente substituída por geração directa. O modo
        compact encadeia múltiplas chamadas de refinamento com um
        orçamento de contexto que o LlamaIndex decide internamente —
        o que impediria de equalizar contexto, latência e custo entre
        condições. O que continua a ser avaliado do LlamaIndex é o
        que distingue a framework: o node parsing e o retrieval.
        """
        index = self._get_index()

        # ── Retrieval (uma única vez) ────────────────────────
        t0 = time.time()
        retriever = index.as_retriever(similarity_top_k=top_k)
        source_nodes = retriever.retrieve(query)
        tempo_retrieval = time.time() - t0

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
                "method": "llamaindex-denso",
            })

        # ── Orçamento de contexto equalizado ─────────────────
        blocos = [
            f"--- FONTE ({m['tipo']} {m['numero']}) ---\n{c}"
            for c, m in zip(contextos, metadados)
        ]
        contexto_fmt, n_usados, n_tokens = montar_contexto(blocos)

        # ── Geração ──────────────────────────────────────────
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