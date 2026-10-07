"""
rag_langchainQwen.py - LangChain com embedder Qwen3
====================================================================================
"""

import os
import time

from config import (
    qdrant_client,
    montar_contexto,
    SYSTEM_PROMPT,
    GEN_MODEL,
    QDRANT_URL,
)
from runner import FrameworkBase
from dre_loader import carregar_corpus_dre, corpus_fingerprint

from frameworks.rag_UnitLevelSegmentationQwen import _get_embedder

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

COLLECTION = "benchmark_langchain_qwen"     


# =====================================================================
# Adapter: Qwen3Embedder -> interface Embeddings do LangChain
# =====================================================================
class LangChainQwenEmbeddings(Embeddings):
    """Adapta o Qwen3Embedder partilhado à interface do LangChain."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _get_embedder().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return _get_embedder().embed_query(text)


class Framework(FrameworkBase):

    nome = "langchain_qwen"

    def __init__(self):
        self.collection = COLLECTION
        self.embeddings = LangChainQwenEmbeddings()
        self.nome_run = "langchain-qwen"
        self.llm = ChatGroq(model=GEN_MODEL, api_key=os.environ.get("GROQ_API_KEY"))

        # MESMO splitter do langchain-E5 — sem teto, chunking nativo intacto.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "CONTEXTO LEGAL:\n{contexto}\n\nPERGUNTA: {pergunta}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()
        self._vectorstore = None

    def config_ingestao(self) -> dict:
        # embedder no fingerprint -> índice distinto do langchain-E5
        return {"pipeline": "langchain-recursive-qwen", "chunk_size": 1000,
                "overlap": 200, "embedder": "qwen3-0.6b", "dim": 1024}

    def descricao(self) -> dict:
        return {"embedder": "qwen3-0.6b", "dim": 1024, "splitter": "recursive-1000-200"}

    def _get_vectorstore(self) -> QdrantVectorStore:
        if self._vectorstore is None:
            self._vectorstore = QdrantVectorStore(
                client=qdrant_client,
                collection_name=self.collection,
                embedding=self.embeddings,
            )
        return self._vectorstore

    def ingerir(self) -> dict:
        docs = carregar_corpus_dre()
        print(f"    [LangChain-Qwen] {len(docs)} documentos DRE carregados")

        lc_docs = [
            Document(
                page_content=d["texto"],
                metadata={
                    "doc_id": d["id"],
                    "titulo": d["titulo"],
                    "tipo": d["tipo"],
                    "numero": d["numero"],
                    "data_publicacao": d["data_publicacao"],
                },
            )
            for d in docs
        ]

        chunks = self.splitter.split_documents(lc_docs)
        print(f"    [LangChain-Qwen] {len(chunks)} chunks após splitting "
              f"(chunk_size=1000, overlap=200)")

        print(f"    [LangChain-Qwen] A indexar no Qdrant ({self.collection})...")
        self._vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            url=QDRANT_URL,
            collection_name=self.collection,
            force_recreate=True,
        )

        return {
            "n_chunks": len(chunks),
            "n_documentos": len(docs),
            "corpus_fingerprint": corpus_fingerprint(docs),
        }

    def recuperar(self, query: str, top_k: int = 5) -> dict:
        vs = self._get_vectorstore()
        t0 = time.time()
        resultados = vs.similarity_search_with_score(query, k=top_k)
        tempo_retrieval = time.time() - t0

        contextos, metadados = [], []
        for doc, score in resultados:
            contextos.append(doc.page_content)
            metadados.append({
                "doc_id": doc.metadata.get("doc_id", "?"),
                "titulo": doc.metadata.get("titulo", ""),
                "tipo": doc.metadata.get("tipo", ""),
                "numero": doc.metadata.get("numero", ""),
                "score": float(score),
                "method": "langchain-qwen-denso",
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
        vs = self._get_vectorstore()
        t0 = time.time()
        resultados = vs.similarity_search_with_score(query, k=top_k)
        tempo_retrieval = time.time() - t0

        contextos, metadados = [], []
        for doc, score in resultados:
            contextos.append(doc.page_content)
            metadados.append({
                "doc_id": doc.metadata.get("doc_id", "?"),
                "titulo": doc.metadata.get("titulo", ""),
                "tipo": doc.metadata.get("tipo", ""),
                "numero": doc.metadata.get("numero", ""),
                "score": float(score),
                "method": "langchain-qwen-denso",
            })

        blocos = [
            f"--- FONTE ({m['tipo']} {m['numero']}) ---\n{c}"
            for c, m in zip(contextos, metadados)
        ]
        contexto_formatado, n_usados, n_tokens = montar_contexto(blocos)

        t0 = time.time()
        resposta = self.chain.invoke({
            "contexto": contexto_formatado,
            "pergunta": query,
        })
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