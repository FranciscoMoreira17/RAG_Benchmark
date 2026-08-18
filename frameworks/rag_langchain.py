"""
rag_langchain.py — Implementação LangChain para o Benchmark
==============================================================
Usa os componentes nativos do LangChain:
  - PyMuPDFLoader para ingestão de PDFs
  - RecursiveCharacterTextSplitter para chunking
  - QdrantVectorStore para armazenamento
  - ChatGoogleGenerativeAI para geração
  - LCEL chain (prompt | llm | parser)

A segmentação e o chunking são decisões DO LANGCHAIN — não usamos
a segmentação por artigo do OKF nem nenhuma lógica custom.
Isto garante que estamos a avaliar a framework, não o nosso código.
"""

import os
import time

from config import (
    embedding_model,
    qdrant_client,
    get_collection_name,
    montar_contexto,
    SYSTEM_PROMPT,
    GEN_MODEL,
    GEN_TEMPERATURE,
    QDRANT_URL,
    EMBEDDING_DIM,
)
from runner import FrameworkBase
from dre_loader import carregar_corpus_dre, corpus_fingerprint

# ── LangChain imports ────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore


# =====================================================================
# Adapter: E5Embedder → interface Embeddings do LangChain
# =====================================================================
class LangChainE5Embeddings(Embeddings):
    """
    Adapta o E5Embedder partilhado (config.py) à interface
    Embeddings do LangChain. Garante que os mesmos vetores são
    usados independentemente da framework.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embedding_model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return embedding_model.embed_query(text)


# =====================================================================
# Framework LangChain
# =====================================================================
class Framework(FrameworkBase):

    nome = "langchain"

    def __init__(self):
        self.collection = get_collection_name(self.nome)

        # Embedding adapter
        self.embeddings = LangChainE5Embeddings()

        # LLM
        #self.llm = ChatGoogleGenerativeAI(
        #    model=LLM_MODEL,
        #    temperature=LLM_TEMPERATURE,
        #)

        self.llm = ChatGroq(model=GEN_MODEL, api_key=os.environ.get("GROQ_API_KEY"))

        # Text splitter (decisão nativa do LangChain)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "CONTEXTO LEGAL:\n{contexto}\n\nPERGUNTA: {pergunta}"),
        ])

        # Chain LCEL
        self.chain = self.prompt | self.llm | StrOutputParser()

        # Vector store (inicializado após ingestão ou ao responder)
        self._vectorstore = None

    def config_ingestao(self) -> dict:
        return {"pipeline": "langchain-recursive", "chunk_size": 1000, "overlap": 200}

    def _get_vectorstore(self) -> QdrantVectorStore:
        """Obtém ou cria a referência ao vector store."""
        if self._vectorstore is None:
            self._vectorstore = QdrantVectorStore(
                client=qdrant_client,
                collection_name=self.collection,
                embedding=self.embeddings,
            )
        return self._vectorstore

    # ─────────────────────────────────────────────────────────
    # Ingestão
    # ─────────────────────────────────────────────────────────
    def ingerir(self) -> dict:
        """
        Pipeline nativo do LangChain sobre o corpus DRE:
          documento inteiro → RecursiveCharacterTextSplitter → Qdrant

        Cada chunk herda o doc_id (guid) do documento-pai — é o alvo
        de retrieval. O chunking é a decisão nativa do LangChain (1000
        chars, overlap 200); não se alinha à segmentação estrutural do
        OKF, porque é justamente essa diferença que o benchmark mede.
        """
        docs = carregar_corpus_dre()
        print(f"    [LangChain] {len(docs)} documentos DRE carregados")

        # Um Document LangChain por documento DRE, com guid nos metadados
        lc_docs = [
            Document(
                page_content=d["texto"],
                metadata={
                    "doc_id": d["id"],          # guid — alvo de retrieval
                    "titulo": d["titulo"],
                    "tipo": d["tipo"],
                    "numero": d["numero"],
                    "data_publicacao": d["data_publicacao"],
                },
            )
            for d in docs
        ]

        chunks = self.splitter.split_documents(lc_docs)
        print(f"    [LangChain] {len(chunks)} chunks após splitting "
              f"(chunk_size=1000, overlap=200)")

        print(f"    [LangChain] A indexar no Qdrant ({self.collection})...")
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

    # ─────────────────────────────────────────────────────────
    # Retrieval + Geração
    # ─────────────────────────────────────────────────────────
    def recuperar(self, query: str, top_k: int = 5) -> dict:
        """Só retrieval — sem geração, para a fase de avaliação de retrieval."""
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
                "method": "langchain-denso",
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
        Pipeline:
          1. similarity_search_with_score no Qdrant
          2. Formatar contexto
          3. Invocar chain LCEL
        """
        vs = self._get_vectorstore()

        # Retrieval
        t0 = time.time()
        resultados = vs.similarity_search_with_score(query, k=top_k)
        tempo_retrieval = time.time() - t0

        # Extrair contextos e metadados (alvo por guid)
        contextos = []
        metadados = []
        for doc, score in resultados:
            contextos.append(doc.page_content)
            metadados.append({
                "doc_id": doc.metadata.get("doc_id", "?"),
                "titulo": doc.metadata.get("titulo", ""),
                "tipo": doc.metadata.get("tipo", ""),
                "numero": doc.metadata.get("numero", ""),
                "score": float(score),
                "method": "langchain-denso",
            })

        blocos = [
            f"--- FONTE ({m['tipo']} {m['numero']}) ---\n{c}"
            for c, m in zip(contextos, metadados)
        ]
        contexto_formatado, n_usados, n_tokens = montar_contexto(blocos)

        # Geração
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