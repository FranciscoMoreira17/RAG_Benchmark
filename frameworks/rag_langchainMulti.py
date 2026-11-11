"""
rag_langchainMulti.py - LangChain com o embedder escolhido na CLI
=================================================================
Junta rag_langchain.py (E5) e rag_langchainQwen.py (Qwen3) num só ficheiro.
O pipeline é o nativo do LangChain e é igual para todos os embedders:
  - RecursiveCharacterTextSplitter (1000 chars, overlap 200)
  - QdrantVectorStore (retrieval denso)
  - ChatGroq + LCEL chain (prompt | llm | parser)

Uso:
    python runner.py --framework langchain_multi --embedder e5
    python runner.py --framework langchain_multi --embedder qwen

Compatibilidade: com e5 e qwen, a collection, o nome da condição e o
fingerprint são IGUAIS aos dos ficheiros originais, por isso os índices
já construídos e os resultados existentes continuam a ser reaproveitados.
Um embedder novo (registado em embedders.py) segue a mesma regra do qwen:
    collection  benchmark_langchain_<chave>
    condição    langchain-<chave>
"""

import os
import time

from config import (
    qdrant_client,
    get_collection_name,
    SYSTEM_PROMPT,
    GEN_MODEL,
    QDRANT_URL,
    montar_contexto_detalhe,
    GEN_CANDIDATOS,
)
from runner import FrameworkBase
from dre_loader import carregar_corpus_dre, corpus_fingerprint
from embedders import get_embedder_spec, carregar_embedder

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class LangChainEmbeddings(Embeddings):
    """Adapta qualquer embedder do registo à interface do LangChain."""

    def __init__(self, chave: str):
        self.chave = chave

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return carregar_embedder(self.chave).embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return carregar_embedder(self.chave).embed_query(text)


class Framework(FrameworkBase):

    def __init__(self, embedder: str = "e5"):
        self.spec = get_embedder_spec(embedder)
        self.legado_e5 = (self.spec.chave == "e5")

        # Identidade da condição. O E5 mantém os nomes do rag_langchain.py
        # original; os restantes seguem a regra do rag_langchainQwen.py.
        if self.legado_e5:
            self.nome = "langchain"
            self.nome_run = "langchain"
            self.collection = get_collection_name("langchain")
            self._method = "langchain-denso"
        else:
            chave = self.spec.chave
            self.nome = f"langchain_{chave}"
            self.nome_run = f"langchain-{chave}"
            self.collection = f"benchmark_langchain_{chave}"
            self._method = f"langchain-{chave}-denso"

        self.embeddings = LangChainEmbeddings(self.spec.chave)
        # Igual aos ficheiros originais: o ChatGroq usa a temperatura por
        # omissão do langchain_groq (não passa GEN_TEMPERATURE).
        self.llm = ChatGroq(model=GEN_MODEL, api_key=os.environ.get("GROQ_API_KEY"))

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "CONTEXTO LEGAL:\n{contexto}\n\nPERGUNTA: {pergunta}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()
        self._vectorstore = None

    # ─────────────────────────────────────────────────────────
    # Identificação (fingerprint e snapshot dos resultados)
    # ─────────────────────────────────────────────────────────
    def config_ingestao(self) -> dict:
        if self.legado_e5:
            return {"pipeline": "langchain-recursive",
                    "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}
        return {"pipeline": f"langchain-recursive-{self.spec.chave}",
                "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP,
                "embedder": self.spec.tag, "dim": self.spec.dim}

    def descricao(self) -> dict:
        if self.legado_e5:
            return {}
        return {"embedder": self.spec.tag, "dim": self.spec.dim,
                "splitter": f"recursive-{CHUNK_SIZE}-{CHUNK_OVERLAP}"}

    # ─────────────────────────────────────────────────────────
    # Ingestão
    # ─────────────────────────────────────────────────────────
    def _get_vectorstore(self) -> QdrantVectorStore:
        if self._vectorstore is None:
            self._vectorstore = QdrantVectorStore(
                client=qdrant_client,
                collection_name=self.collection,
                embedding=self.embeddings,
            )
        return self._vectorstore

    def ingerir(self) -> dict:
        etiqueta = f"[LangChain-{self.spec.chave}]"
        docs = carregar_corpus_dre()
        print(f"    {etiqueta} {len(docs)} documentos DRE carregados")

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
        print(f"    {etiqueta} {len(chunks)} chunks após splitting "
              f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

        # force_recreate: a collection é criada com a dimensão do embedder
        # escolhido (o runner pode tê-la criado antes com a dimensão do E5).
        print(f"    {etiqueta} A indexar no Qdrant ({self.collection})...")
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
    def _pesquisar(self, query: str, k: int):
        vs = self._get_vectorstore()
        t0 = time.time()
        resultados = vs.similarity_search_with_score(query, k=k)
        tempo = time.time() - t0

        contextos, metadados = [], []
        for doc, score in resultados:
            contextos.append(doc.page_content)
            metadados.append({
                "doc_id": doc.metadata.get("doc_id", "?"),
                "titulo": doc.metadata.get("titulo", ""),
                "tipo": doc.metadata.get("tipo", ""),
                "numero": doc.metadata.get("numero", ""),
                "score": float(score),
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
        """Retrieval de GEN_CANDIDATOS → orçamento de contexto → chain LCEL."""
        contextos, metadados, tempo_retrieval = self._pesquisar(query, GEN_CANDIDATOS)

        blocos = [
            f"--- FONTE ({m['tipo']} {m['numero']}) ---\n{c}"
            for c, m in zip(contextos, metadados)
        ]
        contexto_formatado, n_usados, n_tokens, blocos_usados = montar_contexto_detalhe(blocos)

        t0 = time.time()
        resposta = self.chain.invoke({
            "contexto": contexto_formatado,
            "pergunta": query,
        })
        tempo_geracao = time.time() - t0

        return {
            "resposta": resposta,
            "contextos": blocos_usados,
            "metadados": metadados[:n_usados],
            "tempo_retrieval_s": tempo_retrieval,
            "tempo_geracao_s": tempo_geracao,
            "n_blocos_recuperados": len(blocos),
            "n_blocos_usados": n_usados,
            "tokens_contexto": n_tokens,
        }
