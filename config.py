"""
config.py — Configuração partilhada do Benchmark
==================================================
Tudo o que TEM de ser idêntico entre condições experimentais vive aqui:
  - Modelo de embedding (E5) e respectivos prefixos
  - Orçamento de contexto em tokens (equalização entre frameworks)
  - Prompt de sistema
  - Cliente Qdrant

Cada framework adapta estes componentes à sua interface, mas nunca
os substitui — é isso que garante que a variável em estudo é a
estratégia de indexação/retrieval, e não o embedder ou o parser.
"""

import os
import glob
import json
import hashlib
from dotenv import load_dotenv

load_dotenv()

# Constantes
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

DOCS_DIR = os.getenv("DOCS_DIR", "./documents")

# ── Modelos ──────────────────────────────────────────────────
GEN_MODEL = os.getenv("GEN_MODEL", "qwen/qwen3.8-27b")
GEN_PROVIDER = os.getenv("GEN_PROVIDER", "groq")
GEN_TEMPERATURE = 0
GEN_MAX_TOKENS = 1000

EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "openai/gpt-oss-20b")

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini/gemini-2.5-flash")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

CONTEXT_BUDGET_TOKENS = 4096     # fixo: equalização entre frameworks
GEN_CANDIDATOS = 20

# Rate limiting
SLEEP_ENTRE_QUERIES = float(os.getenv("SLEEP_ENTRE_QUERIES", "15"))
BM25_LANGUAGE = "portuguese"
BM25_MODEL = "Qdrant/bm25"

# Candidatos por ramo (denso / BM25) antes da fusão RRF
PREFETCH = 50

# ── Embedder Qwen3 (frameworks *Qwen) ────────────────────────
# QWEN_MODEL, QWEN_DIM e QWEN_TAG entram no fingerprint dos índices Qwen:
# mudar qualquer um deles obriga a re-ingerir essas collections.
QWEN_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
QWEN_DIM = 1024
QWEN_TAG = "qwen3-0.6b"

# Collection Qdrant de cada framework (chave = Framework.nome em minúsculas).
# Só langchain/llamaindex aceitam override pelo .env.
_COLLECTIONS = {
    # E5 (EMBEDDING_DIM)
    "langchain": os.getenv("QDRANT_COLLECTION_LANGCHAIN", "benchmark_langchain"),
    "llamaindex": os.getenv("QDRANT_COLLECTION_LLAMAINDEX", "benchmark_llamaindex"),
    "uls": "benchmark_hibrido",
    "estrutural": "benchmark_estrutural",
    "overlap": "benchmark_overlap",
    # Qwen3 (QWEN_DIM)
    "langchain_qwen": "benchmark_langchain_qwen",
    "llamaindex_qwen": "benchmark_llamaindex_qwen",
    "uls_qwen": "benchmark_hibrido_qwen_orig",
    "uls_qwen512": "benchmark_hibrido_qwen_512",
    "ulsv2_qwen": "benchmark_ulsv2_qwen",
    "documento_qwen": "benchmark_documento_qwen",
    "estrutural_qwen": "benchmark_estrutural_qwen",
    "overlap_qwen": "benchmark_overlap_qwen",
}


def get_collection_name(framework: str) -> str:
    framework = framework.lower().strip()
    if framework not in _COLLECTIONS:
        raise ValueError(
            f"Framework '{framework}' desconhecida. "
            f"Disponíveis: {list(_COLLECTIONS.keys())}"
        )
    return _COLLECTIONS[framework]



try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def contar_tokens(texto: str) -> int:
        return len(_enc.encode(texto, disallowed_special=()))

    TOKENIZER_NAME = "tiktoken/cl100k_base"
except Exception:
    def contar_tokens(texto: str) -> int:
        return max(1, len(texto) // 4)

    TOKENIZER_NAME = "heuristica/chars-div-4"




def _truncar_tokens(texto: str, max_tokens: int) -> str:
    """Maior prefixo de `texto` com <= max_tokens (pesquisa binária; serve para qualquer contador)."""
    if contar_tokens(texto) <= max_tokens:
        return texto
    lo, hi = 0, len(texto)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if contar_tokens(texto[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return texto[:lo]


def montar_contexto_detalhe(blocos: list[str], budget: int = CONTEXT_BUDGET_TOKENS,
                            min_resto: int = 50):
    """
    Preenche o orçamento por ordem de ranking. O primeiro bloco que não cabe
    é truncado ao espaço restante e o processo para.
    Devolve (contexto, n_blocos_usados, n_tokens, blocos_usados).
    """
    usados, total = [], 0
    for b in blocos:
        n = contar_tokens(b)
        if total + n <= budget:
            usados.append(b)
            total += n
            continue
        resto = budget - total
        if resto > min_resto:
            t = _truncar_tokens(b, resto)
            usados.append(t)
            total += contar_tokens(t)
        break
    return "\n\n".join(usados), len(usados), total, usados


def montar_contexto(blocos, budget=CONTEXT_BUDGET_TOKENS):
    """Compatibilidade: mesma assinatura de antes."""
    c, n, t, _ = montar_contexto_detalhe(blocos, budget)
    return c, n, t


# Phoenix Tracing
def iniciar_tracing(projeto: str = "rag-benchmark"):

    try:
        from phoenix.otel import register
        register(
            project_name=projeto,
            endpoint=os.getenv("PHOENIX_GRPC_ENDPOINT", "http://localhost:4317"),
            auto_instrument=True,
        )
        print(f"  [PHOENIX] Tracing ativo → projeto '{projeto}'")
    except Exception as e:
        print(f"  [PHOENIX] Não disponível: {e}")


# =====================================================================
# Qdrant Client
# =====================================================================
from qdrant_client import QdrantClient

qdrant_client = QdrantClient(url=QDRANT_URL)


def garantir_collection(nome: str, recriar: bool = False):
    """Cria a collection se não existir. Se recriar=True, apaga primeiro."""
    from qdrant_client.models import VectorParams, Distance

    existe = qdrant_client.collection_exists(nome)
    if existe and recriar:
        qdrant_client.delete_collection(nome)
        existe = False
    if not existe:
        qdrant_client.create_collection(
            collection_name=nome,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


# =====================================================================
# Embedding Model (E5 com prefixos) 
from sentence_transformers import SentenceTransformer


class E5Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        import torch, os
        torch.set_num_threads(os.cpu_count() or 8)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"  [E5] Modelo: {model_name} | dim: {self.dim}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        vectors = self.model.encode(
            prefixed, normalize_embeddings=True,
            show_progress_bar=True, batch_size=64,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:          # ← este falta
        vector = self.model.encode(
            [f"query: {text}"], normalize_embeddings=True,
        )
        return vector[0].tolist()


embedding_model = E5Embedder()


# =====================================================================
# Clientes LLM
from openai import OpenAI


def get_groq_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url=GROQ_BASE_URL,
    )


def gerar_resposta(prompt: str, system_prompt: str = "") -> str:
    """Geração via Groq (idêntica em todas as condições)."""
    client = get_groq_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=messages,
        temperature=GEN_TEMPERATURE,
        max_tokens=GEN_MAX_TOKENS,
    )
    return resp.choices[0].message.content


SYSTEM_PROMPT = (
    "És um assistente jurídico especializado em legislação portuguesa. "
    "Responde EXCLUSIVAMENTE com base no contexto fornecido. "
    "Se a informação não estiver no contexto, diz explicitamente que "
    "não encontraste essa informação nos documentos disponíveis. "
    "Responde de forma direta e concisa, numa ou duas frases, sem preâmbulos, "
    "indo diretamente ao facto perguntado e usando a terminologia do documento. "
    "Quando a informação provier de um articulado legal, indica o artigo e o "
    "diploma; quando provier de um anúncio, aviso ou outro ato administrativo, "
    "identifica o documento-fonte."
)


# =====================================================================
# Utilitários de ficheiros
# =====================================================================
def listar_pdfs() -> list[str]:
    pdfs = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    if not pdfs:
        raise FileNotFoundError(f"Nenhum PDF em '{DOCS_DIR}/'")
    return pdfs


# =====================================================================
# Identificador do parser (entra no fingerprint dos índices — não alterar)
# =====================================================================
PARSER_NAME = "docling+tesseract-por"


# =====================================================================
# Fingerprint da configuração de ingestão
# =====================================================================
def fingerprint_ingestao(extra: dict | None = None) -> str:

    base = {
        "parser": PARSER_NAME,
        "embedding": EMBEDDING_MODEL_NAME,
        "dim": EMBEDDING_DIM,
    }
    if extra:
        base.update(extra)
    payload = json.dumps(base, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def snapshot_config(extra: dict | None = None) -> dict:
    """Configuração real em runtime, para gravar nos resultados."""
    cfg = {
        "gen_model": GEN_MODEL,
        "gen_provider": GEN_PROVIDER,
        "gen_temperature": GEN_TEMPERATURE,
        "extract_model": EXTRACT_MODEL,
        "judge_model": JUDGE_MODEL,
        "embedding": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "parser": PARSER_NAME,
        "tokenizer_orcamento": TOKENIZER_NAME,
        "context_budget_tokens": CONTEXT_BUDGET_TOKENS,
        "vector_db": "qdrant",
    }
    if extra:
        cfg.update(extra)
    return cfg
