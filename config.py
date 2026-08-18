"""
config.py — Configuração partilhada do Benchmark
==================================================
Tudo o que TEM de ser idêntico entre condições experimentais vive aqui:
  - Modelo de embedding (E5) e respectivos prefixos
  - Orçamento de contexto em tokens (equalização entre frameworks)
  - Prompt de sistema
  - Extracção de texto (Docling + OCR selectivo)
  - Cliente Qdrant

Cada framework adapta estes componentes à sua interface, mas nunca
os substitui — é isso que garante que a variável em estudo é a
estratégia de indexação/retrieval, e não o embedder ou o parser.
"""

import os
import re
import glob
import json
import hashlib
from dotenv import load_dotenv

load_dotenv()

# Validação
_REQUIRED = ["GOOGLE_API_KEY"]
for var in _REQUIRED:
    if var not in os.environ:
        raise EnvironmentError(f"Variável '{var}' não definida no .env")

# Constantes
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

DOCS_DIR = os.getenv("DOCS_DIR", "./documents")

# ── Modelos ──────────────────────────────────────────────────
GEN_MODEL = os.getenv("GEN_MODEL", "llama-3.3-70b-versatile")
GEN_PROVIDER = os.getenv("GEN_PROVIDER", "groq")
GEN_TEMPERATURE = 0

EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "llama-3.1-8b-instant")

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini/gemini-2.5-flash")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

CONTEXT_BUDGET_TOKENS = int(os.getenv("CONTEXT_BUDGET_TOKENS", "6000"))

# Rate limiting
SLEEP_ENTRE_QUERIES = float(os.getenv("SLEEP_ENTRE_QUERIES", "15"))


_COLLECTIONS = {
    "langchain": os.getenv("QDRANT_COLLECTION_LANGCHAIN", "benchmark_langchain"),
    "llamaindex": os.getenv("QDRANT_COLLECTION_LLAMAINDEX", "benchmark_llamaindex"),
    "hibrido": os.getenv("QDRANT_COLLECTION_HIBRIDO", "benchmark_hibrido"),
    "estrutural": os.getenv("QDRANT_COLLECTION_ESTRUTURAL", "benchmark_estrutural"),
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


def montar_contexto(
    blocos: list[str],
    budget: int = CONTEXT_BUDGET_TOKENS,
) -> tuple[str, int, int]:
    """
    Preenche o orçamento de tokens por ordem de ranking.

    Blocos que não cabem inteiros são descartados (não truncados),
    para não entregar ao gerador artigos cortados a meio — o que
    penalizaria injustamente as condições com chunks grandes.

    Devolve (contexto_formatado, n_blocos_usados, n_tokens_usados).
    """
    usados, total = [], 0
    for b in blocos:
        n = contar_tokens(b)
        if total + n > budget:
            continue
        usados.append(b)
        total += n
    return "\n\n".join(usados), len(usados), total


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


from google import genai

google_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


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
        max_tokens=1000,
    )
    return resp.choices[0].message.content


SYSTEM_PROMPT = (
    "És um assistente jurídico especializado em legislação portuguesa. "
    "Responde EXCLUSIVAMENTE com base no contexto fornecido. "
    "Se a informação não estiver no contexto, diz explicitamente que "
    "não encontraste essa informação nos documentos disponíveis. "
    "Cita sempre o artigo e diploma de onde retiras a informação."
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
# Extracção de texto — Docling com OCR selectivo
# =====================================================================
#from docling.datamodel.base_models import InputFormat
#from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
#from docling.document_converter import DocumentConverter, PdfFormatOption

PARSER_NAME = "docling+tesseract-por"
PARSER_VERSAO = "v2"          
MIN_CHARS_POR_PAGINA = 120    
MIN_RACIO_PAGINAS_COM_TEXTO = 0.95

_CACHE_DIR = os.path.join(DOCS_DIR, ".cache_texto")
_RELATORIO_EXTRACCAO = os.path.join(DOCS_DIR, ".relatorio_extraccao.json")


# ##def _criar_converter(force_ocr: bool = False):
#     opts = PdfPipelineOptions()
#     opts.do_ocr = True
#     opts.ocr_options = TesseractCliOcrOptions(
#         tesseract_cmd=os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
#         lang=["por"],
#         force_full_page_ocr=force_ocr,
#     )
#     return DocumentConverter(
#         format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
#     )


# #_converter_normal = None
# #_converter_ocr = None


# ##def _diagnosticar_paginas(pdf_path: str) -> dict:
#     """Densidade de texto nativo por página."""
#     import pymupdf
#     doc = pymupdf.open(pdf_path)
#     chars = [len(p.get_text().strip()) for p in doc]
#     doc.close()
#     n = len(chars) or 1
#     com_texto = sum(1 for c in chars if c >= MIN_CHARS_POR_PAGINA)
#     return {
#         "n_paginas": len(chars),
#         "paginas_com_texto": com_texto,
#         "racio": com_texto / n,
#         "chars_por_pagina": sum(chars) / n,
#         "paginas_vazias": [i + 1 for i, c in enumerate(chars) if c < MIN_CHARS_POR_PAGINA],
#     }


# ##def extrair_texto_docs(pdf_path: str, usar_cache: bool = True) -> str:
#     """
#     PDF → Markdown via Docling.

#     A decisão de OCR é tomada por densidade de texto, não por
#     presença. Um documento onde uma parte das páginas está
#     digitalizada vai por OCR completo: é preferível assumir o custo
#     e a degradação de OCR em páginas digitais do que perder
#     silenciosamente as páginas em imagem.

#     Após a conversão há uma verificação de sanidade: se o resultado
#     for anormalmente curto face ao número de páginas, repete-se com
#     OCR forçado. Sem esta verificação, uma falha de extracção
#     propaga-se para o índice sem deixar rasto.
#     """
#     global _converter_normal, _converter_ocr

#     os.makedirs(_CACHE_DIR, exist_ok=True)

#     with open(pdf_path, "rb") as f:
#         h_ficheiro = hashlib.md5(f.read()).hexdigest()[:12]
#     h_config = hashlib.md5(
#         f"{PARSER_NAME}|{PARSER_VERSAO}|{MIN_CHARS_POR_PAGINA}|"
#         f"{MIN_RACIO_PAGINAS_COM_TEXTO}".encode()
#     ).hexdigest()[:8]

#     nome = os.path.basename(pdf_path)
#     cache_path = os.path.join(_CACHE_DIR, f"{nome}.{h_ficheiro}.{h_config}.md")

#     if usar_cache and os.path.exists(cache_path):
#         with open(cache_path, "r", encoding="utf-8") as f:
#             md = f.read()
#         print(f"      [Docling] {nome}: {len(md)} chars (cache)")
#         return md

#     diag = _diagnosticar_paginas(pdf_path)
#     forcar_ocr = diag["racio"] < MIN_RACIO_PAGINAS_COM_TEXTO

#     def _converter(forcar: bool) -> str:
#         global _converter_normal, _converter_ocr
#         if forcar:
#             if _converter_ocr is None:
#                 _converter_ocr = _criar_converter(force_ocr=True)
#             conv = _converter_ocr
#         else:
#             if _converter_normal is None:
#                 _converter_normal = _criar_converter(force_ocr=False)
#             conv = _converter_normal
#         texto = conv.convert(pdf_path).document.export_to_markdown()
#         return re.sub(r"<!--\s*image\s*-->", "", texto).strip()

#     md = _converter(forcar_ocr)

#     # ── Verificação de sanidade ──────────────────────────────
#     esperado_min = MIN_CHARS_POR_PAGINA * diag["n_paginas"] * 0.5
#     reprocessado = False
#     if not forcar_ocr and len(md) < esperado_min:
#         print(f"      [Docling] {nome}: saída curta ({len(md)} chars) — a repetir com OCR")
#         md = _converter(True)
#         forcar_ocr, reprocessado = True, True

#     if len(md) < MIN_CHARS_POR_PAGINA:
#         raise RuntimeError(
#             f"Extracção falhou para {nome}: {len(md)} chars após OCR. "
#             f"Verifica o Tesseract e o idioma 'por'."
#         )

#     with open(cache_path, "w", encoding="utf-8") as f:
#         f.write(md)

#     _registar_extraccao(nome, {**diag, "chars_extraidos": len(md),
#                                "ocr_forcado": forcar_ocr,
#                                "reprocessado": reprocessado})

#     print(f"      [Docling] {nome}: {len(md)} chars | {diag['n_paginas']} pág "
#           f"| texto nativo {diag['racio']:.0%} | OCR={'total' if forcar_ocr else 'selectivo'}")
#     return md




def _registar_extraccao(nome: str, dados: dict):
    """Relatório de extracção — evidência da qualidade do parsing."""
    reg = {}
    if os.path.exists(_RELATORIO_EXTRACCAO):
        with open(_RELATORIO_EXTRACCAO, "r", encoding="utf-8") as f:
            reg = json.load(f)
    reg[nome] = dados
    with open(_RELATORIO_EXTRACCAO, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


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