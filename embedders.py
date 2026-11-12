"""
embedders.py — Registo dos modelos de embedding
================================================
Cada entrada de EMBEDDERS é um modelo que pode ser escolhido na CLI:

    python runner.py --framework langchain_multi --embedder qwen

Para acrescentar um modelo novo:
  1. Escrever uma função-fábrica que devolva um objeto com
         embed_documents(list[str]) -> list[list[float]]
         embed_query(str) -> list[float]
     (a mesma interface do E5Embedder e do Qwen3Embedder).
  2. Registar uma EmbedderSpec em EMBEDDERS.

A `tag` entra no fingerprint do índice e no nome da condição/collection,
por isso tem de ser única por modelo e não deve mudar depois de ingerir.
"""

from dataclasses import dataclass
from typing import Callable

from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM, QWEN_MODEL, QWEN_DIM, QWEN_TAG


@dataclass(frozen=True)
class EmbedderSpec:
    chave: str                      # valor de --embedder
    tag: str                        # etiqueta curta (fingerprint, descrição)
    modelo: str                     # identificador do modelo (HuggingFace)
    dim: int                        # dimensão dos vetores densos
    fabrica: Callable[[], object]   # cria o embedder (só chamada quando é usado)


# ── Fábricas (import tardio: o modelo só é carregado se for escolhido) ──
def _e5():
    from config import embedding_model
    return embedding_model


def _qwen():
    return Qwen3Embedder()


# ── Qwen3-Embedding ─────────────────────────────────────────
# Cópia exata da classe de frameworks/old/rag_UnitLevelSegmentationQwen.py
# (mesmos parâmetros ⇒ mesmos vetores dos índices *_qwen já construídos).
# As condições ULS512 / ULSv2 / documento mantêm a sua própria cópia.
QWEN_MAX_SEQ = 2048
QWEN_BATCH = 4


class Qwen3Embedder:

    def __init__(self, model_name: str = QWEN_MODEL, max_seq: int = QWEN_MAX_SEQ):
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


EMBEDDERS: dict[str, EmbedderSpec] = {
    "e5": EmbedderSpec("e5", "e5", EMBEDDING_MODEL_NAME, EMBEDDING_DIM, _e5),
    "qwen": EmbedderSpec("qwen", QWEN_TAG, QWEN_MODEL, QWEN_DIM, _qwen),
    # Terceiro modelo — exemplo:
    # "bge": EmbedderSpec("bge", "bge-m3", "BAAI/bge-m3", 1024, _bge),
}


_cache: dict[str, object] = {}


def get_embedder_spec(chave: str) -> EmbedderSpec:
    if chave not in EMBEDDERS:
        raise ValueError(f"Embedder '{chave}' desconhecido. Disponíveis: {list(EMBEDDERS)}")
    return EMBEDDERS[chave]


def carregar_embedder(chave: str):
    """Instância do embedder (criada uma vez por processo)."""
    if chave not in _cache:
        _cache[chave] = get_embedder_spec(chave).fabrica()
    return _cache[chave]
