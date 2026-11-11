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
    from frameworks.rag_UnitLevelSegmentationQwen import _get_embedder
    return _get_embedder()


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
