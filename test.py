from qdrant_client import models as qm
from fastembed import SparseTextEmbedding
from config import qdrant_client

COL = "benchmark_hibrido_qwen_orig"   # a collection do ULS-Qwen

# 1. Em que língua está o BM25 guardado no Qdrant?
p = qdrant_client.scroll(COL, limit=1, with_payload=["texto", "texto_embebido"],
                         with_vectors=["bm25"])[0][0]
texto = p.payload.get("texto_embebido") or p.payload["texto"]
guardado = set(p.vector["bm25"].indices)
for lang in ("portuguese", "english"):
    sv = next(iter(SparseTextEmbedding("Qdrant/bm25", language=lang).embed([texto])))
    print(f"documento {lang:<10}: igual ao guardado = {set(sv.indices.tolist()) == guardado}")

# 2. Em que língua está o framework a processar as perguntas?
from frameworks.rag_UnitLevelSegmentationQwen import Framework
fw = Framework("esparso")
print("perguntas:", fw._get_sparse().model.language)