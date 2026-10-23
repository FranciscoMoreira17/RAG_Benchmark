"""
atualizar_bm25.py — recalcula SÓ o vetor esparso "bm25" de uma collection
==========================================================================
Os vetores densos (E5/Qwen) NÃO são tocados: usa-se update_vectors, que
substitui apenas o vetor com nome "bm25" de cada ponto.

O texto vem do payload:
  - "texto_embebido" se existir (estrutural, ULS v2: o BM25 foi indexado
    sobre cabeçalho + texto), senão
  - "texto" (ULS, documento inteiro, tamanho fixo/overlap).
Assim o BM25 novo é calculado exatamente sobre o mesmo texto do antigo,
mudando apenas a língua (stemmer + stopwords).

    python atualizar_bm25.py --collection benchmark_hibrido_qwen_orig
    python atualizar_bm25.py --collection benchmark_estrutural_qwen --snapshot
    python atualizar_bm25.py --listar        # mostra as collections com vetor bm25

Depois: alterar o _get_sparse dos frameworks para language="portuguese"
e voltar a correr SÓ as variantes esparso e híbrido.
"""
import argparse
import time

from qdrant_client import models as qm
from fastembed import SparseTextEmbedding

from config import qdrant_client

BM25_MODEL = "Qdrant/bm25"
LOTE = 256


def tem_bm25(nome: str) -> bool:
    info = qdrant_client.get_collection(nome)
    esparsos = info.config.params.sparse_vectors or {}
    return "bm25" in esparsos


def listar():
    for c in qdrant_client.get_collections().collections:
        if tem_bm25(c.name):
            n = qdrant_client.get_collection(c.name).points_count
            print(f"  {c.name:<40} {n:>8} pontos")


def atualizar(nome: str, lingua: str, snapshot: bool):
    if not tem_bm25(nome):
        raise SystemExit(f"'{nome}' não tem vetor esparso 'bm25'.")

    if snapshot:
        s = qdrant_client.create_snapshot(collection_name=nome)
        print(f"  snapshot criado: {s.name}")

    sparse = SparseTextEmbedding(model_name=BM25_MODEL, language=lingua)
    total = qdrant_client.get_collection(nome).points_count
    print(f"  [{nome}] {total} pontos | BM25 language={lingua}")

    feitos, campo_usado, offset, t0 = 0, None, None, time.time()
    while True:
        pontos, offset = qdrant_client.scroll(
            collection_name=nome, limit=LOTE, offset=offset,
            with_payload=["texto", "texto_embebido"], with_vectors=False,
        )
        if not pontos:
            break

        textos = []
        for p in pontos:
            pl = p.payload or {}
            campo = "texto_embebido" if pl.get("texto_embebido") else "texto"
            campo_usado = campo_usado or campo
            if campo != campo_usado:
                raise SystemExit("Mistura de 'texto' e 'texto_embebido' na mesma collection — verificar.")
            textos.append(pl.get(campo) or "")

        esparsos = list(sparse.embed(textos, batch_size=64))
        qdrant_client.update_vectors(
            collection_name=nome,
            points=[
                qm.PointVectors(
                    id=p.id,
                    vector={"bm25": qm.SparseVector(indices=sv.indices.tolist(),
                                                     values=sv.values.tolist())},
                )
                for p, sv in zip(pontos, esparsos)
            ],
            wait=True,
        )
        feitos += len(pontos)
        print(f"    {feitos}/{total}  ({time.time() - t0:.0f}s)", end="\r")
        if offset is None:
            break

    print(f"\n  [{nome}] concluído: {feitos} pontos | campo de texto: {campo_usado}")
    if feitos != total:
        print(f"  [AVISO] atualizados {feitos} de {total} pontos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection")
    ap.add_argument("--language", default="portuguese")
    ap.add_argument("--snapshot", action="store_true",
                    help="cria um snapshot da collection antes de alterar")
    ap.add_argument("--listar", action="store_true")
    args = ap.parse_args()

    if args.listar or not args.collection:
        listar()
        return
    atualizar(args.collection, args.language, args.snapshot)


if __name__ == "__main__":
    main()