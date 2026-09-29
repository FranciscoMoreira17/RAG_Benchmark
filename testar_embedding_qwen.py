"""
testar_embedding_unitlevel.py — Tempo de embedding Qwen3 sobre UnitLevelSegmentation
====================================================================================
Agora com TETO DE TOKENS na segmentacao: nenhum chunk excede --max-seq, por isso
nao ha outliers gigantes a rebentar a memoria e o batch pode ser maior.

O teto e aplicado com o TOKENIZER REAL do Qwen3 (nao a estimativa por chars),
para o corte cair no sitio certo.

    python testar_embedding_unitlevel.py --device cuda                    # T4 Colab
    python testar_embedding_unitlevel.py --device cuda --batch-size 16
    python testar_embedding_unitlevel.py --device cuda --teto 1024

Requisitos:
    pip install -U "sentence-transformers>=2.7.0" "transformers>=4.51.0"
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import random
import statistics
import time

MODELO = "Qwen/Qwen3-Embedding-0.6B"
LIMITE_512 = 512
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="documentos aleatorios")
    ap.add_argument("--device", default="cpu", help="cpu ou cuda")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="sobe se sobrar memoria; desce se der OOM")
    ap.add_argument("--max-seq", type=int, default=2048,
                    help="janela e TETO de tokens (iguais: nenhum chunk excede a janela)")
    args = ap.parse_args()

    from dre_loader import carregar_corpus_dre
    from dre_segmentador import segmentar_corpus

    docs = carregar_corpus_dre(relatorio=False)
    amostra = random.Random(SEED).sample(docs, min(args.n, len(docs)))
    print(f"Corpus: {len(docs)} docs  |  amostra: {len(amostra)} docs (seed={SEED})\n")

    # Carregar modelo primeiro: precisamos do tokenizer para o teto.
    print(f"A carregar {MODELO} em {args.device}... (1.a vez descarrega pesos)")
    t0 = time.time()
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODELO, device=args.device)
    model.max_seq_length = args.max_seq
    model.tokenizer.model_max_length = args.max_seq
    print(f"  modelo carregado em {time.time()-t0:.1f}s  |  dim="
          f"{model.get_sentence_embedding_dimension()}\n")

    # contador de tokens REAL (o tokenizer do Qwen3), injetado no segmentador
    def contar_tokens(texto):
        return len(model.tokenizer(texto, add_special_tokens=True,
                                   truncation=False)["input_ids"])

    print(f"A segmentar com UnitLevelSegmentation (teto={args.max_seq} tokens)...")
    segmentos, rel = segmentar_corpus(amostra, teto_tokens=args.max_seq,
                                      contar_tokens=contar_tokens)
    textos = [s["texto"] for s in segmentos]
    print(f"  -> {len(textos)} chunks a partir de {len(amostra)} documentos\n")

    print("A contar tokens dos chunks (confirmar que o teto pegou)...")
    n_tok = [contar_tokens(t) for t in textos]
    acima_512 = sum(1 for n in n_tok if n > LIMITE_512)
    acima_teto = sum(1 for n in n_tok if n > args.max_seq)
    print(f"  tokens/chunk: media {statistics.mean(n_tok):.0f}, "
          f"mediana {statistics.median(n_tok):.0f}, max {max(n_tok)}")
    print(f"  chunks > 512 tokens : {acima_512}/{len(n_tok)} ({acima_512/len(n_tok):.0%})")
    print(f"  chunks > teto ({args.max_seq}): {acima_teto}/{len(n_tok)} "
          f"({acima_teto/len(n_tok):.0%})  <- deve ser 0%\n")

    n_batches = (len(textos) + args.batch_size - 1) // args.batch_size
    print(f"Embedding de {len(textos)} chunks em {n_batches} batches "
          f"de {args.batch_size}...")

    print("  aquecimento (1 chunk)...")
    with torch.no_grad():
        _ = model.encode(textos[:1], show_progress_bar=False, normalize_embeddings=True)

    t0 = time.time()
    with torch.no_grad():
        _ = model.encode(textos, batch_size=args.batch_size,
                         show_progress_bar=True, normalize_embeddings=True)
    dt = time.time() - t0

    if args.device == "cuda":
        torch.cuda.empty_cache()

    print("\n=== RESULTADO ===")
    print(f"chunks vetorizados: {len(textos)}")
    print(f"tempo total de embedding: {dt:.1f}s")
    print(f"tempo por chunk: {dt/len(textos)*1000:.0f} ms")
    print(f"velocidade: {len(textos)/dt:.1f} chunks/s")

    fator = len(docs) / len(amostra)
    print(f"\nExtrapolacao para os {len(docs)} docs (x {fator:.1f}):")
    print(f"  ~ {len(textos)*fator:,.0f} chunks  |  "
          f"~ {dt*fator/60:.1f} min  ({dt*fator/3600:.2f} h)")
    print("  (com o teto, os chunks sao homogeneos -> extrapolacao mais fiavel)")


if __name__ == "__main__":
    main()