"""
testar_embedding_qwen.py — Tempo de embedding Qwen3 sobre UnitLevelSegmentation
====================================================================================
Com TETO DE TOKENS na segmentacao (nenhum chunk excede --max-seq) e as tres
correcoes que tornam o Qwen3 viavel numa GPU local:

  - float16            : dobra a velocidade no T4 e na RTX 2060 (Tensor Cores).
  - attn_implementation="eager" : contorna o bug "CUDA driver error: device not
                         ready" do kernel SDPA com CUDA 13.x em GPUs Turing.
  - teto de tokens     : chunks homogeneos <= max-seq, cabem em 8 GB de VRAM.

    python testar_embedding_qwen.py --device cuda                 # RTX 2060 local
    python testar_embedding_qwen.py --device cuda --batch-size 8
    python testar_embedding_qwen.py --device cuda --teto 1024
    python testar_embedding_qwen.py --device cuda --attn sdpa     # tentar kernel rapido

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
    ap.add_argument("--batch-size", type=int, default=8,
                    help="8 cabe nos 8 GB da 2060; sobe/desce conforme a VRAM")
    ap.add_argument("--max-seq", type=int, default=2048,
                    help="janela e TETO de tokens (iguais: nenhum chunk excede a janela)")
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"],
                    help="'eager' (compativel; default) ou 'sdpa' (rapido, pode falhar em CUDA 13/Turing)")
    args = ap.parse_args()

    from dre_loader import carregar_corpus_dre
    # NOTA: o teu import aponta para dre_segmentador_qwen; mantido tal como o tens.
    from dre_segmentador_v2 import segmentar_corpus

    docs = carregar_corpus_dre(relatorio=False)
    amostra = random.Random(SEED).sample(docs, min(args.n, len(docs)))
    print(f"Corpus: {len(docs)} docs  |  amostra: {len(amostra)} docs (seed={SEED})\n")

    print(f"A carregar {MODELO} em {args.device}... (1.a vez descarrega pesos)")
    t0 = time.time()
    import torch

    # float16 so no GPU (em CPU nao acelera e pode dar erro)
    usa_fp16 = (args.device == "cuda")
    model_kwargs = {"attn_implementation": args.attn}
    if usa_fp16:
        model_kwargs["torch_dtype"] = torch.float16

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODELO, device=args.device, model_kwargs=model_kwargs)
    model.max_seq_length = args.max_seq
    model.tokenizer.model_max_length = args.max_seq

    dtype = next(model.parameters()).dtype
    dev = next(model.parameters()).device
    print(f"  modelo carregado em {time.time()-t0:.1f}s  |  device={dev}  "
          f"|  dtype={dtype}  |  attn={args.attn}  |  dim="
          f"{model.get_sentence_embedding_dimension()}\n")

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