"""
testar_embedding_unitlevel.py — Tempo de embedding Qwen3 sobre UnitLevelSegmentation
====================================================================================
Faz SÓ o que interessa medir:

  1. Pega em 100 documentos aleatórios do corpus DRE.
  2. Segmenta-os com a estratégia UnitLevelSegmentation (segmentar_corpus).
  3. Conta quantos chunks se formaram e quantos passam de 512 tokens.
  4. Faz o embedding desses chunks com o Qwen3-0.6B (janela 8192).
  5. Mostra os batches a correr e o tempo total.

NÃO escreve no Qdrant. NÃO faz esparso. Só mede o embedding denso.

    python testar_embedding_unitlevel.py                 # 100 docs, CPU
    python testar_embedding_unitlevel.py --device cuda   # no Colab com GPU
    python testar_embedding_unitlevel.py --n 100 --batch-size 16

Requisitos:
    pip install -U "sentence-transformers>=2.7.0" "transformers>=4.51.0"
"""
import argparse
import random
import statistics
import time

MODELO = "Qwen/Qwen3-Embedding-0.6B"
LIMITE_512 = 512          # janela do E5-small — referência de truncagem
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="documentos aleatórios")
    ap.add_argument("--device", default="cpu", help="cpu ou cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=8192, help="janela máx. em tokens")
    args = ap.parse_args()

    from dre_loader import carregar_corpus_dre
    from dre_segmentador import segmentar_corpus

    # ── 1. Amostra de 100 documentos ────────────────────────────────
    docs = carregar_corpus_dre(relatorio=False)
    amostra = random.Random(SEED).sample(docs, min(args.n, len(docs)))
    print(f"Corpus: {len(docs)} docs  |  amostra: {len(amostra)} docs (seed={SEED})\n")

    # ── 2. Segmentação UnitLevelSegmentation ────────────────────────
    print("A segmentar com UnitLevelSegmentation...")
    segmentos, rel = segmentar_corpus(amostra)
    textos = [s["texto"] for s in segmentos]
    print(f"  → {len(textos)} chunks a partir de {len(amostra)} documentos\n")

    # ── 3. Carregar modelo e tokenizer ──────────────────────────────
    print(f"A carregar {MODELO} em {args.device}... (1.ª vez descarrega pesos)")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODELO, device=args.device)
    model.max_seq_length = args.max_seq
    print(f"  modelo carregado em {time.time()-t0:.1f}s  |  dim="
          f"{model.get_sentence_embedding_dimension()}\n")

    # ── 3b. Contagem de tokens: quantos chunks passam de 512? ───────
    print("A contar tokens dos chunks...")
    n_tok = [len(model.tokenizer(t, add_special_tokens=True, truncation=False)
                 ["input_ids"]) for t in textos]
    acima_512 = sum(1 for n in n_tok if n > LIMITE_512)
    acima_max = sum(1 for n in n_tok if n > args.max_seq)
    print(f"  tokens/chunk: média {statistics.mean(n_tok):.0f}, "
          f"mediana {statistics.median(n_tok):.0f}, max {max(n_tok)}")
    print(f"  chunks > 512 tokens (truncados no E5-small): "
          f"{acima_512}/{len(n_tok)} ({acima_512/len(n_tok):.0%})")
    print(f"  chunks > {args.max_seq} tokens (truncados mesmo no Qwen3): "
          f"{acima_max}/{len(n_tok)} ({acima_max/len(n_tok):.0%})\n")

    # ── 4. Embedding, com batches à vista ───────────────────────────
    n_batches = (len(textos) + args.batch_size - 1) // args.batch_size
    print(f"Embedding de {len(textos)} chunks em {n_batches} batches "
          f"de {args.batch_size}...")

    print("  aquecimento (1 chunk)...")
    _ = model.encode(textos[:1], show_progress_bar=False, normalize_embeddings=True)

    t0 = time.time()
    _ = model.encode(
        textos,
        batch_size=args.batch_size,
        show_progress_bar=True,      # mostra a barra de batches a avançar
        normalize_embeddings=True,
    )
    dt = time.time() - t0

    # ── 5. Resultado ────────────────────────────────────────────────
    print("\n=== RESULTADO ===")
    print(f"chunks vetorizados: {len(textos)}")
    print(f"tempo total de embedding: {dt:.1f}s")
    print(f"tempo por chunk: {dt/len(textos)*1000:.0f} ms")
    print(f"velocidade: {len(textos)/dt:.1f} chunks/s")

    # extrapolação simples para o corpus completo (opcional, informativa)
    fator = len(docs) / len(amostra)
    print(f"\nExtrapolação grosseira para os {len(docs)} docs "
          f"(× {fator:.1f}):")
    print(f"  ≈ {len(textos)*fator:,.0f} chunks  |  "
          f"≈ {dt*fator/60:.1f} min  ({dt*fator/3600:.2f} h)")
    print("  (assume que a amostra representa a distribuição de comprimentos)")


if __name__ == "__main__":
    main()