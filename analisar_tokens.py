"""
analisar_tokens_segmentacao.py — Onde estão os segmentos grandes de mais?
=========================================================================
Responde à pergunta que decide o teto de tokens da UnitLevelSegmentation:

  1. Distribuição de tokens por DOCUMENTO (sem segmentar).
  2. Distribuição de tokens por SEGMENTO, separada por RAMO (bloco/artigo/janela).
  3. Quantos segmentos de cada ramo passam de um TETO candidato.

Só conta tokens — NÃO vetoriza, NÃO usa GPU, NÃO escreve no Qdrant.
Corre em segundos no teu PC.

    python analisar_tokens_segmentacao.py                 # corpus todo, teto 2048
    python analisar_tokens_segmentacao.py --teto 1024
    python analisar_tokens_segmentacao.py --n 500         # amostra, mais rápido

Requisitos:
    pip install -U "transformers>=4.51.0"
"""
import argparse
import random
import statistics
from collections import defaultdict

MODELO = "Qwen/Qwen3-Embedding-0.6B"   # tokenizer de referência (o do embedder alvo)
SEED = 42


def percentis(vals, ps=(50, 90, 95, 99)):
    if not vals:
        return {p: 0 for p in ps}
    s = sorted(vals)
    out = {}
    for p in ps:
        idx = min(len(s) - 1, int(p / 100 * len(s)))
        out[p] = s[idx]
    return out


def linha_dist(nome, vals, teto):
    if not vals:
        print(f"  {nome:<10} (sem segmentos)")
        return
    pc = percentis(vals)
    acima = sum(1 for v in vals if v > teto)
    print(f"  {nome:<10} n={len(vals):>6}  média={statistics.mean(vals):>6.0f}  "
          f"mediana={pc[50]:>6}  p90={pc[90]:>6}  p95={pc[95]:>6}  "
          f"p99={pc[99]:>7}  max={max(vals):>7}  "
          f">teto={acima:>5} ({acima/len(vals):>4.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teto", type=int, default=2048, help="teto candidato em tokens")
    ap.add_argument("--n", type=int, default=None, help="amostra (default: corpus todo)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from dre_loader import carregar_corpus_dre
    from dre_segmentador import segmentar_documento

    print(f"A carregar tokenizer {MODELO}...")
    tok = AutoTokenizer.from_pretrained(MODELO)

    def n_tokens(texto):
        return len(tok(texto, add_special_tokens=True, truncation=False)["input_ids"])

    docs = carregar_corpus_dre(relatorio=False)
    if args.n:
        docs = random.Random(SEED).sample(docs, min(args.n, len(docs)))
    print(f"Documentos a analisar: {len(docs)}  |  teto candidato: {args.teto} tokens\n")

    # ── 1. Tokens por DOCUMENTO (sem segmentar) ─────────────────────
    print("=== 1. TOKENS POR DOCUMENTO (documento inteiro, sem segmentar) ===")
    tok_docs = [n_tokens(d["texto"]) for d in docs]
    linha_dist("documento", tok_docs, args.teto)
    acima_512 = sum(1 for v in tok_docs if v > 512)
    acima_8192 = sum(1 for v in tok_docs if v > 8192)
    print(f"  documentos > 512 tokens : {acima_512}/{len(tok_docs)} "
          f"({acima_512/len(tok_docs):.0%})")
    print(f"  documentos > 8192 tokens: {acima_8192}/{len(tok_docs)} "
          f"({acima_8192/len(tok_docs):.0%})\n")

    # ── 2. Tokens por SEGMENTO, separados por RAMO ──────────────────
    print("=== 2. TOKENS POR SEGMENTO, por ramo da UnitLevelSegmentation ===")
    por_ramo = defaultdict(list)
    todos_segs = []
    for d in docs:
        for s in segmentar_documento(d):
            n = n_tokens(s["texto"])
            por_ramo[s["nivel_segmentacao"]].append(n)
            todos_segs.append(n)

    for ramo in ("bloco", "artigo", "janela"):
        linha_dist(ramo, por_ramo.get(ramo, []), args.teto)
    print()
    linha_dist("TODOS", todos_segs, args.teto)

    # ── 3. Veredicto sobre o teto ───────────────────────────────────
    print(f"\n=== 3. IMPACTO DO TETO DE {args.teto} TOKENS ===")
    total_segs = len(todos_segs)
    for ramo in ("bloco", "artigo", "janela"):
        vals = por_ramo.get(ramo, [])
        if not vals:
            continue
        acima = sum(1 for v in vals if v > args.teto)
        print(f"  {ramo:<10} {acima:>5}/{len(vals):<6} segmentos passam do teto "
              f"({acima/len(vals):.0%} do ramo)")
    acima_tot = sum(1 for v in todos_segs if v > args.teto)
    print(f"  {'TOTAL':<10} {acima_tot:>5}/{total_segs:<6} segmentos passam do teto "
          f"({acima_tot/total_segs:.1%} do corpus)")

    print("\nLeitura:")
    print("  - Se quase todos os que passam do teto forem do ramo 'artigo', o teto")
    print("    parte artigos (foge ao propósito). Decide: sub-dividir por número")
    print("    interno, ou deixar artigos longos inteiros como exceção.")
    print("  - Se forem do ramo 'bloco'/'janela', o teto é inócuo ao propósito —")
    print("    só afina o tamanho da janela. Baixar o teto é seguro.")
    print("  - O nº de segmentos > teto é também quantos sub-cortes extra terias,")
    print("    o que aumenta o total de chunks (e o tempo de embedding).")


if __name__ == "__main__":
    main()