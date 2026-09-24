"""
medir_tokens_chunk.py — Calibração do tamanho de chunk para overlap_chunking
============================================================================
Mede a distribuição REAL de tokens (tokenizer do E5) para chunks de vários
tamanhos em caracteres, sobre o corpus DRE. Objetivo: escolher o maior
tamanho de chunk cujo P95/P99 de tokens fique ABAIXO de 512 (a janela do
E5), garantindo que nada é truncado silenciosamente no embedding denso.

    python medir_tokens_chunk.py

Corre na tua máquina (precisa do tokenizer do E5 em cache e do corpus).
"""
from statistics import quantiles, mean, median
from transformers import AutoTokenizer
from dre_loader import carregar_corpus_dre

MODELO = "intfloat/multilingual-e5-small"
LIMITE_JANELA = 512
TAMANHOS_TESTE = [1200, 1300, 1400, 1500, 1600]   # caracteres por chunk a testar
OVERLAP_PCT = 0.15

tok = AutoTokenizer.from_pretrained(MODELO)

def ntok(texto: str) -> int:
    # sem truncar, para vermos o comprimento REAL
    return len(tok.encode(texto, add_special_tokens=True, truncation=False))

def janelar(texto: str, tam: int, overlap: int) -> list[str]:
    passo = tam - overlap
    segs, i = [], 0
    while i < len(texto):
        w = texto[i:i + tam].strip()
        if w:
            segs.append(w)
        i += passo
    return segs

def p(vals, q):
    # percentil q (0..100) simples
    if not vals:
        return 0
    s = sorted(vals)
    k = min(len(s) - 1, int(round((q / 100) * (len(s) - 1))))
    return s[k]

def main():
    docs = carregar_corpus_dre(relatorio=False)
    textos = [d["texto"] for d in docs]
    print(f"Corpus: {len(textos)} documentos\n")
    print(f"{'tam_chars':>10} {'overlap':>8} {'n_chunks':>9} "
          f"{'tok_media':>10} {'tok_med':>8} {'tok_P95':>8} "
          f"{'tok_P99':>8} {'tok_max':>8} {'%>512':>7}")
    print("-" * 82)

    for tam in TAMANHOS_TESTE:
        overlap = int(tam * OVERLAP_PCT)
        toks = []
        for t in textos:
            for chunk in janelar(t, tam, overlap):
                toks.append(ntok(chunk))
        if not toks:
            continue
        n_excede = sum(1 for x in toks if x > LIMITE_JANELA)
        pct_excede = 100 * n_excede / len(toks)
        print(f"{tam:>10} {overlap:>8} {len(toks):>9} "
              f"{mean(toks):>10.1f} {median(toks):>8.0f} "
              f"{p(toks,95):>8} {p(toks,99):>8} {max(toks):>8} "
              f"{pct_excede:>6.2f}%")

    print("\nEscolhe o MAIOR tam_chars cujo tok_P99 fique <= 512 "
          "(ou %>512 seja ~0). Esse é o tamanho seguro.")
    print("Se todos excederem, o overlap_chunking precisa de chunks menores.")

if __name__ == "__main__":
    main()