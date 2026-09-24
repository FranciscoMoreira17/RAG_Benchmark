"""
tempo_embedding_corpus.py — Quanto tempo demora vetorizar o corpus DRE completo?
=================================================================================
Corre o embedding numa AMOSTRA e extrapola diretamente para os N documentos do
corpus, imprimindo o tempo total estimado em minutos/horas.

NÃO escreve no Qdrant. Serve só para dimensionar o custo antes de te
comprometeres com a ingestão completa.

Onde correr: idealmente no hardware onde vais fazer a ingestão real (Colab/Kaggle
com GPU). Em CPU também corre, mas o número será o do CPU, não do GPU.

    # amostra de 100 docs, janela 8192, extrapola para o corpus todo:
    python tempo_embedding_corpus.py --n 100 --device cuda

    # testar a janela de 32k (custo maior por doc longo):
    python tempo_embedding_corpus.py --n 100 --device cuda --max-seq 32768

    # comparar segmentação: passa o nome de uma config registada
    python tempo_embedding_corpus.py --n 100 --device cuda --config langchain_chunks

Requisitos:
    pip install -U "sentence-transformers>=2.7.0" "transformers>=4.51.0"
"""
import argparse
import random
import statistics
import time

MODELO = "Qwen/Qwen3-Embedding-0.6B"
SEED = 42


# ---------------------------------------------------------------------
# CONFIGURAÇÕES de segmentação — cada uma transforma docs numa lista de
# textos a vetorizar. Liga aqui as tuas estratégias (sentence, artigo,
# janela, bloco, late chunking...). O tempo estimado é POR CONFIGURAÇÃO,
# porque cada uma gera um número e um tamanho de segmentos diferente.
# ---------------------------------------------------------------------
def cfg_documento_inteiro(docs):
    return [d["texto"] for d in docs]


def cfg_langchain_chunks(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    sp = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    out = []
    for d in docs:
        out.extend(sp.split_text(d["texto"]))
    return out


CONFIGURACOES = {
    "documento_inteiro": cfg_documento_inteiro,
    "langchain_chunks": cfg_langchain_chunks,
    # "llamaindex_sentence": cfg_llamaindex_sentence,   # <- liga as tuas aqui
    # "por_artigo": cfg_por_artigo,
    # "late_chunking": cfg_late_chunking,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="documentos na amostra")
    ap.add_argument("--device", default="cpu", help="cpu ou cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=8192, help="janela em tokens")
    ap.add_argument("--config", default="documento_inteiro",
                    choices=list(CONFIGURACOES),
                    help="estratégia de segmentação a medir")
    args = ap.parse_args()

    from dre_loader import carregar_corpus_dre
    from sentence_transformers import SentenceTransformer

    docs = carregar_corpus_dre(relatorio=False)
    n_docs_corpus = len(docs)
    print(f"Corpus completo: {n_docs_corpus} documentos")
    print(f"Configuração: {args.config}  |  device={args.device}  "
          f"|  max_seq={args.max_seq}\n")

    # Amostra ALEATÓRIA de documentos, depois segmentada pela config escolhida.
    amostra_docs = random.Random(SEED).sample(docs, min(args.n, n_docs_corpus))
    segmentar = CONFIGURACOES[args.config]

    # Segmentos da amostra e (por proporção) estimativa dos segmentos do corpus.
    segs_amostra = segmentar(amostra_docs)
    fator_corpus = n_docs_corpus / len(amostra_docs)
    n_segs_corpus_est = len(segs_amostra) * fator_corpus
    print(f"Amostra: {len(amostra_docs)} docs → {len(segs_amostra)} segmentos "
          f"(≈ {n_segs_corpus_est:,.0f} segmentos no corpus completo)\n")

    print(f"A carregar {MODELO} em {args.device}...")
    t0 = time.time()
    model = SentenceTransformer(MODELO, device=args.device)
    model.max_seq_length = args.max_seq
    print(f"Modelo carregado em {time.time()-t0:.1f}s "
          f"(pago uma vez, não escala com o corpus)\n")

    print("Aquecimento...")
    _ = model.encode(segs_amostra[:args.batch_size], batch_size=args.batch_size,
                     show_progress_bar=False, normalize_embeddings=True)

    print(f"A vetorizar {len(segs_amostra)} segmentos da amostra...")
    t0 = time.time()
    _ = model.encode(segs_amostra, batch_size=args.batch_size,
                     show_progress_bar=True, normalize_embeddings=True)
    dt = time.time() - t0

    # ── EXTRAPOLAÇÃO para o corpus completo ──────────────────────────
    seg_por_s = len(segs_amostra) / dt if dt > 0 else 0
    t_corpus = dt * fator_corpus     # tempo escala com o nº de segmentos

    print("\n=== TEMPO PARA O CORPUS COMPLETO ===")
    print(f"Amostra: {len(segs_amostra)} segmentos em {dt:.1f}s "
          f"({seg_por_s:.1f} seg/s)")
    print(f"Corpus:  ≈ {n_segs_corpus_est:,.0f} segmentos")
    print(f"\n  TEMPO ESTIMADO (embedding): {t_corpus:.0f}s  "
          f"= {t_corpus/60:.1f} min  = {t_corpus/3600:.2f} h")
    print(f"  (+ ~{time.time()-t0-dt:.0f}s de carregamento do modelo, uma vez)")

    print("\nNotas:")
    print("  - A extrapolação assume que a amostra representa a distribuição de")
    print("    comprimentos do corpus. Aumenta --n se a cauda de docs longos for")
    print("    grande (no DRE é: alguns docs têm dezenas de milhares de tokens).")
    print("  - O tempo é DESTA config e DESTA janela. Muda a segmentação ou o")
    print("    max_seq e volta a medir — não extrapoles de uma para a outra.")
    print("  - Em CPU este número é o do CPU. Corre com --device cuda no Colab")
    print("    para o número que interessa à ingestão real.")


if __name__ == "__main__":
    main()