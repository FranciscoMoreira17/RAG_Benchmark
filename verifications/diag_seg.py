"""
_diag_seg.py - Diagnóstico da segmentação
Corre no teu ambiente:  python _diag_seg.py
"""
import re
from collections import Counter
from statistics import mean, median

from dre_loader import carregar_corpus_dre

RE_ARTIGO = re.compile(
    r"(?:^|\n)#{0,6}\s*\**\s*(Art(?:igo)?\.?\s*(\d+)[\.\-]?\s*[º°oO]?(?:[\-–][A-Z])?)",
    re.IGNORECASE,
)
RE_SECCAO = re.compile(r"(?:^|\n)\s*(\d+(?:\.\d+)?)\s*[-–]\s+([A-ZÀ-Ú][^\n]{3,80})")

docs = carregar_corpus_dre(relatorio=False)

tipos_artigo = Counter()
seg_counts = []
exemplos_falso = []

for d in docs:
    texto = d["texto"]
    arts = list(RE_ARTIGO.finditer(texto))
    seccoes_l1 = [m for m in RE_SECCAO.finditer(texto) if "." not in m.group(1)]
    if arts:
        tipos_artigo[d["tipo"]] += 1
        if d["tipo"] in ("Anúncio","Aviso","Louvor","Edital") and len(exemplos_falso) < 5:
            exemplos_falso.append((d["tipo"], d["titulo"][:50], [m.group(1) for m in arts[:3]]))
    if len(seccoes_l1) >= 2:
        seg_counts.append((d["tipo"], len(seccoes_l1), d["n_chars"]))

print("\n=== TIPOS que disparam RE_ARTIGO ===")
for t, n in tipos_artigo.most_common(10):
    print(f"  {t:<20} {n}")

print("\n=== 'artigo' em tipos SEM articulado (falsos positivos) ===")
for tipo, tit, matches in exemplos_falso:
    print(f"  [{tipo}] {tit}  →  {matches}")

print("\n=== Inflação de secções (top 15) ===")
seg_counts.sort(key=lambda x: -x[1])
for tipo, ns, nc in seg_counts[:15]:
    print(f"  {tipo:<15} {ns} secções em {nc} chars")

if seg_counts:
    todas = [x[1] for x in seg_counts]
    print(f"\nSecções L1/doc (quando >=2): mediana={median(todas)} media={mean(todas):.1f} max={max(todas)}")