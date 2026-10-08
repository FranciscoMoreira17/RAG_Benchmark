"""
ver_segmentador_v2.py - inspeciona a segmentação v2 de um documento

    python ver_segmentador_v2.py                 # regulamento longo aleatório
    python ver_segmentador_v2.py --id 6423298    # documento específico
"""
import argparse, random, re
from transformers import AutoTokenizer
from dre_loader import carregar_corpus_dre
from dre_segmentador import segmentar_documento as seg_v1
from dre_segmentador_v2 import (segmentar_documento as seg_v2,
                                _tem_articulado_real, RE_ARTIGO)

TETO = 2048
ap = argparse.ArgumentParser()
ap.add_argument("--id")
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
contar = lambda t: len(tok(t, add_special_tokens=True, truncation=False)["input_ids"])

docs = carregar_corpus_dre(relatorio=False)
if args.id:
    doc = next(d for d in docs if d["id"] == args.id)
else:
    cands = [d for d in docs if d["tipo"] == "Regulamento"
             and len(d["texto"]) > 8000 and _tem_articulado_real(d["texto"])]
    doc = random.Random(args.seed).choice(cands)

print(f"DOCUMENTO: {doc['titulo']}  (id {doc['id']})")
print(f"  {len(doc['texto'])} chars | {contar(doc['texto'])} tokens | "
      f"articulado: {_tem_articulado_real(doc['texto'])}\n")

segs = seg_v2(doc, teto_tokens=TETO, contar_tokens=contar)
print(f"{'#':>3} {'rótulo':<28} {'nível':<14} {'tok_embed':>9} {'chars':>7}  início")
for i, s in enumerate(segs):
    ini = re.sub(r"\s+", " ", s["texto"])[:60]
    print(f"{i:>3} {s['rotulo']:<28} {s['nivel_segmentacao']:<14} "
          f"{contar(s['texto_embed']):>9} {len(s['texto']):>7}  {ini}")

# ── verificações ──
max_tok = max(contar(s["texto_embed"]) for s in segs)
arts_doc = [m.group(1) for m in RE_ARTIGO.finditer(doc["texto"])]
arts_segs = [m.group(1) for s in segs for m in RE_ARTIGO.finditer(s["texto"])]
em_falta = sorted(set(arts_doc) - set(arts_segs), key=int)
v1 = seg_v1(doc)

print("\nVERIFICAÇÕES")
print(f"  maior segmento (com cabeçalho): {max_tok} tokens -> "
      f"{'OK' if max_tok <= TETO else 'PASSA O TETO'}")
print(f"  artigos no documento: {len(arts_doc)} | artigos em falta nos segmentos: "
      f"{em_falta if em_falta else 'nenhum'}")
print(f"  segmentos: v1 = {len(v1)}  |  v2 = {len(segs)}")
print(f"  v1 com menos de 500 chars: {sum(len(s['texto']) < 500 for s in v1)}")
print(f"\nCABEÇALHO (início do texto_embed do 1.º segmento):\n  "
      f"{segs[0]['texto_embed'][:300]}")