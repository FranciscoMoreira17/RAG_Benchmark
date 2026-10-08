import json, glob
from collections import defaultdict
from dre_loader import carregar_corpus_dre
from dre_segmentador import segmentar_documento
from avaliacao import metricas_retrieval

docs = {d["id"]: d for d in carregar_corpus_dre(relatorio=False)}
cache = {}
def ramo(did):
    if did not in cache:
        cache[did] = segmentar_documento(docs[did])[-1]["nivel_segmentacao"]
    return cache[did]

for f in sorted(glob.glob("resultados/ThirdResults-Qwen/*.json")):
    dados = json.load(open(f, encoding="utf-8"))
    if "resultados" not in dados:
        continue
    cond = dados.get("condicao", dados.get("framework"))
    por = defaultdict(list)
    for r in dados["resultados"]:
        gold = r.get("documentos_esperados") or []
        if gold:
            por[ramo(gold[0])].append(r)
    linha = {k: (len(v), round(metricas_retrieval(v, k=5)["hit_rate@5"], 2))
             for k, v in sorted(por.items())}
    print(f"{cond:42s} {linha}")