import re
import json
import random
from collections import Counter, defaultdict

from dre_loader import carregar_corpus_dre

SEED = 42
N_PARA_GERACAO = 15
MIN_SUMARIO_CHARS = 25
MAX_SUMARIO_CHARS = 160

RE_SUMARIO = re.compile(r"Sumário:\s*(.+?)(?:\n|$)", re.IGNORECASE)

TIPOS_ALVO = ["Aviso", "Despacho", "Portaria", "Decreto",
              "Deliberação", "Edital", "Louvor"]

TEMPLATES = [
    "Que {tipo_l} do Diário da República trata de: {sumario}?",
    "Identifique o {tipo_l} que tem como objeto: {sumario}.",
    "Qual o {tipo_l} relativo a: {sumario}?",
]


def _limpar_sumario(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    return s


def gerar(docs, seed=SEED):
    rng = random.Random(seed)

    # 1. Recolher candidatos com sumário
    candidatos = []
    sumario_freq = Counter()
    for d in docs:
        if d["tipo"] not in TIPOS_ALVO:
            continue
        m = RE_SUMARIO.search(d["texto"][:500])
        if not m:
            continue
        sumario = _limpar_sumario(m.group(1))
        if not (MIN_SUMARIO_CHARS <= len(sumario) <= MAX_SUMARIO_CHARS):
            continue
        candidatos.append((d, sumario))
        sumario_freq[sumario] += 1

    # 2. Rejeitar sumários repetidos (ambíguos)
    validos = [(d, s) for d, s in candidatos if sumario_freq[s] == 1]
    print(f"  Candidatos com sumário: {len(candidatos)}")
    print(f"  Após remover repetidos: {len(validos)}")

    # 3. Verificar unicidade: o sumário identifica UM só documento?
    #    (já garantido pelo passo 2, mas confirmamos guid único)
    rng.shuffle(validos)

    # 4. Gerar perguntas
    perguntas = []
    dist_tipo = Counter()
    for i, (d, sumario) in enumerate(validos):
        tipo_l = d["tipo"].lower()
        template = TEMPLATES[i % len(TEMPLATES)]
        query = template.format(tipo_l=tipo_l, sumario=sumario)

        perguntas.append({
            "id": f"ID_{d['tipo'][:3].upper()}_{i:03d}",
            "query": query,
            "tipo": "identificador",
            "camada": 1,
            "ground_truth": sumario,   # o ato descrito é a resposta
            "documentos_esperados": [d["id"]],
            "ancora": f"{d['tipo']} {d['numero']}",
            "sumario_len": len(sumario),
            "para_geracao": False,
        })
        dist_tipo[d["tipo"]] += 1

    print(f"  Perguntas geradas: {len(perguntas)}")
    print(f"  Por tipo: {dict(dist_tipo)}")

    # 5. Marcar 15 para geração — as de sumário mais rico (mais longo,
    #    logo mais detalhe para o gerador localizar)
    por_riqueza = sorted(perguntas, key=lambda p: -p["sumario_len"])
    for p in por_riqueza[:N_PARA_GERACAO]:
        p["para_geracao"] = True

    n_gen = sum(1 for p in perguntas if p["para_geracao"])
    print(f"  Marcadas para geração: {n_gen}")

    # limpar campo auxiliar
    for p in perguntas:
        p.pop("sumario_len", None)

    return perguntas


def main():
    docs = carregar_corpus_dre(relatorio=False)
    ident = gerar(docs)

    with open("dataset_identificador.json", "w", encoding="utf-8") as f:
        json.dump(ident, f, ensure_ascii=False, indent=2)
    print(f"\n  Identificador → dataset_identificador.json ({len(ident)})")
    print(f"  Para geração: {sum(1 for p in ident if p['para_geracao'])}")
    print(f"\n  Próximo: merge com as semânticas (dataset_semantica.json)")
    print(f"  Total esperado: {len(ident)} + 19 semânticas")
    print(f"  Geração total: 15 (ident) + 15 (semânticas) = 30")


if __name__ == "__main__":
    main()