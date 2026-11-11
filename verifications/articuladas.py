"""
amostrar_articulados.py — candidatos para o subconjunto "articulado"
=====================================================================
Escolhe aleatoriamente (seed fixa) documentos do corpus que:
  1. têm articulado real (_tem_articulado_real), e
  2. NÃO cabem num bloco (len > LIMIAR_BLOCO_UNICO = 8000 caracteres),
  3. ainda não estão no dataset atual.

Amostragem ESTRATIFICADA por tipo de documento (alocação proporcional,
mínimo 1 por tipo elegível), para não ficar só com regulamentos.

Gera N + reservas: as reservas servem para substituir documentos que
se revelem inadequados (texto partido, sem conteúdo perguntável, etc.).
A substituição deve seguir a ordem das reservas, para a escolha
continuar aleatória e documentável.

    python amostrar_articulados.py
    python amostrar_articulados.py --n 30 --reservas 15 --seed 42

Saída: dataset_articulado_candidatos.json (NÃO altera o dataset atual).
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from dre_loader import carregar_corpus_dre
from dre_segmentador import _tem_articulado_real, RE_ARTIGO, LIMIAR_BLOCO_UNICO

DATASET = "dataset_dre_manual.json"
SAIDA = "dataset_articulado_candidatos.json"


def alocar(contagens: dict, n: int) -> dict:
    """Alocação proporcional com mínimo 1 por tipo (maiores restos)."""
    tipos = sorted(contagens, key=lambda t: -contagens[t])
    if n < len(tipos):                      # mais tipos do que vagas
        return {t: 1 for t in tipos[:n]}
    aloc = {t: 1 for t in tipos}
    resto_n = n - len(tipos)
    total = sum(contagens.values())
    quotas = {t: resto_n * contagens[t] / total for t in tipos}
    for t in tipos:
        aloc[t] += int(quotas[t])
    faltam = n - sum(aloc.values())
    for t in sorted(tipos, key=lambda t: -(quotas[t] - int(quotas[t])))[:faltam]:
        aloc[t] += 1
    # não pedir mais do que existe
    for t in tipos:
        aloc[t] = min(aloc[t], contagens[t])
    return aloc


def registo(d: dict, qid: str, reserva: bool) -> dict:
    n_art = len(RE_ARTIGO.findall(d["texto"]))
    return {
        "id": qid,
        "query": "",
        "tipo": "",
        "camada": None,
        "ground_truth": "",
        "documentos_esperados": [d["id"]],
        "ancora": f"{d['tipo']} {d['numero']}".strip(),
        "para_geracao": False,
        "subconjunto": "articulado",
        "artigo_esperado": "",
        "_reserva": reserva,
        "_titulo": d["titulo"],
        "_tipo_doc": d["tipo"],
        "_numero": d["numero"],
        "_data": d["data_publicacao"],
        "_url": d["url"],
        "_n_chars": len(d["texto"]),
        "_n_artigos": n_art,
        "_conteudo": d["texto"],            # texto COMPLETO: a pergunta visa um artigo concreto
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--reservas", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=LIMIAR_BLOCO_UNICO,
                    help="só documentos MAIORES do que isto (default: limiar do bloco)")
    args = ap.parse_args()

    docs = carregar_corpus_dre(relatorio=False)

    dataset = json.loads(Path(DATASET).read_text(encoding="utf-8"))
    ja_usados = {g for q in dataset for g in (q.get("documentos_esperados") or [])}
    ids_num = [int(q["id"]) for q in dataset if str(q.get("id", "")).isdigit()]
    prox_id = (max(ids_num) if ids_num else len(dataset)) + 1
    largura = max(3, len(str(prox_id + args.n + args.reservas)))

    elegiveis = defaultdict(list)
    for d in docs:
        if (d["id"] not in ja_usados
                and len(d["texto"]) > args.min_chars
                and _tem_articulado_real(d["texto"])):
            elegiveis[d["tipo"]].append(d)

    contagens = {t: len(v) for t, v in elegiveis.items()}
    print(f"Elegíveis (articulado, > {args.min_chars} chars, fora do dataset): "
          f"{sum(contagens.values())}")
    for t, c in sorted(contagens.items(), key=lambda x: -x[1]):
        print(f"  {t:<15} {c}")

    rng = random.Random(args.seed)
    for t in elegiveis:                      # ordem determinística antes de baralhar
        elegiveis[t].sort(key=lambda d: d["id"])
        rng.shuffle(elegiveis[t])

    aloc = alocar(contagens, args.n)
    principais, sobras = [], []
    for t in sorted(elegiveis):
        principais += elegiveis[t][:aloc.get(t, 0)]
        sobras += elegiveis[t][aloc.get(t, 0):]
    rng.shuffle(principais)
    rng.shuffle(sobras)
    reservas = sobras[:args.reservas]

    saida = []
    for i, d in enumerate(principais + reservas):
        qid = str(prox_id + i).zfill(largura)
        saida.append(registo(d, qid, reserva=i >= len(principais)))

    Path(SAIDA).write_text(json.dumps(saida, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print(f"\nAlocação (seed {args.seed}):")
    for t, a in sorted(aloc.items(), key=lambda x: -x[1]):
        print(f"  {t:<15} {a}")
    print(f"\n{len(principais)} principais + {len(reservas)} reservas -> {SAIDA}")
    print(f"IDs: {saida[0]['id']} a {saida[-1]['id']}")


if __name__ == "__main__":
    main()