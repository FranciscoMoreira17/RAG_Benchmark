"""
dre_loader.py - Carregamento do corpus do Diário da República
==============================================================
Lê os JSON extraídos do DRE e devolve documentos normalizados,
prontos para indexação. O identificador de cada documento é o
`guid` - único, mecânico, e é o alvo de retrieval no benchmark.

Decisões de filtragem:
  - Documentos com `error` de scraping → excluídos
  - `article_text` vazio ou < MIN_CHARS → excluídos (sem conteúdo)
  - `article_text` > MAX_CHARS → excluídos
  - guids duplicados → mantém-se a primeira ocorrência


"""

import os
import re
import glob
import json
import hashlib
import statistics
from collections import Counter, defaultdict

MIN_CHARS = int(os.getenv("DRE_MIN_CHARS", "50"))
MAX_CHARS = int(os.getenv("DRE_MAX_CHARS", "500000"))

DRE_DIR = os.getenv("DRE_DIR", "./documents/dre_data")

_MESES = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
    "outubro": 10, "novembro": 11, "dezembro": 12,
}

_RE_NUMERO = re.compile(r"(\d+(?:-[A-Za-z]+)?/\d{4})")
_RE_DATA = re.compile(r"\bde\s+(\d{1,2})\s+de\s+([A-Za-zçÇãõéêáíóú]+)", re.IGNORECASE)


def parse_titulo(titulo: str) -> dict:
    """Extrai tipo, número/ano e data de publicação do título."""
    titulo = titulo or ""
    tipo = titulo.split()[0] if titulo.split() else "?"

    m_num = _RE_NUMERO.search(titulo)
    numero = m_num.group(1) if m_num else ""

    data = ""
    m_data = _RE_DATA.search(titulo)
    if m_num and m_data:
        ano = numero.split("/")[1]
        mes = _MESES.get(m_data.group(2).lower(), 0)
        if mes:
            data = f"{ano}-{mes:02d}-{int(m_data.group(1)):02d}"

    return {"tipo": tipo, "numero": numero, "data_publicacao": data}


def carregar_corpus_dre(
    dre_dir: str = DRE_DIR,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
    relatorio: bool = True,
) -> list[dict]:
    """
    Devolve lista de documentos normalizados:
      {
        "id": guid,               
        "titulo": str,
        "tipo": str,              # Anúncio, Aviso, Despacho, ...
        "numero": str,            # ex: "131/2026"
        "data_publicacao": str,   # ISO "2026-01-03" (pode ser "")
        "texto": str,             # article_text
        "url": str,
        "n_chars": int,
        "ficheiro_origem": str,
      }
    """
    ficheiros = sorted(glob.glob(os.path.join(dre_dir, "*.json")))
    if not ficheiros:
        raise FileNotFoundError(f"Nenhum JSON em '{dre_dir}/'")

    docs: list[dict] = []
    vistos_guid: set[str] = set()

    cont = Counter()
    tipos = Counter()
    tamanhos: list[int] = []
    excluidos_grandes: list[tuple[str, int]] = []
    ficheiros_partidos: list[str] = []

    for f in ficheiros:
        try:
            with open(f, encoding="utf-8") as fh:
                items = json.load(fh)
        except json.JSONDecodeError as e:
            ficheiros_partidos.append(f"{os.path.basename(f)}: {str(e)[:60]}")
            cont["ficheiro_partido"] += 1
            continue

        if not isinstance(items, list):
            items = [items]

        for it in items:
            cont["total_lidos"] += 1
            if not isinstance(it, dict):
                cont["nao_dict"] += 1
                continue

            if it.get("error"):
                cont["erro_scraping"] += 1
                continue

            guid = str(it.get("guid", "")).strip()
            if not guid:
                cont["sem_guid"] += 1
                continue
            if guid in vistos_guid:
                cont["guid_duplicado"] += 1
                continue

            texto = (it.get("article_text") or "").strip()
            n = len(texto)
            if n < min_chars:
                cont["vazio_ou_curto"] += 1
                continue
            if n > max_chars:
                cont["grande_demais"] += 1
                excluidos_grandes.append((it.get("title", "")[:60], n))
                continue

            meta = parse_titulo(it.get("title", ""))
            vistos_guid.add(guid)
            tamanhos.append(n)
            tipos[meta["tipo"]] += 1

            docs.append({
                "id": guid,
                "titulo": it.get("title", ""),
                "tipo": meta["tipo"],
                "numero": meta["numero"],
                "data_publicacao": meta["data_publicacao"],
                "texto": texto,
                "url": it.get("url", ""),
                "n_chars": n,
                "ficheiro_origem": os.path.basename(f),
            })
            cont["aceites"] += 1

    rel = {
        "n_ficheiros": len(ficheiros),
        "ficheiros_partidos": ficheiros_partidos,
        "contagens": dict(cont),
        "limiares": {"min_chars": min_chars, "max_chars": max_chars},
        "distribuicao_tipos": dict(tipos.most_common()),
        "tamanhos": {
            "min": min(tamanhos) if tamanhos else 0,
            "max": max(tamanhos) if tamanhos else 0,
            "mediana": statistics.median(tamanhos) if tamanhos else 0,
            "media": statistics.mean(tamanhos) if tamanhos else 0,
            "curtos_lt2000": sum(1 for t in tamanhos if t < 2000),
            "medios_2000_8000": sum(1 for t in tamanhos if 2000 <= t <= 8000),
            "longos_gt8000": sum(1 for t in tamanhos if t > 8000),
        },
        "excluidos_grandes_amostra": excluidos_grandes[:20],
    }

    print(f"  [DRE] {cont['aceites']} documentos aceites de {cont['total_lidos']} lidos")
    print(f"  [DRE] Excluídos: {cont['erro_scraping']} erro scraping, "
          f"{cont['vazio_ou_curto']} vazios/curtos, "
          f"{cont['grande_demais']} grandes demais, "
          f"{cont['guid_duplicado']} duplicados")

    if relatorio:
        destino = os.path.join(dre_dir, ".relatorio_corpus.json")
        with open(destino, "w", encoding="utf-8") as fh:
            json.dump(rel, fh, ensure_ascii=False, indent=2)
        print(f"  [DRE] Relatório → {destino}")

    return docs


def corpus_fingerprint(docs: list[dict]) -> str:
    guids = sorted(d["id"] for d in docs)
    payload = json.dumps(
        {"guids": guids, "min": MIN_CHARS, "max": MAX_CHARS},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


if __name__ == "__main__":
    docs = carregar_corpus_dre()
    print(f"\n  Total: {len(docs)} documentos")
    print(f"  Fingerprint: {corpus_fingerprint(docs)}")
    print(f"\n  Amostra:")
    for d in docs[:3]:
        print(f"    [{d['id']}] {d['tipo']} {d['numero']} "
              f"({d['data_publicacao']}) — {d['n_chars']} chars")