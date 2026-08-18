"""
dre_segmentador.py - Segmentação consciente da estrutura
==============================================================
Reformulação do OKF para corpus heterogéneo. 

  1. ARTIGO  - só quando há sequencia real de artigos (>=2, começando
               em 1 ou 2, maioria dos saltos +1). Isto aceita leis,
               decretos-lei, regulamentos e os avisos/despachos que
               publicam estatutos ou regulamentos internos.
  2. BLOCO   - documento inteiro como uma unidade. Documentos curtos
               sem articulado (a maioria: anúncios, avisos simples,
               louvores, editais).
  3. JANELA  - documentos longos (>LIMIAR) sem articulado: janelas
               com overlap.

"""

import re
from collections import Counter

LIMIAR_BLOCO_UNICO = 8000
JANELA_CHARS = 4000
JANELA_OVERLAP = 400
MIN_SEGMENTO_CHARS = 40
MIN_RACIO_SEQUENCIA = 0.6     

RE_ARTIGO = re.compile(
    r"(?:^|\n)#{0,6}\s*\**\s*Art(?:igo)?\.?\s*(\d+)[\.\-]?\s*[º°oO]?(?:[\-–][A-Z])?",
    re.IGNORECASE,
)


def _tem_articulado_real(texto: str) -> bool:
    """
    True se o texto tem uma sequência genuína de artigos, não
    apenas remissões dispersas para legislação externa.
    """
    nums = [int(m.group(1)) for m in RE_ARTIGO.finditer(texto)]
    if len(nums) < 2 or nums[0] > 2:
        return False
    saltos_ok = sum(1 for i in range(1, len(nums)) if nums[i] - nums[i - 1] in (0, 1))
    return (saltos_ok / (len(nums) - 1)) >= MIN_RACIO_SEQUENCIA


def _por_artigo(texto: str) -> list[dict]:
    matches = list(RE_ARTIGO.finditer(texto))
    segs = []
    if matches[0].start() > 0:
        pre = texto[:matches[0].start()].strip()
        if len(pre) >= MIN_SEGMENTO_CHARS:
            segs.append({"texto": pre, "rotulo": "preambulo", "nivel": "artigo"})
    for i, m in enumerate(matches):
        fim = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
        conteudo = texto[m.start():fim].strip()
        if len(conteudo) >= MIN_SEGMENTO_CHARS:
            segs.append({
                "texto": conteudo,
                "rotulo": f"Artigo {m.group(1)}",
                "nivel": "artigo",
            })
    return segs


def _por_janela(texto: str) -> list[dict]:
    segs, i, n = [], 0, len(texto)
    while i < n:
        janela = texto[i:i + JANELA_CHARS].strip()
        if janela:
            segs.append({"texto": janela, "rotulo": f"janela@{i}", "nivel": "janela"})
        i += JANELA_CHARS - JANELA_OVERLAP
    return segs


def segmentar_documento(doc: dict) -> list[dict]:
    texto = doc["texto"]

    if _tem_articulado_real(texto):
        segmentos = _por_artigo(texto)
    elif len(texto) <= LIMIAR_BLOCO_UNICO:
        segmentos = [{"texto": texto, "rotulo": "documento", "nivel": "bloco"}]
    else:
        segmentos = _por_janela(texto)

    saida = []
    for j, s in enumerate(segmentos):
        saida.append({
            "doc_id": doc["id"],
            "chunk_id": f"{doc['id']}::{j}",
            "texto": s["texto"],
            "rotulo": s["rotulo"],
            "nivel_segmentacao": s["nivel"],
            "titulo": doc["titulo"],
            "tipo": doc["tipo"],
            "numero": doc["numero"],
            "data_publicacao": doc["data_publicacao"],
            "url": doc["url"],
        })
    return saida


def segmentar_corpus(docs: list[dict]) -> tuple[list[dict], dict]:
    todos, niveis = [], Counter()
    dist_segs = []
    for d in docs:
        segs = segmentar_documento(d)
        todos.extend(segs)
        dist_segs.append(len(segs))
        nivel_doc = next((s["nivel_segmentacao"] for s in segs
                          if s["rotulo"] not in ("preambulo",)), "bloco")
        niveis[nivel_doc] += 1

    from statistics import mean, median
    relatorio = {
        "n_documentos": len(docs),
        "n_segmentos": len(todos),
        "segmentos_por_doc_media": mean(dist_segs) if dist_segs else 0,
        "segmentos_por_doc_mediana": median(dist_segs) if dist_segs else 0,
        "segmentos_por_doc_max": max(dist_segs) if dist_segs else 0,
        "distribuicao_nivel_documento": dict(niveis),
    }
    print(f"  [SEG] {len(docs)} docs → {len(todos)} segmentos "
          f"(media {relatorio['segmentos_por_doc_media']:.1f}/doc, "
          f"mediana {relatorio['segmentos_por_doc_mediana']:.0f}, "
          f"max {relatorio['segmentos_por_doc_max']})")
    print(f"  [SEG] Nível: {dict(niveis)}")
    return todos, relatorio


if __name__ == "__main__":
    import json
    from dre_loader import carregar_corpus_dre
    docs = carregar_corpus_dre(relatorio=False)
    segs, rel = segmentar_corpus(docs)
    print(json.dumps(rel, ensure_ascii=False, indent=2))