"""
dre_segmentador.py - Segmentacao consciente da estrutura (com TETO de tokens)
=============================================================================
Reformulacao do OKF para corpus heterogeneo.

  1. ARTIGO  - so quando ha sequencia real de artigos (>=2, comecando
               em 1 ou 2, maioria dos saltos +1).
  2. BLOCO   - documento inteiro como uma unidade (documentos curtos).
  3. JANELA  - documentos longos sem articulado: janelas com overlap.

NOVO: TETO DE TOKENS
--------------------
Nenhum segmento excede TETO_TOKENS. A segmentacao estrutural (artigo/bloco/
janela) e aplicada primeiro, INTACTA; depois, qualquer segmento que passe do
teto e sub-dividido preservando ao maximo a estrutura:

  - artigo longo  -> sub-divide pela numeracao interna (1., 2., 3., alineas);
                     se um pedaco ainda passar (tabelas/listas), janela.
  - bloco/janela  -> janela com overlap (nao ha estrutura interna a preservar).

O teto conta-se em TOKENS. Para nao acoplar este modulo ao tokenizer do
embedder, a contagem e injetavel: passa-se `contar_tokens` a segmentar_*;
sem ela, usa-se uma estimativa por caracteres (~4 chars/token).
"""

import re
from collections import Counter

LIMIAR_BLOCO_UNICO = 8000     
JANELA_CHARS = 4000
JANELA_OVERLAP = 400
MIN_SEGMENTO_CHARS = 40
MIN_RACIO_SEQUENCIA = 0.6

TETO_TOKENS = 2048            # nenhum segmento deve exceder isto
SUBJANELA_OVERLAP_TOKENS = 100  # overlap ao sub-dividir por janela (em tokens aprox.)

RE_ARTIGO = re.compile(
    r"(?:^|\n)#{0,6}\s*\**\s*Art(?:igo)?\.?\s*(\d+)[\.\-]?\s*[º°oO]?(?:[\-–][A-Z])?",
    re.IGNORECASE,
)

RE_NUM_INTERNO = re.compile(r"(?:^|\n)\s*(?:\d+\s*[\.\-\u2013]|[a-z]\))\s", re.IGNORECASE)


def _tokens_por_chars(texto: str) -> int:
    """Estimativa barata: ~4 caracteres por token. Usada se nao houver tokenizer."""
    return max(1, len(texto) // 4)


def _default_contador():
    return _tokens_por_chars


def _tem_articulado_real(texto: str) -> bool:
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


def _por_janela_chars(texto: str, janela_chars=JANELA_CHARS,
                      overlap_chars=JANELA_OVERLAP, rotulo_base="janela",
                      nivel="janela") -> list[dict]:
    segs, i, n = [], 0, len(texto)
    passo = max(1, janela_chars - overlap_chars)
    while i < n:
        pedaco = texto[i:i + janela_chars].strip()
        if pedaco:
            segs.append({"texto": pedaco, "rotulo": f"{rotulo_base}@{i}", "nivel": nivel})
        i += passo
    return segs


# ── NOVO: aplicar o teto de tokens a um segmento ─────────────────────
def _dividir_por_numeracao(texto: str) -> list[str]:
    """Divide um artigo pelas suas subunidades numeradas (1., 2., a), b)...)."""
    cortes = [m.start() for m in RE_NUM_INTERNO.finditer(texto)]
    if len(cortes) < 2:
        return [texto]                      # sem numeracao util
    if cortes[0] > 0:
        cortes = [0] + cortes
    pedacos = []
    for k in range(len(cortes)):
        fim = cortes[k + 1] if k + 1 < len(cortes) else len(texto)
        p = texto[cortes[k]:fim].strip()
        if p:
            pedacos.append(p)
    return pedacos


def _janela_por_tokens(texto: str, teto: int, contar) -> list[str]:
    """
    Janela com overlap dimensionada em TOKENS. Converte o teto de tokens
    para um tamanho aproximado de caracteres usando a densidade real do
    proprio texto (chars/token), para o corte cair perto do teto.
    """
    n_tok = contar(texto)
    if n_tok <= teto:
        return [texto]
    dens = len(texto) / max(1, n_tok)                 # chars por token deste texto
    janela_chars = max(200, int(teto * dens * 0.95))  # 5% de margem
    overlap_chars = int(SUBJANELA_OVERLAP_TOKENS * dens)
    partes = [s["texto"] for s in _por_janela_chars(
        texto, janela_chars, overlap_chars)]
    # garantir teto mesmo apos a conversao (texto denso pode enganar)
    saida = []
    for p in partes:
        if contar(p) <= teto:
            saida.append(p)
        else:
            # corte cego adicional por caracteres, ja bem abaixo do teto
            jc = max(200, int(janela_chars * 0.7))
            saida.extend(s["texto"] for s in _por_janela_chars(p, jc, overlap_chars))
    return saida


def _aplicar_teto(seg: dict, teto: int, contar) -> list[dict]:
    """
    Recebe um segmento (artigo/bloco/janela) e devolve 1+ segmentos, nenhum
    acima do teto. Preserva a estrutura: artigo -> numeracao -> janela.
    """
    if contar(seg["texto"]) <= teto:
        return [seg]

    nivel = seg["nivel"]
    rotulo = seg["rotulo"]

    if nivel == "artigo" and rotulo != "preambulo":
        # 1) tentar pela numeracao interna do artigo
        pedacos, atual = [], ""
        for sub in _dividir_por_numeracao(seg["texto"]):
            cand = (atual + "\n" + sub).strip() if atual else sub
            if contar(cand) <= teto:
                atual = cand
            else:
                if atual:
                    pedacos.append(atual)
                atual = sub
        if atual:
            pedacos.append(atual)
        # 2) pedacos que ainda passem (tabelas) -> janela por tokens
        finais = []
        for p in pedacos:
            finais.extend(_janela_por_tokens(p, teto, contar))
        textos = finais
    else:
        # bloco / janela / preambulo -> janela por tokens
        textos = _janela_por_tokens(seg["texto"], teto, contar)

    return [{"texto": t,
             "rotulo": f"{rotulo}~{i}" if len(textos) > 1 else rotulo,
             "nivel": nivel}
            for i, t in enumerate(textos)]


def segmentar_documento(doc: dict, teto_tokens: int = TETO_TOKENS,
                        contar_tokens=None) -> list[dict]:
    contar = contar_tokens or _default_contador()
    texto = doc["texto"]

    if _tem_articulado_real(texto):
        segmentos = _por_artigo(texto)
    elif len(texto) <= LIMIAR_BLOCO_UNICO:
        segmentos = [{"texto": texto, "rotulo": "documento", "nivel": "bloco"}]
    else:
        segmentos = _por_janela_chars(texto)

    # NOVO: garantir que nenhum segmento excede o teto de tokens
    if teto_tokens:
        expandidos = []
        for s in segmentos:
            expandidos.extend(_aplicar_teto(s, teto_tokens, contar))
        segmentos = expandidos

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


def segmentar_corpus(docs: list[dict], teto_tokens: int = TETO_TOKENS,
                     contar_tokens=None) -> tuple[list[dict], dict]:
    todos, niveis = [], Counter()
    dist_segs = []
    for d in docs:
        segs = segmentar_documento(d, teto_tokens=teto_tokens,
                                   contar_tokens=contar_tokens)
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
        "teto_tokens": teto_tokens,
    }
    print(f"  [SEG] {len(docs)} docs -> {len(todos)} segmentos "
          f"(media {relatorio['segmentos_por_doc_media']:.1f}/doc, "
          f"mediana {relatorio['segmentos_por_doc_mediana']:.0f}, "
          f"max {relatorio['segmentos_por_doc_max']}, teto={teto_tokens} tok)")
    print(f"  [SEG] Nivel: {dict(niveis)}")
    return todos, relatorio


if __name__ == "__main__":
    import json
    from dre_loader import carregar_corpus_dre
    docs = carregar_corpus_dre(relatorio=False)
    segs, rel = segmentar_corpus(docs)
    print(json.dumps(rel, ensure_ascii=False, indent=2))