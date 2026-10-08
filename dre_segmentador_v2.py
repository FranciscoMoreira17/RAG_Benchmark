"""
dre_segmentador_v2.py - ULS v2: a maior unidade que cabe na janela
===================================================================
Princípio: cada segmento é a MAIOR unidade estrutural que cabe na janela
do embedder (teto_tokens). Por ordem:

  1. DOCUMENTO - se o documento inteiro cabe na janela, é um só segmento
                 (com ou sem articulado). Sem cabeçalho (o início do texto
                 já identifica o documento).
  2. GRUPO DE ARTIGOS - documento com articulado que não cabe: artigos
                 CONSECUTIVOS agrupados até encher a janela. Nenhum artigo
                 fica isolado só por ser curto. Um artigo que sozinho não
                 cabe é dividido pela numeração interna (1., 2., a)...) e,
                 em último caso, por janela de tokens.
  3. JANELA    - documento longo sem articulado: janelas com overlap,
                 dimensionadas em tokens.

Nos ramos 2 e 3, cada segmento leva à frente um CABEÇALHO do diploma
(título + início do texto), porque perdeu o contexto do documento.
O cabeçalho conta para o teto.

Cada segmento devolve:
  texto       - texto cru (para o contexto do gerador)
  texto_embed - cabeçalho + texto (o que se vetoriza e indexa no BM25)
"""

import re
from collections import Counter

TETO_TOKENS = 2048
OVERLAP_TOKENS = 100          # overlap das janelas (ramo 3 e cortes de último recurso)
CABECALHO_MAX_CHARS = 200     # início do texto usado no cabeçalho
MIN_RACIO_SEQUENCIA = 0.6

RE_ARTIGO = re.compile(
    r"(?:^|\n)#{0,6}\s*\**\s*Art(?:igo)?\.?\s*(\d+)[\.\-]?\s*[º°oO]?(?:[\-–][A-Z])?",
    re.IGNORECASE,
)
RE_NUM_INTERNO = re.compile(r"(?:^|\n)\s*(?:\d+\s*[\.\-–]|[a-z]\))\s", re.IGNORECASE)


def _tokens_por_chars(texto: str) -> int:
    return max(1, len(texto) // 4)


def _tem_articulado_real(texto: str) -> bool:
    nums = [int(m.group(1)) for m in RE_ARTIGO.finditer(texto)]
    if len(nums) < 2 or nums[0] > 2:
        return False
    saltos_ok = sum(1 for i in range(1, len(nums)) if nums[i] - nums[i - 1] in (0, 1))
    return (saltos_ok / (len(nums) - 1)) >= MIN_RACIO_SEQUENCIA


# ── cabeçalho ────────────────────────────────────────────────────────
RE_NOME_DIPLOMA = re.compile(
    r"\b(?:Regulamento|C[óo]digo|Estatutos?|Normas?)\s+(?:[Mm]unicipal\s+)?"
    r"(?:d[aoe]s?|para|sobre|relativo)\s+[^\n;:.]{5,150}"
)

def construir_cabecalho(doc: dict) -> str:
    texto = doc["texto"]
    m = RE_NOME_DIPLOMA.search(texto[:4000])
    assunto = re.sub(r"\s+", " ", m.group(0)).strip() if m else ""
    inicio = re.sub(r"\s+", " ", texto).strip()
    if len(inicio) > CABECALHO_MAX_CHARS:
        corte = inicio.rfind(" ", 0, CABECALHO_MAX_CHARS)
        inicio = inicio[:corte if corte > 0 else CABECALHO_MAX_CHARS] + " …"
    partes = [f"[{doc['titulo']}]"]
    if assunto:
        partes.append(f"Assunto: {assunto}.")
    partes.append(inicio)
    return " ".join(partes)


# ── unidades ─────────────────────────────────────────────────────────
def _unidades_artigo(texto: str) -> list[tuple[str, str]]:
    """(rotulo, texto) para preâmbulo + cada artigo. NÃO descarta artigos curtos."""
    matches = list(RE_ARTIGO.finditer(texto))
    unidades = []
    if matches[0].start() > 0:
        pre = texto[:matches[0].start()].strip()
        if pre:
            unidades.append(("preambulo", pre))
    for i, m in enumerate(matches):
        fim = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
        conteudo = texto[m.start():fim].strip()
        if conteudo:
            unidades.append((f"Artigo {m.group(1)}", conteudo))
    return unidades


def _janela_por_tokens(texto: str, teto: int, contar) -> list[str]:
    """Janelas com overlap, dimensionadas em tokens pela densidade do próprio texto."""
    n_tok = contar(texto)
    if n_tok <= teto:
        return [texto]
    dens = len(texto) / max(1, n_tok)                       # chars por token
    jan = max(200, int(teto * dens * 0.95))
    ovl = int(OVERLAP_TOKENS * dens)
    passo = max(1, jan - ovl)
    saida, i = [], 0
    while i < len(texto):
        p = texto[i:i + jan].strip()
        if p:
            if contar(p) > teto:                            # texto denso: corta mais
                saida.extend(_janela_por_tokens(p[: int(len(p) * 0.7)], teto, contar))
                saida.extend(_janela_por_tokens(p[int(len(p) * 0.7) - ovl:], teto, contar))
            else:
                saida.append(p)
        i += passo
    return saida


def _partir_artigo(rotulo: str, texto: str, teto: int, contar) -> list[tuple[str, str]]:
    """Artigo que sozinho excede o teto: numeração interna, depois janela."""
    subs = [m.start() for m in RE_NUM_INTERNO.finditer(texto)]
    partes = []
    if len(subs) >= 2:
        cortes = ([0] if subs[0] > 0 else []) + subs
        atual = ""
        for k, ini in enumerate(cortes):
            fim = cortes[k + 1] if k + 1 < len(cortes) else len(texto)
            sub = texto[ini:fim].strip()
            cand = f"{atual}\n{sub}".strip() if atual else sub
            if contar(cand) <= teto:
                atual = cand
            else:
                if atual:
                    partes.append(atual)
                atual = sub
        if atual:
            partes.append(atual)
    else:
        partes = [texto]
    finais = []
    for p in partes:
        finais.extend(_janela_por_tokens(p, teto, contar))
    return [(f"{rotulo}~{i}" if len(finais) > 1 else rotulo, t)
            for i, t in enumerate(finais)]


def _agrupar_artigos(unidades, teto: int, contar) -> list[tuple[str, str]]:
    """Agrupa unidades consecutivas até encher o teto (soma de tokens + margem)."""
    grupos, atual, rot_ini, rot_fim, tok_atual = [], [], None, None, 0

    def fechar():
        if atual:
            rot = rot_ini if rot_ini == rot_fim else f"{rot_ini} a {rot_fim}"
            grupos.append((rot, "\n".join(atual)))

    for rotulo, texto in unidades:
        n = contar(texto)
        if n > teto:                                       # artigo gigante
            fechar()
            atual, rot_ini, rot_fim, tok_atual = [], None, None, 0
            grupos.extend(_partir_artigo(rotulo, texto, teto, contar))
            continue
        if atual and tok_atual + n + 1 > teto:
            fechar()
            atual, rot_ini, tok_atual = [], None, 0
        if not atual:
            rot_ini = rotulo
        atual.append(texto)
        rot_fim = rotulo
        tok_atual += n + 1
    fechar()

    # verificação final: a soma de tokens é aproximada
    seguros = []
    for rot, txt in grupos:
        if contar(txt) <= teto:
            seguros.append((rot, txt))
        else:
            seguros.extend((f"{rot}~{i}", t) for i, t in
                           enumerate(_janela_por_tokens(txt, teto, contar)))
    return seguros


# ── API ──────────────────────────────────────────────────────────────
def segmentar_documento(doc: dict, teto_tokens: int = TETO_TOKENS,
                        contar_tokens=None) -> list[dict]:
    contar = contar_tokens or _tokens_por_chars
    texto = doc["texto"]

    if contar(texto) <= teto_tokens:
        segs = [("documento", texto, "documento", False)]
    else:
        cab = construir_cabecalho(doc)
        teto_util = max(256, teto_tokens - contar(cab) - 2)
        if _tem_articulado_real(texto):
            grupos = _agrupar_artigos(_unidades_artigo(texto), teto_util, contar)
            segs = [(rot, t, "grupo_artigos", True) for rot, t in grupos]
        else:
            janelas = _janela_por_tokens(texto, teto_util, contar)
            segs = [(f"janela@{i}", t, "janela", True) for i, t in enumerate(janelas)]

    saida = []
    for j, (rotulo, t, nivel, com_cab) in enumerate(segs):
        saida.append({
            "doc_id": doc["id"],
            "chunk_id": f"{doc['id']}::{j}",
            "texto": t,
            "texto_embed": f"{construir_cabecalho(doc)}\n\n{t}" if com_cab else t,
            "rotulo": rotulo,
            "nivel_segmentacao": nivel,
            "titulo": doc["titulo"],
            "tipo": doc["tipo"],
            "numero": doc["numero"],
            "data_publicacao": doc["data_publicacao"],
            "url": doc["url"],
        })
    return saida


def segmentar_corpus(docs: list[dict], teto_tokens: int = TETO_TOKENS,
                     contar_tokens=None) -> tuple[list[dict], dict]:
    todos, niveis, dist = [], Counter(), []
    for d in docs:
        segs = segmentar_documento(d, teto_tokens, contar_tokens)
        todos.extend(segs)
        dist.append(len(segs))
        niveis[segs[0]["nivel_segmentacao"]] += 1

    from statistics import mean, median
    rel = {
        "n_documentos": len(docs),
        "n_segmentos": len(todos),
        "segmentos_por_doc_media": mean(dist) if dist else 0,
        "segmentos_por_doc_mediana": median(dist) if dist else 0,
        "segmentos_por_doc_max": max(dist) if dist else 0,
        "distribuicao_nivel_documento": dict(niveis),
        "teto_tokens": teto_tokens,
    }
    print(f"  [SEG-v2] {len(docs)} docs -> {len(todos)} segmentos "
          f"(media {rel['segmentos_por_doc_media']:.1f}/doc, "
          f"max {rel['segmentos_por_doc_max']}, teto={teto_tokens} tok)")
    print(f"  [SEG-v2] Nivel por documento: {dict(niveis)}")
    return todos, rel


if __name__ == "__main__":
    import json
    from dre_loader import carregar_corpus_dre
    docs = carregar_corpus_dre(relatorio=False)
    segs, rel = segmentar_corpus(docs)          # estimativa chars/4 (sem tokenizer)
    print(json.dumps(rel, ensure_ascii=False, indent=2))