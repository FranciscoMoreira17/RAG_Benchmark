"""
rag_estruturalMulti.py - RAG Estrutural com o embedder escolhido na CLI
=======================================================================
Junta rag_estrutural.py (E5) e rag_estruturalQwen.py (Qwen3) num só ficheiro
(originais em frameworks/old/). Só muda o embedder denso; enriquecimento,
grafo, BM25 e retrieval são iguais para todos.

Uso:
    python runner.py --framework estrutural_multi --embedder e5   --variante enriquecido
    python runner.py --framework estrutural_multi --embedder qwen --variante enriquecido_grafo

Compatibilidade: com e5 e qwen, collection e nome da condição são IGUAIS aos
dos ficheiros originais. O fingerprint NÃO: o enriquecimento v2 (2026-11-12)
muda o texto embebido, por isso os índices antigos são reconstruídos.
Um embedder novo segue a regra do qwen:
    collection  benchmark_estrutural_<chave>
    condição    estrutural-<chave>-<variante>

Método domain-specific para legislação portuguesa. Estende a
segmentação estrutural DRE (já usada pelo híbrido) com duas técnicas,
AMBAS SEM LLM:

  1. ENRIQUECIMENTO DA REPRESENTAÇÃO (contextual embedding determinístico)
     Prepend ao texto, antes do embedding, um cabeçalho de frontmatter
     com metadados extraídos por REGEX: título do DRE (tipo, subtipo,
     número, data), organização (entidade adjudicante ou 1.ª organização
     nomeada; na falta, cargo de quem assina) e assunto ("Sumário:" ou
     linha de título do texto). Ex.:
       "Aviso 2036/2026/2, de 2 de Fevereiro | Entidade: Câmara Municipal
        de Sintra | Assunto: ..."
     Adaptação da técnica de Contextual Retrieval (Anthropic) usando
     metadados estruturados em vez de descrições geradas por LLM.

  2. EXPANSÃO POR GRAFO
     As citações entre diplomas são explícitas no texto ("nos termos da
     Lei 35/2014"), extraídas por regex. Formam um grafo de citações.
     No retrieval, seguir as citações de um documento recuperado e
     acrescentar os diplomas citados ao contexto.

Formato intermédio: bundles compatíveis com OKF (markdown + frontmatter
YAML). NOTA: OKF é o formato de armazenamento, não o método.

Variantes (lidas contra o UnitLevelSegmentation-hibrido já medido, que é a base:
estrutural + texto cru + híbrido, sem enriquecimento, sem grafo):

  enriquecido        → estrutural + frontmatter no embedding + híbrido
  enriquecido_grafo  → o anterior + expansão por grafo de citações

Retrieval: híbrido (denso + BM25, fusão RRF), como as outras.
"""

import re
import time
import unicodedata
import hashlib

from config import (
    qdrant_client,
    montar_contexto,
    get_collection_name,
    get_groq_client,
    SYSTEM_PROMPT,
    GEN_MODEL,
    GEN_TEMPERATURE,
    GEN_MAX_TOKENS,
    BM25_MODEL,
    BM25_LANGUAGE,
    PREFETCH,
)
from runner import FrameworkBase
from dre_loader import carregar_corpus_dre, corpus_fingerprint
from embedders import get_embedder_spec, carregar_embedder
from dre_segmentador import segmentar_corpus

from qdrant_client import models as qm
from fastembed import SparseTextEmbedding

MAX_GRAFO_EXPANSAO = 3
ENRIQUECIMENTO_VERSAO = 2     # entra no fingerprint (ver config_ingestao)

VARIANTES = {
    "enriquecido":       {"grafo": False},
    "enriquecido_grafo": {"grafo": True},
}

# ── Regex de extracção (sem LLM) ─────────────────────────────
RE_SUMARIO = re.compile(r"Sumário:\s*(.+?)(?:\n|$)", re.IGNORECASE)
# Anúncios de procedimento: "Designação da entidade adjudicante: Município de X"
# (por vezes "NIF e designação ...:\n506811913 - Município de X").
RE_ENTIDADE = re.compile(r"entidade adjudicante:\s*(?:\d{9}\s*-\s*)?(.+)", re.IGNORECASE)

# Restantes tipos: 1.ª organização nomeada no texto (ex.: "Câmara Municipal de
# Sintra", "Agrupamento de Escolas de X", "Unidade Local de Saúde de Y").
# Nome = palavra-chave + palavras com maiúscula/ligações, SEM mudar de linha.
_ORGS = (
    r"Câmara Municipal|Município|Assembleia Municipal|Junta de Freguesia|União das Freguesias|"
    r"Freguesia|Serviços Municipalizados|Comunidade Intermunicipal|Área Metropolitana|"
    r"Universidade|Instituto Politécnico|Instituto Superior|Escola Superior|Instituto|"
    r"Agrupamento de Escolas|Escola Secundária|Unidade Local de Saúde|Centro Hospitalar|Hospital|"
    r"Administração Regional de Saúde|Direção-Geral|Direção Regional|Inspeção-Geral|Ministério|"
    r"Secretaria Regional|Secretaria-Geral|Autoridade|Agência|Comissão|Conselho Superior|"
    r"Ordem dos|Ordem das|Tribunal|Polícia|Guarda Nacional Republicana|Exército|Marinha|"
    r"Força Aérea|Banco de Portugal|Entidade Reguladora|Fundação|Associação"
)
_CAP = r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ.'-]*"
_LIG = r"(?:de|da|do|das|dos|e|d')"
RE_ORGANIZACAO = re.compile(
    rf"\b(?:{_ORGS})\b(?:,?[ \t]+(?:{_LIG}[ \t]+)?{_CAP}|[ \t]+{_LIG}[ \t]+{_CAP})*"
)
# Assinatura no fim: "15 de janeiro de 2026.-O Presidente da Câmara, Nome.\n319955772"
RE_ASSINATURA = re.compile(
    r"(?:\d{4}|\d{2}/\d{2}/\d{4})[ \t]*\.?[ \t]*[-–—][ \t]*(?:O|A|Os|As)[ \t]+([^,\n]{4,150}?),"
    r"[^\n]{2,150}?\.?\s*(?:\n\s*\d{6,}\s*)?$"
)
# Linha de assunto (2.ª linha) só se não for início de texto corrido.
RE_BOILERPLATE = re.compile(
    r"^(?:nos termos|em cumprimento|por despacho|por deliberação|por (?:meu )?despacho|faz-se|"
    r"torna-se|para os devidos|para efeitos|ao abrigo|considerando|louvo|de acordo|na sequência|"
    r"em conformidade|sob proposta|o conselho|a câmara|o presidente|a presidente|eu,|"
    r"modelo de anúncio|\d|[a-z]\)|[ivx]+\b)",
    re.IGNORECASE,
)
RE_FIM_CORTADO = re.compile(r"\b(?:da|do|de|na|no|dos|das|pelo|pela|o|a)$")
# Anúncios de procedimento: o assunto é o objeto do contrato.
RE_CONTRATO = re.compile(r"Designação do contrato:\s*(.+)", re.IGNORECASE)
# Expressões apanhadas por RE_ORGANIZACAO que não são organizações.
RE_NAO_ORGANIZACAO = re.compile(r"^Comissão de Serviço\b", re.IGNORECASE)
RE_CITACAO = re.compile(
    r"(decreto[\s\-]lei|decreto|lei|portaria|despacho|regulamento|aviso|resolução)\s*"
    r"(?:n\.?\s*[º°]?\s*)?(\d+(?:-[A-Za-z]+)?/\d{4})",
    re.IGNORECASE,
)
# Número real do diploma no título. Sufixo alfabético = região e faz parte
# do número ("Decreto Legislativo Regional 9/2021/A" ≠ Decreto 9/2021);
# sufixo numérico = série do DR ("Aviso 44/2026/2") e é ignorado.
RE_NUMERO_TITULO = re.compile(r"(\d+(?:-[A-Za-z]+)?/\d{4}(?:/[A-Za-z]+\b)?)")


def _limpar(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _extrair_entidade(texto: str) -> str | None:
    """Entidade adjudicante (anúncios) ou 1.ª organização nomeada no texto."""
    m = RE_ENTIDADE.search(texto)
    if m and len(_limpar(m.group(1))) > 3:
        return _limpar(m.group(1)).strip(" ,.;:")[:120]
    for zona in (texto[:1500], texto[-500:]):
        curtos = []
        for m in RE_ORGANIZACAO.finditer(zona):
            nome = _limpar(m.group(0)).strip(" ,.;:")
            if RE_NAO_ORGANIZACAO.match(nome):
                continue
            if re.search(rf"[ \t]{_LIG}[ \t]+{_CAP}", nome):   # "... de Sintra"
                return nome[:120]
            if len(nome.split()) >= 2:                         # "Câmara Municipal"
                curtos.append(nome)
        if curtos:
            return curtos[0][:120]
    return None


def _extrair_emitente(texto: str) -> str | None:
    """Cargo de quem assina (ex.: 'Presidente da Câmara'), pela assinatura final."""
    m = RE_ASSINATURA.search(texto[-400:].rstrip())
    return _limpar(m.group(1))[:100] if m else None


def _extrair_assunto(texto: str) -> str | None:
    """'Sumário:' explícito, objeto do contrato (anúncios) ou, na falta, a
    linha de título a seguir ao cabeçalho."""
    sm = RE_SUMARIO.search(texto[:500])
    if sm:
        return _limpar(sm.group(1))[:200]
    ct = RE_CONTRATO.search(texto)
    if ct and len(_limpar(ct.group(1))) > 3:
        return _limpar(ct.group(1))[:200]
    linhas = [ln.strip() for ln in texto.split("\n") if ln.strip()]
    if len(linhas) < 2:
        return None
    l2 = linhas[1]
    if (15 <= len(l2) <= 250 and not RE_BOILERPLATE.match(l2) and not l2.endswith(":")
            and not RE_FIM_CORTADO.search(l2)
            and "entidade adjudicante" not in l2.lower()
            and not re.search(r",\s*(?:Presidente|Diretor|Vereador)", l2)):
        return _limpar(l2)[:200]
    return None


def extrair_frontmatter(doc: dict) -> dict:
    """
    Metadados estruturados por regex — o frontmatter OKF (v2).
    O título do DRE já traz tipo, subtipo, número e data
    ("Aviso (extrato) 4621/2026/2, de 3 de Março").
    """
    texto = doc["texto"]
    fm = {
        "titulo": _limpar(doc.get("titulo") or ""),
        "tipo": doc["tipo"],
        "numero": doc["numero"],
        "data": doc["data_publicacao"],
    }
    assunto = _extrair_assunto(texto)
    if assunto:
        fm["sumario"] = assunto
    entidade = _extrair_entidade(texto)
    if entidade:
        fm["entidade"] = entidade
    else:
        emitente = _extrair_emitente(texto)
        if emitente:
            fm["emitente"] = emitente
    return fm


def extrair_citacoes(doc: dict) -> list[str]:
    """
    Números de diplomas citados no texto, EXCLUINDO o número próprio
    (senão criaria auto-loop no grafo).
    """
    proprio = doc["numero"]
    cits = set()
    for m in RE_CITACAO.finditer(doc["texto"]):
        num = m.group(2)
        if num != proprio:
            cits.add(num)
    return sorted(cits)


def _norm_tipo(t: str) -> str:
    """'Decreto-Lei' / 'decreto lei' → 'decreto-lei'; 'Resolução' → 'resolucao'."""
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[\s\-]+", "-", t.strip())


def _chave_diploma(doc: dict) -> tuple[str, str]:
    """Identidade de um diploma no grafo: (tipo normalizado, número real)."""
    m = RE_NUMERO_TITULO.search(doc.get("titulo") or "")
    numero = m.group(1) if m else doc["numero"]
    return _norm_tipo(doc["tipo"] or ""), numero


def extrair_citacoes_tipadas(doc: dict) -> list[tuple[str, str]]:
    """Citações como (tipo normalizado, número), sem a auto-citação."""
    proprio = _chave_diploma(doc)
    cits = {(_norm_tipo(m.group(1)), m.group(2)) for m in RE_CITACAO.finditer(doc["texto"])}
    cits.discard(proprio)
    return sorted(cits)


def construir_grafo(docs: list[dict]) -> tuple[dict, dict]:
    """
    Grafo de citações, determinístico (depende só do corpus):
      doc_por_chave:    (tipo, número) → doc_id, só para chaves que
                        identificam UM documento (ambíguas são descartadas)
      citacoes_por_doc: doc_id → [(tipo, número) citados]
    """
    por_chave: dict[tuple[str, str], list[str]] = {}
    for d in docs:
        if d["numero"]:
            por_chave.setdefault(_chave_diploma(d), []).append(d["id"])
    doc_por_chave = {k: ids[0] for k, ids in por_chave.items() if len(ids) == 1}
    citacoes_por_doc = {d["id"]: extrair_citacoes_tipadas(d) for d in docs}
    return doc_por_chave, citacoes_por_doc


def texto_enriquecido(doc: dict, fm: dict) -> str:
    """Texto com cabeçalho de frontmatter prepended (para embedding)."""
    if fm.get("titulo"):
        partes = [fm["titulo"]]          # inclui tipo, número e data
    else:
        partes = [f"Tipo: {fm['tipo']}"]
        if fm.get("numero"):
            partes.append(f"Número: {fm['numero']}")
    if fm.get("entidade"):
        partes.append(f"Entidade: {fm['entidade']}")
    elif fm.get("emitente"):
        partes.append(f"Emitente: {fm['emitente']}")
    if fm.get("sumario"):
        partes.append(f"Assunto: {fm['sumario']}")
    cabecalho = " | ".join(partes)
    return f"{cabecalho}\n\n{doc['texto']}"


def _point_id(chunk_id: str) -> int:
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:15], 16)


class Framework(FrameworkBase):

    usa_qdrant = True

    def __init__(self, variante: str = "enriquecido", embedder: str = "e5"):
        if variante not in VARIANTES:
            raise ValueError(f"Variante '{variante}' inválida. {list(VARIANTES)}")
        self.variante = variante
        self.usa_grafo = VARIANTES[variante]["grafo"]
        self.spec = get_embedder_spec(embedder)
        self.legado_e5 = (self.spec.chave == "e5")
        # Identidade da condição: o E5 mantém os nomes do ficheiro original;
        # os restantes seguem a regra do ficheiro *Qwen original.
        if self.legado_e5:
            self.nome = "estrutural"
            self.collection = get_collection_name("estrutural")
            self.nome_run = f"estrutural-{variante}"
        else:
            chave = self.spec.chave
            self.nome = f"estrutural_{chave}"
            self.collection = f"benchmark_estrutural_{chave}"
            self.nome_run = f"estrutural-{chave}-{variante}"
        self._groq = None
        self._sparse = None
        # grafo de citações (construído do corpus, ver construir_grafo):
        # (tipo, número) → doc_id  e  doc_id → [(tipo, número) citados]
        self._citacoes_por_doc = {}
        self._doc_por_chave = {}

    def _get_sparse(self):
        if self._sparse is None:
            self._sparse = SparseTextEmbedding(model_name=BM25_MODEL, language=BM25_LANGUAGE)
        return self._sparse

    def _get_groq(self):
        if self._groq is None:
            self._groq = get_groq_client()
        return self._groq

    def _embedder(self):
        return carregar_embedder(self.spec.chave)

    def config_ingestao(self) -> dict:
        # As duas variantes partilham índice (o enriquecimento é igual;
        # só difere o uso do grafo no retrieval, não na indexação).
        # "enriquecimento": 2 (2026-11-12) — cabeçalho com título, organização
        # e assunto; muda o texto embebido ⇒ fingerprint novo ⇒ re-ingestão.
        if not self.legado_e5:
            return {"pipeline": f"estrutural-enriquecido-{self.spec.chave}",
                    "bm25": BM25_MODEL, "embedder": self.spec.tag,
                    "enriquecimento": ENRIQUECIMENTO_VERSAO}
        return {"pipeline": "estrutural-enriquecido", "bm25": BM25_MODEL,
                "enriquecimento": ENRIQUECIMENTO_VERSAO}

    def descricao(self) -> dict:
        d = {"variante": self.variante, "grafo": self.usa_grafo,
             "retrieval": "hibrido-rrf"}
        if self.usa_grafo:
            # v2 (2026-11-12): ligação por (tipo, número), chaves ambíguas
            # descartadas, segmento do diploma citado escolhido pela query.
            d["grafo_versao"] = 2
        return d

    # ─────────────────────────────────────────────────────────
    # Ingestão
    # ─────────────────────────────────────────────────────────
    def ingerir(self) -> dict:
        docs = carregar_corpus_dre()
        segmentos, rel_seg = segmentar_corpus(docs)

        # Frontmatter + citações por documento (as citações só vão para o
        # payload, como informação; o grafo usado no retrieval é construído
        # a partir do corpus em _garantir_grafo).
        fm_por_doc, cit_por_doc = {}, {}
        for d in docs:
            fm_por_doc[d["id"]] = extrair_frontmatter(d)
            cit_por_doc[d["id"]] = extrair_citacoes(d)

        n_com_citacoes = sum(1 for c in cit_por_doc.values() if c)
        print(f"    [EST] Grafo: {n_com_citacoes}/{len(docs)} docs com citações")

        # Texto enriquecido por segmento (herda frontmatter do doc-pai)
        textos_embed = []
        for s in segmentos:
            fm = fm_por_doc.get(s["doc_id"], {"tipo": s["tipo"], "numero": s["numero"]})
            # todos os segmentos levam o cabeçalho completo do doc-pai
            # (título, entidade/emitente, assunto) — contextual embedding
            doc_stub = {"texto": s["texto"], "tipo": s["tipo"], "numero": s["numero"]}
            textos_embed.append(texto_enriquecido(doc_stub, fm))

        print(f"    [EST] Embedding denso ({self.spec.tag}) de {len(textos_embed)} segmentos enriquecidos...")
        densos = self._embedder().embed_documents(textos_embed)

        print("    [EST] Embedding esparso (BM25)...")
        sparse = self._get_sparse()
        esparsos = list(sparse.embed(textos_embed, batch_size=64))

        if qdrant_client.collection_exists(self.collection):
            qdrant_client.delete_collection(self.collection)
        qdrant_client.create_collection(
            collection_name=self.collection,
            vectors_config={"dense": qm.VectorParams(size=self.spec.dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )

        print(f"    [EST] A indexar {len(segmentos)} pontos...")
        pontos = []
        for i, (seg, dv, sv) in enumerate(zip(segmentos, densos, esparsos)):
            payload = {
                "doc_id": seg["doc_id"],
                "chunk_id": seg["chunk_id"],
                "texto": seg["texto"],                 # texto CRU no payload
                "texto_embebido": textos_embed[i],     # o que foi embebido
                "titulo": seg["titulo"],
                "tipo": seg["tipo"],
                "numero": seg["numero"],
                "citacoes": ",".join(cit_por_doc.get(seg["doc_id"], [])),
                "nivel_segmentacao": seg["nivel_segmentacao"],
            }
            pontos.append(qm.PointStruct(
                id=_point_id(seg["chunk_id"]),
                vector={"dense": dv,
                        "bm25": qm.SparseVector(indices=sv.indices.tolist(),
                                                values=sv.values.tolist())},
                payload=payload,
            ))

        for i in range(0, len(pontos), 100):
            qdrant_client.upsert(collection_name=self.collection,
                                 points=pontos[i:i + 100], wait=True)

        print(f"    [EST] {len(pontos)} pontos → {self.collection}")
        return {
            "n_chunks": len(pontos),
            "n_documentos": len(docs),
            "docs_com_citacoes": n_com_citacoes,
            "segmentacao": rel_seg["distribuicao_nivel_documento"],
            "corpus_fingerprint": corpus_fingerprint(docs),
        }

    # ─────────────────────────────────────────────────────────
    # Índice do grafo (reconstruído se ingestão foi saltada)
    # ─────────────────────────────────────────────────────────
    def _garantir_grafo(self):
        if self._doc_por_chave:
            return
        # Construído sempre do corpus (mesmo resultado com ou sem ingestão).
        self._doc_por_chave, self._citacoes_por_doc = construir_grafo(
            carregar_corpus_dre(relatorio=False))
        n_lig = sum(1 for cits in self._citacoes_por_doc.values()
                    for c in cits if c in self._doc_por_chave)
        print(f"    [EST] Grafo: {len(self._doc_por_chave)} diplomas identificáveis, "
              f"{n_lig} citações resolvidas no corpus")

    # ─────────────────────────────────────────────────────────
    # Retrieval híbrido + expansão por grafo
    # ─────────────────────────────────────────────────────────
    def _retrieve_hibrido(self, qv, query: str, top_k: int):
        sparse = self._get_sparse()
        sv = next(iter(sparse.query_embed(query)))
        sparse_vec = qm.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
        return qdrant_client.query_points(
            collection_name=self.collection,
            prefetch=[
                qm.Prefetch(query=qv, using="dense", limit=PREFETCH),
                qm.Prefetch(query=sparse_vec, using="bm25", limit=PREFETCH),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k, with_payload=True,
        ).points

    def _expandir_grafo(self, resultados, qv) -> list[dict]:
        """
        Segue as citações dos documentos recuperados (por ordem do ranking).
        Só liga a diplomas identificados por (tipo, número) sem ambiguidade;
        do diploma citado entra o segmento mais próximo da query (denso).
        """
        if not self.usa_grafo:
            return []
        self._garantir_grafo()

        ja = {r.payload.get("doc_id") for r in resultados}
        extra = []
        for r in resultados:
            if len(extra) >= MAX_GRAFO_EXPANSAO:
                break
            did = r.payload.get("doc_id")
            for chave in self._citacoes_por_doc.get(did, []):
                if len(extra) >= MAX_GRAFO_EXPANSAO:
                    break
                alvo_id = self._doc_por_chave.get(chave)
                if alvo_id and alvo_id not in ja:
                    pts = qdrant_client.query_points(
                        collection_name=self.collection,
                        query=qv, using="dense",
                        query_filter=qm.Filter(must=[
                            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=alvo_id))
                        ]),
                        limit=1, with_payload=True,
                    ).points
                    if pts:
                        extra.append(pts[0].payload)
                        ja.add(alvo_id)
        return extra

    def _montar(self, query, top_k):
        qv = self._embedder().embed_query(query)
        resultados = self._retrieve_hibrido(qv, query, top_k)
        contextos, metadados = [], []
        for r in resultados:
            pl = r.payload
            contextos.append(pl["texto"])
            metadados.append({
                "doc_id": pl.get("doc_id", "?"),
                "titulo": pl.get("titulo", ""),
                "tipo": pl.get("tipo", ""),
                "numero": pl.get("numero", ""),
                "score": float(r.score),
                "method": "estrutural-hibrido",
            })
        # expansão por grafo
        for pl in self._expandir_grafo(resultados, qv):
            contextos.append(pl["texto"])
            metadados.append({
                "doc_id": pl.get("doc_id", "?"),
                "titulo": pl.get("titulo", ""),
                "tipo": pl.get("tipo", ""),
                "numero": pl.get("numero", ""),
                "score": 0.0,
                "method": "grafo_expansao",
            })
        return contextos, metadados

    def recuperar(self, query: str, top_k: int = 5) -> dict:
        t0 = time.time()
        contextos, metadados = self._montar(query, top_k)
        tempo = time.time() - t0
        return {
            "resposta": "",
            "contextos": contextos,
            "metadados": metadados,
            "tempo_retrieval_s": tempo,
            "tempo_geracao_s": 0.0,
            "n_blocos_recuperados": len(contextos),
            "n_blocos_usados": len(contextos),
            "tokens_contexto": 0,
        }

    def responder(self, query: str, top_k: int = 5) -> dict:
        t0 = time.time()
        contextos, metadados = self._montar(query, top_k)
        tempo_retrieval = time.time() - t0

        blocos = [
            f"--- FONTE ({m['tipo']} {m['numero']}) ---\n{c}"
            for c, m in zip(contextos, metadados)
        ]
        contexto_fmt, n_usados, n_tokens = montar_contexto(blocos)

        t0 = time.time()
        resp = self._get_groq().chat.completions.create(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"CONTEXTO LEGAL:\n{contexto_fmt}\n\nPERGUNTA: {query}"},
            ],
            temperature=GEN_TEMPERATURE, max_tokens=GEN_MAX_TOKENS,
        )
        resposta = resp.choices[0].message.content
        tempo_geracao = time.time() - t0

        return {
            "resposta": resposta,
            "contextos": contextos[:n_usados],
            "metadados": metadados[:n_usados],
            "tempo_retrieval_s": tempo_retrieval,
            "tempo_geracao_s": tempo_geracao,
            "n_blocos_recuperados": len(blocos),
            "n_blocos_usados": n_usados,
            "tokens_contexto": n_tokens,
        }