"""
runner.py - Orquestrador do Benchmark

Executa uma condição experimental: ingestão → queries → exportação.

Uso:
    python runner.py --framework langchain
    python runner.py --framework llamaindex
    python runner.py --framework hibrido --variante denso/esparso/hibrido
    python runner.py --framework estrutural --variante enriquecido/enriquecido_grafo 
    python runner.py --framework langchain --so-retrieval (Apenas realizar retrieval)
    python runner.py --framework todas
    python runner.py --framework langchain --repeticoes 3
"""

import os
import sys
import json
import time
import importlib
from datetime import datetime, timezone

_BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARK_DIR)

from config import (
    get_collection_name,
    qdrant_client,
    garantir_collection,
    snapshot_config,
    fingerprint_ingestao,
    SLEEP_ENTRE_QUERIES,
)

RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), "resultados")

DATASET_PATH = os.getenv("DATASET_PATH", os.path.join(_BENCHMARK_DIR, "dataset_dre.json"))


def carregar_dataset():
    """Carrega o dataset de perguntas DRE (alvo por guid)."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        ds = json.load(f)
    if not ds:
        raise ValueError(f"Dataset vazio: {DATASET_PATH}")
    return ds


class FrameworkBase:
    """
    Contrato de cada condição experimental.

    Atributos opcionais:
      - collection: str      (default: get_collection_name(nome))
      - nome_run: str        (default: nome) — identifica a condição
      - usa_qdrant: bool     (default: True)

    """

    nome: str = ""
    usa_qdrant: bool = True

    def ingerir(self) -> dict:
        raise NotImplementedError

    def responder(self, query: str, top_k: int = 5) -> dict:
        raise NotImplementedError

    def config_ingestao(self) -> dict:
        return {}

    def descricao(self) -> dict:
        return {}



def _path_fingerprint(collection: str) -> str:
    os.makedirs(os.path.join(_BENCHMARK_DIR, ".index_meta"), exist_ok=True)
    return os.path.join(_BENCHMARK_DIR, ".index_meta", f"{collection}.json")


def _fingerprint_guardado(collection: str) -> str | None:
    p = _path_fingerprint(collection)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("fingerprint")


def _guardar_fingerprint(collection: str, fp: str, extra: dict):
    with open(_path_fingerprint(collection), "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fp, "config": extra,
                   "quando": datetime.now(timezone.utc).isoformat()},
                  f, ensure_ascii=False, indent=2)


def _precisa_ingerir(framework: FrameworkBase, collection: str, forcar: bool) -> bool:
    """
    Re-indexa se: (a) foi pedido, (b) a collection está vazia, ou
    (c) a configuração de ingestão mudou desde a última construção.

    O caso (c) é o que impede avaliar um índice construído com um
    parser ou modelo de embedding diferente do que está reportado.
    """
    if forcar:
        print("  Ingestão forçada (--reingerir)")
        return True

    if not framework.usa_qdrant:
        return True

    garantir_collection(collection)
    n = qdrant_client.get_collection(collection).points_count
    if n == 0:
        return True

    fp_atual = fingerprint_ingestao(framework.config_ingestao())
    fp_antigo = _fingerprint_guardado(collection)

    if fp_antigo is None:
        print(f"  Collection tem {n} pontos mas sem fingerprint registado.")
        print("  A reconstruir por segurança (proveniência desconhecida).")
        return True

    if fp_antigo != fp_atual:
        print(f"  Configuração de ingestão mudou ({fp_antigo} → {fp_atual}).")
        print("  A reconstruir o índice.")
        return True

    print(f"  Collection com {n} pontos e fingerprint coincidente — skip ingestão")
    return False


# =====================================================================
# Execução
# =====================================================================
def executar_benchmark(
    framework: FrameworkBase,
    top_k: int = 5,
    reingerir: bool = False,
    repeticao: int = 1,
    so_retrieval: bool = False,
    retomar: bool = False,
) -> dict:
    nome = framework.nome
    nome_run = getattr(framework, "nome_run", nome)
    collection = getattr(framework, "collection", None) or get_collection_name(nome)

    print(f"\n{'=' * 62}")
    print(f"  CONDIÇÃO: {nome_run.upper()}   (repetição {repeticao})")
    print(f"  Collection: {collection}")
    if framework.descricao():
        print(f"  Config: {framework.descricao()}")
    print(f"{'=' * 62}")

    if _precisa_ingerir(framework, collection, reingerir):
        print("\n  A ingerir documentos...")
        t0 = time.time()
        metricas_ingestao = framework.ingerir()
        metricas_ingestao["tempo_s"] = time.time() - t0
        metricas_ingestao["skip"] = False
        _guardar_fingerprint(
            collection,
            fingerprint_ingestao(framework.config_ingestao()),
            framework.config_ingestao(),
        )
        print(f"  Ingestão: {metricas_ingestao.get('n_chunks')} chunks "
              f"em {metricas_ingestao['tempo_s']:.1f}s")
    else:
        n = qdrant_client.get_collection(collection).points_count if framework.usa_qdrant else 0
        metricas_ingestao = {"n_chunks": n, "tempo_s": 0.0, "skip": True}

    dataset = carregar_dataset()
    print(f"\n  Dataset: {len(dataset)} queries")

    respostas_previas = {}
    if retomar:
        import glob as _glob
        padrao = os.path.join(RESULTADOS_DIR, f"llamaindex_r1_20260729_095358.json")
        anteriores = sorted(_glob.glob(padrao))
        if anteriores:
            with open(anteriores[-1], "r", encoding="utf-8") as f:
                prev = json.load(f)
            for r in prev.get("resultados", []):
                # válida = não é erro; guarda por id
                if not r.get("resposta", "").startswith("ERRO"):
                    respostas_previas[r["id"]] = r
            print(f"  [RETOMA] {len(respostas_previas)} respostas válidas reutilizadas "
                  f"de {os.path.basename(anteriores[-1])}")
            print(f"  [RETOMA] Só serão (re)geradas as que faltam ou falharam.")

    resultados = []
    for i, item in enumerate(dataset, 1):
        query = item["query"]
        print(f"\n  [{i}/{len(dataset)}] {query[:78]}...")

        base = {
            "id": item["id"],
            "query": item["query"],
            "tipo": item["tipo"],
            "camada": item.get("camada"),
            "ground_truth": item["ground_truth"],
            "documentos_esperados": item.get("documentos_esperados", []),
            "para_geracao": item.get("para_geracao", False),
        }

        # Geração só nas perguntas marcadas para_geracao=True.
        # As restantes fazem apenas retrieval
        gerar = item.get("para_geracao", False) and not so_retrieval

        if gerar and item["id"] in respostas_previas:
            resultados.append(respostas_previas[item["id"]])
            print(f"    [RETOMA] reutilizada (não consome quota)")
            continue

        try:
            t0 = time.time()
            if gerar:
                r = framework.responder(query, top_k=top_k)
            elif hasattr(framework, "recuperar"):
                r = framework.recuperar(query, top_k=top_k)
                r.setdefault("resposta", "")
                r.setdefault("tempo_geracao_s", 0.0)
            else:
                r = framework.responder(query, top_k=top_k)
            tempo_total = time.time() - t0

            resultados.append({
                **base,
                "resposta": r["resposta"],
                "contextos": r["contextos"],
                "metadados": r["metadados"],
                "tempo_retrieval_s": r["tempo_retrieval_s"],
                "tempo_geracao_s": r["tempo_geracao_s"],
                "tempo_total_s": tempo_total,
                "n_blocos_recuperados": r.get("n_blocos_recuperados", len(r["contextos"])),
                "n_blocos_usados": r.get("n_blocos_usados", len(r["contextos"])),
                "tokens_contexto": r.get("tokens_contexto"),
            })
            marca = "GEN" if gerar else "ret"
            print(f"    [{marca}] {r.get('n_blocos_usados', len(r['contextos']))} blocos "
                  f"| {r.get('tokens_contexto', '?')} tokens | {tempo_total:.2f}s")
            if gerar:
                time.sleep(SLEEP_ENTRE_QUERIES)

        except Exception as e:
            print(f"    → ERRO: {e}")
            resultados.append({
                **base,
                "resposta": f"ERRO: {e}",
                "contextos": [], "metadados": [],
                "tempo_retrieval_s": 0.0, "tempo_geracao_s": 0.0,
                "tempo_total_s": 0.0,
                "n_blocos_recuperados": 0, "n_blocos_usados": 0,
                "tokens_contexto": 0,
            })

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ficheiro = os.path.join(RESULTADOS_DIR, f"{nome_run}_r{repeticao}_{ts}.json")

    export = {
        "framework": nome,
        "condicao": nome_run,
        "repeticao": repeticao,
        "timestamp": ts,
        # Configuração lida em runtime — não escrita à mão.
        "config": snapshot_config({
            "collection": collection,
            "top_k": top_k,
            **framework.descricao(),
        }),
        "ingestao": metricas_ingestao,
        "resultados": resultados,
    }

    with open(ficheiro, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    n_err = sum(1 for r in resultados if r["resposta"].startswith("ERRO"))
    toks = [r["tokens_contexto"] for r in resultados if r.get("tokens_contexto")]
    print(f"\n{'=' * 62}")
    print(f"  Exportado: {ficheiro}")
    print(f"  Queries: {len(resultados)} | Erros: {n_err}")
    if toks:
        print(f"  Tokens de contexto: média {sum(toks)/len(toks):.0f} "
              f"| min {min(toks)} | max {max(toks)}")
    print(f"{'=' * 62}")

    return export


# =====================================================================
# CLI
# =====================================================================
_FRAMEWORKS = {
    "langchain": "frameworks.rag_langchain",
    "llamaindex": "frameworks.rag_llamaindex",
    "hibrido": "frameworks.rag_hibrido",
    "estrutural": "frameworks.rag_estrutural",
}


def carregar_framework(nome: str, variante: str | None = None) -> FrameworkBase:
    if nome not in _FRAMEWORKS:
        raise ValueError(f"Framework '{nome}' desconhecida. "
                         f"Disponíveis: {list(_FRAMEWORKS.keys())}")
    modulo = importlib.import_module(_FRAMEWORKS[nome])
    if variante:
        return modulo.Framework(variante=variante)
    return modulo.Framework()


def main():
    import argparse
    from config import iniciar_tracing

    p = argparse.ArgumentParser(description="Benchmark RAG — legislação portuguesa")
    p.add_argument("--framework", required=True,
                   choices=list(_FRAMEWORKS.keys()) + ["todas"])
    p.add_argument("--variante", default=None,
                   help="Variante de ablação (só para ragokf)")
    p.add_argument("--ablacao", action="store_true",
                   help="Corre todas as variantes de ablação do ragokf")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--repeticoes", type=int, default=1,
                   help="Execuções independentes, para estimar variância")
    p.add_argument("--reingerir", action="store_true")
    p.add_argument("--so-retrieval", action="store_true",
                   help="Só retrieval: sem gerador, sem sleep, sem quota")
    p.add_argument("--retomar", action="store_true",
                   help="Reutiliza respostas válidas de corrida anterior; só gera as que faltam")
    args = p.parse_args()

    def corre(nome, variante=None):
        for rep in range(1, args.repeticoes + 1):
            fw = carregar_framework(nome, variante)
            iniciar_tracing(projeto=f"rag-benchmark-{getattr(fw, 'nome_run', nome)}")
            executar_benchmark(
                fw, top_k=args.top_k,
                reingerir=(args.reingerir and rep == 1),
                repeticao=rep,
                so_retrieval=args.so_retrieval,
                retomar=args.retomar,
            )

    if args.ablacao:
        from frameworks.rag_okf import VARIANTES
        for v in VARIANTES:
            corre("ragokf", v)
    elif args.framework == "todas":
        for nome in _FRAMEWORKS:
            corre(nome)
    else:
        corre(args.framework, args.variante)


if __name__ == "__main__":
    main()