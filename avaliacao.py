"""
avaliacao.py - Avaliação do Benchmark
======================================================
O corpus DRE é heterogéneo e a unidade de recuperação é o DOCUMENTO,
identificado pelo seu `guid`. O alvo de retrieval de cada pergunta
está em `documentos_esperados` (lista de guids). 

Camadas, por ordem de peso:
  1. RETRIEVAL (determinístico, sem LLM) - Precision@k, Recall@k,
     MRR, nDCG@k, hit_rate@k. Alvo = guid. Estratificado por tipo
     de pergunta (identificador / semantica) e por camada.
  2. GERAÇÃO com juiz LLM - Só sobre perguntas para_geracao=True.
  3. CLÁSSICAS (BLEU, ROUGE-L, BERTScore).
  4. Bootstrap de IC - obrigatório com n pequeno.

Uso:
    python avaliacao.py --pasta resultados/ --so-retrieval
    python avaliacao.py --ficheiro resultados/hibrido-denso_r1_....json
    python avaliacao.py --comparar
    python avaliacao.py --ficheiro X.json --langsmith
"""

import os
import sys
import json
import glob
import math
import random
import argparse
import statistics
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

load_dotenv()

RESULTADOS_DIR = os.path.join(_DIR, "resultados")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini/gemini-2.5-flash")

JUDGE_MAX_WORKERS = int(os.getenv("JUDGE_MAX_WORKERS", "1"))
JUDGE_MODEL_GROQ = os.getenv("JUDGE_MODEL_GROQ", "openai/gpt-oss-120b")
JUDGE_TIMEOUT = int(os.getenv("JUDGE_TIMEOUT", "180"))

# Segundos a dormir antes de CADA chamada ao juiz. Com TPM=8000 do
# gpt-oss-120b e contextos grandes, ~8s evita bater no limite.
JUDGE_SLEEP = float(os.getenv("JUDGE_SLEEP", "8"))


def _get_client_groq():
    """Cliente Groq (OpenAI-compatible) para o juiz próprio."""
    from openai import OpenAI
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )


def _make_judge_llm():
    """
    ChatOpenAI apontado ao Groq, com um rate limiter que dorme antes
    de cada pedido.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.rate_limiters import InMemoryRateLimiter

    # requests_per_second baixo = espaçamento forçado.
    # 1/JUDGE_SLEEP pedidos por segundo → ~JUDGE_SLEEP s entre pedidos.
    limiter = InMemoryRateLimiter(
        requests_per_second=max(0.01, 1.0 / JUDGE_SLEEP),
        check_every_n_seconds=0.5,
        max_bucket_size=1,
    )
    return ChatOpenAI(
        model=JUDGE_MODEL_GROQ,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
        rate_limiter=limiter,
    )


# =====================================================================
# Extracção de guids recuperados (uniforme entre condições)
# =====================================================================
def guids_recuperados(resultado: dict) -> list[str]:
    """
    Lista de doc_id (guid) por ordem de ranking. Todas as condições
    escrevem `doc_id` no payload/metadados.
    """
    return [m.get("doc_id", "") for m in resultado.get("metadados", [])]


# =====================================================================
# Bootstrap
# =====================================================================
def bootstrap_ic(valores: list[float], n: int = 2000, alpha: float = 0.05) -> list[float]:
    if not valores:
        return [0.0, 0.0]
    if len(valores) == 1:
        return [round(valores[0], 4), round(valores[0], 4)]
    rng = random.Random(42)
    m = len(valores)
    medias = []
    for _ in range(n):
        amostra = [valores[rng.randrange(m)] for _ in range(m)]
        medias.append(sum(amostra) / m)
    medias.sort()
    lo = medias[int((alpha / 2) * n)]
    hi = medias[int((1 - alpha / 2) * n) - 1]
    return [round(lo, 4), round(hi, 4)]


# =====================================================================
# 1. MÉTRICAS DE RETRIEVAL
# =====================================================================
def _dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def metricas_retrieval(resultados: list[dict], k: int = 5) -> dict:
    """
    Precision@k, Recall@k, MRR, nDCG@k, hit_rate@k, com alvo por guid.
    Só considera perguntas com documentos_esperados não vazio.
    """
    agg = defaultdict(list)
    for r in resultados:
        alvos = set(r.get("documentos_esperados") or [])
        if not alvos:
            continue

        vistos, docs_unicos = set(), []
        for g in guids_recuperados(r):
            if g and g not in vistos:
                vistos.add(g)
                docs_unicos.append(g)
        recuperados = docs_unicos[:k]

        rels = [1.0 if g in alvos else 0.0 for g in recuperados]
        acertos = len(alvos & set(recuperados))

        agg["precision"].append(sum(rels) / max(1, len(recuperados)))
        agg["recall"].append(acertos / len(alvos))
        agg["mrr"].append(next((1/(i+1) for i,x in enumerate(rels) if x>0), 0.0))
        n_rel = int(sum(rels))
        ideal = _dcg([1.0]*min(n_rel, k)) if n_rel > 0 else 0.0
        agg["ndcg"].append(_dcg(rels)/ideal if ideal > 0 else 0.0)
        agg["hit"].append(1.0 if acertos > 0 else 0.0)

    if not agg["precision"]:
        return {}

    return {
        f"precision@{k}": statistics.mean(agg["precision"]),
        f"recall@{k}": statistics.mean(agg["recall"]),
        "mrr": statistics.mean(agg["mrr"]),
        f"ndcg@{k}": statistics.mean(agg["ndcg"]),
        f"hit_rate@{k}": statistics.mean(agg["hit"]),
        "n_queries": len(agg["precision"]),
        "ic95_recall": bootstrap_ic(agg["recall"]),
        "ic95_hit": bootstrap_ic(agg["hit"]),
    }


def metricas_estratificadas(resultados: list[dict], k: int = 5) -> dict:
    """
    Estratifica por `tipo` (identificador / semantica) e por `camada`.
    É aqui que aparece o resultado interessante: espera-se que o BM25
    domine nas de identificador e o denso/híbrido nas semânticas.
    """
    out = {}
    por_tipo = defaultdict(list)
    for r in resultados:
        por_tipo[r.get("tipo", "?")].append(r)
    for tipo, rs in por_tipo.items():
        m = metricas_retrieval(rs, k)
        if m:
            out[f"tipo={tipo}"] = m
    return out



def verificar_orcamento(resultados: list[dict]) -> dict:
    toks = [r.get("tokens_contexto") for r in resultados if r.get("tokens_contexto")]
    blocos = [r.get("n_blocos_usados") for r in resultados if r.get("n_blocos_usados")]
    if not toks:
        return {}
    return {
        "tokens_media": statistics.mean(toks),
        "tokens_desvio": statistics.pstdev(toks) if len(toks) > 1 else 0.0,
        "tokens_min": min(toks),
        "tokens_max": max(toks),
        "blocos_media": statistics.mean(blocos) if blocos else 0,
    }


# =====================================================================
# 2. GERAÇÃO - juiz LLM (só perguntas para_geracao=True)
# =====================================================================
def _validos_geracao(resultados: list[dict]) -> list[dict]:
    """Perguntas marcadas para geração, sem erro, com ground truth."""
    return [r for r in resultados
            if r.get("para_geracao")
            and not r["resposta"].startswith("ERRO")
            and r.get("ground_truth")]


def _sem_erro(resultados: list[dict]) -> list[dict]:
    """Reference-free: só exclui erros (não exige ground truth)."""
    return [r for r in resultados if not r["resposta"].startswith("ERRO")]

# Não está a ser utilizado de momento.
def avaliar_ragas(resultados: list[dict]) -> dict:
    try:
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import (
            Faithfulness, ResponseRelevancy,
            LLMContextPrecisionWithoutReference, LLMContextRecall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        print(f"    [RAGAS] Dependências em falta: {e}")
        return {}

    validos = _validos_geracao(resultados)
    if not validos:
        print("    [RAGAS] Sem perguntas para_geracao válidas")
        return {}

    # Juiz via Groq com rate limiter (dorme antes de cada chamada)
    llm = LangchainLLMWrapper(_make_judge_llm())
    # Embedder local (E5) para as métricas que precisam de embeddings
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
    ))

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=r["query"],
            response=r["resposta"],
            retrieved_contexts=r["contextos"] or [""],
            reference=r["ground_truth"],
        ) for r in validos
    ])

    metricas = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=emb),
        LLMContextPrecisionWithoutReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]

    print(f"    [RAGAS] {len(validos)} perguntas, juiz={JUDGE_MODEL_GROQ} (Groq)...")
    try:
        run_cfg = RunConfig(max_workers=JUDGE_MAX_WORKERS, timeout=JUDGE_TIMEOUT,
                            max_retries=5)
        res = evaluate(dataset=dataset, metrics=metricas, llm=llm,
                       embeddings=emb, run_config=run_cfg)
        df = res.to_pandas()
        out = {}
        for col in ("faithfulness", "answer_relevancy",
                    "llm_context_precision_without_reference", "context_recall"):
            if col in df.columns:
                serie = df[col].dropna().tolist()
                if serie:
                    out[col] = float(sum(serie) / len(serie))
                    out[f"{col}_ic95"] = bootstrap_ic(serie)
        return out
    except Exception as e:
        print(f"    [RAGAS] Erro: {e}")
        return {}

# Não está a ser utilizado de momento.
def avaliar_ragchecker(resultados: list[dict]) -> dict:
    try:
        from ragchecker import RAGResults, RAGChecker
        from ragchecker.metrics import all_metrics
    except ImportError:
        print("    [RAGChecker] Não instalado — skip")
        return {}

    if not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

    validos = _validos_geracao(resultados)
    if not validos:
        return {}

    payload = [{
        "query_id": r["id"],
        "query": r["query"],
        "gt_answer": r["ground_truth"],
        "response": r["resposta"],
        "retrieved_context": [{"text": c} for c in (r["contextos"] or [""])],
    } for r in validos]

    print(f"    [RAGChecker] {len(payload)} perguntas, juiz=groq/{JUDGE_MODEL_GROQ}...")
    # litellm respeita estas envs: limita pedidos por minuto e ativa
    # o backoff automático em rate limits.
    os.environ["LITELLM_LOG"] = "ERROR"
    try:
        import litellm
        litellm.drop_params = True
        litellm.num_retries = 10          # backoff automático em rate limit
        litellm.request_timeout = JUDGE_TIMEOUT
    except Exception:
        pass
    try:
        checker = RAGChecker(
            extractor_name=f"groq/{JUDGE_MODEL_GROQ}", checker_name=f"groq/{JUDGE_MODEL_GROQ}",
            batch_size_extractor=1, batch_size_checker=1,
        )
        dados = RAGResults.from_json(json.dumps({"results": payload}, ensure_ascii=False))
        checker.evaluate(dados, all_metrics)
        df = pd.DataFrame([r.metrics for r in dados.results])
        out = {}
        for col in df.columns:
            serie = df[col].dropna().tolist()
            if serie:
                out[col] = float(sum(serie) / len(serie))
        return out
    except Exception as e:
        print(f"    [RAGChecker] Erro: {e}")
        return {}


def avaliar_phoenix(resultados: list[dict]) -> dict:
    try:
        from phoenix.evals import HallucinationEvaluator, QAEvaluator, run_evals
        from phoenix.evals.models import OpenAIModel
    except ImportError as e:
        print(f"    [Phoenix] evals indisponíveis: {e}")
        return {}

    validos = _validos_geracao(resultados)
    if not validos:
        return {}

    df = pd.DataFrame([{
        "input": r["query"],
        "output": r["resposta"],
        "reference": "\n\n".join(r["contextos"]) or " ",
    } for r in validos])

    try:
        modelo = OpenAIModel(
            model=JUDGE_MODEL_GROQ,
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
        halluc_df, qa_df = run_evals(
            dataframe=df,
            evaluators=[HallucinationEvaluator(modelo), QAEvaluator(modelo)],
            provide_explanation=True,
            concurrency=1,
        )
        return {
            "phoenix_factual_rate": float((halluc_df["label"] == "factual").mean()),
            "phoenix_qa_correctness": float((qa_df["label"] == "correct").mean()),
        }
    except Exception as e:
        print(f"    [Phoenix] Erro: {e}")
        return {}


# =====================================================================
# 3. CLÁSSICAS (anexo)
# =====================================================================
def avaliar_classicas(resultados: list[dict]) -> dict:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    validos = _validos_geracao(resultados)
    if not validos:
        return {}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method1

    bleus, rouges = [], []
    for r in validos:
        bleus.append(sentence_bleu([r["ground_truth"].split()],
                                   r["resposta"].split(),
                                   smoothing_function=smooth))
        rouges.append(scorer.score(r["ground_truth"], r["resposta"])["rougeL"].fmeasure)

    out = {
        "bleu": statistics.mean(bleus),
        "rouge_l": statistics.mean(rouges),
        "rouge_l_ic95": bootstrap_ic(rouges),
        "_nota": ("Sobreposição lexical com referência única. Anexo, "
                  "não métrica principal."),
    }
    try:
        from bert_score import score as bert_fn
        print("    [Clássicas] BERTScore...")
        _, _, F1 = bert_fn([r["resposta"] for r in validos],
                           [r["ground_truth"] for r in validos],
                           lang="pt", verbose=False)
        out["bertscore_f1"] = float(F1.mean())
    except Exception as e:
        print(f"    [Clássicas] BERTScore indisponível: {e}")
    return out


# =====================================================================
# LangSmith
# =====================================================================
def avaliar_com_langsmith(caminho: str):
    try:
        from langsmith import Client
        from langsmith.evaluation import evaluate
    except ImportError:
        print("    [LangSmith] Não instalado — skip")
        return

    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        print("    [LangSmith] Sem API key — skip")
        return

    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)

    condicao = dados.get("condicao", dados["framework"])
    resultados = dados["resultados"]
    client = Client()

    nome_ds = os.getenv("LANGSMITH_DATASET", "rag-benchmark-dre")
    if client.has_dataset(dataset_name=nome_ds):
        ds = client.read_dataset(dataset_name=nome_ds)
    else:
        ds = client.create_dataset(dataset_name=nome_ds)
        client.create_examples(
            dataset_id=ds.id,
            examples=[{
                "inputs": {"query": r["query"]},
                "outputs": {
                    "ground_truth": r.get("ground_truth", ""),
                    "documentos_esperados": r.get("documentos_esperados", []),
                },
                "metadata": {"tipo": r["tipo"], "id": r["id"],
                             "camada": r.get("camada")},
            } for r in resultados if r.get("documentos_esperados")],
        )
        print(f"    [LangSmith] Dataset criado: {nome_ds}")

    por_query = {r["query"]: r for r in resultados}

    def target(inputs: dict) -> dict:
        r = por_query.get(inputs["query"], {})
        return {
            "resposta": r.get("resposta", ""),
            "metadados": r.get("metadados", []),
            "tokens_contexto": r.get("tokens_contexto", 0),
        }

    def hit_guid(outputs, reference_outputs):
        alvos = set(reference_outputs.get("documentos_esperados") or [])
        if not alvos:
            return {"key": "hit_guid", "score": None}
        obtidos = {m.get("doc_id") for m in outputs.get("metadados", [])}
        return {"key": "hit_guid", "score": 1.0 if alvos & obtidos else 0.0}

    def recall_guid(outputs, reference_outputs):
        alvos = set(reference_outputs.get("documentos_esperados") or [])
        if not alvos:
            return {"key": "recall_guid", "score": None}
        obtidos = {m.get("doc_id") for m in outputs.get("metadados", [])}
        return {"key": "recall_guid", "score": len(alvos & obtidos) / len(alvos)}

    def sem_erro(outputs):
        return {"key": "sem_erro",
                "score": 0.0 if outputs["resposta"].startswith("ERRO") else 1.0}

    evaluate(
        target, data=nome_ds,
        evaluators=[hit_guid, recall_guid, sem_erro],
        experiment_prefix=condicao,
        metadata={"condicao": condicao, **dados.get("config", {})},
        max_concurrency=4,
    )
    print(f"    [LangSmith] Experiência registada: {condicao}")


# =====================================================================
# Avaliação de um ficheiro
# =====================================================================
def avaliar_ficheiro(caminho: str, so_retrieval: bool = False,
                     usar_langsmith: bool = False, k: int = 5,
                     sem_ragchecker: bool = False) -> dict:
    print(f"\n{'=' * 62}")
    print(f"  AVALIAÇÃO: {os.path.basename(caminho)}")
    print(f"{'=' * 62}")

    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)

    resultados = dados["resultados"]
    validos = _sem_erro(resultados)

    resumo = {
        "condicao": dados.get("condicao", dados["framework"]),
        "framework": dados["framework"],
        "repeticao": dados.get("repeticao", 1),
        "config": dados.get("config", {}),
        "n_queries": len(resultados),
        "n_erros": len(resultados) - len(validos),
        "orcamento_contexto": verificar_orcamento(resultados),
        "tempos": {
            "retrieval_medio_s": statistics.mean([r["tempo_retrieval_s"] for r in validos]) if validos else 0,
            "geracao_media_s": statistics.mean([r["tempo_geracao_s"] for r in validos]) if validos else 0,
            "total_medio_s": statistics.mean([r["tempo_total_s"] for r in validos]) if validos else 0,
        },
    }

    print("\n  [1] Retrieval (guid, determinístico)...")
    resumo["retrieval"] = metricas_retrieval(resultados, k=k)
    resumo["retrieval_estratificado"] = metricas_estratificadas(resultados, k=k)

    if not so_retrieval:
        from juiz_llm import avaliar_llm_juiz
        print("\n  [2] Juiz LLM (faithfulness, relevancy, correctness)...")
        r = avaliar_llm_juiz(resultados, _get_client_groq)
        if r:
            resumo["juiz_llm"] = r
        print("\n  [3] Clássicas (BLEU/ROUGE/BERTScore, anexo)...")
        resumo["classicas"] = avaliar_classicas(resultados)

    if usar_langsmith:
        print("\n  [6] LangSmith...")
        avaliar_com_langsmith(caminho)

    destino = caminho.replace(".json", "_avaliacao.json")
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    imprimir_resumo(resumo)
    print(f"\n  Guardado: {destino}")
    return resumo


def imprimir_resumo(resumo: dict):
    print(f"\n  Condição: {resumo['condicao']}  "
          f"({resumo['n_queries']} queries, {resumo['n_erros']} erros)")

    orc = resumo.get("orcamento_contexto") or {}
    if orc:
        print(f"  Contexto: {orc['tokens_media']:.0f} tokens médios "
              f"(σ={orc['tokens_desvio']:.0f})")

    m = resumo.get("retrieval") or {}
    if m:
        print(f"\n  Retrieval (todas, n={m['n_queries']}):")
        for kk, vv in m.items():
            if kk in ("n_queries",) or kk.startswith("ic95"):
                continue
            print(f"    {kk:<20} {vv:.4f}")
        print(f"    recall IC95%         {m.get('ic95_recall')}")

    for chave, mm in (resumo.get("retrieval_estratificado") or {}).items():
        print(f"\n  Retrieval [{chave}]  n={mm['n_queries']}  "
              f"hit@5={mm['hit_rate@5']:.3f}  recall@5={mm['recall@5']:.3f}  "
              f"ndcg@5={mm['ndcg@5']:.3f}")

    for bloco in ("juiz_llm", "classicas"):
        if bloco in resumo and resumo[bloco]:
            print(f"\n  {bloco.upper()}:")
            for kk, vv in resumo[bloco].items():
                if kk.startswith("_") or kk.endswith("_ic95"):
                    continue
                if isinstance(vv, (int, float)):
                    print(f"    {kk:<36} {vv:.4f}")


# =====================================================================
# Comparação
# =====================================================================
def comparar():
    ficheiros = glob.glob(os.path.join(RESULTADOS_DIR, "*_avaliacao.json"))
    if not ficheiros:
        print("Sem avaliações. Corre primeiro --pasta resultados/")
        return

    por_condicao = defaultdict(list)
    for p in sorted(ficheiros):
        with open(p, "r", encoding="utf-8") as f:
            r = json.load(f)
        por_condicao[r["condicao"]].append(r)

    linhas = []
    for cond, reps in sorted(por_condicao.items()):
        def media(getter):
            vals = [v for v in (getter(r) for r in reps) if v is not None]
            return statistics.mean(vals) if vals else None

        def desvio(getter):
            vals = [v for v in (getter(r) for r in reps) if v is not None]
            return statistics.pstdev(vals) if len(vals) > 1 else 0.0

        def estrat(r, chave, metrica):
            return (r.get("retrieval_estratificado", {}).get(chave) or {}).get(metrica)

        linhas.append({
            "condicao": cond,
            "n_exec": len(reps),
            "precision@5": media(lambda r: (r.get("retrieval") or {}).get("precision@5")),
            "hit@5": media(lambda r: (r.get("retrieval") or {}).get("hit_rate@5")),
            "hit@5_sd": desvio(lambda r: (r.get("retrieval") or {}).get("hit_rate@5")),
            "recall@5": media(lambda r: (r.get("retrieval") or {}).get("recall@5")),
            "ndcg@5": media(lambda r: (r.get("retrieval") or {}).get("ndcg@5")),
            "mrr": media(lambda r: (r.get("retrieval") or {}).get("mrr")),
            "hit_ident": media(lambda r: estrat(r, "tipo=identificador", "hit_rate@5")),
            "hit_seman": media(lambda r: estrat(r, "tipo=semantica", "hit_rate@5")),
            "faithful": media(lambda r: (r.get("juiz_llm") or {}).get("faithfulness")),
            "relevancy": media(lambda r: (r.get("juiz_llm") or {}).get("answer_relevancy")),
            "correctness": media(lambda r: (r.get("juiz_llm") or {}).get("correctness")),
            "tokens_ctx": media(lambda r: (r.get("orcamento_contexto") or {}).get("tokens_media")),
            "latencia_s": media(lambda r: (r.get("tempos") or {}).get("total_medio_s")),
        })

    df = pd.DataFrame(linhas)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)

    print(f"\n{'=' * 110}")
    print("  COMPARAÇÃO DE CONDIÇÕES")
    print(f"{'=' * 110}\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n  Leitura:")
    print("   • hit@5 global é a métrica principal de retrieval.")
    print("   • hit_ident vs hit_seman revela onde cada estratégia ganha:")
    print("     espera-se BM25 forte em identificador, denso/híbrido em semântica.")
    print("   • Diferenças < desvio entre execuções não são interpretáveis.")
    print("   • tokens_ctx semelhante entre condições ⇒ comparação não confundida.")

    destino = os.path.join(RESULTADOS_DIR, "comparacao_final.csv")
    df.to_csv(destino, index=False)
    print(f"\n  Exportado: {destino}")


def main():
    p = argparse.ArgumentParser(description="Avaliação do Benchmark RAG (guid)")
    p.add_argument("--ficheiro")
    p.add_argument("--pasta")
    p.add_argument("--comparar", action="store_true")
    p.add_argument("--so-retrieval", action="store_true")
    p.add_argument("--sem-ragchecker", action="store_true",
                   help="Salta o RAGChecker (lento e redundante com RAGAS/Phoenix)")
    p.add_argument("--langsmith", action="store_true")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    if args.comparar:
        comparar()
    elif args.ficheiro:
        avaliar_ficheiro(args.ficheiro, args.so_retrieval, args.langsmith, args.k,
                         sem_ragchecker=args.sem_ragchecker)
    elif args.pasta:
        fs = [f for f in glob.glob(os.path.join(args.pasta, "*.json"))
              if "_avaliacao" not in f and "comparacao" not in f]
        for f in sorted(fs):
            avaliar_ficheiro(f, args.so_retrieval, args.langsmith, args.k,
                             sem_ragchecker=args.sem_ragchecker)
        comparar()
    else:
        p.print_help()


if __name__ == "__main__":
    main()