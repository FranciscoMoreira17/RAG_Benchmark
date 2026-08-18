"""
juiz_llm.py - Juiz LLM-as-judge próprio
============================================================
Avalia respostas geradas usando um LLM juiz via Groq, com prompt
explícito e controlado. Substitui o RAGAS e o Phoenix, que davam
conflitos de dependências no ambiente.

Métricas (todas 0. a .1, avaliadas pelo juiz):
  - faithfulness      : a resposta é suportada PELO CONTEXTO recuperado?
                        (mede alucinação: 1 = tudo suportado, 0 = inventado)
  - answer_relevancy  : a resposta responde à PERGUNTA?
  - correctness        : a resposta bate com o GROUND TRUTH?

"""

import os
import re
import json
import time

JUIZ_MODELO = os.getenv("JUDGE_MODEL_GROQ", "openai/gpt-oss-120b")
JUIZ_SLEEP = float(os.getenv("JUDGE_SLEEP", "10"))
JUIZ_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "6"))

PROMPT_AVALIACAO = """És um avaliador rigoroso de sistemas de resposta a perguntas sobre \
legislação portuguesa. Avalias UMA resposta segundo três critérios, com base \
apenas na informação fornecida. Não uses conhecimento externo.

CRITÉRIOS (cada um de 0.0 a 1.0):

1. fidelidade (faithfulness): A RESPOSTA é integralmente suportada pelo CONTEXTO?
   - 1.0 = toda a informação da resposta está no contexto.
   - 0.5 = parte está no contexto, parte não.
   - 0.0 = a resposta afirma coisas que o contexto não suporta (alucinação).

2. relevancia (answer_relevancy): A RESPOSTA responde diretamente à PERGUNTA?
   - 1.0 = responde de forma completa e focada.
   - 0.5 = responde parcialmente ou com informação a mais/irrelevante.
   - 0.0 = não responde à pergunta.

3. correcao (correctness): A RESPOSTA está de acordo com a REFERÊNCIA correta?
   - 1.0 = factualmente equivalente à referência.
   - 0.5 = parcialmente correta.
   - 0.0 = contradiz ou falha a referência.

PERGUNTA:
{pergunta}

CONTEXTO RECUPERADO:
{contexto}

REFERÊNCIA (resposta correta):
{referencia}

RESPOSTA A AVALIAR:
{resposta}

Responde APENAS com um objeto JSON válido, sem texto antes ou depois, no formato:
{{"fidelidade": <float>, "relevancia": <float>, "correcao": <float>, "justificacao": "<breve>"}}"""


def _extrair_json(texto: str) -> dict | None:
    """Extrai o primeiro objeto JSON do texto do juiz."""
    m = re.search(r"\{.*\}", texto, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # tentar limpar fences e reparar
        limpo = re.sub(r"```json|```", "", m.group(0)).strip()
        try:
            return json.loads(limpo)
        except json.JSONDecodeError:
            return None


def _chamar_juiz(client, pergunta, contexto, referencia, resposta) -> dict | None:
    prompt = PROMPT_AVALIACAO.format(
        pergunta=pergunta,
        contexto=contexto[:4000],      
        referencia=referencia,
        resposta=resposta,
    )
    for tentativa in range(JUIZ_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=JUIZ_MODELO,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1200,
            )
            msg = resp.choices[0].message
            texto = msg.content or ""
            dados = _extrair_json(texto)
            if not dados:
                reasoning = getattr(msg, "reasoning", None) or ""
                dados = _extrair_json(reasoning)
            if dados and all(k in dados for k in ("fidelidade", "relevancia", "correcao")):
                return {
                    "faithfulness": float(dados["fidelidade"]),
                    "answer_relevancy": float(dados["relevancia"]),
                    "correctness": float(dados["correcao"]),
                }
            return None
        except Exception as e:
            msg = str(e)
            if "rate_limit" in msg or "429" in msg:
                espera = JUIZ_SLEEP * (tentativa + 1)
                print(f"      [juiz] rate limit — espera {espera:.0f}s")
                time.sleep(espera)
                continue
            print(f"      [juiz] erro: {msg[:80]}")
            return None
    return None


def avaliar_llm_juiz(resultados: list[dict], get_client) -> dict:
    import statistics

    validos = [r for r in resultados
               if r.get("para_geracao")
               and not r["resposta"].startswith("ERRO")
               and r.get("ground_truth")]

    if not validos:
        print("    [JUIZ] Sem perguntas para_geracao válidas")
        return {}

    client = get_client()
    print(f"    [JUIZ] {len(validos)} perguntas, modelo={JUIZ_MODELO} "
          f"(sleep {JUIZ_SLEEP}s entre chamadas)")

    fis, rels, cors = [], [], []
    n_falhas = 0
    for i, r in enumerate(validos, 1):
        contexto = "\n\n".join(r.get("contextos", []) or [""])
        notas = _chamar_juiz(
            client, r["query"], contexto, r["ground_truth"], r["resposta"]
        )
        if notas:
            fis.append(notas["faithfulness"])
            rels.append(notas["answer_relevancy"])
            cors.append(notas["correctness"])
        else:
            n_falhas += 1
        print(f"      [{i}/{len(validos)}] "
              f"{'ok' if notas else 'falha'}")
        time.sleep(JUIZ_SLEEP)   

    if not fis:
        print("    [JUIZ] Nenhuma avaliação válida")
        return {}

    def resumo(vals):
        return {
            "media": statistics.mean(vals),
            "desvio": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    print(f"    [JUIZ] concluído: {len(fis)} avaliadas, {n_falhas} falhas")
    return {
        "faithfulness": statistics.mean(fis),
        "answer_relevancy": statistics.mean(rels),
        "correctness": statistics.mean(cors),
        "_detalhe": {
            "faithfulness": resumo(fis),
            "answer_relevancy": resumo(rels),
            "correctness": resumo(cors),
            "n_falhas_parse": n_falhas,
        },
        "_juiz": JUIZ_MODELO,
    }