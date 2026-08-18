# RAG Benchmark - Legislação Portuguesa (DRE)

Benchmark comparativo de estratégias de Retrieval-Augmented Generation (RAG) aplicadas a documentos legislativos portugueses do Diário da República Eletrónico (DRE). 

## Objetivo

Avaliar sistematicamente o impacto da segmentação, do enriquecimento e da expansão por grafo na qualidade do retrieval e da geração de respostas sobre legislação portuguesa.

## Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                    docker-compose.yml                   │
│                                                         │
│  ┌───────────┐   ┌───────────┐   ┌───────────────────┐  │
│  │  Qdrant   │   │  Phoenix  │   │      Ollama       │  │
│  │  :6333    │   │  :6006    │   │     :11434        │  │
│  │  :6334    │   │  :4317    │   │                   │  │
│  └───────────┘   └───────────┘   └───────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │                │                  │
         ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                     Python Pipeline                     │
│                                                         │
│  config.py ─── runner.py ─── avaliacao.py               │
│      │             │              │                     │
│      ▼             ▼              ▼                     │
│  Embedding    Frameworks     juiz_llm.py                │
│  E5-large     (Strategy)     (LLM Judge)                │
└─────────────────────────────────────────────────────────┘
```

## Stack Tecnológico

| Componente | Tecnologia |
|---|---|
| Embedding | `intfloat/multilingual-e5-small` (384 dims, prefixos `query:`/`passage:`) |
| Vector Store | Qdrant (cosine similarity) |
| Tracing | Arize Phoenix (OTLP) |
| Gerador | Groq `llama-3.3-70b-versatile` |
| Juiz LLM | `gpt-oss-120b` (faithfulness, relevancy, correctness) |
| Enrichment LLM | Groq `llama-3.1-8b-instant` |
| OCR | Tesseract (para PDFs digitalizados do DRE) |
| Orquestração | Docker Compose (Qdrant + Phoenix + Ollama) |


## Collections Qdrant

Cada framework/condição utiliza uma collection dedicada no Qdrant. Todas partilham a mesma configuração vetorial: 1024 dimensões, distância coseno.

| Collection | Framework | Notas |
|---|---|---|
| `benchmark_langchain` | LangChain | Chunking genérico (RecursiveCharacterTextSplitter) |
| `benchmark_llamaindex` | LlamaIndex | Chunking por frase (SentenceSplitter) |
| `benchmark_estrutural` | OKF (Estrutural) | Segmentação por artigo; partilhada entre variantes `enriquecido` e `enriquecido_grafo` |
| `benchmark_hibrido` | Artigo (OKF) | Texto cru |

As collections são criadas automaticamente pelo serviço `init-qdrant` no Docker Compose. Para recriá-las manualmente:


## Condições Experimentais

O benchmark avalia **7 condições de retrieval** e **4 condições de geração**, organizadas numa escada de ablação:

### Retrieval (7 condições)

| Condição | Segmentação | Embedding | Retrieval | Grafo | Collection |
|---|---|---|---|---|---|
| `hibrido-denso` | Artigo (OKF) | Texto cru | Denso | Não | `benchmark_hibrido` |
| `hibrido-esparso` | Artigo (OKF) | Texto cru | BM25 (esparso) | Não | `benchmark_hibrido` |
| `hibrido-hibrido` | Artigo (OKF) | Texto cru | Denso + BM25 (RRF) | Não | `benchmark_hibrido` |
| `estrutural-enriquecido` | Artigo (OKF) | Frontmatter prepended | Denso + BM25 (RRF) | Não | `benchmark_estrutural` |
| `estrutural-enriquecido_grafo` | Artigo (OKF) | Frontmatter prepended | Denso + BM25 (RRF) | Sim | `benchmark_estrutural` |
| `langchain` | RecursiveCharacter | Texto cru | Denso | Não | `benchmark_langchain` |
| `llamaindex` | SentenceSplitter | Texto cru | Denso | Não | `benchmark_llamaindex` |

### Geração (4 condições)

As condições com geração avaliada são: `estrutural-enriquecido`, `hibrido-hibrido`, `langchain`, `llamaindex`.

### Escada de Ablação (OKF)

A lógica da ablação é isolar uma variável por degrau:

1. **hibrido-hibrido** → base: segmentação estrutural, texto cru, retrieval híbrido, sem grafo
2. **estrutural-enriquecido** → acrescenta frontmatter ao embedding; comparado com (1), isola o enriquecimento
3. **estrutural-enriquecido_grafo** → acrescenta expansão por citações; comparado com (2), isola o grafo

## Dataset

O dataset contém **256 queries** com ground truth (Não verificado por especialista), organizadas em duas camadas:

- **Camada 1 - Identificador** (237 queries): perguntas factuais geradas programaticamente a partir dos campos dos documentos (entidade, número, preço base, prazo). Ground truth extraído diretamente dos metadados.
- **Camada 2 - Semântica** (19 queries): perguntas de compreensão propostas manualmente, com ground truth validado contra o texto fonte.

Destas 256, apenas ~30 são marcadas com `para_geracao: true` para avaliação da geração (controlo de quota do LLM).

## Métricas de Avaliação

### Retrieval

| Métrica | Descrição |
|---|---|
| `precision@5` | Fração de documentos relevantes nos top-5 |
| `recall@5` | Fração de documentos relevantes recuperados nos top-5 |
| `hit_rate@5` | Pelo menos 1 documento relevante nos top-5 |
| `mrr` | Mean Reciprocal Rank - posição média do primeiro documento relevante |
| `ndcg@5` | Normalized Discounted Cumulative Gain |

As métricas são desagregadas por tipo de query (`identificador` vs `semantica`).

### Geração - Juiz LLM (`juiz_llm.py`)

| Métrica | Descrição |
|---|---|
| `faithfulness` | A resposta é fiel ao contexto recuperado (não alucina)? |
| `answer_relevancy` | A resposta é relevante para a pergunta? |
| `correctness` | A resposta está factualmente correta face ao ground truth? |

O juiz utiliza o modelo `gpt-oss-120b` (reasoning model). Configuração: `max_tokens=1200` com fallback para o campo `reasoning` caso `content` venha vazio (comportamento de modelos de raciocínio).

### Geração - Métricas Clássicas

| Métrica | Descrição |
|---|---|
| `bleu` | Sobreposição de n-gramas (Papineni et al., 2002) |
| `rouge_l` | Longest Common Subsequence (Lin, 2004) |
| `bertscore_f1` | Similaridade semântica via embeddings contextuais (Zhang et al., 2020) |

## Utilização

### 1. Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Chaves de API: `GOOGLE_API_KEY`, `GROQ_API_KEY`

### 2. Infraestrutura

```bash
# Subir os serviços
docker compose up -d

# Verificar collections
docker compose logs init-qdrant
curl http://localhost:6333/collections

# (Opcional) Modelo local para Ollama
docker exec ollama ollama pull qwen3:14b
```

### 3. Configuração

```bash
cp .env.example .env
# Editar .env com as API keys
```

### 4. Dataset

```bash
# Gerar template (preencher manualmente com queries e ground truth)
python dataset.py --gerar-template
```

### 5. Benchmark - Ingestão e Retrieval

```bash
# LangChain
python runner.py --framework langchain --reingerir --so-retrieval

# LlamaIndex
python runner.py --framework llamaindex --reingerir --so-retrieval

# OKF Estrutural - variante enriquecido (constrói o índice)
python runner.py --framework estrutural --variante enriquecido --reingerir --so-retrieval

# OKF Estrutural - variante enriquecido + grafo (reutiliza o índice)
python runner.py --framework estrutural --variante enriquecido_grafo --so-retrieval

# OKF Híbrido (3 estratégias de retrieval sobre o mesmo índice)
python runner.py --framework estrutural --variante hibrido-denso --so-retrieval
python runner.py --framework estrutural --variante hibrido-esparso --so-retrieval
python runner.py --framework estrutural --variante hibrido-hibrido --so-retrieval
```

### 6. Benchmark - Geração

```bash
# Cada geração consome ~79k tokens/dia (limite 100k do Llama 70B)
# Recomendação: uma condição por dia
python runner.py --framework estrutural --variante enriquecido
python runner.py --framework langchain
python runner.py --framework llamaindex
python runner.py --framework estrutural --variante hibrido-hibrido
```

### 7. Avaliação

```bash
# Avaliação de retrieval (todas as condições)
python avaliacao.py --pasta resultados/ --so-retrieval

# Avaliação completa de uma condição (retrieval + juiz LLM + clássicas)
python avaliacao.py --ficheiro resultados/estrutural-enriquecido_r1_<timestamp>.json

# Tabela comparativa final
python avaliacao.py --comparar
```

### CLI do Runner - Referência

```bash
python runner.py --framework <nome>           # Framework específica
python runner.py --framework <nome> --variante <var>  # Com variante
python runner.py --framework todas            # Todas sequencialmente
python runner.py --framework <nome> --reingerir       # Forçar re-indexação
python runner.py --framework <nome> --so-retrieval    # Só retrieval (sem geração)
python runner.py --framework <nome> --repeticoes 3    # Múltiplas repetições
python runner.py --framework <nome> --ablacao         # Todas as variantes OKF
```

## Padrões de Design

- **Strategy Pattern** - cada framework implementa `FrameworkBase` (interface com `ingerir()` e `responder()`); o `runner.py` seleciona a estratégia concreta via argparse
- **CLI com Argument Parsing** - `argparse` como ponto de entrada central, com `--framework`, `--variante`, `--reingerir`, `--so-retrieval`, `--repeticoes`, `--ablacao`, `--comparar`
- **Fingerprint de Ingestão** - controlo de coerência do índice via hash da configuração de ingestão; evita re-indexações desnecessárias ou uso acidental de um índice desatualizado
- **Retoma de Geração** - o runner pode retomar gerações interrompidas (por rate limit) sem desperdiçar quota nas queries já respondidas

## Notas Importantes

**Quotas de API.** A geração com Llama 70B via Groq tem um limite de ~100k tokens/dia, o que permite aproximadamente uma condição completa por dia (30 queries × ~2.600 tokens de contexto). O juiz LLM (`gpt-oss-120b`) tem quota separada de 200k tokens/dia. Planear a execução em dias consecutivos.

**PDFs digitalizados.** Alguns PDFs do DRE são digitalizações de imagem sem texto extraível. O pipeline OKF integra Tesseract OCR automaticamente para estes casos. As frameworks genéricas (LangChain, LlamaIndex) podem falhar silenciosamente nestes documentos.

**Segmentação por artigo.** A decisão arquitetural central do OKF: um artigo = um chunk. Isto garante que a unidade semântica jurídica nunca é fragmentada entre chunks, ao contrário do chunking genérico (RecursiveCharacterTextSplitter, SentenceSplitter) que corta arbitrariamente dentro de artigos.

**Embedding com prefixos E5.** O modelo `multilingual-e5-small` exige prefixos `query:` nas queries e `passage:` nos documentos. Omitir estes prefixos degrada severamente a qualidade do retrieval. Todas as frameworks utilizam a mesma classe `E5Embedder` do `config.py` para garantir consistência.

## Licença

Projeto académico - ESTG, Politécnico do Porto.