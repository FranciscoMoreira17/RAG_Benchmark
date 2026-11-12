# RAG Estrutural — descrição do método (versão 2, 2026-11-12)

Este documento descreve o funcionamento actual da condição **RAG Estrutural**
(`frameworks/rag_estruturalMulti.py`) e as alterações introduzidas em 2026-11-12, com o
objectivo de servir de base à actualização da secção de metodologia do artigo científico.
Os valores numéricos apresentados foram medidos sobre o corpus real (8243 documentos do
Diário da República, 25 812 segmentos).

---

## 1. Visão geral

O RAG Estrutural é um método específico do domínio jurídico-administrativo português que
estende a segmentação estrutural (Unit-Level Segmentation, ULS) com duas técnicas, ambas
**determinísticas e sem recurso a modelos de linguagem**:

1. **Enriquecimento da representação** (*contextual embedding* determinístico) — antes da
   vectorização, é anteposto a cada segmento um cabeçalho com metadados do documento de
   origem, extraídos por expressões regulares.
2. **Expansão por grafo de citações** — no momento da recuperação, as citações explícitas
   entre diplomas são seguidas e os diplomas citados são acrescentados ao contexto.

| Variante | Segmentação | Enriquecimento | Grafo | Recuperação |
|---|---|---|---|---|
| `enriquecido` | ULS | sim | não | híbrida (densa + BM25, RRF) |
| `enriquecido_grafo` | ULS | sim | sim | híbrida (densa + BM25, RRF) |

A linha de base de comparação é a condição `UnitLevelSegmentation-hibrido` (mesma
segmentação e recuperação, texto sem enriquecimento e sem grafo). As duas variantes partilham
o mesmo índice, já que o grafo actua apenas na recuperação. O método é avaliado com dois
modelos de *embedding*: `intfloat/multilingual-e5-small` (384 dimensões, janela de 512
*tokens*) e `Qwen/Qwen3-Embedding-0.6B` (1024 dimensões, janela de 2048 *tokens*).

---

## 2. Indexação

### 2.1 Segmentação

A segmentação é a da ULS (`dre_segmentador.py`), aplicada por documento:

- **Artigo** — quando existe articulado real (pelo menos dois artigos, a começar em 1 ou 2,
  com pelo menos 60% de saltos consecutivos), cada artigo constitui um segmento.
- **Documento** — documentos sem articulado com até 8000 caracteres constituem um único
  segmento.
- **Janela** — documentos sem articulado e com mais de 8000 caracteres são divididos em janelas
  de 4000 caracteres com sobreposição de 400.

### 2.2 Enriquecimento (frontmatter, versão 2)

A cada segmento é anteposto um cabeçalho, construído a partir do **documento-pai** e
partilhado por todos os seus segmentos. O formato é:

```
<título DRE> | Entidade: <organização> | Assunto: <assunto>

<texto do segmento>
```

Os campos são extraídos da seguinte forma:

| Campo | Fonte (por ordem de prioridade) |
|---|---|
| **Título** | Título oficial do DRE, que inclui tipo, subtipo, número e data (p. ex., «Aviso (extrato) 4621/2026/2, de 3 de Março»). |
| **Entidade** | (1) Entidade adjudicante («Designação da entidade adjudicante:»), nos anúncios de procedimento; (2) primeira organização nomeada no início (1500 caracteres) ou no fim (500 caracteres) do texto, identificada por uma lista de palavras-chave institucionais (Câmara Municipal, Município, Junta/União de Freguesias, Universidade, Instituto Politécnico, Agrupamento de Escolas, Unidade Local de Saúde, Direção-Geral, Ministério, entre outras), seguida de palavras com maiúscula inicial e conectores (*de*, *da*, *do*, …), sem atravessar mudanças de linha. É preferida a forma completa («Câmara Municipal de Sintra») à forma curta («Câmara Municipal»). |
| **Emitente** | Apenas quando não é encontrada entidade: cargo de quem assina, extraído da fórmula de assinatura final («…2026.-O Presidente da Câmara, Nome.»). |
| **Assunto** | (1) Linha «Sumário:», nos primeiros 500 caracteres; (2) «Designação do contrato:», nos anúncios de procedimento; (3) a segunda linha do texto (a seguir ao cabeçalho do diploma), desde que tenha entre 15 e 250 caracteres e não corresponda a texto corrido. É rejeitada a linha que começa por fórmulas como «Nos termos», «Em cumprimento», «Por despacho», «Torna-se público», a que termina num conector (linha cortada) e a que contém a assinatura ou a entidade adjudicante. |

Exemplos reais de cabeçalhos:

```
Aviso 947/2026/2, de 19 de Janeiro | Entidade: Câmara Municipal de Loures
Anúncio de Procedimento 5369/2026, de 5 de Março | Entidade: Município de Redondo | Assunto: AQUISIÇÃO DE EQUIPAMENTO DE VÍDEO, IMAGEM E TECNOLOGIA PARA O AUDITÓRIO MUNICIPAL DE REDONDO
Despacho 3586/2021, de 7 de Abril | Entidade: Direção-Geral das Artes | Assunto: Designa Pedro Nuno de Pinho Matos Barbosa para exercer, em regime de substituição, o cargo de subdiretor-geral das Artes.
Aviso 7586/2026/2, de 2 de Abril | Entidade: Serviços Municipalizados de Viseu | Assunto: Mobilidade Intercarreiras de trabalhadora dos Serviços Municipalizados de Viseu
```

O texto enriquecido (cabeçalho + segmento) alimenta **ambas** as representações do índice:
o vector denso (*embedding*) e o vector esparso (BM25, `Qdrant/bm25`, língua portuguesa,
modificador IDF). No *payload* do Qdrant é guardado o **texto cru** do segmento, que é o
texto entregue ao gerador. O texto enriquecido fica também guardado (`texto_embebido`) para
efeitos de auditoria.

### 2.3 Cobertura do enriquecimento

| Tipo | N.º docs | Entidade | Emitente | Assunto |
|---|---:|---:|---:|---:|
| Anúncio | 3998 | 3995 | 1 | 3969 |
| Aviso | 2538 | 1719 | 805 | 1274 |
| Despacho | 1004 | 724 | 249 | 383 |
| Regulamento | 213 | 204 | 2 | 134 |
| Edital | 119 | 111 | 8 | 50 |
| Louvor | 97 | 68 | 27 | 11 |
| Deliberação | 64 | 57 | 6 | 18 |
| Decreto-lei | 54 | 6 | 0 | 44 |
| Declaração | 47 | 30 | 16 | 21 |
| Resolução | 44 | 16 | 0 | 33 |
| **Total (todos os tipos)** | **8243** | **6965 (84,5%)** | **1124 (13,6%)** | **5975 (72,5%)** |

Apenas 59 documentos (0,7%) ficam só com o título no cabeçalho.

**Custo em *tokens*** (tokenizador do E5): o cabeçalho tem uma mediana de 43 *tokens*
(P95 = 66; máximo = 121). Com o E5, 40,4% dos segmentos excedem a janela de 512 *tokens*.
Este excesso é herdado sobretudo da segmentação, que admite segmentos até 8000 caracteres.
Apenas 351 segmentos (1,4%) passam a exceder a janela por causa do cabeçalho. Com o Qwen3,
5,6% dos segmentos excedem os 2048 *tokens*. Como o cabeçalho é anteposto, a truncagem
elimina o fim do segmento e nunca o cabeçalho.

---

## 3. Recuperação

### 3.1 Recuperação híbrida

A pergunta é codificada nas duas representações. São obtidos 50 candidatos em cada ramo
(`PREFETCH = 50`: denso por cosseno, esparso por BM25), que são fundidos por
*Reciprocal Rank Fusion* (RRF) [Cormack et al., 2009]. São devolvidos os `top_k = 5`
segmentos.

### 3.2 Expansão por grafo de citações (versão 2)

**Construção do grafo.** O grafo é construído a partir do corpus, de forma determinística,
sempre que a condição é executada. Não depende da ordem de inserção no índice.

1. **Extracção de citações.** Uma expressão regular identifica referências do tipo
   «⟨tipo⟩ [n.º] ⟨número⟩/⟨ano⟩», com o tipo em {decreto-lei, decreto, lei, portaria,
   despacho, regulamento, aviso, resolução}. Cada citação é representada pelo par
   **(tipo normalizado, número)**. A normalização converte para minúsculas, remove os
   diacríticos e unifica espaços e hífenes (p. ex., «Decreto Lei» e «Decreto-Lei» dão
   `decreto-lei`).
2. **Identidade de cada documento.** Cada documento do corpus é identificado pelo mesmo par
   (tipo normalizado, número). O número é lido do título. Um sufixo alfabético faz parte do
   número, pois identifica a região autónoma («Decreto Legislativo Regional 9/2021/A» ≠
   Decreto 9/2021). Um sufixo numérico é ignorado, pois indica a série do Diário da República
   («Aviso 44/2026/2» = Aviso 44/2026).
3. **Resolução sem ambiguidade.** Uma citação só é ligada a um documento quando o par
   (tipo, número) identifica um único documento do corpus. As chaves ambíguas são
   descartadas. As auto-citações são excluídas.

**Expansão.** Os segmentos recuperados são percorridos pela ordem do *ranking*. Para cada um,
são seguidas as citações do documento de origem que resolvem para um documento ainda ausente
do contexto, até ao máximo de **3 documentos adicionais** (`MAX_GRAFO_EXPANSAO`). De cada
documento citado é acrescentado **o segmento mais próximo da pergunta**, obtido por pesquisa
densa filtrada pelo identificador do documento. Os segmentos acrescentados ficam no fim da
lista, com `score = 0` e `method = "grafo_expansao"`.

**Estatísticas do grafo (corpus completo):**

| Medida | Valor |
|---|---:|
| Citações tipadas extraídas | 9398 |
| Citações resolvidas para um documento do corpus | 66 |
| Documentos com pelo menos uma ligação válida | 64 |
| Ligações para um diploma de tipo diferente do citado | 0 |
| Ligações por tipo citado | despacho 35, regulamento 24, aviso 4, decreto-lei 2, decreto 1 |

A esmagadora maioria das citações (≈99,3%) refere-se a legislação que não faz parte do corpus
(p. ex., Lei 35/2014, Decreto-Lei 305/2009). Por isso, o grafo é muito esparso e a variante
`enriquecido_grafo` coincide com `enriquecido` na maioria das perguntas.

### 3.3 Geração

O gerador recebe os segmentos (os recuperados e os da expansão) formatados como
«--- FONTE (tipo número) ---», dentro de um orçamento de contexto fixo de 4096 *tokens*, igual
para todas as condições. Os segmentos que excedem o orçamento são descartados a partir do
fim, pelo que os da expansão são os primeiros a sair.

---

## 4. Diferenças face à versão anterior (v1)

| Aspecto | v1 (versão usada nos resultados anteriores) | v2 (actual) |
|---|---|---|
| Cabeçalho | `Tipo | Número | Entidade | Assunto` | `Título DRE | Entidade/Emitente | Assunto` |
| Entidade | Só «entidade adjudicante» (anúncios): 3994 docs | Adjudicante, 1.ª organização ou cargo do signatário: 8089 docs |
| Assunto | Só «Sumário:»: 291 docs (3,5%) | Sumário, objecto do contrato ou linha de título: 5975 docs (72,5%) |
| Docs só com tipo/número/título | ≈48% | 0,7% |
| Chave do grafo | Apenas o número (p. ex., `2/2026`) | Par (tipo, número), com número real do título |
| Números ambíguos | O 1.º documento encontrado ganha | Descartados |
| Ligações resolvidas | 150, das quais ≈125 (83%) para um diploma errado | 66, das quais 0 para um diploma errado |
| Determinismo do grafo | Dependia de a ingestão ter sido executada ou saltada (ordem do corpus vs. ordem do índice) | Construído sempre do corpus; determinístico |
| Segmento do diploma citado | Arbitrário (primeiro devolvido pelo *scroll*) | O mais próximo da pergunta (denso, filtrado) |

**Consequência:** os resultados das quatro condições `estrutural-*` (E5 e Qwen3, com e sem
grafo) obtidos com a v1 não são comparáveis com os da v2 e têm de ser repetidos. O índice é
reconstruído automaticamente, porque a versão do enriquecimento (`"enriquecimento": 2`) entra
na impressão digital (*fingerprint*) da ingestão. Os resultados da v2 registam
`grafo_versao: 2` na descrição da condição.

---

## 5. Limitações a reportar

1. **Extracção por regras.** Os metadados vêm de heurísticas lexicais. A organização extraída
   é a primeira mencionada, que nem sempre coincide com a entidade emitente (p. ex., um
   despacho que menciona outra instituição antes da sua). O assunto pode ficar vazio
   (27,5% dos documentos), sobretudo em avisos e despachos sem linha de título.
2. **Grafo esparso.** Como o corpus não inclui a legislação de base citada, só 64 documentos
   têm ligações. O efeito da expansão é, por isso, necessariamente limitado.
3. **Descarte de ambiguidades.** A política conservadora de descartar chaves ambíguas evita
   ligações erradas à custa de cobertura.
4. **Interacção com a métrica.** A avaliação de recuperação considera documentos distintos
   pela ordem do *ranking*. Quando os 5 segmentos recuperados pertencem a menos de 5
   documentos, os documentos da expansão podem entrar no top-5 distinto e influenciar as
   métricas @5. Quando pertencem a 5 documentos distintos, os da expansão só contam para
   K > 5.
5. **Truncagem no E5.** 40,4% dos segmentos excedem a janela de 512 *tokens* do E5. A
   limitação é comum à ULS-E5 e decorre da segmentação, não do enriquecimento.
6. **Enriquecimento em ambos os ramos.** O cabeçalho entra também no BM25. A comparação com
   a ULS-híbrida mede, portanto, o efeito conjunto nas duas representações.

---

## 6. Alterações a fazer no artigo

- [ ] Metodologia — enriquecimento: substituir a lista «tipo, número, entidade, sumário» pela
      descrição da §2.2 (título DRE, entidade/emitente, assunto e respectivas prioridades).
- [ ] Metodologia — indicar que o enriquecimento é aplicado a **todos** os segmentos e aos
      **dois** ramos (denso e BM25).
- [ ] Metodologia — incluir a tabela de cobertura (§2.3) ou, pelo menos, os totais
      (84,5% entidade; 72,5% assunto).
- [ ] Metodologia — grafo: descrever a chave (tipo, número), a normalização, o tratamento dos
      sufixos (região vs. série), o descarte de ambiguidades e a escolha do segmento por
      similaridade com a pergunta.
- [ ] Metodologia — reportar as estatísticas do grafo (9398 citações; 66 resolvidas;
      64 documentos).
- [ ] Resultados — substituir todos os valores das condições `estrutural-*` pelos da v2,
      após nova ingestão e execução.
- [ ] Discussão — interpretar o (provavelmente reduzido) efeito do grafo à luz da sua
      esparsidade.
- [ ] Limitações — incluir os pontos da §5.
- [ ] Se o artigo referir *Contextual Retrieval* (Anthropic), manter a ressalva de que aqui
      o contexto é extraído por regras e não gerado por modelo de linguagem.

---

## 7. Reprodutibilidade

```bash
python runner.py --framework estrutural_multi --embedder e5   --variante enriquecido
python runner.py --framework estrutural_multi --embedder e5   --variante enriquecido_grafo
python runner.py --framework estrutural_multi --embedder qwen --variante enriquecido
python runner.py --framework estrutural_multi --embedder qwen --variante enriquecido_grafo
```

| Condição (`nome_run`) | Collection |
|---|---|
| `estrutural-enriquecido`, `estrutural-enriquecido_grafo` | `benchmark_estrutural` |
| `estrutural-qwen-enriquecido`, `estrutural-qwen-enriquecido_grafo` | `benchmark_estrutural_qwen` |

Parâmetros fixos: `PREFETCH = 50`, `top_k = 5`, `MAX_GRAFO_EXPANSAO = 3`,
`CONTEXT_BUDGET_TOKENS = 4096`, BM25 `Qdrant/bm25` (português), RRF.

---

## Referências citadas

⚠️ *Verificar disponibilidade e dados antes de incluir no documento.*

**Cormack, Clarke & Büttcher (2009) — Reciprocal Rank Fusion**
DOI: https://doi.org/10.1145/1571941.1572114

```bibtex
@inproceedings{Cormack:2009:RRF,
  author    = {Cormack, Gordon V. and Clarke, Charles L. A. and B{\"u}ttcher, Stefan},
  title     = {Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods},
  booktitle = {Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year      = {2009},
  pages     = {758--759},
  publisher = {ACM},
  doi       = {10.1145/1571941.1572114}
}
```

**Robertson & Zaragoza (2009) — BM25**
DOI: https://doi.org/10.1561/1500000019

```bibtex
@article{Robertson:2009:BM25,
  author  = {Robertson, Stephen and Zaragoza, Hugo},
  title   = {The Probabilistic Relevance Framework: {BM25} and Beyond},
  journal = {Foundations and Trends in Information Retrieval},
  year    = {2009},
  volume  = {3},
  number  = {4},
  pages   = {333--389},
  doi     = {10.1561/1500000019}
}
```

**Anthropic (2024) — Contextual Retrieval** (publicação técnica, sem DOI)
Acesso: https://www.anthropic.com/news/contextual-retrieval

```bibtex
@misc{Anthropic:2024:ContextualRetrieval,
  author       = {{Anthropic}},
  title        = {Introducing Contextual Retrieval},
  year         = {2024},
  howpublished = {\url{https://www.anthropic.com/news/contextual-retrieval}},
  note         = {Acedido em 2026-11-12}
}
```
