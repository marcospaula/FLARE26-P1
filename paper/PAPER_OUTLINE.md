# Paper — Outline / Esqueleto de Trabalho

> **Status:** rascunho inicial (2026-06-14). Língua de redação final: **inglês**
> (alvo são venues internacionais); este outline está em PT para alinhamento.
> **Tipo de paper (honesto):** aplicado / de sistemas — *não* é paper de método
> novo de ML. Contribuição = padrão arquitetural + avaliação empírica num
> domínio real. Alvos plausíveis: workshop de Document AI / Legal NLP / RAG,
> trilha industrial, ou preprint arXiv + workshop.

---

## Título (candidatos)

1. **"Knowing When the Answer Isn't There: Ontology-Gated Extraction for
   False-Positive-Free Multi-Document Audit"**
2. "Abstention-First RAG: Reducing False Divergences in Cross-Document
   Comparison of Legal Contracts"
3. "The Glass-Box Auditor: Deterministic, Abstaining Retrieval for
   Trustworthy Document Comparison"

*(preferência atual: #1 — coloca a abstenção como tese central)*

---

## Tese central (a frase do paper)

> Em auditoria multi-documento, o erro mais caro não é errar o valor — é
> **inventar uma divergência que não existe** (falso-positivo) ou **deixar de
> apontar uma que existe** (falso-negativo). Mostramos que **condicionar a
> extração a uma checagem de compatibilidade ontológica explícita**, com
> detecção de *lacuna de evidência*, reduz drasticamente as divergências
> falso-positivas frente a um RAG padrão, mantendo a decisão final
> determinística e auditável.

---

## Abstract (rascunho v0)

Sistemas RAG aplicados à comparação de documentos tendem a "alucinar"
respostas quando a informação pedida não está presente, produzindo
**divergências falso-positivas** que corroem a confiança em contextos de
auditoria. Apresentamos o FLARE26, um auditor neuro-simbólico "Caixa de Vidro"
cuja etapa de extração é **ontologicamente restrita**: o modelo é forçado a
declarar se o trecho recuperado é do mesmo *tipo* da pergunta e se não viola
restrições negativas, abstendo-se (LACUNA DE EVIDÊNCIA) quando não há resposta.
A decisão de consenso/divergência entre N documentos é então tomada por um
juiz **determinístico** sobre os dados tipados. Avaliamos em um corpus de
contratos e editais governamentais brasileiros [N pares] e mostramos redução
de X% na taxa de divergência falso-positiva frente a um baseline de RAG +
extração livre, sem perda relevante de recall sobre respostas existentes.

---

## 1. Introdução
- Contexto: comparação/auditoria de documentos (contratos, editais) em escala.
- Dor real: RAG padrão **prefere responder a abster-se** → falso-positivos de
  divergência (ex.: pergunta sobre "multa por inexecução total" responde com a
  cláusula de "multa por atraso" — caso real do nosso corpus).
- Custo do erro em auditoria ≠ custo em QA: falso-positivo gera retrabalho
  jurídico / decisão errada.
- Contribuições:
  1. Esquema de **extração ontologicamente restrita** com abstenção explícita.
  2. **Juiz N-way determinístico** sobre dados tipados (auditável, O(N)).
  3. **Benchmark** de abstenção em domínio jurídico BR + métrica de
     **divergência falso-positiva**.

## 2. Trabalhos relacionados (posicionamento)
- RAG e suas alucinações em extração estruturada.
- Saída estruturada / function-calling (Instructor, Guardrails, Pydantic).
- **Selective prediction / abstention** em NLP e LLMs (o vizinho mais próximo —
  é aqui que ancoramos a originalidade).
- Faithfulness / groundedness em RAG.
- Legal NLP / document comparison.
- *Gap que exploramos:* abstenção tratada como **pré-condição da comparação
  cruzada**, não como pós-filtro de QA.

## 3. Método
- 3.1 Visão geral do pipeline (M1.5 retrieval híbrido → M2 extração → M4 juiz
  → M5 síntese). Figura do pipeline.
- 3.2 **Extração ontologicamente restrita (núcleo do paper):**
  - schema tipado com `houve_compatibilidade_ontologica`,
    `violou_restricao_do_usuario`, e colapso para `NÃO LOCALIZADO`/
    `LACUNA DE EVIDÊNCIA`.
  - por que isso muda o incentivo do modelo (declarar tipo antes de responder).
- 3.3 **Juiz N-way determinístico:** agrupamento por equivalência
  (normalização numérica), veredito Consenso/Divergência/Lacuna, O(N).
- 3.4 Caixa de Vidro: proveniência e determinismo (reprodutibilidade).

## 4. Avaliação
- 4.1 Corpus: contratos sintéticos + editais governamentais reais (PT-BR).
- 4.2 **Protocolo de rotulagem:** para cada par (documento, pergunta), rótulo
  ouro = resposta correta **ou** `ABSTAIN` (quando o doc não contém o dado).
- 4.3 Baselines: (a) RAG + extração livre (sem gating); (b) RAG + JSON estrito
  sem checagem ontológica.
- 4.4 **Métricas:**
  - Acurácia da resposta quando ela existe.
  - **Abstenção:** precision/recall de detectar corretamente a ausência.
  - **★ Taxa de divergência falso-positiva** (métrica-manchete): em pares onde
    o gold é "mesma coisa / ausente", quantas vezes o sistema grita
    DIVERGÊNCIA indevida.
  - Custo (chamadas de LLM economizadas pelo juiz simbólico) — métrica
    secundária.

## 5. Resultados
- (placeholder) Tabela principal: baseline vs FLARE26 nas métricas acima.
- Análise de erros: onde a abstenção falha (respostas textuais ambíguas →
  limitação do agrupamento simbólico).

## 6. Discussão e limitações (honestidade)
- Agrupamento N-way puramente simbólico super-divide texto equivalente.
- Corpus parcialmente sintético; generalização a outros domínios.
- Dependência de um LLM proprietário (gpt-4o-mini) na extração.
- Não é método novo de ML — é padrão arquitetural; valor está na avaliação.

## 7. Conclusão e trabalhos futuros
- Passe neural opcional para fundir grupos textuais equivalentes.
- Generalização do gating ontológico para outros domínios (config injetável,
  já iniciada via `GatilhosLexicais`).

---

## Plano de execução (paper = frente de métricas)
1. **[próximo]** Refatorar M2/M1.5 para serem chamáveis fora do Streamlit
   (harness de avaliação sem UI). → `eval/`.
2. Construir o **dataset ouro** (`eval/gold_dataset.json`) a partir do corpus
   `documentos/` (já mapeamos quais docs têm atraso/inexecução/multa/prazo).
3. Implementar baselines e o script de métricas (`eval/run_eval.py`).
4. Rodar, preencher §5, e escrever o draft EN.
