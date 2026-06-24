# Paper — Outline / Esqueleto de Trabalho

> **⚠️ SUPERADO (2026-06-22):** o paper foi reenquadrado no ângulo metodológico
> ("pitfalls de avaliação de abstenção"). O texto vigente está em `draft.md` /
> `draft.tex` (título: *When the Baseline Also Abstains*). Este outline fica como
> registro histórico do raciocínio.

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

### 5.1 Resultado principal (benchmark de abstenção, 30 pares; 13 ABSTAIN, 17 com resposta)

Extrator gpt-4o-mini, temperatura 0, seed fixo. BASELINE = extração livre
(sem gating ontológico); FLARE26 = extração com gating de tipo+escopo.
Números agregados (média ± desvio): BASELINE sobre 3 execuções, FLARE26
estrita sobre 8 execuções, self-consistency por bootstrap (k=5, 300 reamostras).

| Sistema | ★ Falso-positivo | Recall de abstenção | Recall de resposta |
|---------|------------------|---------------------|--------------------|
| BASELINE (extração livre)            | 38% ± 0% | 62% ± 0%  | 100% ± 0% |
| FLARE26 (gating estrita)             | **0% ± 0%** | **100% ± 0%** | 82% ± 4% |
| FLARE26 + self-consistency (k=5)     | **0% ± 0%** | **100% ± 0%** | **92% ± 3%** |

**Leitura:** a gating ontológica **elimina as divergências falso-positivas**
(38%→0%, ± 0% — estável, não é sorte de uma execução), capturando as distinções
juros≠multa, garantia≠pagamento e inexecução≠atraso. A self-consistency
**recupera o recall a 92% ± 3% mantendo 0% de falso-positivo** (§5.4).

### 5.2 A fronteira precision/recall (ablação da regra de escopo)

| Regra de escopo | Falso-positivo | Recall de resposta |
|-----------------|----------------|--------------------|
| Estrita (igualdade de condição) | **0%** | 71% |
| Por interseção (subconjunto/superconjunto) | 38% | 88% |

Afrouxar o escopo para aceitar sub/superconjuntos recupera recall, mas
**reabre os falso-positivos** (ex.: "multa por atraso" passa a responder
"multa por inexecução"). A versão estrita é o ponto de operação adequado a
auditoria: **nunca afirmar uma divergência falsa** vale mais que recall.

### 5.3 Análise de erros (super-abstenções do FLARE26)
- **Fronteira legítima:** texto mais específico (`atraso > 10 dias` p/ pergunta
  "atraso") ou mais geral (`inadimplemento parcial ou total` p/ "total") que a
  pergunta — intocável sem reabrir falso-positivos.
- **Não-determinismo:** casos como "retenção 5%/10%" oscilam entre responder e
  abster — corrigível por self-consistency (§5.4), sem trocar precisão.

### 5.4 Self-consistency recupera recall a custo ZERO de precisão

Achado-chave do diagnóstico (5 execuções por item): as **abstenções
ontológicas são estáveis** — 13/13 casos ABSTAIN tiveram **0 vazamento** em 5
execuções. O ruído está apenas nas super-abstenções (respostas legítimas que
oscilam, ex.: retenção respondeu 2/5). Como os ABSTAIN nunca vazam, pode-se
usar a política mais permissiva — **"responder se ≥1 de k execuções responder"**
— sem reabrir falso-positivo.

| Sistema | Falso-positivo | Recall de resposta |
|---------|----------------|--------------------|
| FLARE26 (amostra única, 8 execuções) | 0% ± 0% | 82% ± 4% |
| FLARE26 + self-consistency (k=5, ≥1) | **0% ± 0%** | **92% ± 3%** |

O ganho de recall (82%→92%) é real e o desvio é pequeno (± 3-4%). As respostas
borderline têm taxa-base baixa, então k=5 as recupera de forma probabilística;
**o 0% de falso-positivo, em contraste, é exato e estável** (0% ± 0% em todas as
execuções), pois nenhuma abstenção ontológica vaza. k maior recupera mais
respostas borderline, a custo linear de chamadas de LLM.

## 6. Discussão e limitações (honestidade)
- **Recall < 100%:** preço da garantia de 0% falso-positivo. Com self-consistency
  (k=5) o recall sobe a 92% ± 3%; super-abstenção residual em escopos
  sub/superconjunto é a fronteira legítima. Trade-off explícito.
- **Não-determinismo do LLM:** mitigado por seed + self-consistency; resultados
  já reportados como média ± desvio (3-8 execuções + bootstrap).
- Corpus parcialmente sintético; generalização a outros domínios a validar.
- Dependência de um LLM proprietário (gpt-4o-mini) na extração.
- Não é método novo de ML — é padrão arquitetural; valor está na avaliação.

## 7. Conclusão e trabalhos futuros
- Ampliar o corpus (incl. editais governamentais reais já presentes) e validar
  os rótulos com um segundo anotador.
- Caracterizar custo: chamadas de LLM economizadas pelo juiz simbólico vs
  o custo 5x da self-consistency; varrer k para a curva recall × custo.
- Replicar com LLM aberto (não-proprietário) para reprodutibilidade.
- Passe neural opcional para fundir grupos textuais equivalentes no juiz N-way.

---

## Plano de execução (paper = frente de métricas)
1. ~~Refatorar M2/M1.5 para serem chamáveis fora do Streamlit~~ ✅ (`flare26_pipeline.py`).
2. ~~Construir o **dataset ouro**~~ ✅ (`eval/gold_dataset.json`, 30 pares verificados).
3. ~~Implementar baselines e o script de métricas~~ ✅ (`eval/run_eval.py`).
4. ~~Rodar e preencher §5~~ ✅ (baseline 38% vs FLARE26 0% falso-positivo).
5. ~~Self-consistency (votação) para recuperar recall a 0% FP~~ ✅ (k=5 → 92% recall, 0% FP).
6. ~~Rodar k repetições (média ± desvio)~~ ✅ (`eval/run_eval_agg.py`; tabela em §5.1).
7. **[próximo]** Escrever o draft EN a partir deste outline.
8. Ampliar/validar corpus (2º anotador; editais grandes) e curva recall × custo (varredura de k).
