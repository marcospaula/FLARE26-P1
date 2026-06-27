# Avaliação — Benchmark de Abstenção Ontológica

Sustenta o paper (ver `../paper/draft.md`). A métrica-manchete é a
**taxa de divergência falso-positiva**, operacionalizada **por-item** em
`run_eval.py` como *"o sistema respondeu quando o gold era `ABSTAIN`"* — uma
alucinação que, numa comparação multi-documento, fabricaria uma divergência.

## Protocolo de rotulagem (gold)

Cada item de `gold_dataset.json` é um par **(pergunta, documento)** com:
- `gold`: a resposta correta **ou** o literal `"ABSTAIN"` quando o documento
  **não contém** a informação pedida (caso de lacuna de evidência).
- `evidencia`: trecho literal do PDF que justifica o rótulo (vazio se ABSTAIN).
- `verificado`: `true` só depois de conferir lendo o PDF (não inferir pelo nome
  do arquivo!).

Os **pares de comparação N-way** explícitos (gold de veredito CONSENSO /
DIVERGÊNCIA / LACUNA) ainda **não** estão montados — ver `_todo` em
`gold_dataset.json`. A manchete atual **não depende** deles: é medida por-item
(acima), derivável diretamente dos rótulos por documento.

## Métricas e harnesses (implementados)
- `run_eval.py` — baseline (extração livre) vs FLARE26 (gated) no gold set;
  reporta ★ taxa de falso-positivo (alucinou em `ABSTAIN`), recall de abstenção
  e recall de resposta. `FLARE_CONSENSO=k` ativa a votação.
- `run_eval_repeat.py` + `run_eval_agg.py` — repetições + agregação
  (média ± desvio via bootstrap); sustentam a Pitfall A (§4.2).
- `run_eval_curve.py` — curva recall × custo.
- `run_eval_grid.py` — varredura **(k, t)** da política de voto ("responder se
  ≥ t de k"); produz o ponto de operação **k=10, t=4** do paper.

## Baselines
- **Baseline**: RAG + extração livre (sem gating ontológico).
- **FLARE26**: pipeline completo com gating + abstenção.

(Ambos em `run_eval.py`.)

## Status do harness
M1.5 (retrieval) e M2 (extração) já são **headless** em `../flare26_pipeline.py`
(não dependem do Streamlit; `app_flare26_pdf.py` é só a casca de UI).
Pendente: montar e versionar os pares de comparação N-way (ver `_todo` no gold).
