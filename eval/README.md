# Avaliação — Benchmark de Abstenção Ontológica

Sustenta o paper (ver `../paper/PAPER_OUTLINE.md`). A métrica-manchete é a
**taxa de divergência falso-positiva**: quantas vezes o sistema grita
DIVERGÊNCIA quando, na verdade, os documentos concordam ou simplesmente não
contêm o dado.

## Protocolo de rotulagem (gold)

Cada item de `gold_dataset.json` é um par **(pergunta, documento)** com:
- `gold`: a resposta correta **ou** o literal `"ABSTAIN"` quando o documento
  **não contém** a informação pedida (caso de lacuna de evidência).
- `evidencia`: trecho literal do PDF que justifica o rótulo (vazio se ABSTAIN).
- `verificado`: `true` só depois de conferir lendo o PDF (não inferir pelo nome
  do arquivo!).

A partir desses rótulos por documento, derivamos os **pares de comparação**
N-way e o gold do veredito (CONSENSO / DIVERGÊNCIA / LACUNA).

## Métricas (eval/run_eval.py — a implementar)
1. Acurácia da resposta quando `gold != ABSTAIN`.
2. Abstenção: precision/recall de prever `ABSTAIN` corretamente.
3. ★ Taxa de divergência falso-positiva (manchete).
4. Custo: chamadas de LLM economizadas pelo juiz simbólico.

## Baselines a comparar
- B0: RAG + extração livre (sem gating ontológico).
- B1: RAG + JSON estrito, sem checagem de compatibilidade ontológica.
- FLARE26: pipeline completo com gating + abstenção.

## Pré-requisito técnico
Tornar M1.5 (retrieval) e M2 (extração) **chamáveis fora do Streamlit** para o
harness rodar headless. Hoje vivem em `app_flare26_pdf.py` acoplados a `st`.
