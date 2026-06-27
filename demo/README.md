# FLARE26 — Demo (offline, custo zero)

Vitrine read-only do benchmark e dos resultados **pré-computados** já versionados
no repositório. **Não chama nenhum LLM** (não usa OpenAI, não gasta API), então
roda publicamente sem custo nem risco de abuso.

Mostra: explorador do benchmark com o comportamento real do FLARE por item (N
execuções cacheadas), o resultado-manchete e os gráficos da Pitfall A (curva
bootstrap + varredura do limiar de voto).

## Rodar localmente

```bash
pip install -r demo/requirements.txt
streamlit run demo/streamlit_app.py
```

## Publicar grátis (Streamlit Community Cloud)

1. Acesse https://share.streamlit.io e faça login com a conta do GitHub.
2. **New app** → repositório `marcospaula/FLARE26-P1`, branch `master`.
3. **Main file path:** `demo/streamlit_app.py`.
4. Deploy. (As dependências vêm de `demo/requirements.txt` — só `streamlit` e
   `pandas`, build rápido. Nenhuma chave/segredo é necessária.)

O app lê os dados de `../eval/` por caminho relativo ao próprio arquivo, então
funciona tanto local quanto no deploy.
