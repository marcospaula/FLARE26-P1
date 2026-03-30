import streamlit as st
from ddgs import DDGS
from goose3 import Goose
from chat_flare26 import extract_atomic_claim, motor_de_conflito_m4
import time
import json
from datetime import datetime
from pathlib import Path
import hashlib

# ========================================
# CONFIGURAÇÕES INICIAIS STREAMLIT
# ========================================
# O set_page_config deve ser sempre o primeiro comando do Streamlit
st.set_page_config(page_title="FLARE26 - Web Engine", page_icon="🌐", layout="wide")
st.markdown(
    """<style>.stDeployButton {display:none;} #MainMenu {visibility: hidden;}</style>""",
    unsafe_allow_html=True
)

st.title("🌐 FLARE26: Web Engine (Motor Blindado MVP)")
st.markdown("Extração Profunda e Validação M4 contra Alucinações de LLM.")
st.divider()

# ========================================
# LEDGER PROVENIENCIA REAL - FLARE26-P1 M4
# ========================================
@st.cache_data
def init_ledger():
    ledger_path = Path("flare26_ledger.json")
    if ledger_path.exists():
        with open(ledger_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            "proveniencia": {
                "timestamp": datetime.now().isoformat(),
                "alegacoes": 0,
                "conflitos_detectados": 0,
                "score_global": 0.0
            },
            "grafo_claims": []
        }

def hash_claim(claim):
    return hashlib.sha256(claim.encode()).hexdigest()[:8]

def adicionar_claim_real(claim_texto, fonte_url, confiabilidade=1.0, conflito_com=None):
    ledger = init_ledger()
    claim_id = hash_claim(claim_texto)
    
    # 🛑 TRAVA ANTI-DUPLICAÇÃO DE NÓS DO GRAFO
    for claim_existente in ledger["grafo_claims"]:
        if claim_existente["id"] == claim_id:
            return claim_existente, []

    novo_claim = {
        "id": claim_id,
        "claim": claim_texto,
        "fonte": fonte_url,
        "confiabilidade": confiabilidade,
        "timestamp": datetime.now().isoformat(),
        "estado": "validada" if confiabilidade > 0.7 else "conflito_suspeito",
        "conflito_com": conflito_com or []
    }
    
    conflitos = []
    for claim_existente in ledger["grafo_claims"]:
        if abs(claim_existente["confiabilidade"] - confiabilidade) > 0.5:
            conflitos.append(claim_existente["id"])
    
    if conflitos:
        novo_claim["estado"] = "CONFLITO"
        ledger["proveniencia"]["conflitos_detectados"] += 1
    
    ledger["grafo_claims"].append(novo_claim)
    ledger["proveniencia"]["alegacoes"] += 1
    
    with open("flare26_ledger.json", 'w', encoding='utf-8') as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
    
    return novo_claim, conflitos

# ========================================
# SIDEBAR COM LEDGER + TESTES
# ========================================
st.sidebar.title("⚙️ Configurações FLARE26")

st.sidebar.header("🪙 Ledger Providência M4")
if st.sidebar.button("🚀 Testar Ledger Real", type="primary", help="Rastreia claims reais com hash + conflitos"):
    with st.spinner("🔗 Rastreando proveniência persistente..."):
        claim1, _ = adicionar_claim_real(
            "VL-JEPA usa Y-Encoder para embeddings multimodais",
            "arxiv.org/2507.01078",
            0.95
        )
        claim2, conflitos = adicionar_claim_real(
            "VL-JEPA supera GPT-4 em todas tarefas",
            "blog-ia-fake.com",
            0.25
        )
        st.sidebar.success("✅ Ledger salvo em flare26_ledger.json!")
        st.sidebar.json({"Novo Claim": claim1, "Conflitos": conflitos})

# ========================================
# BUSCA E ANÁLISE
# ========================================
def ler_texto_completo(url, fallback_snippet):
    try:
        # User-Agent para evitar que sites bloqueiem a nossa requisição
        g = Goose({'browser_user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        article = g.extract(url=url)
        # Se extraiu bem o texto, usa o texto completo
        if article.cleaned_text and len(article.cleaned_text) > 150:
            return article.cleaned_text
        # Fallback: mescla snippet com a descrição meta do site
        return fallback_snippet + " " + (article.meta_description or "")
    except:
        return fallback_snippet

tema_busca = st.text_input(
    "🔍 Qual assunto ou métrica você deseja pesquisar na internet hoje?",
    placeholder="Ex: Microsoft Corp lucro quarter 2025"
)

if st.button("Buscar e Analisar Evidências", type="primary", use_container_width=True):
    if not tema_busca:
        st.warning("Por favor, digite um tema para pesquisar.")
    else:
        with st.status("Iniciando varredura profunda e segura...", expanded=True) as status:
            try:
                st.write("📡 Conectando ao DuckDuckGo (Nova API)...")
                ddgs = DDGS()
                time.sleep(1)

                resultados = []
                
                # 1. Tenta forçar busca de notícias primeiro para pegar balanços e dados exatos
                try:
                    resultados = list(ddgs.news(query=tema_busca, max_results=4))
                except Exception:
                    pass

                # 2. Se não achar notícias, tenta a web geral
                if len(resultados) < 2:
                    st.write("⚠️ Poucas notícias. Tentando web geral...")
                    try:
                        resultados = list(ddgs.text(query=tema_busca, max_results=4))
                    except Exception:
                        pass

                if len(resultados) < 2:
                    status.update(label="Falha na Busca Web", state="error", expanded=True)
                    st.error("🛑 O DuckDuckGo bloqueou nossa pesquisa ou não encontrou nada. Tente um termo mais simples.")
                    st.stop()

                resultados = resultados[:2]
                st.write("✅ Extraindo links...")

                # A API do DDG tem chaves diferentes para 'news' e 'text'
                url_1 = resultados[0].get('url', resultados[0].get('href', ''))
                url_2 = resultados[1].get('url', resultados[1].get('href', ''))

                body_1 = resultados[0].get('body', resultados[0].get('title', ''))
                body_2 = resultados[1].get('body', resultados[1].get('title', ''))

                st.write("📖 Lendo texto integral (Fallback Seguro)...")
                texto_1 = ler_texto_completo(url_1, body_1)
                texto_2 = ler_texto_completo(url_2, body_2)

                f1_id = resultados[0].get('source', resultados[0].get('title', 'Fonte 1'))
                f2_id = resultados[1].get('source', resultados[1].get('title', 'Fonte 2'))

                st.write("🧠 Acionando OpenAI GPT-4o-mini (Extração de Precisão)...")
                # Ampliado para 2000 caracteres para abranger artigos maiores
                claim_1 = extract_atomic_claim(texto_1[:2000], f1_id, tema_busca)
                claim_2 = extract_atomic_claim(texto_2[:2000], f2_id, tema_busca)

                if not claim_1 or not claim_2:
                    status.update(label="Erro Pydantic", state="error", expanded=True)
                    st.error("Falha grave na formatação estruturada do LLM.")
                    st.stop()

                # DEBUG
                st.write(f"🔍 DEBUG M2 Fonte 1 Encontrou: {claim_1['claim_data'].get('dados_encontrados')}")
                st.write(f"🔍 DEBUG M2 Fonte 2 Encontrou: {claim_2['claim_data'].get('dados_encontrados')}")

                # Trava Anti-Alucinação
                if claim_1['claim_data'].get('dados_encontrados') is False or claim_2['claim_data'].get('dados_encontrados') is False:
                    status.update(label="Aviso Anti-Alucinação Acionado", state="error", expanded=True)
                    st.error("🛑 A OpenAI Rejeitou a Extração. Os textos da internet não continham a resposta explícita. O Motor impediu a invenção de dados.")
                    st.info(f"**Trecho da Fonte 1 repassado para IA:** {texto_1[:400]}...")
                    st.info(f"**Trecho da Fonte 2 repassado para IA:** {texto_2[:400]}...")
                    st.stop()

                st.write("🪙 Registrando claims no Ledger Providência...")
                # O retorno é ignorado aqui intencionalmente para não sujar a tela principal
                _ = adicionar_claim_real(claim_1['claim_data'].get('objeto', ''), f1_id, 0.9)
                _ = adicionar_claim_real(claim_2['claim_data'].get('objeto', ''), f2_id, 0.9)

                st.write("⚖️ Passando os dados no Motor de Conflito M4...")
                status.update(label="Análise Concluída!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Erro Crítico de Execução", state="error", expanded=True)
                st.error(f"Erro inesperado no sistema principal: {e}")
                st.stop()

        # Renderização Final
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**{f1_id}**\n\n{texto_1[:300]}...")
        with col2:
            st.info(f"**{f2_id}**\n\n{texto_2[:300]}...")

        st.divider()
        st.subheader("🤖 Diagnóstico do Motor M4")

        resultado_m4 = motor_de_conflito_m4(claim_1, claim_2)
        da, db = claim_1['claim_data'], claim_2['claim_data']
        tema = str(da.get('sujeito') or 'Tema').title()

        if 'DISPUTA' in resultado_m4:
            st.error(f"**Disputa Detectada:** Valores diferentes sobre '{tema}'.")
            st.markdown(f"- **{f1_id}**: {da.get('objeto')}")
            st.markdown(f"- **{f2_id}**: {db.get('objeto')}")
        elif 'ATUALIZACAO' in resultado_m4:
            st.warning(f"**Evolução Histórica em '{tema}':**")
            st.markdown(f"- Antigo ({da.get('tempo')}): {da.get('objeto')} ({f1_id})")
            st.markdown(f"- Recente ({db.get('tempo')}): {db.get('objeto')} ({f2_id})")
        elif 'COEXISTENCIA' in resultado_m4:
            st.warning(f"**Coexistência (Divergência de Escopo/Texto):**")
            st.markdown(f"- **[{da.get('escopo')}]**: {da.get('objeto')} ({f1_id})")
            st.markdown(f"- **[{db.get('escopo')}]**: {db.get('objeto')} ({f2_id})")
        else:
            st.success("**Consenso Alcançado:** As fontes concordam.")
            st.markdown(f"- O valor/fato apurado é: **{da.get('objeto')}**")