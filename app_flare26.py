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
# CONFIGURAÇÕES STREAMLIT
# ========================================
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
    
    # 🛑 TRAVA ANTI-DUPLICAÇÃO
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
# BUSCA E ANÁLISE
# ========================================
def ler_texto_completo(url, fallback_snippet):
    try:
        g = Goose({'browser_user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        article = g.extract(url=url)
        if article.cleaned_text and len(article.cleaned_text) > 150:
            return article.cleaned_text
        return fallback_snippet + " " + (article.meta_description or "")
    except:
        return fallback_snippet

tema_busca = st.text_input(
    "🔍 Qual assunto ou métrica você deseja pesquisar na internet hoje?",
    placeholder="Ex: quem ganhou o oscar de melhor ator em 2024"
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
                try:
                    resultados = list(ddgs.news(query=tema_busca, max_results=4))
                except Exception:
                    pass

                if len(resultados) < 2:
                    st.write("⚠️ Poucas notícias. Tentando web geral...")
                    try:
                        resultados = list(ddgs.text(query=tema_busca, max_results=4))
                    except Exception:
                        pass

                if len(resultados) < 2:
                    status.update(label="Falha na Busca Web", state="error", expanded=True)
                    st.error("🛑 O DuckDuckGo bloqueou nossa pesquisa ou não encontrou nada.")
                    st.stop()

                resultados = resultados[:2]
                st.write("✅ Extraindo links...")

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
                claim_1 = extract_atomic_claim(texto_1[:2000], f1_id, tema_busca)
                claim_2 = extract_atomic_claim(texto_2[:2000], f2_id, tema_busca)

                if not claim_1 or not claim_2:
                    status.update(label="Erro Pydantic", state="error", expanded=True)
                    st.error("Falha na formatação do LLM.")
                    st.stop()

                st.write(f"🔍 DEBUG M2 Fonte 1 Encontrou: {claim_1['claim_data'].get('dados_encontrados')}")
                st.write(f"🔍 DEBUG M2 Fonte 2 Encontrou: {claim_2['claim_data'].get('dados_encontrados')}")

                # ATENÇÃO: Retiramos o st.stop() agressivo daqui e delegamos a inteligência pro M4!

                st.write("🪙 Registrando claims no Ledger Providência...")
                if claim_1['claim_data'].get('dados_encontrados'):
                    _ = adicionar_claim_real(claim_1['claim_data'].get('objeto', ''), f1_id, 0.9)
                if claim_2['claim_data'].get('dados_encontrados'):
                    _ = adicionar_claim_real(claim_2['claim_data'].get('objeto', ''), f2_id, 0.9)

                st.write("⚖️ Passando os dados no Motor de Conflito M4...")
                status.update(label="Análise Concluída!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Erro Crítico de Execução", state="error", expanded=True)
                st.error(f"Erro inesperado no sistema principal: {e}")
                st.stop()

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**{f1_id}**\n\n{texto_1[:300]}...")
        with col2:
            st.info(f"**{f2_id}**\n\n{texto_2[:300]}...")

        st.divider()
        st.subheader("🤖 Diagnóstico do Motor M4")

        # Chama a nova inteligência que detecta as assimetrias e falhas totais
        resultado_m4 = motor_de_conflito_m4(claim_1, claim_2)
        da = claim_1['claim_data']
        db = claim_2['claim_data']
        tema = str(da.get('sujeito') or 'Tema').title()

        if resultado_m4 == "FALHA_TOTAL":
            st.error("🛑 Aviso Anti-Alucinação: Nenhuma das fontes conteve a resposta explícita. O Motor impediu a invenção de dados.")
            st.info(f"**Contexto da Fonte 1:** {da.get('contexto_da_fonte')}")
            st.info(f"**Contexto da Fonte 2:** {db.get('contexto_da_fonte')}")
            
        elif resultado_m4 == "COMPLEMENTO_ASSIMETRICO_A":
            st.success("🧩 Síntese Complementar: A resposta foi encontrada e enriquecida por contexto periférico da outra fonte.")
            st.markdown(f"- **Fato Principal ({f1_id})**: {da.get('objeto')} (Escopo: {da.get('escopo')})")
            st.markdown(f"- **Contexto Agregado ({f2_id})**: *{db.get('contexto_da_fonte')}*")
            
        elif resultado_m4 == "COMPLEMENTO_ASSIMETRICO_B":
            st.success("🧩 Síntese Complementar: A resposta foi encontrada e enriquecida por contexto periférico da outra fonte.")
            st.markdown(f"- **Fato Principal ({f2_id})**: {db.get('objeto')} (Escopo: {db.get('escopo')})")
            st.markdown(f"- **Contexto Agregado ({f1_id})**: *{da.get('contexto_da_fonte')}*")

        elif resultado_m4 == "DISPUTA_DIRETA":
            st.error(f"⚔️ **Disputa Detectada:** Fontes discordam sobre '{tema}'.")
            st.markdown(f"- **{f1_id}**: {da.get('objeto')}")
            st.markdown(f"- **{f2_id}**: {db.get('objeto')}")
            
        elif resultado_m4 == "ATUALIZACAO_TEMPORAL":
            st.warning(f"⏳ **Evolução Histórica em '{tema}':**")
            st.markdown(f"- Tempo ({da.get('tempo')}): {da.get('objeto')} ({f1_id})")
            st.markdown(f"- Tempo ({db.get('tempo')}): {db.get('objeto')} ({f2_id})")
            
        elif resultado_m4 == "COEXISTENCIA_ESCOPO":
            st.info(f"⚖️ **Coexistência Pacífica (Divergência de Escopo):**")
            st.markdown(f"- **[{da.get('escopo')}]**: {da.get('objeto')} ({f1_id})")
            st.markdown(f"- **[{db.get('escopo')}]**: {db.get('objeto')} ({f2_id})")
            
        elif resultado_m4 == "CONSENSO_TOTAL":
            st.success("✅ **Consenso Alcançado:** Ambas as fontes concordam.")
            st.markdown(f"- O fato apurado é: **{da.get('objeto')}**")