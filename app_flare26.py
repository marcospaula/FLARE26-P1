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
            # O fato já existe no ledger! Não adiciona duplicado.
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
    
    # Detecta conflito automático
    conflitos = []
    for claim_existente in ledger["grafo_claims"]:
        if abs(claim_existente["confiabilidade"] - confiabilidade) > 0.5:
            conflitos.append(claim_existente["id"])
    
    if conflitos:
        novo_claim["estado"] = "CONFLITO"
        ledger["proveniencia"]["conflitos_detectados"] += 1
    
    ledger["grafo_claims"].append(novo_claim)
    ledger["proveniencia"]["alegacoes"] += 1
    
    # Salva persistente no disco
    with open("flare26_ledger.json", 'w', encoding='utf-8') as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
    
    return novo_claim, conflitos

# ========================================
# CONFIGURAÇÕES STREAMLIT
st.set_page_config(page_title="FLARE26 - Web Engine", page_icon="🌐", layout="wide")
st.markdown("""<style>.stDeployButton {display:none;} #MainMenu {visibility: hidden;}</style>""", unsafe_allow_html=True)

st.title("🌐 FLARE26: Web Engine (Motor Blindado MVP)")
st.markdown("Extração Profunda e Validação M4 contra Alucinações de LLM.")
st.divider()

# ========================================
# SIDEBAR COM LEDGER + TESTES
st.sidebar.title("⚙️ Configurações FLARE26")

# 🆕 LEDGER REAL NA SIDEBAR
st.sidebar.header("🪙 Ledger Providência M4")
if st.sidebar.button("🚀 Testar Ledger Real", type="primary", help="Rastreia claims reais com hash + conflitos"):
    with st.spinner("🔗 Rastreando proveniência persistente..."):
        claim1, _ = adicionar_claim_real(
            "VL-JEPA usa Y-Encoder para embeddings multimodais", 
            "arxiv.org/2507.01078", 0.95
        )
        claim2, conflitos = adicionar_claim_real(
            "VL-JEPA supera GPT-4 em todas tarefas", 
            "blog-ia-fake.com", 0.25
        )
        
        st.sidebar.success("✅ Ledger salvo em flare26_ledger.json!")
        st.sidebar.json({"Novo Claim": claim1, "Conflitos": conflitos})

# ========================================
# CÓDIGO ORIGINAL - BUSCA E ANÁLISE
def ler_texto_completo(url):
    try:
        g = Goose()
        article = g.extract(url=url)
        return article.cleaned_text if article.cleaned_text and len(article.cleaned_text) > 50 else None
    except:
        return None

tema_busca = st.text_input("🔍 Qual assunto ou métrica você deseja pesquisar na internet hoje?", 
                           placeholder="Ex: Lucro Banco do Brasil 2025")

if st.button("Buscar e Analisar Evidências", type="primary", use_container_width=True):
    if not tema_busca:
        st.warning("Por favor, digite um tema para pesquisar.")
    else:
        with st.status("Iniciando varredura profunda e segura...", expanded=True) as status:
            try:
                st.write("📡 Conectando ao DuckDuckGo (Nova API)...")
                ddgs = DDGS()
                time.sleep(1)
                
                tipo_busca = "Notícias"
                resultados = []
                
                try:
                    resultados = list(ddgs.news(query=tema_busca, max_results=5))
                except Exception:
                    pass
                
                if len(resultados) < 2:
                    st.write("⚠️ Poucas notícias ou API restrita. Tentando web geral...")
                    tipo_busca = "Web Geral"
                    try:
                        resultados = list(ddgs.text(query=tema_busca, max_results=5))
                    except Exception:
                        pass
                
                if len(resultados) < 2:
                    status.update(label="Falha na Busca Web", state="error", expanded=True)
                    st.error("🛑 Não foram encontrados resultados suficientes na internet ou fomos bloqueados (Rate Limit). Tente palavras-chave mais objetivas como 'Sede Petrobras cidade' em vez de frases completas.")
                    st.stop()
                    
                resultados = resultados[:2]
                st.write(f"✅ Extraindo links via {tipo_busca}...")

                url_1 = resultados[0].get('url', resultados[0].get('href', ''))
                url_2 = resultados[1].get('url', resultados[1].get('href', ''))
                
                st.write("📖 Lendo texto integral (Goose3)...")
                texto_1 = ler_texto_completo(url_1) or resultados[0].get('body', '')
                texto_2 = ler_texto_completo(url_2) or resultados[1].get('body', '')
                
                f1_id = resultados[0].get('source', resultados[0].get('title', 'Fonte 1'))
                f2_id = resultados[1].get('source', resultados[1].get('title', 'Fonte 2'))

                st.write("🧠 Acionando Qwen 1.5B e Pydantic...")
                claim_1 = extract_atomic_claim(texto_1[:1500], f1_id, tema_busca)
                claim_2 = extract_atomic_claim(texto_2[:1500], f2_id, tema_busca)
                
                if not claim_1 or not claim_2:
                    status.update(label="Erro Pydantic", state="error", expanded=True)
                    st.error("Falha grave na formatação estruturada do LLM.")
                    st.stop()
                    
                if not claim_1['claim_data'].get('dados_encontrados') or not claim_2['claim_data'].get('dados_encontrados'):
                    status.update(label="Aviso Anti-Alucinação Acionado", state="error", expanded=True)
                    st.error("🛑 IA Rejeitou a Extração: Os textos encontrados eram inúteis (ex: dicionários ou erro de leitura do site). A IA foi forçada a abortar a missão para não inventar dados. Tente uma busca focada em empresas ou fatos.")
                    st.stop()

                # REGISTRANDO NO LEDGER REAL
                st.write("🪙 Registrando claims no Ledger Providência...")
                adicionar_claim_real(claim_1['claim_data'].get('objeto', ''), f1_id, 0.9)
                adicionar_claim_real(claim_2['claim_data'].get('objeto', ''), f2_id, 0.9)

                st.write("⚖️ Passando os dados no Motor de Conflito M4...")
                status.update(label="Análise Concluída!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Erro Crítico de Execução", state="error", expanded=True)
                st.error(f"Erro inesperado no sistema principal: {e}")
                st.stop()

        # EXIBIÇÃO FINAL
        col1, col2 = st.columns(2)
        with col1: st.info(f"**{f1_id}**\n\n{texto_1[:300]}...")
        with col2: st.info(f"**{f2_id}**\n\n{texto_2[:300]}...")

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
            st.success(f"**Consenso Alcançado:** As fontes concordam.")
            st.markdown(f"- O valor/fato apurado é: **{da.get('objeto')}**")