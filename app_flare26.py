import streamlit as st
import time
import json
from datetime import datetime
from pathlib import Path
import hashlib
import wikipedia
from chat_flare26 import extract_atomic_claim, motor_de_conflito_m4, fallback_cognitivo_m5

# ========================================
# CONFIGURAÇÕES DA WIKIPÉDIA
# ========================================
# Define o idioma da Wikipédia para Português
wikipedia.set_lang("pt")

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
# FUNÇÃO DE OTIMIZAÇÃO (IA DE BUSCA)
# ========================================
def otimizar_query(pergunta):
    try:
        from openai import OpenAI
        import os
        cliente_rapido = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resposta = cliente_rapido.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente de busca. Extraia apenas as 1 ou 2 entidades/palavras-chave principais da pergunta do usuário para usar numa busca na Wikipedia. Ex: se a pergunta for 'quem inventou o chatgpt', responda apenas 'ChatGPT'. Não use frases completas."},
                {"role": "user", "content": pergunta}
            ],
            temperature=0.0
        )
        return resposta.choices[0].message.content.strip()
    except:
        return pergunta

# ========================================
# INTERFACE PRINCIPAL E NOVO MOTOR M1
# ========================================
tema_busca = st.text_input(
    "🔍 Qual assunto ou métrica você deseja pesquisar na internet hoje?",
    placeholder="Ex: quem inventou o chatgpt"
)

if st.button("Buscar e Analisar Evidências", type="primary", use_container_width=True):
    if not tema_busca:
        st.warning("Por favor, digite um tema para pesquisar.")
    else:
        with st.status("Iniciando varredura profunda e segura (Wikipédia)...", expanded=True) as status:
            try:
                st.write("🧠 Otimizando intenção de busca para o Motor M1...")
                query_otimizada = otimizar_query(tema_busca)
                st.write(f"🎯 Buscando na Enciclopédia por: **'{query_otimizada}'**")
                
                time.sleep(1)

                # Busca as páginas na Wikipédia usando os termos otimizados
                resultados_busca = wikipedia.search(query_otimizada, results=2)

                # Se não achar nada em Português, tenta em Inglês
                if len(resultados_busca) < 2:
                    wikipedia.set_lang("en")
                    resultados_busca = wikipedia.search(query_otimizada, results=2)
                    if len(resultados_busca) < 2:
                        status.update(label="Falha na Busca Web", state="error", expanded=True)
                        st.error("🛑 A Wikipédia não encontrou páginas suficientes sobre este tema.")
                        st.stop()

                st.write(f"✅ Encontrado: {resultados_busca[0]} e {resultados_busca[1]}")
                st.write("📖 Extraindo conteúdo enciclopédico auditável...")

                try:
                    # Usar auto_suggest=False para garantir precisão e .content para pegar tudo
                    pagina_1 = wikipedia.page(resultados_busca[0], auto_suggest=False)
                    texto_1 = pagina_1.content
                    f1_id = pagina_1.title
                except wikipedia.exceptions.DisambiguationError as e:
                    pagina_1 = wikipedia.page(e.options[0], auto_suggest=False)
                    texto_1 = pagina_1.content
                    f1_id = pagina_1.title
                except:
                    texto_1 = wikipedia.summary(resultados_busca[0])
                    f1_id = resultados_busca[0]

                try:
                    pagina_2 = wikipedia.page(resultados_busca[1], auto_suggest=False)
                    texto_2 = pagina_2.content
                    f2_id = pagina_2.title
                except wikipedia.exceptions.DisambiguationError as e:
                    pagina_2 = wikipedia.page(e.options[0], auto_suggest=False)
                    texto_2 = pagina_2.content
                    f2_id = pagina_2.title
                except:
                    texto_2 = wikipedia.summary(resultados_busca[1])
                    f2_id = resultados_busca[1]

                st.write("🧠 Acionando OpenAI GPT-4o-mini (Extração de Precisão)...")
                # Aumentamos o escopo de leitura para 8000 caracteres (pega a história/desenvolvimento)
                claim_1 = extract_atomic_claim(texto_1[:8000], f1_id, tema_busca)
                claim_2 = extract_atomic_claim(texto_2[:8000], f2_id, tema_busca)

                if not claim_1 or not claim_2:
                    status.update(label="Erro Pydantic", state="error", expanded=True)
                    st.error("Falha na formatação do LLM.")
                    st.stop()

                st.write(f"🔍 DEBUG M2 Fonte 1 Encontrou: {claim_1['claim_data'].get('dados_encontrados')}")
                st.write(f"🔍 DEBUG M2 Fonte 2 Encontrou: {claim_2['claim_data'].get('dados_encontrados')}")

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

        # ========================================
        # RENDERIZAÇÃO DE RESULTADOS E M4
        # ========================================
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Fonte: {f1_id} (Wikipédia)**\n\n{texto_1[:400]}...")
        with col2:
            st.info(f"**Fonte: {f2_id} (Wikipédia)**\n\n{texto_2[:400]}...")

        st.divider()
        st.subheader("🤖 Diagnóstico do Motor M4")

        resultado_m4 = motor_de_conflito_m4(claim_1, claim_2)
        da = claim_1['claim_data']
        db = claim_2['claim_data']
        tema = str(da.get('sujeito') or 'Tema').title()

        if resultado_m4 == "FALHA_TOTAL":
            st.error("🛑 **FALHA NO GLASS BOX (Busca Insuficiente)**: O Motor M4 bloqueou a síntese certificada pois os textos enciclopédicos não continham a resposta explícita.")
            st.info(f"**Contexto da Fonte 1:** {da.get('contexto_da_fonte')}")
            st.info(f"**Contexto da Fonte 2:** {db.get('contexto_da_fonte')}")
            
            st.divider()
            
            st.warning("⚠️ **AVISO DE DEGRADAÇÃO (MODO BLACK BOX ATIVADO)** ⚠️\n\nA busca auditável falhou. A resposta abaixo foi gerada a partir da memória interna da IA (GPT-4o). **Ela NÃO possui proveniência garantida e não foi registrada no Ledger.** Use com cautela.")
            
            with st.spinner("Acionando memória paramétrica..."):
                resposta_fallback = fallback_cognitivo_m5(tema_busca)
                st.markdown(f"> 🤖 **Resposta (Sem Garantia de Fonte):**\n\n{resposta_fallback}")
            
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