import streamlit as st
import json
import os
import uuid
import PyPDF2
import requests
import math
from datetime import datetime
from pydantic import BaseModel, Field

# ==========================================
# ⚙️ CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="FLARE26: RAG Auditor", page_icon="⚖️", layout="wide")

# ==========================================
# 💾 SISTEMA DE PROVENIÊNCIA (LEDGER M3)
# ==========================================
def gravar_no_ledger(claim, fonte, contexto, estado, conflito_com=None):
    arquivo_ledger = "flare26_ledger.json"
    if not os.path.exists(arquivo_ledger):
        ledger = {"proveniencia": {"timestamp": "", "alegacoes": 0, "conflitos_detectados": 0}, "grafo_claims": []}
    else:
        with open(arquivo_ledger, "r", encoding="utf-8") as f:
            try:
                ledger = json.load(f)
            except json.JSONDecodeError:
                ledger = {"proveniencia": {"timestamp": "", "alegacoes": 0, "conflitos_detectados": 0}, "grafo_claims": []}

    novo_registro = {
        "id": str(uuid.uuid4())[:8],
        "claim": claim,
        "fonte": fonte,
        "contexto_original": contexto,
        "timestamp": datetime.now().isoformat(),
        "estado": estado,
        "conflito_com": conflito_com if conflito_com else []
    }
    
    ledger["grafo_claims"].append(novo_registro)
    ledger["proveniencia"]["alegacoes"] = len(ledger["grafo_claims"])
    ledger["proveniencia"]["timestamp"] = datetime.now().isoformat()
    if estado == "em_conflito":
        ledger["proveniencia"]["conflitos_detectados"] = sum(1 for c in ledger["grafo_claims"] if c["estado"] == "em_conflito")

    with open(arquivo_ledger, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)

# ==========================================
# 🧠 LENTE PYDANTIC (M2)
# ==========================================
class ExtracaoAtomica(BaseModel):
    valor_encontrado: str = Field(description="O dado exato. Se não achar, escreva APENAS: 'NÃO LOCALIZADO'")
    contexto_da_clausula: str = Field(description="O texto que comprova. Se não achar, escreva: 'LACUNA DE EVIDÊNCIA'")
    confiabilidade: float = Field(description="Nível de certeza de 0.0 a 1.0")

# ==========================================
# 📄 LEITOR DE PDF (M1)
# ==========================================
def extrair_texto_pdf(arquivo):
    texto = ""
    leitor = PyPDF2.PdfReader(arquivo)
    for pagina in leitor.pages:
        texto += pagina.extract_text() + "\n"
    return texto

# ==========================================
# 📐 GATILHO VETORIAL (M1.5) - EMBEDDINGS
# ==========================================
def obter_embedding_ollama(texto):
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": "qwen2.5:1.5b",
        "prompt": texto[:2000],
        "keep_alive": "10m"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("embedding", [])
        return []
    except:
        return []

def calcular_similaridade_cosseno(vec1, vec2):
    if not vec1 or not vec2: return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0: return 0.0
    return dot_product / (magnitude1 * magnitude2)

# ==========================================
# 🤖 EXTRATOR DE IA (M2)
# ==========================================
def extrair_dado_com_ia(texto, pergunta):
    schema_json = ExtracaoAtomica.model_json_schema()
    prompt = f"""
    Você é um auditor forense. Responda APENAS em JSON válido.
    REGRA CRÍTICA: Se a informação exata NÃO estiver no texto, retorne "NÃO LOCALIZADO" no campo valor_encontrado.
    PERGUNTA: {pergunta}
    FORMATO ESPERADO: {json.dumps(schema_json)}
    DOCUMENTO: {texto[:4000]}
    """
    url = "http://localhost:11434/api/generate"
    payload = {"model": "qwen2.5:1.5b", "prompt": prompt, "format": "json", "stream": False, "keep_alive": "10m", "options": {"temperature": 0.0, "num_ctx": 4096}}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            resposta_texto = response.json().get("response", "{}")
            return ExtracaoAtomica(**json.loads(resposta_texto))
        return None
    except:
        return None

# ==========================================
# 📝 SINTETIZADOR CONDICIONADO (M5)
# ==========================================
def gerar_sintese_m5(pergunta, doc_a, val_a, ctx_a, doc_b, val_b, ctx_b, status):
    prompt = f"""
    Você é um auditor jurídico sênior. Escreva UM PARÁGRAFO (máx 4 linhas) sintetizando: "{pergunta}".
    STATUS: {status}
    Doc A ({doc_a}): {val_a} (Contexto: {ctx_a})
    Doc B ({doc_b}): {val_b} (Contexto: {ctx_b})
    REGRAS: 1. Se LACUNA, diga qual é omisso. 2. Não invente. 3. MANTENHA O SÍMBOLO R$.
    """
    url = "http://localhost:11434/api/generate"
    payload = {"model": "qwen2.5:1.5b", "prompt": prompt, "stream": False, "keep_alive": "10m", "options": {"temperature": 0.1, "num_ctx": 2048}}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "Erro na geração.")
        return "Erro na API."
    except:
        return "Erro de conexão."

# ==========================================
# 🖥️ INTERFACE E MOTOR M4 (M6)
# ==========================================
st.title("FLARE26: RAG Auditor (Caixa de Vidro Corporativa)")
st.markdown("Auditoria Neuro-Simbólica com Filtro Vetorial M1.5.")

col_a, col_b = st.columns(2)
with col_a: doc_a = st.file_uploader("Documento Base (A):", type=["pdf"], key="doc_a")
with col_b: doc_b = st.file_uploader("Documento Comparativo (B):", type=["pdf"], key="doc_b")

pergunta = st.text_input("O que você quer auditar?", placeholder="Ex: valor da multa por rescisão")
LIMITE_SEMANTICO = 0.450 # REDUZIDO PARA EVITAR LACUNA FALSA NO CONTRATO D

if st.button("🚀 Iniciar Auditoria", type="primary"):
    if doc_a and doc_b and pergunta:
        with st.status("Auditoria Vetorial em andamento...", expanded=True) as status_ui:
            texto_a = extrair_texto_pdf(doc_a)
            texto_b = extrair_texto_pdf(doc_b)
            
            st.write("📐 M1.5: Calculando Embeddings Matemáticos (O(1))...")
            emb_pergunta = obter_embedding_ollama(pergunta)
            emb_doc_a = obter_embedding_ollama(texto_a)
            emb_doc_b = obter_embedding_ollama(texto_b)
            
            sim_a = calcular_similaridade_cosseno(emb_pergunta, emb_doc_a)
            sim_b = calcular_similaridade_cosseno(emb_pergunta, emb_doc_b)
            
            st.write(f"📊 Score Semântico Doc A: `{sim_a:.3f}`")
            st.write(f"📊 Score Semântico Doc B: `{sim_b:.3f}`")
            
            st.write("🧠 Acionando Extrator Atômico (M2) seletivamente...")
            
            if sim_a < LIMITE_SEMANTICO:
                resultado_A = ExtracaoAtomica(valor_encontrado="NÃO LOCALIZADO", contexto_da_clausula="LACUNA DE EVIDÊNCIA", confiabilidade=0.0)
                st.write("🛑 Inferência M2 abortada para Doc A (Baixa Similaridade).")
            else:
                resultado_A = extrair_dado_com_ia(texto_a, pergunta)
                
            if sim_b < LIMITE_SEMANTICO:
                resultado_B = ExtracaoAtomica(valor_encontrado="NÃO LOCALIZADO", contexto_da_clausula="LACUNA DE EVIDÊNCIA", confiabilidade=0.0)
                st.write("🛑 Inferência M2 abortada para Doc B (Baixa Similaridade).")
            else:
                resultado_B = extrair_dado_com_ia(texto_b, pergunta)
                
            st.write("⚖️ Passando os dados no Juiz de Conflito M4...")
            status_ui.update(label="Auditoria Concluída!", state="complete", expanded=False)
            
        if resultado_A and resultado_B:
            st.header("🤖 Diagnóstico Forense M4")
            valor_A = str(resultado_A.valor_encontrado).strip().upper()
            valor_B = str(resultado_B.valor_encontrado).strip().upper()
            status_conflito = ""

            if "NÃO LOCALIZADO" in valor_A or "NÃO LOCALIZADO" in valor_B:
                status_conflito = "LACUNA DE EVIDÊNCIA"
                st.warning("⚠️ **LACUNA DE EVIDÊNCIA DETECTADA:** A informação não existe em um dos documentos.")
                col1, col2 = st.columns(2)
                with col1: st.info(f"📄 **{doc_a.name}**\n\n**Valor:** `{resultado_A.valor_encontrado}`\n\n_{resultado_A.contexto_da_clausula}_")
                with col2: st.info(f"📄 **{doc_b.name}**\n\n**Valor:** `{resultado_B.valor_encontrado}`\n\n_{resultado_B.contexto_da_clausula}_")
                gravar_no_ledger(pergunta, f"{doc_a.name} / {doc_b.name}", "Falta de dados", "lacuna_evidencia")

            elif valor_A != valor_B:
                status_conflito = "DIVERGÊNCIA CRÍTICA"
                st.error("⚔️ **DIVERGÊNCIA CRÍTICA DETECTADA:** Os documentos contradizem-se.")
                col1, col2 = st.columns(2)
                with col1: st.error(f"📄 **{doc_a.name}**\n\n**Valor:** `{resultado_A.valor_encontrado}`\n\n_{resultado_A.contexto_da_clausula}_")
                with col2: st.error(f"📄 **{doc_b.name}**\n\n**Valor:** `{resultado_B.valor_encontrado}`\n\n_{resultado_B.contexto_da_clausula}_")
                gravar_no_ledger(resultado_A.valor_encontrado, doc_a.name, resultado_A.contexto_da_clausula, "em_conflito", [doc_b.name])
                gravar_no_ledger(resultado_B.valor_encontrado, doc_b.name, resultado_B.contexto_da_clausula, "em_conflito", [doc_a.name])

            else:
                status_conflito = "CONSENSO"
                st.success("✅ **CONCORDÂNCIA CONFIRMADA:** Mesma informação em ambos.")
                st.markdown(f"**Valor Validado:** `{resultado_A.valor_encontrado}`")
                st.info(f"_{resultado_A.contexto_da_clausula}_")
                gravar_no_ledger(resultado_A.valor_encontrado, f"{doc_a.name} e {doc_b.name}", resultado_A.contexto_da_clausula, "validada")

            st.divider()
            st.subheader("📝 Parecer Executivo (M5 Sintetizador)")
            with st.spinner("Redigindo síntese executiva baseada em evidências..."):
                texto_sintese = gerar_sintese_m5(pergunta, doc_a.name, resultado_A.valor_encontrado, resultado_A.contexto_da_clausula, doc_b.name, resultado_B.valor_encontrado, resultado_B.contexto_da_clausula, status_conflito)
                st.info(texto_sintese)
                
            # --- NOVA SEÇÃO: CAIXA DE VIDRO (TABELA CORPORATIVA) ---
            st.markdown("---")
            st.subheader("🕵️‍♂️ Trilha de Auditoria Forense (Caixa de Vidro)")
            
            with st.expander("Ver Provas Matemáticas e Extrações (Ledger M3 e M4)", expanded=False):
                st.markdown("Registro imutável das decisões tomadas pelos módulos neuro-simbólicos:")
                
                # 1. LINHA DE MÉTRICAS (M1.5)
                st.markdown("##### 1. Filtro Vetorial (M1.5)")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric(label="Limite de Corte (Threshold)", value=f"{LIMITE_SEMANTICO:.3f}")
                col_m2.metric(label="Aderência Doc A", value=f"{sim_a:.3f}", delta="Aprovado" if sim_a >= LIMITE_SEMANTICO else "Bloqueado", delta_color="normal" if sim_a >= LIMITE_SEMANTICO else "inverse")
                col_m3.metric(label="Aderência Doc B", value=f"{sim_b:.3f}", delta="Aprovado" if sim_b >= LIMITE_SEMANTICO else "Bloqueado", delta_color="normal" if sim_b >= LIMITE_SEMANTICO else "inverse")
                
                # 2. TABELA DE EXTRAÇÃO (M2 e M4)
                st.markdown("##### 2. Dados Atômicos e Veredito (M2 e M4)")
                dados_tabela = [
                    {"Documento": doc_a.name, "Dado Extraído": resultado_A.valor_encontrado, "Status": "Analisado"},
                    {"Documento": doc_b.name, "Dado Extraído": resultado_B.valor_encontrado, "Status": "Analisado"},
                    {"Documento": "Conclusão (M4)", "Dado Extraído": status_conflito, "Status": "Veredito Final"}
                ]
                st.dataframe(dados_tabela, use_container_width=True, hide_index=True)
                
                # 3. JSON ORIGINAL (Escondido para devs no Streamlit)
                with st.popover("Ver código-fonte (JSON)"):
                    st.json({
                        "M1.5_Filtro": {"corte": LIMITE_SEMANTICO, "A": sim_a, "B": sim_b},
                        "M2_Extrator": {"A": resultado_A.valor_encontrado, "B": resultado_B.valor_encontrado},
                        "M4_Juiz": {"status": status_conflito}
                    })
    else:
        st.warning("⚠️ Preencha a pergunta e carregue os dois PDFs para iniciar.")