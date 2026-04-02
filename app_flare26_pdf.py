import streamlit as st
import json
import os
import uuid
import PyPDF2
import requests
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
# 🤖 EXTRATOR DE IA (M2)
# ==========================================
def extrair_dado_com_ia(texto, pergunta):
    schema_json = ExtracaoAtomica.model_json_schema()
    prompt = f"""
    Você é um auditor forense. Responda APENAS em JSON válido.
    REGRA CRÍTICA: Se a informação exata NÃO estiver no texto, retorne "NÃO LOCALIZADO" no campo valor_encontrado.
    
    PERGUNTA: {pergunta}
    
    FORMATO ESPERADO:
    {json.dumps(schema_json)}
    
    DOCUMENTO:
    {texto[:4000]}
    """
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:1.5b",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.0, "num_ctx": 4096}
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            resposta_texto = response.json().get("response", "{}")
            return ExtracaoAtomica(**json.loads(resposta_texto))
        return None
    except Exception as e:
        st.error(f"Erro: {e}")
        return None

# ==========================================
# 📝 SINTETIZADOR CONDICIONADO (M5)
# ==========================================
def gerar_sintese_m5(pergunta, doc_a, val_a, ctx_a, doc_b, val_b, ctx_b, status):
    prompt = f"""
    Você é um auditor jurídico sênior. 
    Escreva UM ÚNICO PARÁGRAFO (máx 4 linhas) sintetizando o resultado para a pergunta: "{pergunta}".
    
    STATUS DA AUDITORIA: {status}
    Doc A ({doc_a}): {val_a} (Contexto: {ctx_a})
    Doc B ({doc_b}): {val_b} (Contexto: {ctx_b})
    
    REGRAS DE REDAÇÃO:
    1. Se houver LACUNA DE EVIDÊNCIA, diga qual documento é omisso.
    2. NÃO INVENTE DADOS.
    3. MANTENHA O SÍMBOLO MONETÁRIO EXATO (ex: escreva "R$ 5.000", não engula o cifrão "R$" nem escreva apenas "R").
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:1.5b",
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.1, "num_ctx": 2048}
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "Erro na geração.")
        return "Erro na API."
    except Exception as e:
        return f"Erro: {e}"

# ==========================================
# 🖥️ INTERFACE E MOTOR M4 (M6)
# ==========================================
st.title("FLARE26: RAG Auditor (Caixa de Vidro Corporativa)")
st.markdown("Auditoria Cruzada de Contratos com Validação M4 e Síntese M5.")

st.header("1. Selecione os Documentos")
col_a, col_b = st.columns(2)
with col_a:
    doc_a = st.file_uploader("Documento Base (A):", type=["pdf"], key="doc_a")
with col_b:
    doc_b = st.file_uploader("Documento para Comparar (B):", type=["pdf"], key="doc_b")

st.header("2. O que você quer auditar?")
pergunta = st.text_input("Ex: valor da multa por rescisão")

if st.button("🚀 Iniciar Auditoria", type="primary"):
    if doc_a and doc_b and pergunta:
        with st.status("Auditoria em andamento...", expanded=True) as status_ui:
            st.write("📖 Lendo Documentos...")
            texto_a = extrair_texto_pdf(doc_a)
            texto_b = extrair_texto_pdf(doc_b)
            
            # FILTRO SEMÂNTICO (Anti-alucinação)
            # Pega as palavras da pergunta ignorando "valor", "da", "por", "qual", "o"
            palavras_pergunta = [p for p in pergunta.lower().split() if len(p) > 3 and p not in ["qual", "valor"]]
            palavra_chave = palavras_pergunta[0] if palavras_pergunta else "multa"
            
            st.write("🧠 Acionando Extrator Atômico (M2)...")
            
            if palavra_chave not in texto_a.lower():
                resultado_A = ExtracaoAtomica(valor_encontrado="NÃO LOCALIZADO", contexto_da_clausula="LACUNA DE EVIDÊNCIA", confiabilidade=0.0)
            else:
                resultado_A = extrair_dado_com_ia(texto_a, pergunta)
                
            if palavra_chave not in texto_b.lower():
                resultado_B = ExtracaoAtomica(valor_encontrado="NÃO LOCALIZADO", contexto_da_clausula="LACUNA DE EVIDÊNCIA", confiabilidade=0.0)
            else:
                resultado_B = extrair_dado_com_ia(texto_b, pergunta)
                
            st.write("⚖️ Passando os dados no Juiz de Conflito M4...")
            status_ui.update(label="Auditoria Concluída!", state="complete", expanded=False)
            
        if resultado_A and resultado_B:
            st.header("🤖 Diagnóstico Forense M4")
            
            valor_A = str(resultado_A.valor_encontrado).strip().upper()
            valor_B = str(resultado_B.valor_encontrado).strip().upper()
            status_conflito = "" # Variável inicializada para evitar o NameError

            # REGRA 1: LACUNA
            if "NÃO LOCALIZADO" in valor_A or "NÃO LOCALIZADO" in valor_B:
                status_conflito = "LACUNA DE EVIDÊNCIA"
                st.warning("⚠️ **LACUNA DE EVIDÊNCIA DETECTADA:** A informação não existe em um dos documentos.")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📄 **{doc_a.name}**\n\n**Valor:** `{resultado_A.valor_encontrado}`\n\n_{resultado_A.contexto_da_clausula}_")
                with col2:
                    st.info(f"📄 **{doc_b.name}**\n\n**Valor:** `{resultado_B.valor_encontrado}`\n\n_{resultado_B.contexto_da_clausula}_")
                gravar_no_ledger(pergunta, f"{doc_a.name} / {doc_b.name}", "Falta de dados", "lacuna_evidencia")

            # REGRA 2: DIVERGÊNCIA
            elif valor_A != valor_B:
                status_conflito = "DIVERGÊNCIA CRÍTICA"
                st.error("⚔️ **DIVERGÊNCIA CRÍTICA DETECTADA:** Os documentos contradizem-se.")
                col1, col2 = st.columns(2)
                with col1:
                    st.error(f"📄 **{doc_a.name}**\n\n**Valor:** `{resultado_A.valor_encontrado}`\n\n_{resultado_A.contexto_da_clausula}_")
                with col2:
                    st.error(f"📄 **{doc_b.name}**\n\n**Valor:** `{resultado_B.valor_encontrado}`\n\n_{resultado_B.contexto_da_clausula}_")
                gravar_no_ledger(resultado_A.valor_encontrado, doc_a.name, resultado_A.contexto_da_clausula, "em_conflito", [doc_b.name])
                gravar_no_ledger(resultado_B.valor_encontrado, doc_b.name, resultado_B.contexto_da_clausula, "em_conflito", [doc_a.name])

            # REGRA 3: CONSENSO
            else:
                status_conflito = "CONSENSO"
                st.success("✅ **CONCORDÂNCIA CONFIRMADA:** Mesma informação em ambos.")
                st.markdown(f"**Valor Validado:** `{resultado_A.valor_encontrado}`")
                st.info(f"_{resultado_A.contexto_da_clausula}_")
                gravar_no_ledger(resultado_A.valor_encontrado, f"{doc_a.name} e {doc_b.name}", resultado_A.contexto_da_clausula, "validada")

            # ==========================================
            # 📝 SÍNTESE EXECUTIVA (M5)
            # ==========================================
            st.divider()
            st.subheader("📝 Parecer Executivo (M5 Sintetizador)")
            with st.spinner("Redigindo síntese executiva baseada em evidências..."):
                texto_sintese = gerar_sintese_m5(
                    pergunta=pergunta,
                    doc_a=doc_a.name, val_a=resultado_A.valor_encontrado, ctx_a=resultado_A.contexto_da_clausula,
                    doc_b=doc_b.name, val_b=resultado_B.valor_encontrado, ctx_b=resultado_B.contexto_da_clausula,
                    status=status_conflito
                )
                st.info(texto_sintese)
    else:
        st.warning("⚠️ Preencha a pergunta e carregue os dois PDFs para iniciar.")