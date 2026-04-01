import streamlit as st
import json
import os
import uuid
import PyPDF2
import requests
from datetime import datetime
from pydantic import BaseModel, Field

# ==========================================
# ⚙️ CONFIGURAÇÃO DA PÁGINA E ESTILO
# ==========================================
st.set_page_config(page_title="FLARE26: RAG Auditor", page_icon="⚖️", layout="wide")

# ==========================================
# 💾 SISTEMA DE PROVENIÊNCIA (LEDGER)
# ==========================================
def gravar_no_ledger(claim, fonte, contexto, estado, conflito_com=None):
    arquivo_ledger = "flare26_ledger.json"
    
    # Se não existir, cria a estrutura básica
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
        "confiabilidade": 0.99,
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
# 🧠 LENTE PYDANTIC (M2) - BLINDAGEM CONTRA ALUCINAÇÃO
# ==========================================
class ExtracaoAtomica(BaseModel):
    valor_encontrado: str = Field(
        description="O dado exato encontrado (ex: R$ 5.000, 10.000 km, 30 dias). SE A INFORMAÇÃO NÃO ESTIVER NO TEXTO, ESCREVA EXATAMENTE 'NÃO LOCALIZADO'."
    )
    contexto_da_clausula: str = Field(
        description="O parágrafo exato que comprova o valor. SE A INFORMAÇÃO NÃO ESTIVER NO TEXTO, ESCREVA 'LACUNA DE EVIDÊNCIA'."
    )
    confiabilidade: float = Field(
        description="De 0.0 a 1.0. Se a informação não estiver presente, coloque 0.0."
    )

# ==========================================
# 📄 LEITOR DE PDF
# ==========================================
def extrair_texto_pdf(arquivo):
    texto = ""
    leitor = PyPDF2.PdfReader(arquivo)
    for pagina in leitor.pages:
        texto += pagina.extract_text() + "\n"
    return texto

# ==========================================
# 🤖 CONEXÃO COM O MODELO qwen LOCAL
# ==========================================
def extrair_dado_com_ia(texto, pergunta):
    # Pegamos o schema gerado pelo Pydantic para orientar a IA
    schema_json = ExtracaoAtomica.model_json_schema()
    
    prompt = f"""
    [INSTRUÇÃO CRÍTICA]
    Você é um auditor forense extremamente rígido. 
    Seu único objetivo é encontrar a resposta EXATA para a pergunta no documento fornecido.
    
    REGRA DE OURO: 
    Se a pergunta for sobre "multa" e o texto falar apenas de "valor mensal" ou "pagamento", VOCÊ NÃO PODE EXTRAPOLAR. 
    Se a informação exata solicitada NÃO EXISTIR CLARAMENTE no documento, você é OBRIGADO a preencher 'valor_encontrado' como 'NÃO LOCALIZADO' e 'contexto_da_clausula' como 'LACUNA DE EVIDÊNCIA'.
    
    FORMATO ESPERADO (ESTRITAMENTE ESTE JSON):
    {json.dumps(schema_json)}
    
    PERGUNTA: {pergunta}
    
    DOCUMENTO:
    {texto[:4000]}
    """
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:1.5b",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "keep_alive": "10m",  # <-- MANTÉM O MODELO NA RAM POR 10 MINUTOS
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096       # <-- LIMITA O CONTEXTO PARA EVITAR TRAVAMENTOS NA CPU
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            resposta_texto = response.json().get("response", "{}")
            dados_json = json.loads(resposta_texto)
            return ExtracaoAtomica(**dados_json)
        else:
            return None
    except Exception as e:
        st.error(f"Erro na conexão/validação: {e}")
        return None

# ==========================================
# 🖥️ INTERFACE DO USUÁRIO (M6)
# ==========================================
st.title("FLARE26: RAG Auditor (Caixa de Vidro Corporativa)")
st.markdown("Auditoria Cruzada de Contratos e Documentos com Validação M4.")

st.header("1. Selecione os Documentos para Auditoria")
col_a, col_b = st.columns(2)

with col_a:
    doc_a = st.file_uploader("Selecione o Documento Base (A):", type=["pdf"], key="doc_a")

with col_b:
    doc_b = st.file_uploader("Selecione o Documento para Comparar (B):", type=["pdf"], key="doc_b")

st.header("2. O que você quer auditar nestes documentos?")
pergunta = st.text_input("Ex: valor da multa por rescisão, prazo de entrega, nome do contratante")

if st.button("🚀 Iniciar Auditoria", type="primary"):
    if doc_a and doc_b and pergunta:
        with st.status("Auditoria em andamento...", expanded=True) as status:
            st.write(f"📖 Lendo Documento A: {doc_a.name}...")
            texto_a = extrair_texto_pdf(doc_a)
            
            st.write(f"📖 Lendo Documento B: {doc_b.name}...")
            texto_b = extrair_texto_pdf(doc_b)
            
            st.write("🧠 Acionando Extrator Atômico (Lente Pydantic)...")
            resultado_A = extrair_dado_com_ia(texto_a, pergunta)
            st.write(f"🔍 DEBUG M2 {doc_a.name} Encontrou: {resultado_A is not None}")
            
            resultado_B = extrair_dado_com_ia(texto_b, pergunta)
            st.write(f"🔍 DEBUG M2 {doc_b.name} Encontrou: {resultado_B is not None}")
            
            st.write("⚖️ Passando os dados no Juiz de Conflito M4...")
            status.update(label="Auditoria Concluída!", state="complete", expanded=False)
            
        if resultado_A and resultado_B:
            st.header("🤖 Diagnóstico Forense M4")
            
            # ==========================================
            # ⚖️ JUIZ DE CONFLITO (M4) E RENDERIZAÇÃO
            # ==========================================
            valor_A = str(resultado_A.valor_encontrado).strip().upper()
            valor_B = str(resultado_B.valor_encontrado).strip().upper()

            # REGRA 1: DETECÇÃO DE LACUNA DE EVIDÊNCIA (A invenção patenteada)
            if "NÃO LOCALIZADO" in valor_A or "NÃO LOCALIZADO" in valor_B:
                st.warning("⚠️ **LACUNA DE EVIDÊNCIA DETECTADA:** A informação buscada não existe em um dos documentos.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📄 **{doc_a.name}**\n\n**Valor:** `{resultado_A.valor_encontrado}`")
                    st.caption(f"_{resultado_A.contexto_da_clausula}_")
                with col2:
                    st.info(f"📄 **{doc_b.name}**\n\n**Valor:** `{resultado_B.valor_encontrado}`")
                    st.caption(f"_{resultado_B.contexto_da_clausula}_")
                    
                gravar_no_ledger(pergunta, f"{doc_a.name} / {doc_b.name}", "Falta de dados em um dos documentos", "lacuna_evidencia")

            # REGRA 2: DIVERGÊNCIA CRÍTICA
            elif valor_A != valor_B:
                st.error(f"⚔️ **DIVERGÊNCIA CRÍTICA DETECTADA:** Os documentos contradizem-se.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.error(f"📄 **{doc_a.name}**\n\n**Valor Extraído:** `{resultado_A.valor_encontrado}`")
                    st.info(f"📜 **Contexto:**\n\n_{resultado_A.contexto_da_clausula}_")
                with col2:
                    st.error(f"📄 **{doc_b.name}**\n\n**Valor Extraído:** `{resultado_B.valor_encontrado}`")
                    st.info(f"📜 **Contexto:**\n\n_{resultado_B.contexto_da_clausula}_")
                    
                gravar_no_ledger(resultado_A.valor_encontrado, doc_a.name, resultado_A.contexto_da_clausula, "em_conflito", [doc_b.name])
                gravar_no_ledger(resultado_B.valor_encontrado, doc_b.name, resultado_B.contexto_da_clausula, "em_conflito", [doc_a.name])

            # REGRA 3: CONSENSO
            else:
                st.success("✅ **CONCORDÂNCIA CONFIRMADA:** Ambos os documentos apresentam a mesma informação.")
                st.markdown(f"**Valor Validado:** `{resultado_A.valor_encontrado}`")
                st.info(f"📜 **Contexto:**\n\n_{resultado_A.contexto_da_clausula}_")
                
                gravar_no_ledger(resultado_A.valor_encontrado, f"{doc_a.name} e {doc_b.name}", resultado_A.contexto_da_clausula, "validada")