import streamlit as st
import tempfile
import os
from openai import OpenAI
import requests
import json
import statistics
import math
import re
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Inicia a API da OpenAI lendo do secrets.toml
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ==========================================
# CONFIGURAÇÕES GERAIS E MODELOS
# ==========================================
st.set_page_config(page_title="FLARE26: Caixa de Vidro", layout="wide")

OLLAMA_URL = "http://localhost:11434"
MODEL_GERACAO = "qwen2.5:1.5b"  # M2 e M5
MODEL_EMBEDDING = "all-MiniLM-L6-v2"  # M1

# ==========================================
# ESTRUTURA DE DADOS (PYDANTIC) - M2
# ==========================================
class ExtracaoMultivariavel(BaseModel):
    valor_formatado: str = Field(description="O valor exato como aparece no texto")
    valor_numerico: str = Field(description="Apenas os números convertidos do valor")
    unidade_ou_moeda: str = Field(description="A unidade de medida ou moeda do valor")
    condicao_ou_prazo: str = Field(description="Condição de tempo, prazo ou evento de cobrança")
    contexto_da_clausula: str = Field(description="Cópia literal do trecho que baseou a resposta")
    confiabilidade: float = Field(description="Nota de 0.0 a 1.0 sobre a certeza da extração")

def resultado_vazio():
    return ExtracaoMultivariavel(
        valor_formatado="NÃO LOCALIZADO",
        valor_numerico="NÃO LOCALIZADO",
        unidade_ou_moeda="NÃO LOCALIZADO",
        condicao_ou_prazo="NÃO LOCALIZADO",
        contexto_da_clausula="LACUNA DE EVIDÊNCIA",
        confiabilidade=0.0
    )

# ==========================================
# MOTOR VETORIAL E RAG (M1 E M1.5)
# ==========================================
@st.cache_resource
def iniciar_banco_vetorial():
    # Normalizamos na raiz para garantir que L2 vire Cosine real de 0 a 1
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_EMBEDDING,
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = Chroma(
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vector_store, embeddings

vector_store, _ = iniciar_banco_vetorial()

def processar_pdf(arquivo_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(arquivo_pdf.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documentos = loader.load()
    
    for doc in documentos:
        doc.metadata["source"] = arquivo_pdf.name

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documentos)

    vector_store.add_documents(chunks)
    os.remove(tmp_path)

def recuperar_contexto_filtrado(pergunta, nome_documento, top_k=5):
    """
    Módulo M1.5: Filtro Vetorial Adaptativo (Threshold Dinâmico Estatístico)
    Calcula matematicamente o limite dinâmico Z-Score.
    """
    if not vector_store:
        return ""

    resultados_brutos = vector_store.similarity_search_with_score(
        pergunta, k=top_k, filter={"source": nome_documento}
    )

    if not resultados_brutos:
        return ""

    resultados = []
    for doc, distancia in resultados_brutos:
        # Reversão matemática da distância para escore de similaridade (0 a 1)
        similaridade = max(0.0, 1.0 - distancia)
        resultados.append((doc, similaridade))

    scores = [score for _, score in resultados]
    
    if len(scores) < 2:
        limite_corte = max(0.0, scores[0] - 0.05)
    else:
        media_scores = statistics.mean(scores)
        desvio_padrao = statistics.stdev(scores)
        limite_corte = media_scores + (desvio_padrao * 0.5)

    limite_corte = max(0.35, limite_corte) # Piso mínimo

    contextos_validos = []
    maior_score = 0.0

    for doc, score in resultados:
        if score >= limite_corte:
            contextos_validos.append(doc.page_content)
            if score > maior_score:
                maior_score = score

    # Registro de Auditoria M1.5
    if 'm1_5_ledger' not in st.session_state:
        st.session_state['m1_5_ledger'] = {}
    
    st.session_state['m1_5_ledger'][nome_documento] = {
        "limite_corte_calculado": round(limite_corte, 3),
        "maior_score_encontrado": round(maior_score, 3),
        "status": "Aprovado" if contextos_validos else "Reprovado"
    }

    return "\n\n".join(contextos_validos)

# ==========================================
# EXTRATOR NEURAL (M2) - HÍBRIDO E BLINDADO
# ==========================================
def extrair_dado_com_ia(texto, pergunta):
    if not texto.strip():
        return resultado_vazio()

    prompt = f"""
Sua tarefa é extrair dados em formato JSON estrito.
PERGUNTA: {pergunta}

Crie um JSON EXATAMENTE com as chaves abaixo. Se a informação não existir no texto, preencha o valor com "NÃO LOCALIZADO".
NUNCA invente informações.
{{
  "valor_formatado": "",
  "valor_numerico": "",
  "unidade_ou_moeda": "",
  "condicao_ou_prazo": "",
  "contexto_da_clausula": "",
  "confiabilidade": 0.0
}}

TEXTO:
{texto[:3000]}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # O modelo mais rápido e barato que dá conta com folga da extração JSON
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Você é um auditor forense de contratos super restrito. Retorne APENAS um JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        raw_response = response.choices[0].message.content
        
        print("====================================")
        print("RESPOSTA CRUA DO GPT (M2):")
        print(raw_response)
        print("====================================")
        
        data = json.loads(raw_response)
        
        dados_finais = {
            "valor_formatado": str(data.get("valor_formatado", "NÃO LOCALIZADO")),
            "valor_numerico": str(data.get("valor_numerico", "NÃO LOCALIZADO")),
            "unidade_ou_moeda": str(data.get("unidade_ou_moeda", "NÃO LOCALIZADO")),
            "condicao_ou_prazo": str(data.get("condicao_ou_prazo", "NÃO LOCALIZADO")),
            "contexto_da_clausula": str(data.get("contexto_da_clausula", "LACUNA DE EVIDÊNCIA")),
            "confiabilidade": float(data.get("confiabilidade", 0.0))
        }
        return ExtracaoMultivariavel(**dados_finais)
        
    except Exception as e:
        print(f"Erro Fatal M2 (OpenAI): {e}")
        return resultado_vazio()

# ==========================================
# JUIZ DETERMINÍSTICO (M4)
# ==========================================
def comparar_documentos(dado_A: ExtracaoMultivariavel, dado_B: ExtracaoMultivariavel):
    if dado_A.valor_numerico == "NÃO LOCALIZADO" and dado_B.valor_numerico == "NÃO LOCALIZADO":
        return "LACUNA DE EVIDÊNCIA", ["Documento A sem evidência localizável.", "Documento B sem evidência localizável."]
    
    if dado_A.valor_numerico == "NÃO LOCALIZADO":
        return "DIVERGÊNCIA CRÍTICA", ["Apenas Documento B contém a informação."]
        
    if dado_B.valor_numerico == "NÃO LOCALIZADO":
        return "DIVERGÊNCIA CRÍTICA", ["Apenas Documento A contém a informação."]

    motivos_divergencia = []
    
    if dado_A.valor_numerico != dado_B.valor_numerico:
        motivos_divergencia.append(f"Valor numérico divergente: {dado_A.valor_numerico} vs {dado_B.valor_numerico}.")
        
    if dado_A.unidade_ou_moeda != dado_B.unidade_ou_moeda:
        motivos_divergencia.append(f"Moeda/unidade divergente: {dado_A.unidade_ou_moeda} vs {dado_B.unidade_ou_moeda}.")
        
    if dado_A.condicao_ou_prazo.lower() != dado_B.condicao_ou_prazo.lower():
        motivos_divergencia.append(f"Condição/prazo divergente: '{dado_A.condicao_ou_prazo}' vs '{dado_B.condicao_ou_prazo}'.")

    if motivos_divergencia:
        return "DIVERGÊNCIA CRÍTICA", motivos_divergencia
    
    return "CONSENSO TOTAL", ["Os documentos são equivalentes em todas as dimensões analisadas."]

# ==========================================
# SINTETIZADOR DE PARECER (M5)
# ==========================================
def gerar_parecer_executivo(pergunta, doc_A, extracao_A, doc_B, extracao_B, veredito, motivos):
    # Prompt projetado para "prender" o LLM aos fatos
    prompt_sintese = f"""
Você é o M5 Sintetizador, um sistema auditor forense de contratos.
Sua única função é escrever um "Parecer Executivo" imparcial (máximo de 4 frases) explicando as conclusões de uma auditoria documental.

Você está PROIBIDO de inventar qualquer dado, número ou cláusula. Use APENAS as informações abaixo.
Se faltar informação, apenas diga que "há uma lacuna de evidência".

--- INFORMAÇÕES DA AUDITORIA (RAW DATA) ---
Pergunta Auditada: {pergunta}
Veredito do Juiz (M4): {veredito}
Motivos do Juiz: {', '.join(motivos)}

Dados do Contrato X ({doc_A}):
- Valor Formatado: {extracao_A.valor_formatado}
- Moeda: {extracao_A.unidade_ou_moeda}
- Condição/Prazo: {extracao_A.condicao_ou_prazo}

Dados do Contrato Y ({doc_B}):
- Valor Formatado: {extracao_B.valor_formatado}
- Moeda: {extracao_B.unidade_ou_moeda}
- Condição/Prazo: {extracao_B.condicao_ou_prazo}
-------------------------------------------

Escreva o Parecer Executivo de forma profissional e direta:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um auditor sênior redigindo pareceres jurídicos impecáveis e factuais."},
                {"role": "user", "content": prompt_sintese}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Erro M5 (OpenAI): {e}")
        return "O Sintetizador M5 falhou em gerar o parecer executivo devido a uma indisponibilidade de rede ou API."

# ==========================================
# INTERFACE DO USUÁRIO (STREAMLIT)
# ==========================================
st.title("FLARE26: RAG Auditor (Caixa de Vidro Corporativa)")
st.markdown("Auditoria Neuro-Simbólica com Filtro Vetorial Inteligente e Extração Multivariável.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Documento Base (A)")
    arquivo_a = st.file_uploader("Contrato A", type=["pdf"], key="file_a")

with col2:
    st.subheader("Documento Comparativo (B)")
    arquivo_b = st.file_uploader("Contrato B", type=["pdf"], key="file_b")

pergunta_auditoria = st.text_input("O que você quer auditar?", placeholder="Ex: Qual é o valor da multa compensatória e sua condição de cobrança?")

if st.button("Executar Auditoria Forense", type="primary"):
    if arquivo_a and arquivo_b and pergunta_auditoria:
        
        st.session_state['m1_5_ledger'] = {}
        
        with st.spinner("Inicializando motores neuro-simbólicos..."):
            processar_pdf(arquivo_a)
            processar_pdf(arquivo_b)

        with st.spinner("M1.5: Aplicando Limites Vetoriais Dinâmicos..."):
            contexto_a = recuperar_contexto_filtrado(pergunta_auditoria, arquivo_a.name)
            contexto_b = recuperar_contexto_filtrado(pergunta_auditoria, arquivo_b.name)

        with st.spinner("M2: Extraindo Matriz de Dados..."):
            extracao_a = extrair_dado_com_ia(contexto_a, pergunta_auditoria)
            extracao_b = extrair_dado_com_ia(contexto_b, pergunta_auditoria)

        with st.spinner("M4: Julgamento Determinístico..."):
            veredito, motivos = comparar_documentos(extracao_a, extracao_b)
            
        with st.spinner("M5: Gerando Parecer Executivo..."):
            parecer = gerar_parecer_executivo(
                pergunta_auditoria, arquivo_a.name, extracao_a, arquivo_b.name, extracao_b, veredito, motivos
            )

        st.success("Auditoria Concluída!")

        st.header("🤖 Diagnóstico Forense M4")
        
        if veredito == "CONSENSO TOTAL":
            st.success(f"✅ **CONSENSO TOTAL**: {motivos}")
        elif veredito == "LACUNA DE EVIDÊNCIA":
            st.warning("⚠️ **LACUNA DE EVIDÊNCIA**: Não foi possível localizar a informação nos dois documentos.")
        else:
            st.error(f"⚔️ **DIVERGÊNCIA CRÍTICA DETECTADA**: {motivos}")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"**📄 {arquivo_a.name}**")
            st.write(f"- Valor formatado: `{extracao_a.valor_formatado}`")
            st.write(f"- Valor numérico: `{extracao_a.valor_numerico}`")
            st.write(f"- Moeda/Unidade: `{extracao_a.unidade_ou_moeda}`")
            st.write(f"- Condição/Prazo: `{extracao_a.condicao_ou_prazo}`")
            st.caption(f"Contexto Base: *\"{extracao_a.contexto_da_clausula}\"*")
            
        with col_res2:
            st.markdown(f"**📄 {arquivo_b.name}**")
            st.write(f"- Valor formatado: `{extracao_b.valor_formatado}`")
            st.write(f"- Valor numérico: `{extracao_b.valor_numerico}`")
            st.write(f"- Moeda/Unidade: `{extracao_b.unidade_ou_moeda}`")
            st.write(f"- Condição/Prazo: `{extracao_b.condicao_ou_prazo}`")
            st.caption(f"Contexto Base: *\"{extracao_b.contexto_da_clausula}\"*")

        st.divider()
        st.subheader("📝 Parecer Executivo (M5 Sintetizador)")
        st.info(parecer)

        with st.expander("🕵️‍♂️ Trilha de Auditoria Forense (Caixa de Vidro)"):
            st.markdown("### 1. Filtro Vetorial Adaptativo (M1.5)")
            ledger_m1_5 = st.session_state.get('m1_5_ledger', {})
            
            for doc_name, dados_filtro in ledger_m1_5.items():
                st.write(f"**{doc_name}** | Limite Dinâmico Calculado: `{dados_filtro['limite_corte_calculado']}` | Maior Score: `{dados_filtro['maior_score_encontrado']}` | Status: {dados_filtro['status']}")

            st.markdown("### 2. JSON de Proveniência (Raw Data)")
            json_prov = {
                "M1.5_Filtro_Adaptativo": ledger_m1_5,
                "M2_Extrator_Multivariavel": {
                    "doc_A": extracao_a.model_dump(),
                    "doc_B": extracao_b.model_dump()
                },
                "M4_Juiz": {
                    "diagnostico": veredito,
                    "motivos": motivos
                }
            }
            st.json(json_prov)
    else:
        st.warning("Por favor, faça o upload dos dois contratos e insira uma pergunta.")