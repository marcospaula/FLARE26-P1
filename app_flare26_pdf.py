import streamlit as st
import tempfile
import os
from openai import OpenAI
import requests
import json
import statistics
import math
import re
from datetime import datetime
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

# Arquivo físico do Ledger
LEDGER_FILE = "flare26_ledger_v2.json"

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
# SALVAMENTO DO LEDGER LOCAL (FIX)
# ==========================================
def salvar_no_ledger_local(pergunta, doc_a_name, doc_b_name, json_data):
    """
    Grava o JSON de proveniência no arquivo de ledger local
    garantindo que o histórico de testes da patente não se perca.
    """
    ledger_history = []
    
    # Lê o ledger existente, se houver
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, 'r', encoding='utf-8') as f:
                ledger_history = json.load(f)
        except json.JSONDecodeError:
            pass # Se o arquivo estiver corrompido, começa um novo

    # Monta o novo registro
    novo_registro = {
        "id_teste": f"T_{len(ledger_history) + 1}",
        "timestamp": datetime.now().isoformat(),
        "pergunta_auditada": pergunta,
        "documentos": {"doc_A": doc_a_name, "doc_B": doc_b_name},
        "caixa_de_vidro": json_data
    }
    
    # Adiciona e salva
    ledger_history.append(novo_registro)
    
    with open(LEDGER_FILE, 'w', encoding='utf-8') as f:
        json.dump(ledger_history, f, indent=4, ensure_ascii=False)

# ==========================================
# MOTOR VETORIAL E RAG PAI/FILHO (M1 E M1.5)
# ==========================================
# ==========================================
# MOTOR VETORIAL E RAG PAI/FILHO NATIVO (M1 E M1.5)
# ==========================================
@st.cache_resource
def iniciar_banco_vetorial():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_EMBEDDING,
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Único banco vetorial, sem InMemoryStore
    vector_store = Chroma(
        collection_name="flare26_docs",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return vector_store, embeddings

vector_store, _ = iniciar_banco_vetorial()

# Dicionário global (em memória) para guardar os blocos Pais.
# Na Fase 2 isso irá para o SQLite, mas agora resolve o problema do Streamlit.
if "docstore_pai" not in st.session_state:
    st.session_state.docstore_pai = {}

def processar_pdf(arquivo_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(arquivo_pdf.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documentos = loader.load()
    
    # 1. Cria os blocos grandes (Pais) de 2000 caracteres
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs_pai = parent_splitter.split_documents(documentos)
    
    # 2. Cria os blocos pequenos (Filhos) de 400 caracteres
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    docs_filho_para_chroma = []
    
    for i, doc_pai in enumerate(docs_pai):
        # Cria um ID único para cada bloco pai
        parent_id = f"{arquivo_pdf.name}_pai_{i}"
        
        # Salva o bloco pai no dicionário da sessão (nosso próprio InMemoryStore)
        st.session_state.docstore_pai[parent_id] = doc_pai.page_content
        
        # Corta o pai em filhos
        docs_filhos = child_splitter.split_documents([doc_pai])
        
        # Anota em cada filho de quem ele descende
        for filho in docs_filhos:
            filho.metadata["source"] = arquivo_pdf.name
            filho.metadata["parent_id"] = parent_id
            docs_filho_para_chroma.append(filho)

    # 3. Indexa SÓ OS FILHOS no Chroma
    if docs_filho_para_chroma:
        vector_store.add_documents(docs_filho_para_chroma)
        
    os.remove(tmp_path)

def recuperar_contexto_filtrado(pergunta, nome_documento, top_k=5):
    """
    Módulo M1.5: Filtro Vetorial Adaptativo (Engenharia Nativa)
    """
    # Busca os "Filhos" no Chroma
    resultados_brutos = vector_store.similarity_search_with_score(
        pergunta, k=top_k, filter={"source": nome_documento}
    )

    if not resultados_brutos:
        return ""

    resultados = []
    for doc, distancia in resultados_brutos:
        similaridade = max(0.0, 1.0 - distancia)
        resultados.append((doc, similaridade))

    scores = [score for _, score in resultados]
    
    if len(scores) < 2:
        limite_corte = max(0.0, scores[0] - 0.05)
    else:
        media_scores = statistics.mean(scores)
        desvio_padrao = statistics.stdev(scores)
        limite_corte = media_scores + (desvio_padrao * 0.5)

    limite_corte = max(0.35, limite_corte) 

    contextos_validos = []
    maior_score = 0.0
    parent_ids_vistos = set()

    for doc, score in resultados:
        if score >= limite_corte:
            if score > maior_score:
                maior_score = score
                
            # A MÁGICA NATIVA: Lê o ID do pai e busca no nosso dicionário
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in parent_ids_vistos:
                conteudo_pai = st.session_state.docstore_pai.get(parent_id)
                if conteudo_pai:
                    contextos_validos.append(conteudo_pai)
                    parent_ids_vistos.add(parent_id)

    if 'm1_5_ledger' not in st.session_state:
        st.session_state['m1_5_ledger'] = {}
    
    st.session_state['m1_5_ledger'][nome_documento] = {
        "limite_corte_calculado": round(limite_corte, 3),
        "maior_score_encontrado": round(maior_score, 3),
        "status": "Aprovado" if contextos_validos else "Reprovado"
    }

    return "\n\n--- [NOVO BLOCO DE CONTEXTO - PAI] ---\n\n".join(contextos_validos)

# ==========================================
# EXTRATOR NEURAL (M2) - HÍBRIDO E BLINDADO
# ==========================================
# ==========================================
# EXTRATOR NEURAL (M2) - HÍBRIDO E BLINDADO (COM SEED)
# ==========================================
def extrair_dado_com_ia(texto, pergunta):
    if not texto.strip():
        return resultado_vazio()

    prompt = f"""
Sua tarefa é atuar como um Extrator Forense de Dados Universal.
Você deve analisar o TEXTO abaixo e extrair os dados solicitados na PERGUNTA, retornando APENAS um JSON válido.

PERGUNTA DA AUDITORIA: {pergunta}

REGRAS ESTRITAS DE PREENCHIMENTO DO JSON:
1. "valor_formatado": O valor exato, número e unidade juntos, exatamente como escrito no texto (Exemplos: "15 dias", "R$ 5.000,00", "45 kg/m³", "12,5%").
2. "valor_numerico": OBRIGATÓRIO isolar APENAS os algarismos (Exemplos: "15", "5000.00", "45", "12.5"). Se não houver número explícito, retorne "NÃO LOCALIZADO".
3. "unidade_ou_moeda": Apenas a métrica, moeda ou grandeza (Exemplos: "dias", "kg/m³", "USD", "%", "toneladas").
4. "condicao_ou_prazo": O evento gatilho, restrição de engenharia, escopo financeiro ou contexto temporal (Exemplos: "após aprovação", "em temperatura ambiente", "margem líquida anual", "em caso de quebra contratual").
5. "contexto_da_clausula": A cópia LITERAL e EXATA do parágrafo, linha de tabela ou frase do texto que prova a sua extração.
6. Se a informação não existir no texto, preencha todos os campos textuais com "NÃO LOCALIZADO" e a confiabilidade com 0.0.
7. "confiabilidade": Se a informação foi encontrada e extraída, DEVE ser um valor numérico entre 0.8 e 1.0 (sendo 1.0 a certeza absoluta baseada no texto). NUNCA retorne 0.0 se você extraiu a informação.

NUNCA invente informações. NUNCA misture números com palavras no campo "valor_numerico". O sistema é agnóstico: os dados podem ser jurídicos, financeiros, médicos ou de engenharia.

TEXTO PARA AUDITORIA:
{texto[:3000]}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Você é um extrator de dados determinístico. Retorne apenas o JSON estruturado."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            seed=4242  # A mágica do determinismo: força a mesma amostragem randômica na GPU da OpenAI
        )
        
        raw_response = response.choices[0].message.content
        
        print("====================================")
        print("RESPOSTA CRUA DO GPT (M2) COM SEED:")
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
        return "LACUNA DE EVIDÊNCIA", ["Nenhum dos documentos contém a informação solicitada."]
    
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
    prompt_sintese = f"""
Você é o M5 Sintetizador, um sistema auditor forense de documentos.
Sua única função é escrever um "Parecer Executivo" imparcial (máximo de 4 frases) explicando as conclusões de uma auditoria documental.

Você está PROIBIDO de inventar qualquer dado, número ou cláusula. Use APENAS as informações abaixo.
Se faltar informação, apenas diga que "há uma lacuna de evidência".

--- INFORMAÇÕES DA AUDITORIA (RAW DATA) ---
Pergunta Auditada: {pergunta}
Veredito do Juiz (M4): {veredito}
Motivos do Juiz: {', '.join(motivos)}

Dados do Documento X ({doc_A}):
- Valor Formatado: {extracao_A.valor_formatado}
- Moeda/Unidade: {extracao_A.unidade_ou_moeda}
- Condição/Prazo: {extracao_A.condicao_ou_prazo}

Dados do Documento Y ({doc_B}):
- Valor Formatado: {extracao_B.valor_formatado}
- Moeda/Unidade: {extracao_B.unidade_ou_moeda}
- Condição/Prazo: {extracao_B.condicao_ou_prazo}
-------------------------------------------

Escreva o Parecer Executivo de forma profissional e direta:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um auditor sênior redigindo pareceres documentais impecáveis e factuais."},
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
    arquivo_a = st.file_uploader("Documento A", type=["pdf"], key="file_a")

with col2:
    st.subheader("Documento Comparativo (B)")
    arquivo_b = st.file_uploader("Documento B", type=["pdf"], key="file_b")

pergunta_auditoria = st.text_input("O que você quer auditar?", placeholder="Ex: Qual é o valor unitário e a condição de frete?")

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

        with st.expander("🕵️‍♂️ Trilha de Auditoria Forense (Caixa de Vidro)", expanded=True):
            st.markdown("### 1. Filtro Vetorial Adaptativo Híbrido (M1.5)")
            ledger_m1_5 = st.session_state.get('m1_5_ledger', {})
            
            for doc_name, dados_filtro in ledger_m1_5.items():
                st.write(f"**{doc_name}** | Limite Dinâmico Calculado: `{dados_filtro['limite_corte_calculado']}` | Maior Score (Filho): `{dados_filtro['maior_score_encontrado']}` | Status: {dados_filtro['status']}")

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

            # FIX DO LEDGER: Salva no arquivo local JSON imediatamente
            salvar_no_ledger_local(pergunta_auditoria, arquivo_a.name, arquivo_b.name, json_prov)
            st.caption("💾 *Registro salvo automaticamente no Ledger Local (`flare26_ledger_v2.json`)*")

    else:
        st.warning("Por favor, faça o upload dos dois documentos e insira uma pergunta.")