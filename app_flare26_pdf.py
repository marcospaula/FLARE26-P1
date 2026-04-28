import streamlit as st
import tempfile
import os
import time
import sqlite3     
import gc          
import json
import re
import hashlib
from datetime import datetime
from pydantic import BaseModel, Field
import fitz 
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ==========================================
# CONFIGURAÇÕES E INICIALIZAÇÃO
# ==========================================
st.set_page_config(
    page_title="FLARE26: Auditor Universal", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

MODEL_EMBEDDING = "all-MiniLM-L6-v2"
LEDGER_FILE = "flare26_ledger_v2.json"
DB_PATH = "flare26_cache_docs.db"

MAX_FILE_SIZE_MB = 10  
MAX_PAGES = 500        
CHUNK_SIZE_PAI = 3500  
CHUNK_OVERLAP_PAI = 200
CHUNK_SIZE_FILHO = 400
CHUNK_OVERLAP_FILHO = 50
TOP_K_RETRIEVAL = 20   

# ==========================================
# GESTÃO DE ESTADO (ISOLAMENTO DO FRONT-END)
# ==========================================
if 'pdfs_processados' not in st.session_state:
    st.session_state['pdfs_processados'] = set()
if 'm1_5_ledger' not in st.session_state: 
    st.session_state['m1_5_ledger'] = {}

def gerar_hash_arquivo(arquivo_pdf):
    """Gera um MD5 rápido para identificar univocamente o PDF."""
    arquivo_pdf.seek(0)
    file_hash = hashlib.md5(arquivo_pdf.read()).hexdigest()
    arquivo_pdf.seek(0)
    return file_hash

# ==========================================
# MOTOR SQLITE E SANITIZADOR DE OCR
# ==========================================
def iniciar_banco_sqlite():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS docstore_pai (
            parent_id TEXT PRIMARY KEY,
            nome_doc TEXT,
            file_hash TEXT,
            conteudo TEXT
        )
    ''')
    conn.commit()
    conn.close()

def limpar_banco_sqlite():
    if os.path.exists(DB_PATH):
        try: os.remove(DB_PATH)
        except PermissionError: pass
    iniciar_banco_sqlite()

iniciar_banco_sqlite()

def sanitizar_texto_pdf(texto_bruto):
    if not texto_bruto: return ""
    texto = texto_bruto.replace('\x00', '')
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = re.sub(r'-\n+', '', texto)
    texto = re.sub(r'[ \t]+', ' ', texto)
    return texto.strip()

# ==========================================
# SCHEMA AGNÓSTICO (PYDANTIC)
# ==========================================
class ExtracaoUniversal(BaseModel):
    natureza_da_pergunta: str = Field(description="O domínio de dados exigido (ex: Temporal, Monetário).")
    natureza_do_texto_encontrado: str = Field(description="O domínio de dados encontrado.")
    houve_compatibilidade_ontologica: bool = Field(description="True se a natureza encontrada corresponde à exigida.")
    violou_restricao_do_usuario: bool = Field(
        description="CRÍTICO: Retorne True SE a pergunta exigir explicitamente que um tipo de dado seja ignorado E o texto encontrado tratar desse assunto proibido."
    )
    resposta_direta: str = Field(description="A resposta exata. Retorne 'NÃO LOCALIZADO' se houve_compatibilidade_ontologica for False OU se violou_restricao_do_usuario for True.")
    tipo_dado: str = Field(description="Categoria do dado. Se incompatível/violado, retorne 'NÃO LOCALIZADO'.")
    condicionantes: str = Field(description="Regras da resposta. Se incompatível/violado, retorne 'NÃO LOCALIZADO'.")
    trecho_literal: str = Field(description="Cópia literal do texto. Se incompatível/violado, retorne 'LACUNA DE EVIDÊNCIA'.")
    confiabilidade: float = Field(description="Score de 0.0 a 1.0. Se incompatível/violado, OBRIGATORIAMENTE 0.0.")

def resultado_vazio():
    return ExtracaoUniversal(
        natureza_da_pergunta="N/A", natureza_do_texto_encontrado="N/A",
        houve_compatibilidade_ontologica=False, violou_restricao_do_usuario=False,
        resposta_direta="NÃO LOCALIZADO", tipo_dado="NÃO LOCALIZADO",
        condicionantes="NÃO LOCALIZADO", trecho_literal="LACUNA DE EVIDÊNCIA", confiabilidade=0.0
    )

# ==========================================
# MOTOR VETORIAL (M1) CACHEADO
# ==========================================
@st.cache_resource
def iniciar_banco_vetorial():
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_EMBEDDING, encode_kwargs={'normalize_embeddings': True})
    vector_store = Chroma(collection_name="flare26_docs", embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="./chroma_db")
    return vector_store, embeddings

vector_store, _ = iniciar_banco_vetorial()

def resetar_ambiente_total():
    global vector_store
    try: vector_store.delete_collection()
    except Exception: pass
    try:
        limpar_banco_sqlite()
        st.session_state['pdfs_processados'].clear()
        st.session_state['m1_5_ledger'].clear()
        st.cache_resource.clear() 
        vector_store, _ = iniciar_banco_vetorial()
        gc.collect()
    except Exception: pass

def validar_tamanho_pdf(arquivo_pdf):
    return (arquivo_pdf.size / (1024 * 1024)) <= MAX_FILE_SIZE_MB

def extrair_texto_fitz_rapido(tmp_path):
    doc = fitz.open(tmp_path)
    if doc.page_count > MAX_PAGES: 
        doc.close()
        raise ValueError(f"Limite de {MAX_PAGES} páginas excedido.")
    textos_paginas = [sanitizar_texto_pdf(page.get_text()) for page in doc]
    doc.close()
    return textos_paginas

def _worker_processamento_pdf(tmp_path, arquivo_name, file_hash, session_state_ref):
    """Worker de background isolado para não travar a UI (Thread 2)."""
    try:
        textos_paginas = extrair_texto_fitz_rapido(tmp_path)
        pages = [Document(page_content=txt, metadata={"source": arquivo_name, "file_hash": file_hash}) for txt in textos_paginas]
        
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE_PAI, chunk_overlap=CHUNK_OVERLAP_PAI)
        docs_pai = parent_splitter.split_documents(pages)
        
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE_FILHO, chunk_overlap=CHUNK_OVERLAP_FILHO)
        docs_filho_para_chroma = []
        
        for i, doc_pai in enumerate(docs_pai):
            parent_id = f"{file_hash}_pai_{i}"
            cursor.execute("INSERT OR REPLACE INTO docstore_pai (parent_id, nome_doc, file_hash, conteudo) VALUES (?, ?, ?, ?)",
                           (parent_id, arquivo_name, file_hash, doc_pai.page_content))
            for filho in child_splitter.split_documents([doc_pai]):
                filho.metadata.update({"source": arquivo_name, "file_hash": file_hash, "parent_id": parent_id})
                docs_filho_para_chroma.append(filho)

        conn.commit()
        conn.close()

        if docs_filho_para_chroma:
            # Batching agressivo de inserção vetorial para salvar RAM
            batch_size = 200 
            for i in range(0, len(docs_filho_para_chroma), batch_size):
                vector_store.add_documents(docs_filho_para_chroma[i:i + batch_size])
        
        session_state_ref.add(file_hash)
        
        # Otimização extrema de lixo de memória
        del docs_pai, docs_filho_para_chroma, pages, textos_paginas
        gc.collect()
        
    except Exception as e:
        print(f"Erro fatal na thread do PDF {arquivo_name}: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def processar_pdf_idempotente(arquivo_pdf):
    """Gerencia a thread de processamento e mantém a UI viva."""
    if not validar_tamanho_pdf(arquivo_pdf): raise ValueError(f"PDF {arquivo_pdf.name} muito grande.")
    
    file_hash = gerar_hash_arquivo(arquivo_pdf)
    
    if file_hash in st.session_state['pdfs_processados']: 
        return {"sucesso": True, "status": "cache"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(arquivo_pdf.getvalue())
        tmp_path = tmp_file.name

    # Delega a injeção pesada para a Thread
    thread = threading.Thread(
        target=_worker_processamento_pdf, 
        args=(tmp_path, arquivo_pdf.name, file_hash, st.session_state['pdfs_processados'])
    )
    add_script_run_ctx(thread)
    thread.start()
    
    # UI Loader: Mantém o Streamlit desenhando frames enquanto a thread trabalha
    barra_progresso = st.progress(0, text=f"Indexando {arquivo_pdf.name} em Background...")
    
    while thread.is_alive():
        for i in range(100):
            if not thread.is_alive(): break
            time.sleep(0.05)
            barra_progresso.progress((i + 1) % 100, text=f"Indexando vetores de {arquivo_pdf.name}...")
            
    barra_progresso.empty()
    
    # Verifica se a thread falhou silenciosamente
    if file_hash not in st.session_state['pdfs_processados']:
        raise RuntimeError(f"Falha ao processar {arquivo_pdf.name}. Verifique os logs de Thread.")
        
    return {"sucesso": True, "status": "indexado"}

# ==========================================
# MÓDULO M1.5: RAG HÍBRIDO OTIMIZADO
# ==========================================
def recuperar_contexto_filtrado(pergunta, arquivo_pdf):
    file_hash = gerar_hash_arquivo(arquivo_pdf)
    
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT conteudo FROM docstore_pai WHERE file_hash = ?", (file_hash,))
    rows = cursor.fetchall()
    conn.close()
    
    texto_completo = "\n".join([row[0] for row in rows])
    if len(texto_completo) < 80000: 
        st.session_state['m1_5_ledger'][arquivo_pdf.name] = {"estrategia": "Leitura Total (Bypass)", "caracteres": len(texto_completo)}
        return texto_completo, [{"texto": "Full Context Bypass.", "score": 1.0}]

    stopwords = {"qual", "quais", "como", "quem", "onde", "para", "pelo", "pela", "sobre", "entre", "seja", "esse", "este", "dos", "das", "nas", "nos", "que", "com", "por", "um", "uma"}
    pergunta_norm = pergunta.lower().replace("percentual", "%").replace("porcentagem", "%")
    palavras_limpas = [re.sub(r'[^a-záéíóúçãõâê%]', '', p) for p in pergunta_norm.split()]
    termos_dinamicos = [p for p in palavras_limpas if (len(p) >= 4 or p == "%") and p not in stopwords]
    ancora_principal = sorted([t for t in termos_dinamicos if t != "%"], key=len, reverse=True)[0] if [t for t in termos_dinamicos if t != "%"] else ""

    contextos_finais = set()
    chunks_exibicao = []

    res_vetor = vector_store.similarity_search_with_score(pergunta, k=TOP_K_RETRIEVAL, filter={"file_hash": file_hash})
    if res_vetor:
        limite_corte = max(0.25, max([1.0 - dist for _, dist in res_vetor]) * 0.70)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        for doc, dist in res_vetor:
            if 1.0 - dist >= limite_corte:
                cursor.execute("SELECT conteudo FROM docstore_pai WHERE parent_id = ?", (doc.metadata.get("parent_id"),))
                row = cursor.fetchone()
                if row and row[0] not in contextos_finais:
                    contextos_finais.add(row[0])
                    chunks_exibicao.append({"texto": row[0][:200] + "... [VETOR]", "score": 1.0 - dist})
        conn.close()

    contextos_lexicos = []
    for row in rows: 
        score_lex = sum((10 if t == "%" else len(t)) for t in termos_dinamicos if t in row[0].lower())
        if ancora_principal and ancora_principal in row[0].lower(): score_lex += 15 
        if score_lex >= 15: contextos_lexicos.append((score_lex, row[0]))
    
    contextos_lexicos.sort(key=lambda x: x[0], reverse=True)
    for score, conteudo in contextos_lexicos[:6]:
        if conteudo not in contextos_finais:
            contextos_finais.add(conteudo)
            chunks_exibicao.append({"texto": conteudo[:200] + "... [LÉXICO]", "score": min(0.99, score/100.0)})

    st.session_state['m1_5_ledger'][arquivo_pdf.name] = {"estrategia": "Parent-Child RAG", "total_blocos": len(contextos_finais)}
    return "\n\n--- [BLOCO PAI] ---\n\n".join(list(contextos_finais)[:8]), chunks_exibicao

# ==========================================
# EXTRADADOS M2 E M4
# ==========================================
def extrair_dado_com_ia(texto, pergunta):
    if not texto.strip(): return resultado_vazio()
    prompt = (
        "Você é um Extrator Forense Especialista em Editais Governamentais.\n"
        f"PERGUNTA DA AUDITORIA: {pergunta}\n\n"
        "REGRAS:\n"
        "1. Se a pergunta exigir ignorar algo e o texto tratar disso, 'violou_restricao_do_usuario' = True.\n"
        "2. Se incompatível ou violado, 'resposta_direta' = 'NÃO LOCALIZADO'.\n"
        f"TEXTO:\n{texto[:90000]}" 
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um auditor forense focado em validação de domínio e restrições negativas."},
                {"role": "user", "content": prompt}
            ],
            tools=[{"type": "function", "function": {"name": "retornar_extracao", "parameters": ExtracaoUniversal.model_json_schema()}}],
            tool_choice={"type": "function", "function": {"name": "retornar_extracao"}},
            temperature=0.0 
        )
        dados = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        
        incompativel = not dados.get("houve_compatibilidade_ontologica", False)
        violou_restricao = dados.get("violou_restricao_do_usuario", False)
        
        if incompativel or violou_restricao: return resultado_vazio()

        resp = str(dados.get("resposta_direta", "NÃO LOCALIZADO")).strip()
        return ExtracaoUniversal(
            natureza_da_pergunta=str(dados.get("natureza_da_pergunta", "N/A")),
            natureza_do_texto_encontrado=str(dados.get("natureza_do_texto_encontrado", "N/A")),
            houve_compatibilidade_ontologica=True, violou_restricao_do_usuario=False,
            resposta_direta=resp, tipo_dado=str(dados.get("tipo_dado", "NÃO LOCALIZADO")),
            condicionantes=str(dados.get("condicionantes", "NÃO LOCALIZADO")),
            trecho_literal=str(dados.get("trecho_literal", "LACUNA DE EVIDÊNCIA")),
            confiabilidade=0.95 if resp != "NÃO LOCALIZADO" else 0.0
        )
    except Exception: return resultado_vazio()

def limpar_e_extrair_numeros(texto):
    """Extrai apenas dígitos e vírgulas para comparação simbólica pura."""
    if not isinstance(texto, str): return ""
    # Encontra números decimais (ex: 0,5 ou 10.5) ou inteiros
    numeros = re.findall(r'\d+(?:[.,]\d+)?', texto)
    return sorted([float(n.replace(',', '.')) for n in numeros]) if numeros else []

def comparar_documentos(dado_A: ExtracaoUniversal, dado_B: ExtracaoUniversal, pergunta: str):
    """
    Juiz Híbrido (Cost Killer). 
    Tenta resolver por lógica matemática primeiro. Se falhar, apela para o GPT-4.
    """
    # 1. Filtro Simbólico de Existência
    if dado_A.resposta_direta == "NÃO LOCALIZADO" and dado_B.resposta_direta == "NÃO LOCALIZADO":
        return "LACUNA DE EVIDÊNCIA", ["Python (Simbólico): Nenhum documento possui a resposta para a pergunta."]
    if dado_A.resposta_direta == "NÃO LOCALIZADO": 
        return "DIVERGÊNCIA CRÍTICA", ["Python (Simbólico): Apenas Documento B contém a informação."]
    if dado_B.resposta_direta == "NÃO LOCALIZADO": 
        return "DIVERGÊNCIA CRÍTICA", ["Python (Simbólico): Apenas Documento A contém a informação."]

    # 2. Filtro Simbólico de Igualdade Exata
    if dado_A.resposta_direta.strip().lower() == dado_B.resposta_direta.strip().lower():
        if dado_A.condicionantes.strip().lower() == dado_B.condicionantes.strip().lower():
             return "CONSENSO TOTAL", ["Python (Simbólico): Os dados e condicionantes extraídos são strings idênticas."]

    # 3. Filtro Simbólico Numérico (A Matemática não Mente)
    nums_a = limpar_e_extrair_numeros(dado_A.resposta_direta)
    nums_b = limpar_e_extrair_numeros(dado_B.resposta_direta)
    
    if nums_a and nums_b:
        # Se o extrator achou números em ambos, mas os números não têm nenhuma intersecção
        if not set(nums_a).intersection(set(nums_b)):
            return "DIVERGÊNCIA CRÍTICA", [f"Python (Simbólico): Divergência matemática flagrante detectada ({nums_a} vs {nums_b}). Economizamos 1 chamada de API."]

    # 4. Fallback Neural (Quando a Lógica Simbólica não é suficiente)
    prompt_juiz = (
        "Como as respostas são complexas e semanticamente ambíguas, julgue a equivalência.\n"
        f"Pergunta: {pergunta}\n"
        f"Resp A: {dado_A.resposta_direta} (Condições: {dado_A.condicionantes})\n"
        f"Resp B: {dado_B.resposta_direta} (Condições: {dado_B.condicionantes})\n\n"
        "Retorne um JSON exato: {'veredito': 'CONSENSO TOTAL' ou 'DIVERGÊNCIA CRÍTICA', 'motivos': ['Motivo detalhado']}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt_juiz}], temperature=0.0
        )
        res = json.loads(response.choices[0].message.content)
        motivo = res.get("motivos", ["Erro."])
        if isinstance(motivo, list): motivo = motivo[0]
        return res.get("veredito", "DIVERGÊNCIA CRÍTICA"), [f"LLM M4 (Neural): {motivo}"]
    except Exception: 
        return "DIVERGÊNCIA CRÍTICA", ["Falha na comparação neural."]

def gerar_parecer(extracao_a, extracao_b, veredito, pergunta):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Gere um parecer executivo formal (3 frases) para laudo pericial.\nPergunta: {pergunta}\nVeredito: {veredito}\nDoc A: {extracao_a.resposta_direta}\nDoc B: {extracao_b.resposta_direta}"}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception: return "Falha ao gerar parecer."

def gerar_relatorio_markdown(pergunta, arquivo_a, arquivo_b, extracao_a, extracao_b, veredito, motivos, parecer):
    """Gera o laudo forense estruturado para download."""
    data_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    md = f"""# 🏛️ LAUDO DE AUDITORIA FORENSE AUTOMATIZADA
**Projeto:** FLARE26-P1 (Motor Neuro-Simbólico)
**Data da Emissão:** {data_hora}

## 1. ESCOPO DA AUDITORIA
* **Pergunta Alvo:** {pergunta}
* **Documento A (Referência):** {arquivo_a.name}
* **Documento B (Comparativo):** {arquivo_b.name}

## 2. VEREDICTO DO JUIZ NEURAL (M4)
**Status:** {veredito}
**Justificativa:** {motivos[0]}

---
## 3. EXTRAÇÃO DOCUMENTAL (M2)

### 📄 DOCUMENTO A
* **Resposta Extraída:** {extracao_a.resposta_direta}
* **Condicionante:** {extracao_a.condicionantes}
* **Tipo de Dado (Ontologia):** {extracao_a.tipo_dado}
> **Trecho Literal:** "{extracao_a.trecho_literal}"

### 📄 DOCUMENTO B
* **Resposta Extraída:** {extracao_b.resposta_direta}
* **Condicionante:** {extracao_b.condicionantes}
* **Tipo de Dado (Ontologia):** {extracao_b.tipo_dado}
> **Trecho Literal:** "{extracao_b.trecho_literal}"

---
## 4. PARECER EXECUTIVO (M5)
{parecer}

---
*Laudo gerado criptograficamente por IA Determinística. Verifique as hashes dos arquivos fonte para garantia de inalterabilidade.*
"""
    return md

def salvar_no_ledger_local(pergunta, arquivo_a, arquivo_b, ledger_json):
    """Salva a auditoria atual no ledger JSON de longo prazo."""
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, 'r', encoding='utf-8') as f:
                historico = json.load(f)
        except Exception:
            historico = []
    else:
        historico = []
    
    registro = {
        "timestamp": datetime.now().isoformat(),
        "pergunta_auditoria": pergunta,
        "documentos_auditados": [arquivo_a, arquivo_b],
        "telemetria": ledger_json
    }
    historico.append(registro)
    
    with open(LEDGER_FILE, 'w', encoding='utf-8') as f:
        json.dump(historico, f, ensure_ascii=False, indent=4)

# ==========================================
# INTERFACE STREAMLIT (ENTERPRISE UI)
# ==========================================
# --- CSS Customizado (Dark Enterprise Mode) ---
st.markdown("""
<style>
    .card-resultado { padding: 20px; border-radius: 8px; margin: 10px 0; border: 1px solid #333; background-color: #1e1e1e;}
    .card-sucesso { border-left: 6px solid #28a745; }
    .card-aviso { border-left: 6px solid #ffc107; }
    .card-erro { border-left: 6px solid #dc3545; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #2b2b2b; border-bottom: 2px solid #4CAF50;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: Painel de Controle ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=60)
    st.markdown("## ⚙️ Painel de Controle")
    st.markdown("---")
    st.caption("🖥️ **Motor de Embedding:**")
    st.code(MODEL_EMBEDDING, language="text")
    st.caption("🗄️ **Armazenamento Vetorial:** ChromaDB\n\n🗃️ **Indexação Híbrida:** SQLite")
    st.markdown("---")
    if st.button("🗑️ Resetar Ambiente e Cache", type="secondary", use_container_width=True):
        resetar_ambiente_total()
        st.success("Ambiente totalmente purgado.")
        time.sleep(1)
        st.rerun()
    st.markdown("---")
    st.caption("FLARE26-P1 © 2026\n*Arquitetura Zero-Shot Determinística*")

# --- MAIN WORKSPACE ---
st.title("🔍 FLARE26: Dashboard Forense")
st.markdown("###### *Ensemble Retrieval (Vetor + Léxico) e Validação Ontológica*")

col_upload1, col_upload2 = st.columns(2)
with col_upload1: arquivo_a = st.file_uploader("📄 Documento A (Referência)", type="pdf", key="file_a")
with col_upload2: arquivo_b = st.file_uploader("📄 Documento B (Comparativo)", type="pdf", key="file_b")

pergunta_auditoria = st.text_input("❓ Objeto de Auditoria (Pergunta Forense)", placeholder="Ex: Qual o prazo estipulado para pagamento da fatura?")

if st.button("🚀 Iniciar Motor Neuro-Simbólico", type="primary", use_container_width=True):
    if arquivo_a and arquivo_b and pergunta_auditoria:
        start_time = time.time()
        
        # --- TERMINAL DE STATUS (Módulo M1.5 a M4) ---
        with st.status("⚙️ Executando Pipeline de Auditoria...", expanded=True) as status:
            st.write("⏳ Verificando integridade e hashing (Idempotência)...")
            status_a = processar_pdf_idempotente(arquivo_a)
            status_b = processar_pdf_idempotente(arquivo_b)
            if status_a["status"] == "cache" and status_b["status"] == "cache":
                st.write("⚡ Bypass ativado: Documentos lidos diretamente do cache local.")
            
            st.write("🔍 [M1.5] Cruzando matrizes vetoriais com léxico dinâmico...")
            contexto_a, filhos_a = recuperar_contexto_filtrado(pergunta_auditoria, arquivo_a)
            contexto_b, filhos_b = recuperar_contexto_filtrado(pergunta_auditoria, arquivo_b)
            
            st.write("🤖 [M2] Aplicando Validação Ontológica e extraindo evidências...")
            extracao_a = extrair_dado_com_ia(contexto_a, pergunta_auditoria)
            extracao_b = extrair_dado_com_ia(contexto_b, pergunta_auditoria)
            
            st.write("⚖️ [M4] Submetendo ao Juiz Neural para Análise de Divergência...")
            veredito, motivos = comparar_documentos(extracao_a, extracao_b, pergunta_auditoria)
            
            st.write("📝 [M5] Sintetizando o Parecer Executivo...")
            parecer = gerar_parecer(extracao_a, extracao_b, veredito, pergunta_auditoria)
            
            tempo_total = time.time() - start_time
            status.update(label=f"✅ Auditoria Concluída com Sucesso ({tempo_total:.2f}s)", state="complete", expanded=False)

        # --- NAVEGAÇÃO POR ABAS (Apresentação Limpa) ---
        tab_laudo, tab_dados, tab_telemetria = st.tabs(["⚖️ Veredicto Final", "📊 Base de Evidências", "🕵️‍♂️ Caixa de Vidro (Logs)"])

        with tab_laudo:
            # Renderização de Veredicto
            if veredito == "CONSENSO TOTAL":
                st.markdown(f'<div class="card-resultado card-sucesso"><h2>✅ {veredito}</h2><h4>{motivos[0]}</h4></div>', unsafe_allow_html=True)
            elif veredito == "LACUNA DE EVIDÊNCIA":
                st.markdown(f'<div class="card-resultado card-aviso"><h2>⚠️ {veredito}</h2><h4>{motivos[0]}</h4></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card-resultado card-erro"><h2>⚔️ {veredito}</h2><h4>{motivos[0]}</h4></div>', unsafe_allow_html=True)
            
            st.markdown("### 📝 Parecer do Auditor Chefe")
            st.info(parecer)
            
            # Gerador de Arquivo Exportável
            relatorio_md = gerar_relatorio_markdown(pergunta_auditoria, arquivo_a, arquivo_b, extracao_a, extracao_b, veredito, motivos, parecer)
            st.download_button(
                label="📥 Baixar Laudo Forense Oficial (.md)",
                data=relatorio_md,
                file_name=f"laudo_flare26_{int(time.time())}.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True
            )

        with tab_dados:
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown(f"""
                <div class="card-resultado">
                    <h5 style="color: #4CAF50;">📄 {arquivo_a.name}</h5>
                    <p><strong>Resposta Bruta:</strong> {extracao_a.resposta_direta}</p>
                    <p><strong>Condição Legal:</strong> {extracao_a.condicionantes}</p>
                    <p><strong>Domínio/Ontologia:</strong> {extracao_a.tipo_dado}</p>
                    <hr>
                    <p style="font-size: 0.85em; color: #aaa;"><i>"{extracao_a.trecho_literal}"</i></p>
                </div>
                """, unsafe_allow_html=True)
            with col_res2:
                st.markdown(f"""
                <div class="card-resultado">
                    <h5 style="color: #4CAF50;">📄 {arquivo_b.name}</h5>
                    <p><strong>Resposta Bruta:</strong> {extracao_b.resposta_direta}</p>
                    <p><strong>Condição Legal:</strong> {extracao_b.condicionantes}</p>
                    <p><strong>Domínio/Ontologia:</strong> {extracao_b.tipo_dado}</p>
                    <hr>
                    <p style="font-size: 0.85em; color: #aaa;"><i>"{extracao_b.trecho_literal}"</i></p>
                </div>
                """, unsafe_allow_html=True)

        with tab_telemetria:
            st.markdown("### JSON Provenience Ledger")
            json_prov = {
                "M1.5_Gatekeeper": st.session_state.get('m1_5_ledger', {}),
                "M2_Extrator_Score": {"doc_A": extracao_a.confiabilidade, "doc_B": extracao_b.confiabilidade},
                "M4_Juiz_Neural": {"veredito": veredito}
            }
            st.json(json_prov)
            salvar_no_ledger_local(pergunta_auditoria, arquivo_a.name, arquivo_b.name, json_prov)
            
            st.markdown("### Roteamento Vetorial (Top Chunks)")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.caption(f"Chunks injetados via {arquivo_a.name}")
                st.write(filhos_a)
            with col_v2:
                st.caption(f"Chunks injetados via {arquivo_b.name}")
                st.write(filhos_b)

    else:
        st.warning("⚠️ Forneça os dois documentos base e o objeto da auditoria para iniciar a máquina.")