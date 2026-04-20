import streamlit as st
import tempfile
import os
import sqlite3     
import gc          
from openai import OpenAI
import json
import re
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# CONFIGURAÇÕES E INICIALIZAÇÃO
# ==========================================
st.set_page_config(page_title="FLARE26: Auditor Universal", layout="wide")
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
# MOTOR SQLITE E SANITIZADOR DE OCR
# ==========================================
def iniciar_banco_sqlite():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS docstore_pai (
            parent_id TEXT PRIMARY KEY,
            nome_doc TEXT,
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
    resposta_direta: str = Field(description="A resposta exata, direta e sintetizada para a pergunta")
    tipo_dado: str = Field(description="Categoria do dado: 'Monetário', 'Temporal', 'Nominal', 'Percentual', ou 'Texto'")
    condicionantes: str = Field(description="Qualquer regra, exceção ou condição atrelada à resposta")
    trecho_literal: str = Field(description="Cópia literal e exata da cláusula original")
    confiabilidade: float = Field(description="Score de 0.0 a 1.0 da certeza da IA")

def resultado_vazio():
    return ExtracaoUniversal(
        resposta_direta="NÃO LOCALIZADO",
        tipo_dado="NÃO LOCALIZADO",
        condicionantes="NÃO LOCALIZADO",
        trecho_literal="LACUNA DE EVIDÊNCIA",
        confiabilidade=0.0
    )

# ==========================================
# MOTOR VETORIAL (M1) E INGESTÃO DE DISCO
# ==========================================
@st.cache_resource
def iniciar_banco_vetorial():
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_EMBEDDING, encode_kwargs={'normalize_embeddings': True})
    vector_store = Chroma(collection_name="flare26_docs", embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})
    return vector_store, embeddings

vector_store, _ = iniciar_banco_vetorial()

def limpar_banco_vetorial():
    global vector_store
    try:
        vector_store.delete_collection()
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_EMBEDDING, encode_kwargs={'normalize_embeddings': True})
        vector_store = Chroma(collection_name="flare26_docs", embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})
        limpar_banco_sqlite()
        st.session_state['m1_5_ledger'] = {}
        gc.collect()
        return True
    except Exception as e:
        return False

def validar_tamanho_pdf(arquivo_pdf):
    if arquivo_pdf.size / (1024 * 1024) > MAX_FILE_SIZE_MB: return False
    return True

def processar_pdf(arquivo_pdf):
    if not validar_tamanho_pdf(arquivo_pdf): raise ValueError("PDF muito grande.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(arquivo_pdf.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages_all = loader.load()
        if len(pages_all) > MAX_PAGES: raise ValueError(f"Limite de {MAX_PAGES} páginas excedido.")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        batch_size = 50
        
        for batch_start in range(0, len(pages_all), batch_size):
            batch_end = min(batch_start + batch_size, len(pages_all))
            pages = pages_all[batch_start:batch_end]
            
            for p in pages: p.page_content = sanitizar_texto_pdf(p.page_content)
            
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE_PAI, chunk_overlap=CHUNK_OVERLAP_PAI)
            docs_pai = parent_splitter.split_documents(pages)
            
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE_FILHO, chunk_overlap=CHUNK_OVERLAP_FILHO)
            docs_filho_para_chroma = []
            
            for i, doc_pai in enumerate(docs_pai):
                parent_id = f"{arquivo_pdf.name}_pai_{batch_start}_{i}"
                cursor.execute("INSERT OR REPLACE INTO docstore_pai (parent_id, nome_doc, conteudo) VALUES (?, ?, ?)",
                               (parent_id, arquivo_pdf.name, doc_pai.page_content))
                
                for filho in child_splitter.split_documents([doc_pai]):
                    filho.metadata["source"] = arquivo_pdf.name
                    filho.metadata["parent_id"] = parent_id
                    docs_filho_para_chroma.append(filho)

            conn.commit()

            if docs_filho_para_chroma:
                for i in range(0, len(docs_filho_para_chroma), 500):
                    vector_store.add_documents(docs_filho_para_chroma[i:i + 500])
            
            del docs_pai, docs_filho_para_chroma, pages
            gc.collect()
            
        conn.close()
        return {"sucesso": True}
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# ==========================================
# MÓDULO M1.5: RAG HÍBRIDO (ENSEMBLE)
# ==========================================
# ==========================================
# MÓDULO M1.5: RAG HÍBRIDO (ENSEMBLE V2)
# ==========================================
# ==========================================
# MÓDULO M1.5: RAG HÍBRIDO PARENT-CHILD (ENSEMBLE V3)
# ==========================================
# ==========================================
# MÓDULO M1.5: RAG HÍBRIDO (COM BYPASS DE LEITURA TOTAL)
# ==========================================
def recuperar_contexto_filtrado(pergunta, nome_documento):
    """FUSÃO HÍBRIDA V4: Se o documento for pequeno, injeta TUDO no LLM. Senão, usa Parent-Child."""
    
    # 1. TENTA O BYPASS DE LEITURA TOTAL (A MELHOR SOLUÇÃO PARA LLMs MODERNOS)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT conteudo FROM docstore_pai WHERE nome_doc = ?", (nome_documento,))
    rows = cursor.fetchall()
    conn.close()
    
    texto_completo = "\n".join([row[0] for row in rows])
    
    # Se o documento tiver menos de ~40.000 caracteres (Aprox. 20 páginas), MANDAMOS TUDO!
    # O LLM é muito melhor em achar agulha em palheiro do que o Vector/Grep.
    if len(texto_completo) < 80000: 
        if 'm1_5_ledger' not in st.session_state: st.session_state['m1_5_ledger'] = {}
        st.session_state['m1_5_ledger'][nome_documento] = {
            "estrategia": "Leitura Total (Bypass de RAG)",
            "status": "Documento inteiro enviado para o M2",
            "caracteres_lidos": len(texto_completo)
        }
        
        chunks_exibicao = [{"texto": "Documento pequeno o suficiente para Leitura Total (Full Context). Bypass de vetores acionado.", "score": 1.0}]
        return texto_completo, chunks_exibicao

    # 2. SE FOR UM PDF GIGANTE (> 80k chars), CAI NO RAG HÍBRIDO PARENT-CHILD
    stopwords = {"qual", "quais", "como", "quem", "onde", "para", "pelo", "pela", "sobre", "entre", "seja", "esse", "este", "dos", "das", "nas", "nos", "que", "com", "por", "um", "uma"}
    pergunta_norm = pergunta.lower().replace("percentual", "%").replace("porcentagem", "%")
    palavras_limpas = [re.sub(r'[^a-záéíóúçãõâê%]', '', p) for p in pergunta_norm.split()]
    termos_dinamicos = [p for p in palavras_limpas if (len(p) >= 4 or p == "%") and p not in stopwords]

    if not termos_dinamicos: termos_dinamicos = palavras_limpas
    ancora_principal = sorted([t for t in termos_dinamicos if t != "%"], key=len, reverse=True)[0] if [t for t in termos_dinamicos if t != "%"] else ""

    contextos_finais = set()
    chunks_exibicao = []

    res_vetor = vector_store.similarity_search_with_score(pergunta, k=15, filter={"source": nome_documento})
    if res_vetor:
        scores = [1.0 - dist for _, dist in res_vetor]
        limite_corte = max(0.25, max(scores) * 0.70)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for doc, dist in res_vetor:
            if 1.0 - dist >= limite_corte:
                parent_id = doc.metadata.get("parent_id")
                cursor.execute("SELECT conteudo FROM docstore_pai WHERE parent_id = ?", (parent_id,))
                row = cursor.fetchone()
                if row and row[0] not in contextos_finais:
                    contextos_finais.add(row[0])
                    chunks_exibicao.append({"texto": row[0][:400] + "... [VETOR]", "score": 1.0 - dist})
        conn.close()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT conteudo FROM docstore_pai WHERE nome_doc = ?", (nome_documento,))
    rows_fallback = cursor.fetchall()
    conn.close()
    
    contextos_lexicos = []
    for row in rows_fallback:
        conteudo_pai = row[0]
        conteudo_lower = conteudo_pai.lower()
        score_lex = sum((10 if t == "%" else len(t)) for t in termos_dinamicos if t in conteudo_lower)
        if ancora_principal and ancora_principal in conteudo_lower: score_lex += 15 
        if ancora_principal in conteudo_lower and re.search(r'\d+[,\.]\d+|\d+%', conteudo_lower): score_lex += 30 
        
        if score_lex >= 15: contextos_lexicos.append((score_lex, conteudo_pai))
    
    contextos_lexicos.sort(key=lambda x: x[0], reverse=True)
    
    for score, conteudo in contextos_lexicos[:6]: # Pegamos mais blocos de fallback
        if conteudo not in contextos_finais:
            contextos_finais.add(conteudo)
            chunks_exibicao.append({"texto": conteudo[:400] + "... [LÉXICO]", "score": min(0.99, score/100.0)})

    if 'm1_5_ledger' not in st.session_state: st.session_state['m1_5_ledger'] = {}
    st.session_state['m1_5_ledger'][nome_documento] = {"estrategia": "Parent-Child RAG", "total_blocos": len(contextos_finais)}

    # Retorna os blocos fundidos
    return "\n\n--- [BLOCO PAI] ---\n\n".join(list(contextos_finais)[:8]), chunks_exibicao

# ==========================================
# EXTRATOR NEURAL (M2) - AGNÓSTICO E SEGURO
# ==========================================
# ==========================================
# EXTRATOR NEURAL (M2) - CÃO DE CAÇA FORENSE
# ==========================================
def extrair_dado_com_ia(texto, pergunta):
    if not texto.strip(): return resultado_vazio()

    # O PROMPT AGRESSIVO: Proíbe o falso negativo e força a inferência em PDFs sujos.
    prompt = (
        "Você é um Extrator Forense Especialista em Editais Governamentais complexos.\n"
        "Sua missão é extrair dados OBRIGATORIAMENTE, tolerando textos quebrados por OCR.\n\n"
        f"PERGUNTA DA AUDITORIA: {pergunta}\n\n"
        "REGRAS CRÍTICAS DE EXTRAÇÃO (LEIA COM ATENÇÃO):\n"
        "1. Nunca desista fácil. Se o texto mencionar valores como '0,5% a 1%', '20%', '0,5% a 30%', MESMO que a frase esteja gramaticalmente estranha ou truncada, extraia esse número imediatamente como a resposta.\n"
        "2. Se houver uma alínea ou inciso citando 'inexecução' e logo depois uma multa associada a essa alínea, faça a conexão lógica e extraia o valor da multa.\n"
        "3. SÓ use 'NÃO LOCALIZADO' se realmente não existir ABSOLUTAMENTE NENHUM número ou percentual aplicável à infração.\n\n"
        "REGRAS DO JSON:\n"
        "1. 'resposta_direta': Apenas o valor bruto (Ex: '0,5% a 1%' ou '20%').\n"
        "2. 'tipo_dado': 'Percentual' ou 'Monetário'.\n"
        "3. 'condicionantes': Resuma quando a multa é aplicada (Ex: 'Para inexecução total ou parcial').\n"
        "4. 'trecho_literal': A cópia do parágrafo, mesmo que quebrado.\n\n"
        "TEXTO DO DOCUMENTO PARA ANÁLISE:\n"
        f"{texto[:90000]}" # Permite a leitura total de editais de até ~25 páginas
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Você é um auditor forense agressivo. Faça conexões lógicas e extraia o dado obrigatório. Só devolva JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Aumentamos levemente para permitir inferência lógica
            seed=4242
        )
        dados = json.loads(response.choices[0].message.content)
        
        resp = str(dados.get("resposta_direta", "NÃO LOCALIZADO")).strip()
        conf = 0.95 if resp.upper() != "NÃO LOCALIZADO" else 0.0
        
        return ExtracaoUniversal(
            resposta_direta=resp,
            tipo_dado=str(dados.get("tipo_dado", "NÃO LOCALIZADO")),
            condicionantes=str(dados.get("condicionantes", "NÃO LOCALIZADO")),
            trecho_literal=str(dados.get("trecho_literal", "LACUNA DE EVIDÊNCIA")),
            confiabilidade=conf
        )
    except Exception:
        return resultado_vazio()

# ==========================================
# JUIZ NEURAL ESTRUTURADO (M4 AGNÓSTICO)
# ==========================================
def comparar_documentos(dado_A: ExtracaoUniversal, dado_B: ExtracaoUniversal, pergunta: str):
    if dado_A.resposta_direta == "NÃO LOCALIZADO" and dado_B.resposta_direta == "NÃO LOCALIZADO":
        return "LACUNA DE EVIDÊNCIA", ["Nenhum documento possui a resposta para a pergunta."]
    if dado_A.resposta_direta == "NÃO LOCALIZADO": return "DIVERGÊNCIA CRÍTICA", ["Apenas Documento B contém a informação."]
    if dado_B.resposta_direta == "NÃO LOCALIZADO": return "DIVERGÊNCIA CRÍTICA", ["Apenas Documento A contém a informação."]

    prompt_juiz = (
        "Você é um Juiz de Auditoria. Compare as respostas extraídas de dois documentos para a mesma pergunta.\n"
        f"Pergunta: {pergunta}\n"
        f"Resposta A: {dado_A.resposta_direta} (Condições: {dado_A.condicionantes})\n"
        f"Resposta B: {dado_B.resposta_direta} (Condições: {dado_B.condicionantes})\n\n"
        "Eles dizem a mesma coisa do ponto de vista legal ou prático (ex: '20%' vs '20%', '0.5%' vs 'meio por cento')?\n"
        "Retorne um JSON exato:\n"
        '{"veredito": "CONSENSO TOTAL" ou "DIVERGÊNCIA CRÍTICA", "motivos": ["Motivo da diferença"]}'
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt_juiz}],
            temperature=0.0
        )
        resultado = json.loads(response.choices[0].message.content)
        return resultado.get("veredito", "DIVERGÊNCIA CRÍTICA"), resultado.get("motivos", ["Erro na análise."])
    except Exception:
        return "DIVERGÊNCIA CRÍTICA", [f"A: {dado_A.resposta_direta} | B: {dado_B.resposta_direta}"]

# ==========================================
# PARECER EXECUTIVO (M5)
# ==========================================
def gerar_parecer(extracao_a, extracao_b, veredito, pergunta):
    prompt_m5 = (
        "Gere um parecer executivo de 3 frases como auditor sênior.\n"
        f"Pergunta: {pergunta}\n"
        f"Veredicto: {veredito}\n"
        f"Doc A: {extracao_a.resposta_direta}\n"
        f"Doc B: {extracao_b.resposta_direta}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_m5}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception:
        return "O Sintetizador M5 falhou em gerar o parecer executivo."

def salvar_no_ledger_local(pergunta, doc_a_name, doc_b_name, json_data):
    ledger = []
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, 'r') as f: ledger = json.load(f)
        except: pass 
    ledger.append({
        "id": f"T_{len(ledger)+1}", "data": datetime.now().isoformat(),
        "pergunta": pergunta, "caixa_vidro": json_data
    })
    with open(LEDGER_FILE, 'w') as f: json.dump(ledger, f, indent=4, ensure_ascii=False)

# ==========================================
# INTERFACE STREAMLIT
# ==========================================
st.title("🔍 FLARE26: Auditor Forense Universal")
st.markdown("### Arquitetura Enterprise: Ensemble Retrieval (Vetor + Léxico)")

st.markdown("""
<style>
    .card-resultado { padding: 15px; border-radius: 10px; margin: 10px 0; border: 1px solid #444;}
    .card-sucesso { background: #1a3d2e; border-left: 5px solid #28a745; color: #a8e6cf; }
    .card-aviso { background: #3d3520; border-left: 5px solid #ffc107; color: #f9e79f; }
    .card-erro { background: #3d2020; border-left: 5px solid #dc3545; color: #f5b7b1; }
    .card-parecer { background: #2d2d2d; border-left: 5px solid #9b59b6; color: #d2b4de;}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1: arquivo_a = st.file_uploader("📄 Documento A (Referência)", type="pdf")
with col2: arquivo_b = st.file_uploader("📄 Documento B (Comparativo)", type="pdf")

pergunta_auditoria = st.text_input("❓ O que você quer auditar?", placeholder="Ex: Qual o foro eleito? / Qual a multa de atraso?")

if st.button("🚀 Executar Auditoria Híbrida", type="primary"):
    if arquivo_a and arquivo_b and pergunta_auditoria:
        with st.spinner("Limpando ambiente..."): limpar_banco_vetorial()
        
        with st.spinner("Indexando PDFs (Vetor + SQLite)..."):
            processar_pdf(arquivo_a)
            processar_pdf(arquivo_b)
            
        with st.spinner("🔍 RAG Híbrido: Cruzando Vetores com Busca Léxica..."):
            contexto_a, filhos_a = recuperar_contexto_filtrado(pergunta_auditoria, arquivo_a.name)
            contexto_b, filhos_b = recuperar_contexto_filtrado(pergunta_auditoria, arquivo_b.name)
            
        with st.spinner("🤖 Extraindo Respostas (M2)..."):
            extracao_a = extrair_dado_com_ia(contexto_a, pergunta_auditoria)
            extracao_b = extrair_dado_com_ia(contexto_b, pergunta_auditoria)
            
        with st.spinner("⚖️ Julgamento Neural (M4)..."):
            veredito, motivos = comparar_documentos(extracao_a, extracao_b, pergunta_auditoria)
            
        with st.spinner("📝 Sintetizando Parecer..."):
            parecer = gerar_parecer(extracao_a, extracao_b, veredito, pergunta_auditoria)

        st.divider()
        st.header("🤖 Veredicto do Juiz M4")
        if veredito == "CONSENSO TOTAL":
            st.markdown(f'<div class="card-resultado card-sucesso"><h3>✅ {veredito}</h3><p>{motivos[0]}</p></div>', unsafe_allow_html=True)
        elif veredito == "LACUNA DE EVIDÊNCIA":
            st.markdown(f'<div class="card-resultado card-aviso"><h3>⚠️ {veredito}</h3><p>{motivos[0]}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card-resultado card-erro"><h3>⚔️ {veredito}</h3><ul>{"".join(f"<li>{m}</li>" for m in motivos)}</ul></div>', unsafe_allow_html=True)

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"""
            <div class="card-resultado">
                <h4>📄 {arquivo_a.name}</h4>
                <p><strong>Resposta:</strong> {extracao_a.resposta_direta}</p>
                <p><strong>Condição:</strong> {extracao_a.condicionantes}</p>
                <p><strong>Tipo:</strong> {extracao_a.tipo_dado}</p>
                <p><small>"{extracao_a.trecho_literal}"</small></p>
            </div>
            """, unsafe_allow_html=True)
        with col_res2:
            st.markdown(f"""
            <div class="card-resultado">
                <h4>📄 {arquivo_b.name}</h4>
                <p><strong>Resposta:</strong> {extracao_b.resposta_direta}</p>
                <p><strong>Condição:</strong> {extracao_b.condicionantes}</p>
                <p><strong>Tipo:</strong> {extracao_b.tipo_dado}</p>
                <p><small>"{extracao_b.trecho_literal}"</small></p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("📝 Parecer Executivo")
        st.markdown(f'<div class="card-resultado card-parecer"><p>{parecer}</p></div>', unsafe_allow_html=True)

        with st.expander("👁️ Preview Híbrido (Vetor + Léxico)", expanded=False):
            st.markdown("### 📄 Documento A")
            for i, filho in enumerate(filhos_a):
                st.markdown(f"**Chunk {i+1}** (score: `{filho['score']:.3f}`)")
                st.text(filho['texto'])
            st.divider()
            st.markdown("### 📄 Documento B")
            for i, filho in enumerate(filhos_b):
                st.markdown(f"**Chunk {i+1}** (score: `{filho['score']:.3f}`)")
                st.text(filho['texto'])

        with st.expander("🕵️‍♂️ Caixa de Vidro (Logs Universais)", expanded=True):
            json_prov = {
                "M1.5_Gatekeeper": st.session_state.get('m1_5_ledger', {}),
                "M2_Extrator": {"doc_A": extracao_a.model_dump(), "doc_B": extracao_b.model_dump()},
                "M4_Juiz_Neural": {"veredito": veredito, "motivos": motivos}
            }
            st.json(json_prov)
            salvar_no_ledger_local(pergunta_auditoria, arquivo_a.name, arquivo_b.name, json_prov)
    else:
        st.warning("Insira PDFs e uma pergunta válida.")