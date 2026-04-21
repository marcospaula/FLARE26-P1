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
# GESTÃO DE ESTADO (ISOLAMENTO DO FRONT-END)
# ==========================================
if 'pdfs_processados' not in st.session_state:
    st.session_state['pdfs_processados'] = set()
if 'm1_5_ledger' not in st.session_state: 
    st.session_state['m1_5_ledger'] = {}

def gerar_hash_arquivo(arquivo_pdf):
    """Gera um MD5 rápido para identificar univocamente o PDF e evitar reprocessamento."""
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
# SCHEMA AGNÓSTICO (PYDANTIC) - ONTLOGICAL + EXCLUSION CHECKING
# ==========================================
class ExtracaoUniversal(BaseModel):
    natureza_da_pergunta: str = Field(description="O domínio de dados exigido (ex: Temporal, Monetário).")
    natureza_do_texto_encontrado: str = Field(description="O domínio de dados encontrado.")
    houve_compatibilidade_ontologica: bool = Field(description="True se a natureza encontrada corresponde à exigida.")
    violou_restricao_do_usuario: bool = Field(
        description="CRÍTICO: Retorne True SE a pergunta exigir explicitamente que um tipo de dado seja ignorado (ex: 'ignorando prazo de execução') E o texto encontrado tratar exatamente desse assunto proibido. Caso contrário, retorne False."
    )
    resposta_direta: str = Field(description="A resposta exata. Retorne 'NÃO LOCALIZADO' se houve_compatibilidade_ontologica for False OU se violou_restricao_do_usuario for True.")
    tipo_dado: str = Field(description="Categoria do dado. Se incompatível/violado, retorne 'NÃO LOCALIZADO'.")
    condicionantes: str = Field(description="Regras da resposta. Se incompatível/violado, retorne 'NÃO LOCALIZADO'.")
    trecho_literal: str = Field(description="Cópia literal do texto. Se incompatível/violado, retorne 'LACUNA DE EVIDÊNCIA'.")
    confiabilidade: float = Field(description="Score de 0.0 a 1.0. Se incompatível/violado, OBRIGATORIAMENTE 0.0.")

def resultado_vazio():
    return ExtracaoUniversal(
        natureza_da_pergunta="N/A",
        natureza_do_texto_encontrado="N/A",
        houve_compatibilidade_ontologica=False,
        violou_restricao_do_usuario=False,
        resposta_direta="NÃO LOCALIZADO",
        tipo_dado="NÃO LOCALIZADO",
        condicionantes="NÃO LOCALIZADO",
        trecho_literal="LACUNA DE EVIDÊNCIA",
        confiabilidade=0.0
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
    """Limpa absolutamente tudo quando o usuário troca de arquivos."""
    global vector_store
    try:
        vector_store.delete_collection()
    except Exception:
        pass
        
    try:
        limpar_banco_sqlite()
        st.session_state['pdfs_processados'].clear()
        st.session_state['m1_5_ledger'].clear()
        
        st.cache_resource.clear() 
        
        vector_store, _ = iniciar_banco_vetorial()
        gc.collect()
    except Exception as e:
        pass

def validar_tamanho_pdf(arquivo_pdf):
    return (arquivo_pdf.size / (1024 * 1024)) <= MAX_FILE_SIZE_MB

def extrair_texto_fitz_rapido(tmp_path):
    """Extração ultrarrápida usando PyMuPDF (fitz)."""
    doc = fitz.open(tmp_path)
    if doc.page_count > MAX_PAGES: 
        doc.close()
        raise ValueError(f"Limite de {MAX_PAGES} páginas excedido.")
    
    textos_paginas = []
    for page in doc:
        texto = page.get_text()
        textos_paginas.append(sanitizar_texto_pdf(texto))
    doc.close()
    return textos_paginas

def processar_pdf_idempotente(arquivo_pdf):
    if not validar_tamanho_pdf(arquivo_pdf): raise ValueError(f"PDF {arquivo_pdf.name} muito grande.")
    
    file_hash = gerar_hash_arquivo(arquivo_pdf)
    
    if file_hash in st.session_state['pdfs_processados']:
        return {"sucesso": True, "status": "cache"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(arquivo_pdf.getvalue())
        tmp_path = tmp_file.name

    try:
        textos_paginas = extrair_texto_fitz_rapido(tmp_path)
        
        pages = [Document(page_content=txt, metadata={"source": arquivo_pdf.name, "file_hash": file_hash}) for txt in textos_paginas]
        
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE_PAI, chunk_overlap=CHUNK_OVERLAP_PAI)
        docs_pai = parent_splitter.split_documents(pages)
        
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE_FILHO, chunk_overlap=CHUNK_OVERLAP_FILHO)
        docs_filho_para_chroma = []
        
        for i, doc_pai in enumerate(docs_pai):
            parent_id = f"{file_hash}_pai_{i}"
            cursor.execute("INSERT OR REPLACE INTO docstore_pai (parent_id, nome_doc, file_hash, conteudo) VALUES (?, ?, ?, ?)",
                           (parent_id, arquivo_pdf.name, file_hash, doc_pai.page_content))
            
            for filho in child_splitter.split_documents([doc_pai]):
                filho.metadata["source"] = arquivo_pdf.name
                filho.metadata["file_hash"] = file_hash
                filho.metadata["parent_id"] = parent_id
                docs_filho_para_chroma.append(filho)

        conn.commit()
        conn.close()

        if docs_filho_para_chroma:
            batch_size_chroma = 500
            for i in range(0, len(docs_filho_para_chroma), batch_size_chroma):
                vector_store.add_documents(docs_filho_para_chroma[i:i + batch_size_chroma])
        
        st.session_state['pdfs_processados'].add(file_hash)
        
        del docs_pai, docs_filho_para_chroma, pages, textos_paginas
        gc.collect()
        return {"sucesso": True, "status": "indexado"}
        
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

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
        st.session_state['m1_5_ledger'][arquivo_pdf.name] = {
            "estrategia": "Leitura Total (Bypass de RAG)",
            "status": "Documento inteiro enviado para o M2",
            "caracteres_lidos": len(texto_completo)
        }
        chunks_exibicao = [{"texto": "Documento pequeno o suficiente para Leitura Total (Full Context). Bypass de vetores acionado.", "score": 1.0}]
        return texto_completo, chunks_exibicao

    stopwords = {"qual", "quais", "como", "quem", "onde", "para", "pelo", "pela", "sobre", "entre", "seja", "esse", "este", "dos", "das", "nas", "nos", "que", "com", "por", "um", "uma"}
    pergunta_norm = pergunta.lower().replace("percentual", "%").replace("porcentagem", "%")
    palavras_limpas = [re.sub(r'[^a-záéíóúçãõâê%]', '', p) for p in pergunta_norm.split()]
    termos_dinamicos = [p for p in palavras_limpas if (len(p) >= 4 or p == "%") and p not in stopwords]

    if not termos_dinamicos: termos_dinamicos = palavras_limpas
    ancora_principal = sorted([t for t in termos_dinamicos if t != "%"], key=len, reverse=True)[0] if [t for t in termos_dinamicos if t != "%"] else ""

    contextos_finais = set()
    chunks_exibicao = []

    res_vetor = vector_store.similarity_search_with_score(pergunta, k=TOP_K_RETRIEVAL, filter={"file_hash": file_hash})
    if res_vetor:
        scores = [1.0 - dist for _, dist in res_vetor]
        limite_corte = max(0.25, max(scores) * 0.70)
        
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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

    contextos_lexicos = []
    for row in rows: 
        conteudo_pai = row[0]
        conteudo_lower = conteudo_pai.lower()
        score_lex = sum((10 if t == "%" else len(t)) for t in termos_dinamicos if t in conteudo_lower)
        if ancora_principal and ancora_principal in conteudo_lower: score_lex += 15 
        if ancora_principal in conteudo_lower and re.search(r'\d+[,\.]\d+|\d+%', conteudo_lower): score_lex += 30 
        
        if score_lex >= 15: contextos_lexicos.append((score_lex, conteudo_pai))
    
    contextos_lexicos.sort(key=lambda x: x[0], reverse=True)
    
    for score, conteudo in contextos_lexicos[:6]:
        if conteudo not in contextos_finais:
            contextos_finais.add(conteudo)
            chunks_exibicao.append({"texto": conteudo[:400] + "... [LÉXICO]", "score": min(0.99, score/100.0)})

    st.session_state['m1_5_ledger'][arquivo_pdf.name] = {"estrategia": "Parent-Child RAG", "total_blocos": len(contextos_finais)}

    return "\n\n--- [BLOCO PAI] ---\n\n".join(list(contextos_finais)[:8]), chunks_exibicao

# ==========================================
# EXTRADADOS M2 E M4 (ATUALIZADOS COM ONTOLOGIA)
# ==========================================
def extrair_dado_com_ia(texto, pergunta):
    if not texto.strip(): return resultado_vazio()

    prompt = (
        "Você é um Extrator Forense Especialista em Editais Governamentais.\n"
        "Sua missão é extrair dados aplicando Validação Ontológica e Verificação de Exclusões explícitas da pergunta.\n\n"
        f"PERGUNTA DA AUDITORIA: {pergunta}\n\n"
        "REGRAS DE EXTRAÇÃO (LEIA COM ATENÇÃO):\n"
        "1. Se o usuário exigir que uma informação seja ignorada (ex: 'ignorando obras', 'excluindo multas') e o texto SÓ tiver essa informação proibida, você deve definir 'violou_restricao_do_usuario' como True.\n"
        "2. Se 'houve_compatibilidade_ontologica' for False OU 'violou_restricao_do_usuario' for True, a 'resposta_direta' deve ser obrigatoriamente 'NÃO LOCALIZADO'.\n"
        "3. Nunca entregue ao usuário o que ele mandou você ignorar.\n\n"
        "TEXTO DO DOCUMENTO PARA ANÁLISE:\n"
        f"{texto[:90000]}" 
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um auditor forense que aplica validação de domínio e respeita estritamente ordens de exclusão negativa da pergunta."},
                {"role": "user", "content": prompt}
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "retornar_extracao",
                    "description": "Retorna os dados extraídos conforme o schema validado.",
                    "parameters": ExtracaoUniversal.model_json_schema()
                }
            }],
            tool_choice={"type": "function", "function": {"name": "retornar_extracao"}},
            temperature=0.0 
        )
        
        tool_call = response.choices[0].message.tool_calls[0]
        dados = json.loads(tool_call.function.arguments)
        
        # O BLOQUEIO MATEMÁTICO ABSOLUTO:
        incompativel = not dados.get("houve_compatibilidade_ontologica", False)
        violou_restricao = dados.get("violou_restricao_do_usuario", False)
        
        if incompativel or violou_restricao:
            return resultado_vazio()

        resp = str(dados.get("resposta_direta", "NÃO LOCALIZADO")).strip()
        conf = 0.95 if resp.upper() != "NÃO LOCALIZADO" else 0.0
        
        return ExtracaoUniversal(
            natureza_da_pergunta=str(dados.get("natureza_da_pergunta", "N/A")),
            natureza_do_texto_encontrado=str(dados.get("natureza_do_texto_encontrado", "N/A")),
            houve_compatibilidade_ontologica=not incompativel,
            violou_restricao_do_usuario=violou_restricao,
            resposta_direta=resp,
            tipo_dado=str(dados.get("tipo_dado", "NÃO LOCALIZADO")),
            condicionantes=str(dados.get("condicionantes", "NÃO LOCALIZADO")),
            trecho_literal=str(dados.get("trecho_literal", "LACUNA DE EVIDÊNCIA")),
            confiabilidade=conf
        )
    except Exception:
        return resultado_vazio()

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
with col1: arquivo_a = st.file_uploader("📄 Documento A (Referência)", type="pdf", key="file_a")
with col2: arquivo_b = st.file_uploader("📄 Documento B (Comparativo)", type="pdf", key="file_b")

if st.button("🗑️ Limpar Banco e Cache (Trocar Arquivos)", type="secondary"):
    resetar_ambiente_total()
    st.success("Ambiente resetado. Pode subir novos arquivos.")
    st.rerun()

pergunta_auditoria = st.text_input("❓ O que você quer auditar?", placeholder="Ex: Qual o foro eleito? / Qual a multa de atraso?")

if st.button("🚀 Executar Auditoria Híbrida", type="primary"):
    if arquivo_a and arquivo_b and pergunta_auditoria:
        start_time = time.time()  
        
        with st.spinner("Verificando Indexação (Idempotência)..."):
            status_a = processar_pdf_idempotente(arquivo_a)
            status_b = processar_pdf_idempotente(arquivo_b)
            if status_a["status"] == "cache" and status_b["status"] == "cache":
                st.toast("⚡ Leitura Pula Etapa Vetorial! Usando Cache em Disco.", icon="⚡")
            
        with st.spinner("🔍 RAG Híbrido: Cruzando Vetores com Busca Léxica..."):
            contexto_a, filhos_a = recuperar_contexto_filtrado(pergunta_auditoria, arquivo_a)
            contexto_b, filhos_b = recuperar_contexto_filtrado(pergunta_auditoria, arquivo_b)
            
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

        end_time = time.time()
        st.success(f"⏱️ Auditoria concluída em {end_time - start_time:.2f} segundos.")

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

        with st.expander("🕵️‍♂️ Caixa de Vidro (Logs Universais)", expanded=False):
            json_prov = {
                "M1.5_Gatekeeper": st.session_state.get('m1_5_ledger', {}),
                "M2_Extrator": {"doc_A": extracao_a.model_dump(), "doc_B": extracao_b.model_dump()},
                "M4_Juiz_Neural": {"veredito": veredito, "motivos": motivos}
            }
            st.json(json_prov)
            salvar_no_ledger_local(pergunta_auditoria, arquivo_a.name, arquivo_b.name, json_prov)
    else:
        st.warning("Insira PDFs e uma pergunta válida.")