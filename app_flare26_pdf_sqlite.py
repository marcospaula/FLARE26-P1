import streamlit as st
import tempfile
import os
import sqlite3     # MOTOR SQLITE (PURGATÓRIO DE MEMÓRIA)
import gc          # GARBAGE COLLECTOR FORÇADO
from openai import OpenAI
import json
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
MODEL_GERACAO = "qwen2.5:1.5b"  # M2 e M5 local (fallback) - atualmente usando gpt-4o-mini
MODEL_EMBEDDING = "all-MiniLM-L6-v2"  # M1

# Arquivo físico do Ledger
LEDGER_FILE = "flare26_ledger_v2.json"

# ==========================================
# CONFIGURAÇÕES PARA PDFs PESADOS
# ==========================================
MAX_FILE_SIZE_MB = 10  
MAX_PAGES = 500        
CHUNK_SIZE_PAI = 2000  # Reduzido para padrão indústria
CHUNK_OVERLAP_PAI = 150
CHUNK_SIZE_FILHO = 400
CHUNK_OVERLAP_FILHO = 50
TOP_K_RETRIEVAL = 10   # Suficiente

# ==========================================
# MOTOR SQLITE (PURGATÓRIO DE MEMÓRIA)
# ==========================================
DB_PATH = "flare26_cache_docs.db"

def iniciar_banco_sqlite():
    """Cria a tabela de cache no disco para evitar Out Of Memory"""
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
    """Zera o cache no disco antes de uma nova auditoria"""
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except PermissionError:
            pass
    iniciar_banco_sqlite()

# Garante que o banco exista ao iniciar o app
iniciar_banco_sqlite()

# ==========================================
# SANITIZADOR DE OCR (BLINDAGEM LEXICAL)
# ==========================================
def sanitizar_texto_pdf(texto_bruto):
    """
    Remove ruídos de OCR que destroem a matemática vetorial.
    Corrige palavras quebradas, múltiplos espaços e caracteres nulos.
    """
    if not texto_bruto:
        return ""
    
    # 1. Remove caracteres nulos e lixo binário
    texto = texto_bruto.replace('\x00', '')
    
    # 2. Transforma quebras de linha múltiplas em uma só (Preserva parágrafos)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    
    # 3. Corrige o pior defeito de OCR: Letras s o l t a s (ex: "m u l t a")
    texto = re.sub(r'(?<=\b[a-zA-Z])\s(?=[a-zA-Z]\b)', '', texto)
    
    # 4. Remove tabs e espaços múltiplos dentro de uma frase
    texto = re.sub(r'[ \t]+', ' ', texto)
    
    return texto.strip()

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
# SALVAMENTO DO LEDGER LOCAL
# ==========================================
def salvar_no_ledger_local(pergunta, doc_a_name, doc_b_name, json_data):
    ledger_history = []
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, 'r', encoding='utf-8') as f:
                ledger_history = json.load(f)
        except json.JSONDecodeError:
            pass 

    novo_registro = {
        "id_teste": f"T_{len(ledger_history) + 1}",
        "timestamp": datetime.now().isoformat(),
        "pergunta_auditada": pergunta,
        "documentos": {"doc_A": doc_a_name, "doc_B": doc_b_name},
        "caixa_de_vidro": json_data
    }
    
    ledger_history.append(novo_registro)
    with open(LEDGER_FILE, 'w', encoding='utf-8') as f:
        json.dump(ledger_history, f, indent=4, ensure_ascii=False)

# ==========================================
# MOTOR VETORIAL E RAG PAI/FILHO NATIVO (M1)
# ==========================================
@st.cache_resource
def iniciar_banco_vetorial():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_EMBEDDING,
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = Chroma(
        collection_name="flare26_docs",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vector_store, embeddings

vector_store, _ = iniciar_banco_vetorial()

def limpar_banco_vetorial():
    """Limpa ChromaDB + SQLite + força garbage collection"""
    try:
        vector_store.delete_collection()
        # Recria o banco vetorial
        global vector_store
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_EMBEDDING,
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = Chroma(
            collection_name="flare26_docs",
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        # Limpa SQLite
        limpar_banco_sqlite()
        st.session_state['m1_5_ledger'] = {}
        # Força coleta de lixo
        gc.collect()
        return True
    except Exception as e:
        print(f"Erro ao limpar banco: {e}")
        return False

def validar_tamanho_pdf(arquivo_pdf):
    tamanho_mb = arquivo_pdf.size / (1024 * 1024)
    if tamanho_mb > MAX_FILE_SIZE_MB:
        return False, f"Arquivo muito grande ({tamanho_mb:.1f}MB)."
    return True, ""

def processar_pdf(arquivo_pdf):
    """Processa PDF com SQLite (disco) + gc.collect() (memória)"""
    valido, msg = validar_tamanho_pdf(arquivo_pdf)
    if not valido:
        raise ValueError(msg)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(arquivo_pdf.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        total_pages = len(loader.load())
        
        if total_pages > MAX_PAGES:
            raise ValueError(f"PDF muito grande ({total_pages} páginas).")
        
        batch_size = 50
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()[batch_start:batch_end]
            
            # Aplica o Sanitizador ANTES de criar os blocos
            for p in pages:
                p.page_content = sanitizar_texto_pdf(p.page_content)
            
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE_PAI, 
                chunk_overlap=CHUNK_OVERLAP_PAI
            )
            docs_pai = parent_splitter.split_documents(pages)
            
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE_FILHO, 
                chunk_overlap=CHUNK_OVERLAP_FILHO
            )
            
            docs_filho_para_chroma = []
            for i, doc_pai in enumerate(docs_pai):
                parent_id = f"{arquivo_pdf.name}_pai_{batch_start}_{i}"
                
                # Salva o bloco GIGANTE no DISCO (SQLite), não mais na RAM!
                cursor.execute(
                    "INSERT OR REPLACE INTO docstore_pai (parent_id, nome_doc, conteudo) VALUES (?, ?, ?)",
                    (parent_id, arquivo_pdf.name, doc_pai.page_content)
                )
                
                # Corta o filho e anexa o ID da tabela
                docs_filhos = child_splitter.split_documents([doc_pai])
                for filho in docs_filhos:
                    filho.metadata["source"] = arquivo_pdf.name
                    filho.metadata["parent_id"] = parent_id
                    docs_filho_para_chroma.append(filho)

            # Grava no disco
            conn.commit()

            # Indexa os pedaços pequenos na RAM (ChromaDB)
            if docs_filho_para_chroma:
                chroma_batch_size = 500
                for idx in range(0, len(docs_filho_para_chroma), chroma_batch_size):
                    batch = docs_filho_para_chroma[idx:idx + chroma_batch_size]
                    vector_store.add_documents(batch)
            
            # OOM KILLER PREVENTION: Destrói variáveis gigantes do lote
            del docs_pai
            del docs_filhos
            del docs_filho_para_chroma
            del pages
            gc.collect()

        conn.close()
        return {"sucesso": True}
        
    except Exception as e:
        raise RuntimeError(f"Erro ao processar PDF: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ==========================================
# MÓDULO M1.5: FILTRO VETORIAL ADAPTATIVO
# ==========================================
def ler_docstore_pai_sqlite(nome_doc=None):
    """Lê os blocos pais do SQLite (disco) em vez da RAM"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if nome_doc:
        cursor.execute("SELECT parent_id, conteudo FROM docstore_pai WHERE nome_doc = ?", (nome_doc,))
    else:
        cursor.execute("SELECT parent_id, conteudo FROM docstore_pai")
    resultados = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return resultados

def recuperar_contexto_filtrado(pergunta, nome_documento, top_k=None):
    if top_k is None:
        top_k = TOP_K_RETRIEVAL
    
    # SEGREGAÇÃO DE GATILHOS (Gatekeeper)
    termos_comuns = ["contrato", "objeto", "pagamento", "valor", "percentual", "%"]
    termos_criticos = ["multa", "penalidade", "sanção", "inexecução", "atraso", "infração", "descumprimento", "violação", "mora", "compensatória"]
    
    def executar_busca(query):
        resultados_brutos = vector_store.similarity_search_with_score(
            query, k=top_k, filter={"source": nome_documento}
        )
        return [(doc, max(0.0, 1.0 - dist)) for doc, dist in resultados_brutos] if resultados_brutos else []
    
    def verificar_relevancia(contextos):
        if not contextos: return False
        texto_total = " ".join(contextos).lower()
        return any(critico in texto_total for critico in termos_criticos)
    
    def filtrar_resultados(resultados):
        if not resultados: return [], 0.0, 0.0
        scores = [s for _, s in resultados]
        melhor_score = max(scores)
        limite_corte = max(0.25, melhor_score * 0.70)
        
        contextos_validos = []
        maior_score = 0.0
        parent_ids_vistos = set()
        
        for doc, score in resultados:
            if score >= limite_corte:
                if score > maior_score: maior_score = score
                parent_id = doc.metadata.get("parent_id")
                if parent_id and parent_id not in parent_ids_vistos:
                    # Lê do SQLite (disco) em vez da RAM
                    docstore = ler_docstore_pai_sqlite(nome_documento)
                    conteudo_pai = docstore.get(parent_id)
                    if conteudo_pai:
                        contextos_validos.append(conteudo_pai)
                        parent_ids_vistos.add(parent_id)
        
        return contextos_validos, maior_score, limite_corte
    
    def busca_textual_fallback_estruturada(pergunta, nome_doc):
        pergunta_lower = pergunta.lower()
        termos_busca = [p for p in pergunta_lower.split() if len(p) > 3]
        
        # Hiper-gatilhos simplificados (não mais hardcoded)
        hiper_gatilhos = ["inexecução", "multa", "penalidade", "compensatória"]
        
        # Lê do SQLite (disco)
        docstore = ler_docstore_pai_sqlite(nome_doc)
        
        contextos_pontuados = []
        
        for parent_id, conteudo in docstore.items():
            conteudo_lower = conteudo.lower()
            score = 0
            for termo in termos_busca:
                if termo in conteudo_lower: score += 1
            for comum in termos_comuns:
                if comum in conteudo_lower: score += 1
            for critico in termos_criticos:
                if critico in conteudo_lower: score += 3  # Simplificado: +3 em vez de +5
            for hiper in hiper_gatilhos:
                if hiper in conteudo_lower: score += 5   # Simplificado: +5 em vez de +15
                
            if score >= 3:  # Simplificado: >= 3 em vez de >= 5
                contextos_pontuados.append((score, conteudo))
        
        contextos_pontuados.sort(key=lambda x: x[0], reverse=True)
        return [conteudo for score, conteudo in contextos_pontuados[:4]]

    # PIPELINE DE 3 ETAPAS
    resultados = executar_busca(pergunta)
    contextos_validos, maior_score, limite_corte = filtrar_resultados(resultados)
    
    if contextos_validos and not verificar_relevancia(contextos_validos):
        termos_pergunta = [p for p in pergunta.lower().split() if len(p) > 4]
        for termo in termos_pergunta[:5]:
            if termo.lower() not in pergunta.lower()[:30]:
                busca_expandida = f"{pergunta} {termo}"
                resultados_exp = executar_busca(busca_expandida)
                ctx_val, m_score, l_corte = filtrar_resultados(resultados_exp)
                if ctx_val and verificar_relevancia(ctx_val):
                    contextos_validos, maior_score, limite_corte = ctx_val, m_score, l_corte
                    resultados = resultados_exp
                    break
    
    usou_fallback = False
    if not contextos_validos or not verificar_relevancia(contextos_validos):
        contextos_textuais = busca_textual_fallback_estruturada(pergunta, nome_documento)
        if contextos_textuais:
            contextos_validos = contextos_textuais
            maior_score = 0.99
            limite_corte = 0.0
            usou_fallback = True
            resultados = [] # Limpa ruído visual na UI

    if 'm1_5_ledger' not in st.session_state:
        st.session_state['m1_5_ledger'] = {}
    
    st.session_state['m1_5_ledger'][nome_documento] = {
        "limite_corte_calculado": round(limite_corte, 3),
        "maior_score_encontrado": round(maior_score, 3),
        "status": "Aprovado (Fallback Estruturado)" if usou_fallback else ("Aprovado" if contextos_validos else "Reprovado")
    }

    contexto_pai = "\n\n--- [NOVO BLOCO DE CONTEXTO - PAI] ---\n\n".join(contextos_validos)
    
    if usou_fallback:
        filhos_filtrados = [{"texto": c[:400] + "... [Bloco Fallback BM25]", "score": 0.99} for c in contextos_validos]
    else:
        filhos_filtrados = [
            {"texto": doc.page_content, "score": score}
            for doc, score in resultados
            if score >= limite_corte
        ]
    
    return contexto_pai, filhos_filtrados

# ==========================================
# EXTRATOR NEURAL (M2) - HÍBRIDO E BLINDADO
# ==========================================
def extrair_dado_com_ia(texto, pergunta):
    if not texto.strip():
        return resultado_vazio()

    # Engenharia de Prompt Blindada contra Alucinação Monetária
    prompt = f"""
Sua tarefa é atuar como um Extrator Forense de Dados Universal.
Você deve analisar o TEXTO abaixo e extrair os dados solicitados na PERGUNTA, retornando APENAS um JSON válido.

PERGUNTA DA AUDITORIA: {pergunta}

REGRAS ESTRITAS DE PREENCHIMENTO DO JSON:
1. LEIA ATENTAMENTE o texto. Cuidado para não confundir percentuais (%) com Reais (R$).
2. "valor_formatado": O valor exato e unidade (Ex: "20%", "R$ 5.000,00", "0,5% a 1%"). NÃO invente "R$" se o texto usar apenas percentual "%".
3. "valor_numerico": OBRIGATÓRIO isolar APENAS os algarismos (Ex: "20", "5000.00", "0.5 a 1").
4. "unidade_ou_moeda": Apenas a métrica (Ex: "%", "R$").
5. "condicao_ou_prazo": A condição exata, como "no caso de inexecução total".
6. "contexto_da_clausula": A cópia LITERAL do parágrafo que prova a sua extração.
7. Se a informação não existir, preencha tudo com "NÃO LOCALIZADO" e confiabilidade 0.0.

TEXTO PARA AUDITORIA:
{texto[:8000]}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Você é um auditor sênior e matemático. Nunca misture unidades. Retorne apenas o JSON estruturado."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            seed=4242
        )
        
        data = json.loads(response.choices[0].message.content)
        
        return ExtracaoMultivariavel(
            valor_formatado=str(data.get("valor_formatado", "NÃO LOCALIZADO")),
            valor_numerico=str(data.get("valor_numerico", "NÃO LOCALIZADO")),
            unidade_ou_moeda=str(data.get("unidade_ou_moeda", "NÃO LOCALIZADO")),
            condicao_ou_prazo=str(data.get("condicao_ou_prazo", "NÃO LOCALIZADO")),
            contexto_da_clausula=str(data.get("contexto_da_clausula", "LACUNA DE EVIDÊNCIA")),
            confiabilidade=float(data.get("confiabilidade", 0.0))
        )
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
Você é o M5 Sintetizador. Escreva um "Parecer Executivo" imparcial (máximo 4 frases) das conclusões da auditoria.

Pergunta: {pergunta}
Veredito (M4): {veredito}
Motivos: {', '.join(motivos)}

Documento X ({doc_A}): {extracao_A.valor_formatado} | {extracao_A.condicao_ou_prazo}
Documento Y ({doc_B}): {extracao_B.valor_formatado} | {extracao_B.condicao_ou_prazo}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um auditor sênior redigindo pareceres factuais."},
                {"role": "user", "content": prompt_sintese}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "O Sintetizador M5 falhou em gerar o parecer executivo."

# ==========================================
# INTERFACE DO USUÁRIO (STREAMLIT)
# ==========================================
st.markdown("""
<style>
    .card-resultado { padding: 20px; border-radius: 10px; margin: 10px 0; }
    .card-sucesso { background-color: #1a3d2e; border: 1px solid #2ecc71; color: #a8e6cf; }
    .card-aviso { background-color: #3d3520; border: 1px solid #f1c40f; color: #f9e79f; }
    .card-erro { background-color: #3d2020; border: 1px solid #e74c3c; color: #f5b7b1; }
    .card-doc-a { background-color: #1a2a3d; border-left: 4px solid #3498db; color: #aed6f1; }
    .card-doc-b { background-color: #3d2e1a; border-left: 4px solid #e67e22; color: #f5cba7; }
    .card-parecer { background-color: #2d2d2d; border-left: 4px solid #9b59b6; color: #d2b4de; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 FLARE26: RAG Auditor")
st.markdown("**Caixa de Vidro Corporativa** — Auditoria Neuro-Simbólica com Filtro Vetorial Inteligente")
st.divider()

col1, col2 = st.columns(2)

with col1:
    arquivo_a = st.file_uploader("Documento A", type=["pdf"], key="file_a")
with col2:
    arquivo_b = st.file_uploader("Documento B", type=["pdf"], key="file_b")

if arquivo_a:
    valido, msg = validar_tamanho_pdf(arquivo_a)
    if not valido: st.error(f"Doc A: {msg}")
if arquivo_b:
    valido, msg = validar_tamanho_pdf(arquivo_b)
    if not valido: st.error(f"Doc B: {msg}")

pergunta_auditoria = st.text_input("O que você quer auditar?", placeholder="Ex: Qual o valor da multa por inexecução total do objeto?")

if st.button("Executar Auditoria Forense", type="primary"):
    if arquivo_a and arquivo_b and pergunta_auditoria:
        
        with st.spinner("Limpando banco vetorial..."):
            limpar_banco_vetorial()
        
        st.session_state['m1_5_ledger'] = {}
        progress_container = st.container()
        placeholder_etapas = progress_container.empty()
        
        with placeholder_etapas.container():
            st.markdown("### 🔄 Pipeline de Auditoria Neuro-Simbólica")
            progress_bar_1 = st.progress(0, text="Etapa 1/5: Indexando PDFs...")
            processar_pdf(arquivo_a)
            progress_bar_1.progress(50)
            processar_pdf(arquivo_b)
            progress_bar_1.progress(100)
            
            progress_bar_2 = st.progress(0, text="Etapa 2/5: Filtro Vetorial (M1.5)...")
            contexto_a, filhos_a = recuperar_contexto_filtrado(pergunta_auditoria, arquivo_a.name)
            progress_bar_2.progress(50)
            contexto_b, filhos_b = recuperar_contexto_filtrado(pergunta_auditoria, arquivo_b.name)
            progress_bar_2.progress(100)
            
            progress_bar_3 = st.progress(0, text="Etapa 3/5: Extração (M2)...")
            extracao_a = extrair_dado_com_ia(contexto_a, pergunta_auditoria)
            progress_bar_3.progress(50)
            extracao_b = extrair_dado_com_ia(contexto_b, pergunta_auditoria)
            progress_bar_3.progress(100)
            
            progress_bar_4 = st.progress(0, text="Etapa 4/5: Juiz M4...")
            veredito, motivos = comparar_documentos(extracao_a, extracao_b)
            progress_bar_4.progress(100)
            
            progress_bar_5 = st.progress(0, text="Etapa 5/5: Parecer M5...")
            parecer = gerar_parecer_executivo(pergunta_auditoria, arquivo_a.name, extracao_a, arquivo_b.name, extracao_b, veredito, motivos)
            progress_bar_5.progress(100)

        placeholder_etapas.empty()
        st.success("✅ Auditoria Concluída!")

        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        with col_met1: st.metric("📄 Docs Processados", "2")
        with col_met2: st.metric("🎯 Conf. Média", f"{(extracao_a.confiabilidade + extracao_b.confiabilidade) / 2:.0%}")
        with col_met3: st.metric("🔍 Chunks A", len(filhos_a))
        with col_met4: st.metric("🔍 Chunks B", len(filhos_b))
        st.divider()

        with st.expander("👁️ Preview do Contexto Recuperado (Chunks Filtrados)", expanded=False):
            st.markdown("### 📄 Documento A")
            for i, filho in enumerate(filhos_a):
                st.markdown(f"**Chunk {i+1}** (score: `{filho['score']:.3f}`)")
                st.text(filho['texto'])
                st.divider()
            
            st.markdown("### 📄 Documento B")
            for i, filho in enumerate(filhos_b):
                st.markdown(f"**Chunk {i+1}** (score: `{filho['score']:.3f}`)")
                st.text(filho['texto'])
                st.divider()

        st.header("🤖 Diagnóstico Forense M4")
        if veredito == "CONSENSO TOTAL":
            st.markdown(f'<div class="card-resultado card-sucesso"><h3>✅ {veredito}</h3><p>{motivos[0]}</p></div>', unsafe_allow_html=True)
        elif veredito == "LACUNA DE EVIDÊNCIA":
            st.markdown(f'<div class="card-resultado card-aviso"><h3>⚠️ {veredito}</h3><p>{motivos[0]}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card-resultado card-erro"><h3>⚔️ {veredito}</h3><ul>{"".join(f"<li>{m}</li>" for m in motivos)}</ul></div>', unsafe_allow_html=True)

        st.divider()
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"""
            <div class="card-resultado card-doc-a">
                <h4>📄 {arquivo_a.name}</h4>
                <p><strong>Valor:</strong> {extracao_a.valor_formatado}</p>
                <p><strong>Condição:</strong> {extracao_a.condicao_ou_prazo}</p>
                <p><em>"{extracao_a.contexto_da_clausula}"</em></p>
            </div>
            """, unsafe_allow_html=True)
        with col_res2:
            st.markdown(f"""
            <div class="card-resultado card-doc-b">
                <h4>📄 {arquivo_b.name}</h4>
                <p><strong>Valor:</strong> {extracao_b.valor_formatado}</p>
                <p><strong>Condição:</strong> {extracao_b.condicao_ou_prazo}</p>
                <p><em>"{extracao_b.contexto_da_clausula}"</em></p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.subheader("📝 Parecer Executivo (M5)")
        st.markdown(f'<div class="card-resultado card-parecer"><p>{parecer}</p></div>', unsafe_allow_html=True)

        with st.expander("🕵️‍♂️ Trilha de Auditoria Forense (Caixa de Vidro)", expanded=True):
            st.markdown("### 1. Filtro Vetorial Adaptativo Híbrido (M1.5)")
            for doc_name, dados_filtro in st.session_state.get('m1_5_ledger', {}).items():
                st.write(f"**{doc_name}** | Limite: `{dados_filtro['limite_corte_calculado']}` | Score: `{dados_filtro['maior_score_encontrado']}` | Status: {dados_filtro['status']}")

            json_prov = {
                "M1.5_Filtro_Adaptativo": st.session_state.get('m1_5_ledger', {}),
                "M2_Extrator": {"doc_A": extracao_a.model_dump(), "doc_B": extracao_b.model_dump()},
                "M4_Juiz": {"diagnostico": veredito, "motivos": motivos}
            }
            st.json(json_prov)
            salvar_no_ledger_local(pergunta_auditoria, arquivo_a.name, arquivo_b.name, json_prov)
    else:
        st.warning("Faça o upload dos documentos e insira a pergunta.")