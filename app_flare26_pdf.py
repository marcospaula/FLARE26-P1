import streamlit as st
import os
import gc
import time
import json
import threading
from datetime import datetime
from streamlit.runtime.scriptrunner import add_script_run_ctx

import flare26_core as core          # Núcleo determinístico testado (números BR, juízes)
import flare26_pipeline as pipeline  # RAG headless (M1, M1.5, M2) — sem Streamlit

# ==========================================
# CONFIGURAÇÕES E INICIALIZAÇÃO
# ==========================================
st.set_page_config(
    page_title="FLARE26: Auditor Universal",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)
client = pipeline.criar_client_openai(st.secrets["OPENAI_API_KEY"])

# Constantes e schema: fonte única de verdade no pipeline.
MODEL_EMBEDDING = pipeline.MODEL_EMBEDDING
MAX_FILE_SIZE_MB = pipeline.MAX_FILE_SIZE_MB
ExtracaoUniversal = pipeline.ExtracaoUniversal
resultado_vazio = pipeline.resultado_vazio

LEDGER_FILE = "flare26_ledger_v2.json"
DB_PATH = "flare26_cache_docs.db"

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
    file_hash = pipeline.gerar_hash_bytes(arquivo_pdf.read())
    arquivo_pdf.seek(0)
    return file_hash

# ==========================================
# MOTOR SQLITE (delega ao pipeline headless)
# ==========================================
def iniciar_banco_sqlite():
    pipeline.iniciar_banco_sqlite(DB_PATH)

def limpar_banco_sqlite():
    if os.path.exists(DB_PATH):
        try: os.remove(DB_PATH)
        except PermissionError: pass
    iniciar_banco_sqlite()

iniciar_banco_sqlite()

# ==========================================
# MOTOR VETORIAL (M1) CACHEADO
# ==========================================
@st.cache_resource
def iniciar_banco_vetorial():
    return pipeline.criar_vector_store(persist_directory="./chroma_db", model_embedding=MODEL_EMBEDDING)

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

def _worker_processamento_pdf(pdf_bytes, arquivo_name, file_hash, session_state_ref):
    """Worker de background (Thread 2): delega o trabalho pesado ao pipeline."""
    try:
        pipeline.processar_pdf_bytes(pdf_bytes, arquivo_name, db_path=DB_PATH, vector_store=vector_store)
        session_state_ref.add(file_hash)
    except Exception as e:
        print(f"Erro fatal na thread do PDF {arquivo_name}: {e}")

def processar_pdf_idempotente(arquivo_pdf):
    """Gerencia a thread de processamento e mantém a UI viva."""
    if not validar_tamanho_pdf(arquivo_pdf): raise ValueError(f"PDF {arquivo_pdf.name} muito grande.")

    file_hash = gerar_hash_arquivo(arquivo_pdf)
    if file_hash in st.session_state['pdfs_processados']:
        return {"sucesso": True, "status": "cache"}

    pdf_bytes = arquivo_pdf.getvalue()

    # Delega a injeção pesada para a Thread (mantém a UI desenhando frames)
    thread = threading.Thread(
        target=_worker_processamento_pdf,
        args=(pdf_bytes, arquivo_pdf.name, file_hash, st.session_state['pdfs_processados'])
    )
    add_script_run_ctx(thread)
    thread.start()

    barra_progresso = st.progress(0, text=f"Indexando {arquivo_pdf.name} em Background...")
    while thread.is_alive():
        for i in range(100):
            if not thread.is_alive(): break
            time.sleep(0.05)
            barra_progresso.progress((i + 1) % 100, text=f"Indexando vetores de {arquivo_pdf.name}...")
    barra_progresso.empty()

    if file_hash not in st.session_state['pdfs_processados']:
        raise RuntimeError(f"Falha ao processar {arquivo_pdf.name}. Verifique os logs de Thread.")
    return {"sucesso": True, "status": "indexado"}

# ==========================================
# MÓDULO M1.5: RAG HÍBRIDO (delega ao pipeline; grava telemetria no session_state)
# ==========================================
def recuperar_contexto_filtrado(pergunta, arquivo_pdf):
    file_hash = gerar_hash_arquivo(arquivo_pdf)
    contexto, chunks, telemetria = pipeline.recuperar_contexto(
        pergunta, file_hash, db_path=DB_PATH, vector_store=vector_store
    )
    st.session_state['m1_5_ledger'][arquivo_pdf.name] = telemetria
    return contexto, chunks
    if len(texto_completo) < 80000: 
        st.session_state['m1_5_ledger'][arquivo_pdf.name] = {"estrategia": "Leitura Total (Bypass)", "caracteres": len(texto_completo)}
        return texto_completo, [{"texto": "Full Context Bypass.", "score": 1.0}]

# ==========================================
# EXTRAÇÃO M2 (delega ao pipeline headless)
# ==========================================
def extrair_dado_com_ia(texto, pergunta):
    return pipeline.extrair_dado(client, texto, pergunta)

# Delegado ao núcleo testado (flare26_core). Mantido como alias por compatibilidade.
limpar_e_extrair_numeros = core.extrair_numeros_br

# NOTA: o juiz par-a-par com fallback neural foi sucedido pelo juiz N-way
# (core.comparar_n_documentos). Próximo passo: fundir grupos textuais
# semanticamente equivalentes via um passe neural opcional (recuperável no git).

def gerar_parecer_n(pergunta, resultado: core.ResultadoConsenso, extracoes: dict):
    """Parecer executivo (M5) sobre a auditoria de N documentos.

    Ancorado APENAS no resultado simbólico do M4 (enclausuramento neural):
    o LLM redige a narrativa, mas os fatos vêm da matriz determinística.
    """
    linhas_grupos = []
    for g in resultado.grupos:
        linhas_grupos.append(f"- Valor '{g.valor}': {', '.join(g.documentos)}")
    if resultado.lacunas:
        linhas_grupos.append(f"- Sem o dado: {', '.join(resultado.lacunas)}")
    contexto = "\n".join(linhas_grupos)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": (
                "Gere um parecer executivo formal (máx. 4 frases) para laudo pericial "
                "sobre a comparação de múltiplos documentos. NÃO invente números; use "
                "apenas os fatos abaixo.\n"
                f"Pergunta: {pergunta}\n"
                f"Veredito: {resultado.veredito}\n"
                f"Agrupamento:\n{contexto}"
            )}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception:
        return resultado.resumo

def gerar_relatorio_markdown_n(pergunta, resultado: core.ResultadoConsenso,
                               extracoes: dict, parecer: str) -> str:
    """Laudo forense estruturado para auditoria de N documentos."""
    data_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    linhas = [
        "# 🏛️ LAUDO DE AUDITORIA FORENSE AUTOMATIZADA",
        "**Projeto:** FLARE26-P1 (Motor Neuro-Simbólico)",
        f"**Data da Emissão:** {data_hora}",
        "",
        "## 1. ESCOPO DA AUDITORIA",
        f"* **Pergunta Alvo:** {pergunta}",
        f"* **Documentos Auditados:** {len(extracoes)}",
        "",
        "## 2. VEREDICTO DO JUIZ M4 (N-WAY)",
        f"**Status:** {resultado.veredito}",
        f"**Síntese:** {resultado.resumo}",
        "",
        "### Agrupamento de Consenso",
    ]
    for i, g in enumerate(resultado.grupos, 1):
        linhas.append(f"* **Grupo {i} — {g.valor}** ({len(g.documentos)} doc): {', '.join(g.documentos)}")
    if resultado.lacunas:
        linhas.append(f"* **Lacuna de Evidência** ({len(resultado.lacunas)} doc): {', '.join(resultado.lacunas)}")

    linhas += ["", "---", "## 3. EXTRAÇÃO DOCUMENTAL (M2)", ""]
    for nome, ext in extracoes.items():
        linhas += [
            f"### 📄 {nome}",
            f"* **Resposta Extraída:** {ext.resposta_direta}",
            f"* **Condicionante:** {ext.condicionantes}",
            f"* **Tipo de Dado:** {ext.tipo_dado}",
            f"> \"{ext.trecho_literal}\"",
            "",
        ]

    linhas += [
        "---",
        "## 4. PARECER EXECUTIVO (M5)",
        parecer,
        "",
        "---",
        "*Laudo gerado por IA Determinística. Verifique as hashes dos arquivos fonte.*",
    ]
    return "\n".join(linhas)

def salvar_no_ledger_local(pergunta, documentos, ledger_json):
    """Salva a auditoria atual no ledger JSON de longo prazo.

    `documentos` é a lista de nomes auditados (N-way).
    """
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
        "documentos_auditados": list(documentos),
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
st.markdown("###### *Auditoria N-way: Ensemble Retrieval (Vetor + Léxico) e Validação Ontológica*")

arquivos = st.file_uploader(
    "📄 Documentos para auditoria (selecione 2 ou mais PDFs)",
    type="pdf", accept_multiple_files=True, key="files_n"
)

pergunta_auditoria = st.text_input(
    "❓ Objeto de Auditoria (Pergunta Forense)",
    placeholder="Ex: Qual o prazo estipulado para pagamento da fatura?"
)

if st.button("🚀 Iniciar Motor Neuro-Simbólico", type="primary", use_container_width=True):
    if arquivos and len(arquivos) >= 2 and pergunta_auditoria:
        start_time = time.time()
        extracoes = {}          # nome_doc -> ExtracaoUniversal
        chunks_por_doc = {}     # nome_doc -> filhos (telemetria)

        # --- TERMINAL DE STATUS (Pipeline M1.5 a M5) ---
        with st.status("⚙️ Executando Pipeline de Auditoria N-way...", expanded=True) as status:
            for i, arq in enumerate(arquivos, 1):
                st.write(f"⏳ [{i}/{len(arquivos)}] Indexando e extraindo de **{arq.name}**...")
                processar_pdf_idempotente(arq)
                contexto, filhos = recuperar_contexto_filtrado(pergunta_auditoria, arq)
                extracoes[arq.name] = extrair_dado_com_ia(contexto, pergunta_auditoria)
                chunks_por_doc[arq.name] = filhos

            st.write("⚖️ [M4] Juiz N-way: agrupando respostas por consenso determinístico...")
            resultado = core.comparar_n_documentos(
                {nome: ext.resposta_direta for nome, ext in extracoes.items()}
            )

            st.write("📝 [M5] Sintetizando o Parecer Executivo...")
            parecer = gerar_parecer_n(pergunta_auditoria, resultado, extracoes)

            tempo_total = time.time() - start_time
            status.update(label=f"✅ Auditoria de {len(arquivos)} documentos concluída ({tempo_total:.2f}s)",
                          state="complete", expanded=False)

        # Mapa doc -> rótulo do grupo (para a tabela)
        grupo_de = {}
        for idx, g in enumerate(resultado.grupos, 1):
            for d in g.documentos:
                grupo_de[d] = f"Grupo {idx} ({g.valor})"
        for d in resultado.lacunas:
            grupo_de[d] = "Lacuna"

        # --- NAVEGAÇÃO POR ABAS ---
        tab_laudo, tab_dados, tab_telemetria = st.tabs(
            ["⚖️ Veredicto Final", "📊 Base de Evidências", "🕵️‍♂️ Caixa de Vidro (Logs)"]
        )

        with tab_laudo:
            veredito = resultado.veredito
            if veredito == core.CONSENSO:
                st.markdown(f'<div class="card-resultado card-sucesso"><h2>✅ {veredito}</h2><h4>{resultado.resumo}</h4></div>', unsafe_allow_html=True)
            elif veredito == core.LACUNA_EVIDENCIA:
                st.markdown(f'<div class="card-resultado card-aviso"><h2>⚠️ {veredito}</h2><h4>{resultado.resumo}</h4></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card-resultado card-erro"><h2>⚔️ {veredito}</h2><h4>{resultado.resumo}</h4></div>', unsafe_allow_html=True)

            # Métricas de topo
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📄 Documentos", len(extracoes))
            m2.metric("🧩 Grupos de valor", len(resultado.grupos))
            m3.metric("⚠️ Lacunas", len(resultado.lacunas))
            conf_media = (sum(e.confiabilidade for e in extracoes.values()) / len(extracoes)) if extracoes else 0.0
            m4.metric("🎯 Confiança média", f"{conf_media:.0%}")

            st.markdown("### 🧩 Agrupamento de Consenso")
            for idx, g in enumerate(resultado.grupos, 1):
                st.markdown(f"**Grupo {idx} — `{g.valor}`** ({len(g.documentos)} doc): {', '.join(g.documentos)}")
            if resultado.lacunas:
                st.markdown(f"**⚠️ Sem o dado** ({len(resultado.lacunas)} doc): {', '.join(resultado.lacunas)}")

            st.markdown("### 📝 Parecer do Auditor Chefe")
            st.info(parecer)

            relatorio_md = gerar_relatorio_markdown_n(pergunta_auditoria, resultado, extracoes, parecer)
            st.download_button(
                label="📥 Baixar Laudo Forense Oficial (.md)",
                # BOM UTF-8 ("﻿") força editores que defaultam p/ Latin-1/cp1252
                # a detectar UTF-8 e evita mojibake (ó->Ã³, 🏛️->ðï¸).
                data=("﻿" + relatorio_md).encode("utf-8"),
                file_name=f"laudo_flare26_{int(time.time())}.md",
                mime="text/markdown; charset=utf-8", type="primary", use_container_width=True
            )

        with tab_dados:
            tabela = [{
                "Documento": nome,
                "Resposta": ext.resposta_direta,
                "Condicionante": ext.condicionantes,
                "Tipo": ext.tipo_dado,
                "Confiança": f"{ext.confiabilidade:.0%}",
                "Grupo": grupo_de.get(nome, "—"),
            } for nome, ext in extracoes.items()]
            st.dataframe(tabela, use_container_width=True, hide_index=True)

            for nome, ext in extracoes.items():
                with st.expander(f"📄 Trecho literal — {nome}"):
                    st.caption(ext.trecho_literal)

        with tab_telemetria:
            st.markdown("### JSON Provenience Ledger")
            json_prov = {
                "M1.5_Gatekeeper": st.session_state.get('m1_5_ledger', {}),
                "M2_Extrator_Score": {nome: ext.confiabilidade for nome, ext in extracoes.items()},
                "M4_Juiz_NWay": {
                    "veredito": resultado.veredito,
                    "grupos": [{"valor": g.valor, "documentos": list(g.documentos)} for g in resultado.grupos],
                    "lacunas": list(resultado.lacunas),
                },
            }
            st.json(json_prov)
            salvar_no_ledger_local(pergunta_auditoria, list(extracoes.keys()), json_prov)

            st.markdown("### Roteamento Vetorial (Top Chunks por documento)")
            for nome, filhos in chunks_por_doc.items():
                with st.expander(f"Chunks injetados via {nome}"):
                    st.write(filhos)

    else:
        st.warning("⚠️ Forneça **ao menos 2 documentos** e o objeto da auditoria para iniciar a máquina.")