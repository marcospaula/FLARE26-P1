"""
flare26_pipeline — Pipeline de RAG chamável SEM Streamlit.

Concentra o M1 (banco vetorial), M1.5 (retrieval híbrido) e M2 (extração
ontologicamente restrita) de forma headless: nada de `streamlit`, nada de
`st.session_state`. O estado (vector_store, caminho do SQLite, cliente OpenAI)
é injetado por parâmetro, o que permite:
  * o app Streamlit importar e embrulhar com UI/threading;
  * o harness de avaliação (eval/) rodar o mesmo código sem navegador.

A lógica determinística (sanitização, números, juízes) vive em flare26_core.
"""

from __future__ import annotations

import os
import gc
import json
import hashlib
import tempfile

import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import flare26_core as core

# ==========================================================================
# CONFIGURAÇÕES (espelham o app; fonte única de verdade)
# ==========================================================================
MODEL_EMBEDDING = "all-MiniLM-L6-v2"
MODELO_EXTRACAO = "gpt-4o-mini"

MAX_FILE_SIZE_MB = 10
MAX_PAGES = 500
CHUNK_SIZE_PAI = 3500
CHUNK_OVERLAP_PAI = 200
CHUNK_SIZE_FILHO = 400
CHUNK_OVERLAP_FILHO = 50
TOP_K_RETRIEVAL = 20

# Acima deste tamanho de texto, o M1.5 deixa de ler o doc inteiro (bypass) e
# parte para o retrieval híbrido (vetor + léxico).
LIMIAR_BYPASS_CHARS = 80000

COLUNAS_DOCSTORE = {"parent_id", "nome_doc", "file_hash", "conteudo"}


# ==========================================================================
# SCHEMA DE EXTRAÇÃO (M2) — ontologicamente restrito
# ==========================================================================
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


def resultado_vazio() -> ExtracaoUniversal:
    return ExtracaoUniversal(
        natureza_da_pergunta="N/A", natureza_do_texto_encontrado="N/A",
        houve_compatibilidade_ontologica=False, violou_restricao_do_usuario=False,
        resposta_direta="NÃO LOCALIZADO", tipo_dado="NÃO LOCALIZADO",
        condicionantes="NÃO LOCALIZADO", trecho_literal="LACUNA DE EVIDÊNCIA", confiabilidade=0.0
    )


# ==========================================================================
# FÁBRICAS DE RECURSOS (injetáveis)
# ==========================================================================
def criar_client_openai(api_key: str | None = None):
    """Cria o cliente OpenAI. Usa `api_key` ou a env OPENAI_API_KEY."""
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY ausente (passe api_key ou defina a env).")
    return OpenAI(api_key=key)


def criar_vector_store(persist_directory: str = "./chroma_db",
                       model_embedding: str = MODEL_EMBEDDING):
    """Cria (ou abre) o ChromaDB persistente. Retorna (vector_store, embeddings)."""
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=model_embedding, encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = Chroma(
        collection_name="flare26_docs", embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}, persist_directory=persist_directory
    )
    return vector_store, embeddings


# ==========================================================================
# SQLITE DOCSTORE (blocos pais em disco; anti-OOM)
# ==========================================================================
def iniciar_banco_sqlite(db_path: str) -> None:
    import sqlite3
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS docstore_pai (
            parent_id TEXT PRIMARY KEY, nome_doc TEXT, file_hash TEXT, conteudo TEXT
        )
    ''')
    # Guard de migração: caches antigos (versão _sqlite) não têm 'file_hash'.
    colunas = {row[1] for row in cursor.execute("PRAGMA table_info(docstore_pai)")}
    if colunas != COLUNAS_DOCSTORE:
        cursor.execute("DROP TABLE IF EXISTS docstore_pai")
        cursor.execute('''
            CREATE TABLE docstore_pai (
                parent_id TEXT PRIMARY KEY, nome_doc TEXT, file_hash TEXT, conteudo TEXT
            )
        ''')
    conn.commit()
    conn.close()


def _ler_parents_por_hash(db_path: str, file_hash: str) -> list[str]:
    import sqlite3
    conn = sqlite3.connect(db_path, check_same_thread=False)
    rows = conn.execute(
        "SELECT conteudo FROM docstore_pai WHERE file_hash = ?", (file_hash,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


# ==========================================================================
# HASH e EXTRAÇÃO DE TEXTO
# ==========================================================================
def gerar_hash_bytes(pdf_bytes: bytes) -> str:
    return hashlib.md5(pdf_bytes).hexdigest()


def extrair_texto_fitz(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    if doc.page_count > MAX_PAGES:
        doc.close()
        raise ValueError(f"Limite de {MAX_PAGES} páginas excedido.")
    textos = [core.sanitizar_texto_pdf(page.get_text()) for page in doc]
    doc.close()
    return textos


# ==========================================================================
# PROCESSAMENTO DE PDF (headless, síncrono)
# ==========================================================================
def processar_pdf_bytes(pdf_bytes: bytes, nome: str, *, db_path: str,
                        vector_store) -> str:
    """Indexa um PDF (texto -> chunks pai/filho -> SQLite + Chroma).

    Síncrono e sem Streamlit. Retorna o file_hash (MD5) do documento.
    """
    import sqlite3
    file_hash = gerar_hash_bytes(pdf_bytes)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        textos_paginas = extrair_texto_fitz(tmp_path)
        pages = [Document(page_content=t, metadata={"source": nome, "file_hash": file_hash})
                 for t in textos_paginas]

        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_PAI, chunk_overlap=CHUNK_OVERLAP_PAI)
        docs_pai = parent_splitter.split_documents(pages)

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_FILHO, chunk_overlap=CHUNK_OVERLAP_FILHO)
        docs_filho = []
        for i, doc_pai in enumerate(docs_pai):
            parent_id = f"{file_hash}_pai_{i}"
            cursor.execute(
                "INSERT OR REPLACE INTO docstore_pai (parent_id, nome_doc, file_hash, conteudo) VALUES (?, ?, ?, ?)",
                (parent_id, nome, file_hash, doc_pai.page_content))
            for filho in child_splitter.split_documents([doc_pai]):
                filho.metadata.update({"source": nome, "file_hash": file_hash, "parent_id": parent_id})
                docs_filho.append(filho)
        conn.commit()
        conn.close()

        if docs_filho:
            batch = 200
            for i in range(0, len(docs_filho), batch):
                vector_store.add_documents(docs_filho[i:i + batch])

        del docs_pai, docs_filho, pages, textos_paginas
        gc.collect()
        return file_hash
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==========================================================================
# M1.5: RETRIEVAL HÍBRIDO (vetor + léxico)
# ==========================================================================
_STOPWORDS = {"qual", "quais", "como", "quem", "onde", "para", "pelo", "pela",
              "sobre", "entre", "seja", "esse", "este", "dos", "das", "nas",
              "nos", "que", "com", "por", "um", "uma"}


def recuperar_contexto(pergunta: str, file_hash: str, *, db_path: str,
                       vector_store, top_k: int = TOP_K_RETRIEVAL):
    """RAG híbrido sem Streamlit.

    Retorna (contexto: str, chunks: list[dict], telemetria: dict). A telemetria
    substitui o antigo st.session_state['m1_5_ledger'].
    """
    import re
    rows = _ler_parents_por_hash(db_path, file_hash)
    texto_completo = "\n".join(rows)

    # Bypass: documentos pequenos vão inteiros para o extrator.
    if len(texto_completo) < LIMIAR_BYPASS_CHARS:
        telemetria = {"estrategia": "Leitura Total (Bypass)", "caracteres": len(texto_completo)}
        return texto_completo, [{"texto": "Full Context Bypass.", "score": 1.0}], telemetria

    pergunta_norm = pergunta.lower().replace("percentual", "%").replace("porcentagem", "%")
    palavras = [re.sub(r'[^a-záéíóúçãõâê%]', '', p) for p in pergunta_norm.split()]
    termos = [p for p in palavras if (len(p) >= 4 or p == "%") and p not in _STOPWORDS]
    nao_pct = [t for t in termos if t != "%"]
    ancora = sorted(nao_pct, key=len, reverse=True)[0] if nao_pct else ""

    contextos = set()
    chunks = []

    res_vetor = vector_store.similarity_search_with_score(
        pergunta, k=top_k, filter={"file_hash": file_hash})
    if res_vetor:
        import sqlite3
        limite = max(0.25, max(1.0 - d for _, d in res_vetor) * 0.70)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cur = conn.cursor()
        for doc, dist in res_vetor:
            if 1.0 - dist >= limite:
                row = cur.execute(
                    "SELECT conteudo FROM docstore_pai WHERE parent_id = ?",
                    (doc.metadata.get("parent_id"),)).fetchone()
                if row and row[0] not in contextos:
                    contextos.add(row[0])
                    chunks.append({"texto": row[0][:200] + "... [VETOR]", "score": 1.0 - dist})
        conn.close()

    lexicos = []
    for conteudo in rows:
        score = sum((10 if t == "%" else len(t)) for t in termos if t in conteudo.lower())
        if ancora and ancora in conteudo.lower():
            score += 15
        if score >= 15:
            lexicos.append((score, conteudo))
    lexicos.sort(key=lambda x: x[0], reverse=True)
    for score, conteudo in lexicos[:6]:
        if conteudo not in contextos:
            contextos.add(conteudo)
            chunks.append({"texto": conteudo[:200] + "... [LÉXICO]", "score": min(0.99, score / 100.0)})

    telemetria = {"estrategia": "Parent-Child RAG", "total_blocos": len(contextos)}
    contexto = "\n\n--- [BLOCO PAI] ---\n\n".join(list(contextos)[:8])
    return contexto, chunks, telemetria


# ==========================================================================
# M2: EXTRAÇÃO ONTOLOGICAMENTE RESTRITA
# ==========================================================================
def extrair_dado(client, texto: str, pergunta: str, *,
                 modelo: str = MODELO_EXTRACAO) -> ExtracaoUniversal:
    """Extrai a resposta tipada, abstendo-se (NÃO LOCALIZADO) quando o texto é
    ontologicamente incompatível ou viola uma restrição negativa da pergunta.
    """
    if not texto.strip():
        return resultado_vazio()

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
            model=modelo,
            messages=[
                {"role": "system", "content": "Você é um auditor forense focado em validação de domínio e restrições negativas."},
                {"role": "user", "content": prompt},
            ],
            tools=[{"type": "function", "function": {"name": "retornar_extracao", "parameters": ExtracaoUniversal.model_json_schema()}}],
            tool_choice={"type": "function", "function": {"name": "retornar_extracao"}},
            temperature=0.0,
        )
        dados = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

        incompativel = not dados.get("houve_compatibilidade_ontologica", False)
        violou = dados.get("violou_restricao_do_usuario", False)
        if incompativel or violou:
            return resultado_vazio()

        resp = str(dados.get("resposta_direta", "NÃO LOCALIZADO")).strip()
        return ExtracaoUniversal(
            natureza_da_pergunta=str(dados.get("natureza_da_pergunta", "N/A")),
            natureza_do_texto_encontrado=str(dados.get("natureza_do_texto_encontrado", "N/A")),
            houve_compatibilidade_ontologica=True, violou_restricao_do_usuario=False,
            resposta_direta=resp, tipo_dado=str(dados.get("tipo_dado", "NÃO LOCALIZADO")),
            condicionantes=str(dados.get("condicionantes", "NÃO LOCALIZADO")),
            trecho_literal=str(dados.get("trecho_literal", "LACUNA DE EVIDÊNCIA")),
            confiabilidade=0.95 if resp != "NÃO LOCALIZADO" else 0.0,
        )
    except Exception:
        return resultado_vazio()
