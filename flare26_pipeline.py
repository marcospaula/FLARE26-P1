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

# Nº de blocos-pai no contexto final (valores originais; aumentá-los não trouxe
# ganho no piloto real e seria tuning contra amostra pequena).
MAX_BLOCOS_CONTEXTO = 8
MAX_BLOCOS_LEXICOS = 6

COLUNAS_DOCSTORE = {"parent_id", "nome_doc", "file_hash", "conteudo"}


# ==========================================================================
# SCHEMA DE EXTRAÇÃO (M2) — ontologicamente restrito
# ==========================================================================
class ExtracaoUniversal(BaseModel):
    # --- Compatibilidade em DUAS dimensões (genérica, agnóstica de domínio) ---
    natureza_da_pergunta: str = Field(description="TIPO de dado exigido pela pergunta (ex: Temporal, Monetário, Percentual).")
    escopo_da_pergunta: str = Field(description="ESCOPO específico que a pergunta pede: a condição, evento, sujeito ou recorte exato sob o qual o dado é solicitado (ex: 'quando ocorre a condição X').")
    natureza_do_texto_encontrado: str = Field(description="TIPO de dado encontrado no texto candidato.")
    escopo_do_texto_encontrado: str = Field(description="ESCOPO a que a evidência candidata se aplica: a condição/evento/recorte que ela realmente cobre.")
    houve_compatibilidade_ontologica: bool = Field(
        description="True SOMENTE se TIPO e ESCOPO baterem: a evidência precisa ser do mesmo tipo E valer para a MESMA condição/evento/recorte da pergunta. Dois valores do mesmo tipo sob condições diferentes NÃO são compatíveis."
    )
    violou_restricao_do_usuario: bool = Field(
        description="CRÍTICO: Retorne True SE a pergunta exigir explicitamente que um tipo de dado seja ignorado E o texto encontrado tratar desse assunto proibido."
    )
    resposta_direta: str = Field(description="A resposta exata. Retorne 'NÃO LOCALIZADO' se houve_compatibilidade_ontologica for False OU se violou_restricao_do_usuario for True.")
    tipo_dado: str = Field(description="Categoria do dado. Se incompatível/violado, retorne 'NÃO LOCALIZADO'.")
    condicionantes: str = Field(description="Regras da resposta. Se incompatível/violado, retorne 'NÃO LOCALIZADO'.")
    trecho_literal: str = Field(description="Cópia literal do texto. Se incompatível/violado, retorne 'LACUNA DE EVIDÊNCIA'.")
    confiabilidade: float = Field(description="Score de 0.0 a 1.0. Se incompatível/violado, OBRIGATORIAMENTE 0.0.")
    # Sentinela de INFRAESTRUTURA (não é decisão do gate): True quando a extração
    # falhou por erro de API/parse — NÃO por abstenção ontológica. Fica FORA do
    # schema enviado ao LLM (ver _SCHEMA_LLM): o modelo nunca o vê nem o preenche.
    erro_infra: bool = False


def resultado_vazio() -> ExtracaoUniversal:
    """Abstenção genuína: o gate decidiu que não há evidência compatível."""
    return ExtracaoUniversal(
        natureza_da_pergunta="N/A", escopo_da_pergunta="N/A",
        natureza_do_texto_encontrado="N/A", escopo_do_texto_encontrado="N/A",
        houve_compatibilidade_ontologica=False, violou_restricao_do_usuario=False,
        resposta_direta="NÃO LOCALIZADO", tipo_dado="NÃO LOCALIZADO",
        condicionantes="NÃO LOCALIZADO", trecho_literal="LACUNA DE EVIDÊNCIA", confiabilidade=0.0
    )


def resultado_erro() -> ExtracaoUniversal:
    """ERRO DE INFRAESTRUTURA (API/parse falhou), distinto de abstenção.
    O harness de avaliação NÃO deve contá-lo como abstenção (ver eval/run_eval.py):
    um erro de infra não é uma decisão do gate."""
    r = resultado_vazio()
    r.erro_infra = True
    return r


# Schema de function-calling SEM o campo sentinela `erro_infra` (controle interno
# de infraestrutura, não um dado a extrair). Computado uma vez no import.
_SCHEMA_LLM = ExtracaoUniversal.model_json_schema()
_SCHEMA_LLM.get("properties", {}).pop("erro_infra", None)
if "required" in _SCHEMA_LLM:
    _SCHEMA_LLM["required"] = [r for r in _SCHEMA_LLM["required"] if r != "erro_infra"]


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

    # Candidatos rankeados por prioridade (NÃO usar set + truncamento arbitrário:
    # isso descartava o bloco mais relevante em docs grandes). Léxico exato é
    # sinal de precisão forte → prioridade acima da vetorial.
    candidatos = []  # (prioridade, conteudo, chunk_meta)
    vistos = set()

    res_vetor = vector_store.similarity_search_with_score(
        pergunta, k=top_k, filter={"file_hash": file_hash})
    if res_vetor:
        import sqlite3
        limite = max(0.25, max(1.0 - d for _, d in res_vetor) * 0.70)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cur = conn.cursor()
        for doc, dist in res_vetor:
            score_v = 1.0 - dist
            if score_v >= limite:
                row = cur.execute(
                    "SELECT conteudo FROM docstore_pai WHERE parent_id = ?",
                    (doc.metadata.get("parent_id"),)).fetchone()
                if row and row[0] not in vistos:
                    vistos.add(row[0])
                    candidatos.append((score_v, row[0],
                                       {"texto": row[0][:200] + "... [VETOR]", "score": score_v}))
        conn.close()

    lexicos = []
    for conteudo in rows:
        score = sum((10 if t == "%" else len(t)) for t in termos if t in conteudo.lower())
        if ancora and ancora in conteudo.lower():
            score += 15
        if score >= 15:
            lexicos.append((score, conteudo))
    lexicos.sort(key=lambda x: x[0], reverse=True)
    for score, conteudo in lexicos[:MAX_BLOCOS_LEXICOS]:
        if conteudo not in vistos:
            vistos.add(conteudo)
            # prioridade > 1.0 garante que o léxico exato fique acima da vetorial.
            candidatos.append((1.0 + score / 100.0, conteudo,
                               {"texto": conteudo[:200] + "... [LÉXICO]", "score": min(0.99, score / 100.0)}))

    candidatos.sort(key=lambda x: x[0], reverse=True)
    top = candidatos[:MAX_BLOCOS_CONTEXTO]
    chunks = [meta for _, _, meta in top]
    telemetria = {"estrategia": "Parent-Child RAG", "total_blocos": len(top)}
    contexto = "\n\n--- [BLOCO PAI] ---\n\n".join(conteudo for _, conteudo, _ in top)
    return contexto, chunks, telemetria


# ==========================================================================
# M2: EXTRAÇÃO ONTOLOGICAMENTE RESTRITA
# ==========================================================================
# Seed fixo para reduzir não-determinância da extração (reprodutibilidade).
SEED_EXTRACAO = 7

# Regra de compatibilidade — GENÉRICA, sem termos de domínio. O modelo deve
# comparar tipo E escopo; o exemplo é abstrato (condição A vs B) de propósito.
_REGRA_COMPATIBILIDADE = (
    "DEFINIÇÃO DE COMPATIBILIDADE (vale para qualquer domínio):\n"
    "Há compatibilidade ontológica APENAS se as duas condições abaixo forem verdadeiras:\n"
    "  (a) TIPO: o dado encontrado é do mesmo tipo que a pergunta exige.\n"
    "  (b) ESCOPO: a evidência vale para a MESMA condição/evento/sujeito/recorte que a pergunta pede.\n"
    "ATENÇÃO: dois valores do MESMO tipo, porém sob CONDIÇÕES diferentes, NÃO são compatíveis.\n"
    "Exemplo abstrato: se a pergunta pede 'o valor quando ocorre a condição A' e o texto só "
    "traz 'o valor quando ocorre a condição B' (com A ≠ B), então "
    "houve_compatibilidade_ontologica = False — mesmo que ambos sejam do mesmo tipo.\n"
    "Primeiro identifique escopo_da_pergunta e escopo_do_texto_encontrado; só então decida (b)."
)


def extrair_dado(client, texto: str, pergunta: str, *,
                 modelo: str = MODELO_EXTRACAO) -> ExtracaoUniversal:
    """Extrai a resposta tipada, abstendo-se (NÃO LOCALIZADO) quando o texto é
    ontologicamente incompatível (tipo OU escopo divergente) ou viola uma
    restrição negativa da pergunta.
    """
    if not texto.strip():
        return resultado_vazio()

    prompt = (
        "Você é um Extrator Forense de Dados Estruturados (qualquer domínio).\n"
        f"PERGUNTA DA AUDITORIA: {pergunta}\n\n"
        f"{_REGRA_COMPATIBILIDADE}\n\n"
        "REGRAS DE PREENCHIMENTO:\n"
        "1. Se a pergunta exigir ignorar algo e o texto tratar disso, 'violou_restricao_do_usuario' = True.\n"
        "2. Se incompatível (tipo OU escopo) ou violado, 'resposta_direta' = 'NÃO LOCALIZADO'.\n"
        f"TEXTO:\n{texto[:90000]}"
    )
    try:
        response = client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": "Você é um auditor forense agnóstico de domínio, focado em compatibilidade de tipo E escopo e em restrições negativas."},
                {"role": "user", "content": prompt},
            ],
            tools=[{"type": "function", "function": {"name": "retornar_extracao", "parameters": _SCHEMA_LLM}}],
            tool_choice={"type": "function", "function": {"name": "retornar_extracao"}},
            temperature=0.0,
            seed=SEED_EXTRACAO,
        )
        dados = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

        incompativel = not dados.get("houve_compatibilidade_ontologica", False)
        violou = dados.get("violou_restricao_do_usuario", False)
        if incompativel or violou:
            return resultado_vazio()

        resp = str(dados.get("resposta_direta", "NÃO LOCALIZADO")).strip()
        # Confiabilidade vem do MODELO (auto-reportada, NÃO calibrada). Antes era
        # fixada em 0.95 — campo morto que enganava a UI "caixa de vidro". Agora
        # usa o valor real, com guarda de tipo/intervalo; 0.0 se não localizou.
        try:
            conf = max(0.0, min(1.0, float(dados.get("confiabilidade", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        if resp == "NÃO LOCALIZADO":
            conf = 0.0
        return ExtracaoUniversal(
            natureza_da_pergunta=str(dados.get("natureza_da_pergunta", "N/A")),
            escopo_da_pergunta=str(dados.get("escopo_da_pergunta", "N/A")),
            natureza_do_texto_encontrado=str(dados.get("natureza_do_texto_encontrado", "N/A")),
            escopo_do_texto_encontrado=str(dados.get("escopo_do_texto_encontrado", "N/A")),
            houve_compatibilidade_ontologica=True, violou_restricao_do_usuario=False,
            resposta_direta=resp, tipo_dado=str(dados.get("tipo_dado", "NÃO LOCALIZADO")),
            condicionantes=str(dados.get("condicionantes", "NÃO LOCALIZADO")),
            trecho_literal=str(dados.get("trecho_literal", "LACUNA DE EVIDÊNCIA")),
            confiabilidade=conf,
        )
    except Exception:
        # Erro de API/parse: NÃO é abstenção do gate. Devolve sentinela de erro
        # para que o harness não o confunda com uma decisão de abster.
        return resultado_erro()


def extrair_dado_consenso(client, texto: str, pergunta: str, *,
                          k: int = 5, modelo: str = MODELO_EXTRACAO) -> ExtracaoUniversal:
    """Self-consistency com limiar t=1: roda extrair_dado k vezes e RESPONDE se
    QUALQUER execução responder; abstém só se TODAS abstiverem. Entre as que
    responderam, devolve o valor mais frequente.

    ATENÇÃO (ver paper §4.2 — "single-sample illusion"): t=1 ("≥1") NÃO é o ponto
    de operação recomendado. Medições com bootstrap mostram que o gate vaza
    raramente (~1–2% por amostra) e que t=1 AMPLIFICA esse vazamento conforme k
    cresce (≈10% de falso-positivo em k=10, t=1). Para recuperar recall sem
    reabrir falso-positivo, prefira um limiar de maioria (o ponto k=10, t=4 a que
    o paper chega). A varredura (k,t) está em eval/run_eval_grid.py.

    (Uma versão anterior desta docstring afirmava que "≥1" recuperava recall
    "sem reabrir falso-positivo (0%)" — exatamente a ilusão que o paper desmonta.)
    """
    if not texto.strip():
        return resultado_vazio()

    amostras = [extrair_dado(client, texto, pergunta, modelo=modelo) for _ in range(max(1, k))]
    # Amostras com erro de infraestrutura não votam (não são nem resposta nem
    # abstenção). Se TODAS falharam, o resultado é erro — não abstenção.
    validas = [e for e in amostras if not e.erro_infra]
    if not validas:
        return resultado_erro()
    responderam = [e for e in validas if e.resposta_direta != "NÃO LOCALIZADO"]
    if not responderam:
        return resultado_vazio()

    from collections import Counter
    contagem = Counter(e.resposta_direta.strip().lower() for e in responderam)
    valor_top = contagem.most_common(1)[0][0]
    for e in responderam:
        if e.resposta_direta.strip().lower() == valor_top:
            return e
    return responderam[0]
