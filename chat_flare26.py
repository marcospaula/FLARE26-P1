import os
from openai import OpenAI
from pydantic import BaseModel, Field
import json
import re
from thefuzz import fuzz

# Inicializa o cliente OpenAI
client = OpenAI(api_key="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")

# =====================================================================
# 1. SCHEMA MÁXIMO-FLEXÍVEL (ATUALIZADO PARA OPENAI STRUCTURED)
# =====================================================================
class AtomicClaimSchema(BaseModel):
    dados_encontrados: bool = Field(description="Obrigatório. True se o texto contém pelo menos parte da resposta à pergunta do usuário, False se o texto for inútil.")
    sujeito: str = Field(default="Não informado", description="A empresa, pessoa ou entidade. Se não achar, retorne 'Não informado'.")
    relacao: str = Field(default="Não informado", description="O que o sujeito fez. Se não achar, retorne 'Não informado'.")
    objeto: str = Field(default="Não informado", description="O número ou valor principal (ex: $70.1 billion, R$ 4,5 bilhões). Se a pergunta pedir um número e ele não existir, marque dados_encontrados como False.")
    escopo: str = Field(default="Não informado", description="O contexto secundário (ex: Brasil, Global). Se não achar, retorne 'Não informado'.")
    tempo: str = Field(default="Não informado", description="O período de tempo. Se não achar, retorne 'Não informado'.")

# =====================================================================
# MÓDULO M1/M2: EXTRAÇÃO COM OPENAI
# =====================================================================
def extract_atomic_claim(texto_bruto, source_id, pergunta_usuario=""):
    system_prompt = f"""Você é um extrator de dados de precisão focado na pergunta: '{pergunta_usuario}'.
Leia o texto abaixo. Se houver um indício da resposta, extraia o máximo que conseguir.
Se o texto não mencionar a métrica ou entidade perguntada, defina 'dados_encontrados' como false.
Para os campos que você não encontrar no texto (como escopo ou tempo), preencha OBRIGATORIAMENTE com a string exata: "Não informado".
O campo 'objeto' deve conter o número principal e sua unidade (ex: 70.1 billion).
"""

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Texto: {texto_bruto}"}
            ],
            response_format=AtomicClaimSchema,
            temperature=0
        )
        
        claim_obj = response.choices[0].message.parsed
        dados_finais = claim_obj.model_dump()
        
        print(f"[DEBUG M2 - {source_id}] Extração GPT-4 OK: {dados_finais}")
        
        return {
            'source_id': source_id,
            'evidence_span': texto_bruto,
            'claim_data': dados_finais,
        }
    except Exception as e:
        print(f"[DEBUG M2] Erro na OpenAI: {e}")
        return {
            'source_id': source_id,
            'evidence_span': texto_bruto,
            'claim_data': {
                'dados_encontrados': False,
                'sujeito': 'Não informado', 'relacao': 'Não informado',
                'objeto': 'Não informado', 'escopo': 'Não informado',
                'tempo': 'Não informado'
            }
        }


# =====================================================================
# MÓDULO M4: MOTOR DE CONFLITO COM FUZZY MATCHING
# =====================================================================
def normalizar_texto(valor):
    return str(valor or '').strip().lower()

def extrair_numero(valor):
    texto = normalizar_texto(valor)
    m = re.search(r'\d+[\d\.,]*', texto)
    if not m:
        return None
    numero_sujo = m.group(0).replace('.', '').replace(',', '.')
    try:
        num = float(numero_sujo)
    except:
        return None
        
    if 'mil milhões' in texto or 'bilhões' in texto or 'bn' in texto:
        num *= 1_000_000_000
    elif 'milhões' in texto or 'mi' in texto:
        num *= 1_000_000
    elif 'mil' in texto or 'k' in texto:
        num *= 1_000
    return num

def motor_de_conflito_m4(claim_a, claim_b):
    dados_a = claim_a['claim_data']
    dados_b = claim_b['claim_data']

    obj_a = normalizar_texto(dados_a.get('objeto'))
    obj_b = normalizar_texto(dados_b.get('objeto'))
    
    if "informado" in obj_a or "informado" in obj_b or obj_a == "" or obj_b == "":
        return '🔀 DADOS AUSENTES OU INCOMPLETOS'

    tempo_a = normalizar_texto(dados_a.get('tempo'))
    tempo_b = normalizar_texto(dados_b.get('tempo'))
    escopo_a = normalizar_texto(dados_a.get('escopo'))
    escopo_b = normalizar_texto(dados_b.get('escopo'))
    sujeito_a = normalizar_texto(dados_a.get('sujeito'))
    sujeito_b = normalizar_texto(dados_b.get('sujeito'))
    
    num_a = extrair_numero(dados_a.get('objeto'))
    num_b = extrair_numero(dados_b.get('objeto'))

    escopos_similares = (
        fuzz.partial_ratio(escopo_a, escopo_b) >= 60 or 
        escopo_a in escopo_b or escopo_b in escopo_a or
        escopo_a == "não informado" or escopo_b == "não informado"
    )
    
    sujeitos_similares = (
        fuzz.partial_ratio(sujeito_a, sujeito_b) >= 60 or 
        sujeito_a in sujeito_b or sujeito_b in sujeito_a
    )

    contexto_valido = escopos_similares or sujeitos_similares

    if num_a is None or num_b is None:
        if fuzz.ratio(obj_a, obj_b) > 80:
            return '✅ CONSENSO FATO TEXTUAL'
        return '🔀 COEXISTENCIA (TEXTOS DIFERENTES)'

    if contexto_valido:
        if num_a != num_b:
            return '🔴 DISPUTA_DETECTADA'
        return '✅ CONSENSO NUMÉRICO'
    else:
        if num_a == num_b:
             return '✅ CONSENSO NUMÉRICO'
        return '🔀 COEXISTENCIA (ESCOPOS DIFERENTES)'