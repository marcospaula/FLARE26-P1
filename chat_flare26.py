import os
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

# Inicializa o cliente OpenAI usando a variável de ambiente de forma segura
# (O cliente OpenAI automaticamente procura por OPENAI_API_KEY no ambiente)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ========================================
# ESTRUTURA DE DADOS M2 (Agnóstica e Inteligente)
# ========================================
class AtomicClaim(BaseModel):
    dados_encontrados: bool = Field(description="O texto responde diretamente ao que o usuário quer saber?")
    contexto_da_fonte: str = Field(description="Se achou a resposta, resuma o fato. Se NÃO achou, explique sobre o que a fonte está falando (ex: 'É uma biografia focada no início da carreira').")
    sujeito: str = Field(description="A entidade principal da busca (ex: Microsoft, Michael B Jordan).")
    objeto: str = Field(description="A resposta exata encontrada. Se não encontrou, preencha com 'Ausente'.")
    tempo: str = Field(description="Quando aconteceu. Se não houver, 'N/A'.")
    escopo: str = Field(description="O recorte/cenário do fato. Se não houver, 'Geral'.")

# ========================================
# M2 - EXTRAÇÃO ATÔMICA
# ========================================
def extract_atomic_claim(texto_fonte, fonte_id, tema_busca):
    prompt_system = f"""Você é o Motor M2 de Extração Atômica do projeto FLARE26 (Glass Box).
Sua missão é ler o texto recebido e buscar informações sobre: '{tema_busca}'.

REGRAS DE INTELIGÊNCIA:
1. Se a resposta exata estiver no texto, dados_encontrados = True.
2. Se a resposta não estiver, dados_encontrados = False, MAS você deve preencher 'contexto_da_fonte' explicando o que o texto aborda para que o usuário não perca a informação de contexto.
3. Não invente NADA. Extraia apenas o que está no texto."""

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": f"Fonte [{fonte_id}]:\n\n{texto_fonte}"}
            ],
            response_format=AtomicClaim,
            temperature=0.0
        )
        claim = response.choices[0].message.parsed
        return {
            "fonte_id": fonte_id,
            "claim_data": claim.model_dump()
        }
    except Exception as e:
        print(f"Erro no M2: {e}")
        return None

# ========================================
# M4 - MOTOR DE CONFLITO INTELIGENTE
# ========================================
def motor_de_conflito_m4(claim_a, claim_b):
    da = claim_a['claim_data']
    db = claim_b['claim_data']
    
    achou_a = da.get('dados_encontrados')
    achou_b = db.get('dados_encontrados')

    # CASO 1: Nenhuma fonte tem o dado (Falha total real)
    if not achou_a and not achou_b:
        return "FALHA_TOTAL"
        
    # CASO 2: Assimetria (Uma tem o dado alvo, a outra traz apenas contexto)
    if achou_a and not achou_b:
        return "COMPLEMENTO_ASSIMETRICO_A"
    if achou_b and not achou_a:
        return "COMPLEMENTO_ASSIMETRICO_B"

    # CASO 3: Ambas têm o dado alvo (Análise de Conflito)
    # Compara normalização
    obj_a = str(da.get('objeto')).lower().strip()
    obj_b = str(db.get('objeto')).lower().strip()

    if obj_a == obj_b:
        return "CONSENSO_TOTAL"
    
    if da.get('tempo') != db.get('tempo') and da.get('tempo') != 'N/A' and db.get('tempo') != 'N/A':
        return "ATUALIZACAO_TEMPORAL"
        
    if da.get('escopo') != db.get('escopo') and da.get('escopo') != 'Geral' and db.get('escopo') != 'Geral':
        return "COEXISTENCIA_ESCOPO"
        
    return "DISPUTA_DIRETA"