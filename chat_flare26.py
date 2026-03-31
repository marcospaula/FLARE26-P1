import os
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ========================================
# ESTRUTURA DE DADOS M2 (Agnóstica e Inteligente)
# ========================================
class AtomicClaim(BaseModel):
    dados_encontrados: bool = Field(description="O texto responde diretamente ao que o usuário quer saber?")
    contexto_da_fonte: str = Field(description="Se achou a resposta, resuma o fato. Se NÃO achou, explique sobre o que a fonte está falando.")
    sujeito: str = Field(description="A entidade principal da busca.")
    objeto: str = Field(description="A resposta exata encontrada. Se não encontrou, preencha com 'Ausente'.")
    tempo: str = Field(description="Quando aconteceu. Se não houver, 'N/A'.")
    escopo: str = Field(description="O recorte/cenário do fato. Se não houver, 'Geral'.")

# ========================================
# M2 - EXTRAÇÃO ATÔMICA
# ========================================
def extract_atomic_claim(texto_fonte, fonte_id, tema_busca):
    prompt_system = f"""Você é o Motor M2 de Extração Atômica do projeto FLARE26 (Glass Box).
Sua missão é ler o texto recebido e buscar informações que respondam semanticamente a: '{tema_busca}'.

REGRAS DE INTELIGÊNCIA SEMÂNTICA:
1. Seja flexível com sinônimos: se o usuário perguntar por "inventor", considere "criador", "desenvolvedor" ou "pioneiro".
2. FOCO CIRÚRGICO NO OBJETO: Se a pergunta busca 'quem' (uma pessoa/empresa), o campo 'objeto' deve conter APENAS o nome dessa entidade. Se busca um valor/número, o 'objeto' deve ser APENAS o número. Não coloque frases no objeto.
3. Se a resposta exata estiver no texto, coloque dados_encontrados = True.
4. Se o texto abordar o tema mas não trouxer a resposta explícita, coloque dados_encontrados = False, MAS preencha 'contexto_da_fonte'.
5. Não invente fatos. Extraia apenas as informações baseadas no que está escrito no texto fornecido. Se o texto citar múltiplos possíveis criadores (ex: uma disputa histórica), extraia os nomes envolvidos na disputa."""
    
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

    if not achou_a and not achou_b:
        return "FALHA_TOTAL"
        
    if achou_a and not achou_b:
        return "COMPLEMENTO_ASSIMETRICO_A"
    if achou_b and not achou_a:
        return "COMPLEMENTO_ASSIMETRICO_B"

    obj_a = str(da.get('objeto')).lower().strip()
    obj_b = str(db.get('objeto')).lower().strip()

    if obj_a == obj_b:
        return "CONSENSO_TOTAL"
    
    if da.get('tempo') != db.get('tempo') and da.get('tempo') != 'N/A' and db.get('tempo') != 'N/A':
        return "ATUALIZACAO_TEMPORAL"
        
    if da.get('escopo') != db.get('escopo') and da.get('escopo') != 'Geral' and db.get('escopo') != 'Geral':
        return "COEXISTENCIA_ESCOPO"
        
    return "DISPUTA_DIRETA"

# ========================================
# M5 - FALLBACK COGNITIVO (BLACK BOX)
# ========================================
def fallback_cognitivo_m5(pergunta):
    """
    Acionado APENAS quando a busca web falha. 
    Usa a memória paramétrica do LLM de forma tradicional (Caixa Preta).
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é uma IA respondendo a partir de sua base de treinamento. Seja direto, conciso e evite inventar referências bibliográficas que você não pode provar."},
                {"role": "user", "content": pergunta}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Falha na comunicação com o LLM: {e}"