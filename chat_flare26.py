import ollama
from pydantic import BaseModel, Field
import json
import re

# =====================================================================
# 1. DEFINIÇÃO DO SCHEMA (CONTRATO DE SANGUE) COM PYDANTIC
# =====================================================================
class AtomicClaimSchema(BaseModel):
    sujeito: str = Field(..., description="O tema central da métrica. Ex: inflação, população, lucro")
    relacao: str = Field(..., description="O verbo ou ação que conecta o sujeito ao objeto. Ex: atingiu, registrou")
    objeto: str = Field(..., description="O valor numérico bruto extraído, incluindo a unidade. Ex: 4,62%, 30 bilhões de dólares")
    escopo: str = Field(..., description="O local, região ou delimitação material. Ex: Brasil, São Paulo, América do Norte, mundiais")
    tempo: str = Field(..., description="O momento da afirmação. Formato OBRIGATÓRIO: APENAS os 4 dígitos do ano. Ex: 2024")

# =====================================================================
# MÓDULO M1/M2: EXTRAÇÃO ESTRUTURADA (Type-Safe)
# =====================================================================
# =====================================================================
# MÓDULO M1/M2: EXTRAÇÃO ESTRUTURADA (Cão de Guarda Pydantic)
# =====================================================================
def extract_atomic_claim(texto_bruto, source_id):
    system_prompt = """Você é um extrator de dados estrito. Extraia as informações do texto e responda EXCLUSIVAMENTE em um JSON válido.
Use EXATAMENTE as seguintes 5 chaves, todas com valores em texto (string):
1. "sujeito": O tema central (ex: inflação, Microsoft).
2. "relacao": A ação (ex: atingiu, registrou).
3. "objeto": O valor numérico e unidade (ex: 21.900.000.000 dólares, 30.000.000.000 dólares).
4. "escopo": O local ou recorte (ex: América do Norte, globais).
5. "tempo": O ano ou trimestre (ex: 2024).

Retorne APENAS o JSON, sem markdown ou explicações.
"""
    
    try:
        response = ollama.chat(
            model='qwen2.5:1.5b',
            # REMOVEMOS O FORMAT=SCHEMA PARA NÃO BUGAR O MODELO PEQUENO
            options={'temperature': 0}, 
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f'Texto: {texto_bruto}'}
            ]
        )
        
        conteudo = response['message']['content']
        
        # Limpeza rápida caso a IA adicione ```json ... ```
        conteudo = conteudo.replace('```json', '').replace('```', '').strip()
        
        # O CÃO DE GUARDA: Pydantic valida o JSON gerado
        claim_obj = AtomicClaimSchema.model_validate_json(conteudo)
        
        print(f"[DEBUG M2 - {source_id}] Pydantic OK: {claim_obj.model_dump()}")
        
        return {
            'source_id': source_id,
            'evidence_span': texto_bruto,
            'claim_data': claim_obj.model_dump(),
        }
    except Exception as e:
        print(f"[ERRO M2] Falha ao extrair/validar {source_id}: {e}")
        return None

# =====================================================================
# MÓDULO M4: MOTOR DE CONFLITO (Sem IA)
# =====================================================================
def normalizar_texto(valor):
    return str(valor or '').strip().lower()

def extrair_numero(valor):
    texto = normalizar_texto(valor)
    m = re.search(r'\d+[\d\.,]*', texto)
    if not m:
        return None
    numero_sujo = m.group(0)
    numero_limpo = numero_sujo.replace('.', '').replace(',', '.')
    try:
        return float(numero_limpo)
    except:
        return None

def motor_de_conflito_m4(claim_a, claim_b):
    dados_a = claim_a['claim_data']
    dados_b = claim_b['claim_data']

    tempo_a = normalizar_texto(dados_a.get('tempo'))
    tempo_b = normalizar_texto(dados_b.get('tempo'))
    escopo_a = normalizar_texto(dados_a.get('escopo'))
    escopo_b = normalizar_texto(dados_b.get('escopo'))
    
    num_a = extrair_numero(dados_a.get('objeto'))
    num_b = extrair_numero(dados_b.get('objeto'))

    escopos_similares = (escopo_a in escopo_b) or (escopo_b in escopo_a)
    
    if escopos_similares:
        if tempo_a == tempo_b:
            if num_a is not None and num_b is not None and num_a != num_b:
                return '🔴 DISPUTA_DETECTADA'
            return '✅ CONSENSO'
        else:
            if num_a != num_b:
                return '🔄 ATUALIZACAO_DETECTADA'
            else:
                return '✅ CONSENSO'
    else:
        return '🔀 COEXISTENCIA'

# =====================================================================
# MÓDULO M5/M6: SÍNTESE DO CHATBOT (Rastreável)
# =====================================================================
def gerar_resposta_chat(pergunta_usuario, status, claim_a, claim_b):
    print(f"\nUsuário: \"{pergunta_usuario}\"")
    print("="*50)
    
    dados_a = claim_a['claim_data']
    dados_b = claim_b['claim_data']
    tema = dados_a.get('sujeito', 'Tema').title()
    
    print("🤖 Resposta do Assistente FLARE:")
    
    if status == '🔴 DISPUTA_DETECTADA':
        print(f"Encontrei informações conflitantes sobre {tema} para o mesmo período.")
        print("\n## 🔴 BLOCO DE DISPUTA")
        print(f"- De acordo com {claim_a['source_id']}: {dados_a.get('objeto')}")
        print(f"- De acordo com {claim_b['source_id']}: {dados_b.get('objeto')}")
        
    elif status == '🔄 ATUALIZACAO_DETECTADA':
        print(f"Encontrei uma evolução histórica nos dados de {tema}.")
        print("\n## 🔄 BLOCO DE ATUALIZAÇÃO")
        print(f"- Dado antigo ({claim_a['source_id']}, {dados_a.get('tempo')}): {dados_a.get('objeto')}")
        print(f"- Dado recente ({claim_b['source_id']}, {dados_b.get('tempo')}): {dados_b.get('objeto')}")
        
    elif status == '🔀 COEXISTENCIA':
        print(f"Os dados sobre {tema} estão corretos, mas referem-se a escopos diferentes.")
        print("\n## 🔀 BLOCO DE COEXISTÊNCIA (Divergência de Escopo)")
        print(f"- Escopo [{dados_a.get('escopo')}]: {dados_a.get('objeto')} (Fonte: {claim_a['source_id']})")
        print(f"- Escopo [{dados_b.get('escopo')}]: {dados_b.get('objeto')} (Fonte: {claim_b['source_id']})")
        print("\nConclusão: Não há contradição. Os valores coexistem pois abrangem delimitações distintas.")
        
    else:
        print(f"As fontes convergem sobre os dados de {tema}.")
        print("\n## ✅ BLOCO DE CONSENSO")
        print(f"- O valor apurado é: {dados_a.get('objeto')}")

if __name__ == '__main__':
    print("\nIniciando modo interativo Chat-FLARE26 (Blindado com Pydantic)...\n")
    cenarios = [
        {
            "pergunta": "Qual foi o lucro da Microsoft no primeiro trimestre de 2024?",
            "fonte1": {
                "id": "Relatorio_Fiscal_EUA", 
                "texto": "A Microsoft registrou um lucro líquido de 21.900.000.000 dólares no primeiro trimestre de 2024, referente apenas às operações na América do Norte."
            },
            "fonte2": {
                "id": "Relatorio_Fiscal_Global", 
                "texto": "Considerando as operações mundiais, a Microsoft atingiu um lucro de 30.000.000.000 dólares no primeiro trimestre de 2024."
            }
        }
    ]
    
    for cenario in cenarios:
        claim_1 = extract_atomic_claim(cenario["fonte1"]["texto"], cenario["fonte1"]["id"])
        claim_2 = extract_atomic_claim(cenario["fonte2"]["texto"], cenario["fonte2"]["id"])
        if claim_1 and claim_2:
            status = motor_de_conflito_m4(claim_1, claim_2)
            gerar_resposta_chat(cenario["pergunta"], status, claim_1, claim_2)