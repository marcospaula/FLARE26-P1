import ollama
import json
import re

# =====================================================================
# MÓDULO M1/M2: EXTRAÇÃO DE ATOMIC CLAIMS
# =====================================================================
def extract_atomic_claim(texto_bruto, source_id):
    system_prompt = """Você é um extrator de dados estrito. Extraia as informações e responda APENAS com um JSON puro válido.
Use EXATAMENTE as seguintes 5 chaves.
Regras:
1. "sujeito": O tema central da métrica. Ex: "inflação", "população"
2. "relacao": O verbo ou ação. Ex: "atingiu", "registrou"
3. "objeto": O valor numérico com a unidade. Ex: "4,62%", "11.200.000 habitantes"
4. "escopo": O local. Ex: "Brasil", "São Paulo"
5. "tempo": APENAS os 4 dígitos do ano. Ex: "2023"
"""
    
    try:
        response = ollama.chat(
            model='qwen2.5:1.5b',
            format='json', 
            options={'temperature': 0}, 
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f'Texto: {texto_bruto}'}
            ]
        )
        conteudo = response['message']['content']
        claim_data = json.loads(conteudo)
        
        # DEBUG LIGADO PARA VERMOS O QUE A IA FEZ DE ERRADO:
        print(f"[DEBUG M2 - {source_id}] JSON: {claim_data}")
        
        return {
            'source_id': source_id,
            'evidence_span': texto_bruto,
            'claim_data': claim_data,
        }
    except Exception as e:
        print(f"[ERRO M2] Falha ao extrair {source_id}: {e}")
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
    
    num_a = extrair_numero(dados_a.get('objeto'))
    num_b = extrair_numero(dados_b.get('objeto'))

    # Para fins de simulação deste PoC, vamos assumir que os escopos passados na mesma pergunta são compatíveis.
    # O foco aqui é provar as lógicas algébricas de Disputa (Reivindicação 5) vs Atualização (Reivindicação 6)
    
    if tempo_a == tempo_b:
        # Mesmo ano, números diferentes -> DISPUTA
        if num_a is not None and num_b is not None and num_a != num_b:
            return '🔴 DISPUTA_DETECTADA'
        return '✅ CONSENSO_OU_COEXISTENCIA'
    else:
        # Anos diferentes, números diferentes -> ATUALIZAÇÃO
        if num_a != num_b:
            return '🔄 ATUALIZACAO_DETECTADA'
        else:
            return '✅ CONSENSO_OU_COEXISTENCIA'

# =====================================================================
# MÓDULO M5/M6: SÍNTESE DO CHATBOT (Rastreável)
# =====================================================================
def gerar_resposta_chat(pergunta_usuario, status, claim_a, claim_b):
    print(f"\nUsuário: \"{pergunta_usuario}\"")
    print("="*50)
    
    dados_a = claim_a['claim_data']
    dados_b = claim_b['claim_data']
    tema = dados_a.get('sujeito', 'Tema').title()
    escopo = dados_a.get('escopo', 'Região').title()
    
    print("🤖 Resposta do Assistente FLARE:")
    
    if status == '🔴 DISPUTA_DETECTADA':
        print(f"Encontrei informações conflitantes sobre {tema} ({escopo}) para o mesmo período.")
        print("\n## 🔴 BLOCO DE DISPUTA")
        print(f"- De acordo com {claim_a['source_id']}: {dados_a.get('objeto')}")
        print(f"  [Evidência]: \"{claim_a['evidence_span']}\"")
        print(f"- De acordo com {claim_b['source_id']}: {dados_b.get('objeto')}")
        print(f"  [Evidência]: \"{claim_b['evidence_span']}\"")
        print("\nConclusão: Como os dados divergem de forma não resolvida, apresento ambas as fontes para sua auditoria.")
        
    elif status == '🔄 ATUALIZACAO_DETECTADA':
        print(f"Encontrei uma evolução histórica nos dados de {tema} ({escopo}).")
        print("\n## 🔄 BLOCO DE ATUALIZAÇÃO")
        print(f"- O dado mais antigo ({claim_a['source_id']}, {dados_a.get('tempo')}): {dados_a.get('objeto')}")
        print(f"- O dado mais recente ({claim_b['source_id']}, {dados_b.get('tempo')}): {dados_b.get('objeto')}")
        print(f"\nConclusão: Considere o valor atualizado de {dados_b.get('objeto')} (Referência: {dados_b.get('tempo')}).")
        
    else:
        print(f"As fontes convergem sobre os dados de {tema} ({escopo}).")
        print("\n## ✅ BLOCO DE CONSENSO")
        print(f"- O valor apurado é: {dados_a.get('objeto')} (Referência: {dados_a.get('tempo')})")
        print(f"  [Fontes validadas]: {claim_a['source_id']} e {claim_b['source_id']}")

if __name__ == '__main__':
    print("\nIniciando modo interativo Chat-FLARE26...\n")
    cenarios = [
        {
            "pergunta": "Qual foi a inflação do Brasil no ano passado?",
            "fonte1": {"id": "Banco_Central", "texto": "A inflação oficial do Brasil fechou em 4,62% em 2023."},
            "fonte2": {"id": "Portal_Noticias", "texto": "A inflação brasileira atingiu 5,10% no ano de 2023."}
        },
        {
            "pergunta": "Qual a população de São Paulo?",
            "fonte1": {"id": "Censo_2010", "texto": "A população de São Paulo era de 11.200.000 habitantes em 2010."},
            "fonte2": {"id": "Censo_2022", "texto": "A cidade de São Paulo registrou 11.400.000 habitantes no ano de 2022."}
        }
    ]
    for idx, cenario in enumerate(cenarios):
        claim_1 = extract_atomic_claim(cenario["fonte1"]["texto"], cenario["fonte1"]["id"])
        claim_2 = extract_atomic_claim(cenario["fonte2"]["texto"], cenario["fonte2"]["id"])
        if claim_1 and claim_2:
            status = motor_de_conflito_m4(claim_1, claim_2)
            gerar_resposta_chat(cenario["pergunta"], status, claim_1, claim_2)
            if idx < len(cenarios) - 1:
                print("\n" + "-"*60 + "\n")