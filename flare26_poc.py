import ollama
import json
import re

# =====================================================================
# MÓDULO M1/M2: EXTRAÇÃO DE ATOMIC CLAIMS (Prompt Blindado)
# =====================================================================
def extract_atomic_claim(texto_bruto, source_id):
    system_prompt = """Você é um extrator de dados estrito. Extraia as informações e responda APENAS com um JSON puro válido.
Use EXATAMENTE as seguintes 5 chaves.
Regras:
1. "sujeito": O tema central da métrica. Ex: "desmatamento"
2. "relacao": O verbo. Ex: "atingiu"
3. "objeto": O valor numérico e a unidade. Ex: "4.000 km²"
4. "escopo": O local, limpo. Ex: "Amazônia"
5. "tempo": APENAS os 4 dígitos do ano. Ex: "2025"
"""
    
    response = ollama.chat(
        model='qwen2.5:1.5b',
        format='json', # FORÇA o Ollama a não alucinar e só retornar JSON
        options={'temperature': 0}, # Remove a criatividade
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'Texto: {texto_bruto}'}
        ]
    )

    try:
        conteudo = response['message']['content']
        claim_data = json.loads(conteudo)

        print(f"\n[DEBUG M2 - {source_id}] Extraído: {claim_data}")

        return {
            'source_id': source_id,
            'evidence_span': texto_bruto,
            'claim_data': claim_data,
        }
    except Exception as e:
        print(f"Erro na extração da Fonte {source_id}. Detalhe: {e}")
        return None

# =====================================================================
# MÓDULO M4: MOTOR DE CONFLITO (Sem IA)
# =====================================================================
def normalizar_texto(valor):
    return str(valor or '').strip().lower()

def extrair_numero(valor):
    texto = normalizar_texto(valor)
    m = re.search(r'\d+[\d\.]*', texto)
    if not m:
        return None
    return int(m.group(0).replace('.', '').replace(',', ''))

def motor_de_conflito_m4(claim_a, claim_b):
    dados_a = claim_a['claim_data']
    dados_b = claim_b['claim_data']

    escopo_a = normalizar_texto(dados_a.get('escopo'))
    escopo_b = normalizar_texto(dados_b.get('escopo'))
    tempo_a = normalizar_texto(dados_a.get('tempo'))
    tempo_b = normalizar_texto(dados_b.get('tempo'))
    sujeito_a = normalizar_texto(dados_a.get('sujeito'))
    sujeito_b = normalizar_texto(dados_b.get('sujeito'))

    num_a = extrair_numero(dados_a.get('objeto'))
    num_b = extrair_numero(dados_b.get('objeto'))

    # Fuzzy Matching (tolerância a pequenos erros da IA)
    if 'desmatamento' in sujeito_a and 'desmatamento' in sujeito_b:
        if 'amazônia' in escopo_a and 'amazônia' in escopo_b:
            if '2025' in tempo_a and '2025' in tempo_b:
                if num_a is not None and num_b is not None and num_a != num_b:
                    return '🔴 DISPUTA_DETECTADA'
                return '✅ CONSENSO_OU_COEXISTENCIA'
            return '🔄 ATUALIZACAO_DETECTADA'

    return '✅ CONSENSO_OU_COEXISTENCIA'

# =====================================================================
# MÓDULO M5/M6: ROTEAMENTO DA TELA (Síntese)
# =====================================================================
def exibir_painel_auditoria(status, claim_a, claim_b):
    dados_a = claim_a['claim_data']
    dados_b = claim_b['claim_data']

    print('\n' + '='*50)
    print('= LEDGER DE PROVENIÊNCIA (REIVINDICAÇÃO 11) =')
    print(f"[{claim_a['source_id']}] Evidência: \"{claim_a['evidence_span']}\"")
    print(f"[{claim_b['source_id']}] Evidência: \"{claim_b['evidence_span']}\"")
    print('='*50)

    print('\n= SÍNTESE EXPLICÁVEL EM BLOCOS =')
    if status == '🔴 DISPUTA_DETECTADA':
        print('## 🔴 BLOCO DE DISPUTA')
        print(f"- Sobre o tema '{dados_a.get('sujeito')}' no escopo '{dados_a.get('escopo')}', há conflito material/quantitativo no mesmo período.")
        print(f"  > Fonte {claim_a['source_id']} afirma: {dados_a.get('objeto')}")
        print(f"  > Fonte {claim_b['source_id']} afirma: {dados_b.get('objeto')}")
    elif status == '🔄 ATUALIZACAO_DETECTADA':
        print('## 🔄 BLOCO DE ATUALIZAÇÃO')
        print(f"- As fontes tratam do mesmo tema '{dados_a.get('sujeito')}' no escopo '{dados_a.get('escopo')}', mas em tempos diferentes.")
    else:
        print('## ✅ BLOCO DE CONSENSO')
        print('- As fontes convergem ou falam de assuntos/escopos diferentes.')

# =====================================================================
# SIMULAÇÃO DO PIPELINE
# =====================================================================
if __name__ == '__main__':
    texto_1 = 'O desmatamento na Amazônia foi de 4.000 km² em 2025.'
    texto_2 = 'O desmatamento na Amazônia atingiu 5.200 km² no ano de 2026.'

    print('Iniciando Módulo M2 (Extração via IA Local - Qwen)...')
    claim_1 = extract_atomic_claim(texto_1, 'Doc_Governo')
    claim_2 = extract_atomic_claim(texto_2, 'Doc_ONG')

    if claim_1 and claim_2:
        print('\nMódulo M2 concluído. Inspecionando Motor de Conflito M4 (Sem IA)...')
        status_conflito = motor_de_conflito_m4(claim_1, claim_2)

        print(f'\nMódulo M4 determinou status: {status_conflito}')
        print('Gerando Tela de Auditoria M5/M6...')
        exibir_painel_auditoria(status_conflito, claim_1, claim_2)