import ollama
from pydantic import BaseModel, Field
import json
import re

# =====================================================================
# 1. SCHEMA MÁXIMO-FLEXÍVEL (À PROVA DE QWEN 1.5B)
# =====================================================================
class AtomicClaimSchema(BaseModel):
    # Aceitamos boolean, string ou Nulo. O Pydantic engole qualquer coisa.
    dados_encontrados: bool | str | None = Field(default=False)
    sujeito: str | None = Field(default="Não informado")
    relacao: str | None = Field(default="Não informado")
    objeto: str | None = Field(default="Não informado")
    escopo: str | None = Field(default="Não informado")
    tempo: str | None = Field(default="Não informado")

# =====================================================================
# MÓDULO M1/M2: EXTRAÇÃO COM FEW-SHOT PROMPTING (TUTORIAL EMBUTIDO)
# =====================================================================
def extract_atomic_claim(texto_bruto, source_id, pergunta_usuario=""):
    # Aqui aplicamos Few-Shot: Nós ensinamos a IA como jogar o jogo antes dela começar.
    system_prompt = f"""Você é um extrator de dados lendo um texto para responder: '{pergunta_usuario}'.
Você DEVE gerar um JSON respondendo à pergunta baseando-se nos dados.

### EXEMPLO DE COMO VOCÊ DEVE PENSAR E RESPONDER ###
Texto: "A Microsoft atingiu 30 bilhões de dólares em receitas durante o primeiro trimestre do ano fiscal, apenas na região da América do Norte."
Se a pergunta for 'Receita da Microsoft', o seu JSON deve ser EXATAMENTE assim:
{{
"dados_encontrados": true,
"sujeito": "Microsoft",
"relacao": "atingiu receita",
"objeto": "30 bilhões de dólares",
"escopo": "América do Norte",
"tempo": "primeiro trimestre"
}}

### FIM DO EXEMPLO ###

Se achar a resposta no texto abaixo, preencha seguindo o molde do exemplo.
Se o texto for sobre política, cookies, ou não tiver a resposta, responda NADA MAIS ALÉM DE:
{{
"dados_encontrados": false
}}"""

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
        conteudo = conteudo.replace('```json', '').replace('```', '').strip()
        
        claim_obj = AtomicClaimSchema.model_validate_json(conteudo)
        dados_ok = claim_obj.dados_encontrados in [True, "true", "True"]
        dados_finais = claim_obj.model_dump()
        dados_finais['dados_encontrados'] = dados_ok
        
        for k in ['sujeito', 'relacao', 'objeto', 'escopo', 'tempo']:
            if dados_finais[k] is None:
                dados_finais[k] = "Não informado"
                
        print(f"[DEBUG M2 - {source_id}] Extração Limpa OK: {dados_finais}")
        
        return {
            'source_id': source_id,
            'evidence_span': texto_bruto,
            'claim_data': dados_finais,
        }
    except Exception as e:
        print(f"[DEBUG M2 - {source_id}] IA errou a formatação. Erro: {e}")
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
# MÓDULO M4: MOTOR DE CONFLITO E GRANDEZAS
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
    
    # Trava: se a IA respondeu "Não informado", barramos imediatamente
    if "informado" in obj_a or "informado" in obj_b or obj_a == "" or obj_b == "":
        return '🔀 DADOS AUSENTES OU INCOMPLETOS'

    tempo_a = normalizar_texto(dados_a.get('tempo'))
    tempo_b = normalizar_texto(dados_b.get('tempo'))
    escopo_a = normalizar_texto(dados_a.get('escopo'))
    escopo_b = normalizar_texto(dados_b.get('escopo'))
    
    num_a = extrair_numero(dados_a.get('objeto'))
    num_b = extrair_numero(dados_b.get('objeto'))

    # Se a IA extraiu apenas texto puro (não achou números no texto)
    if num_a is None or num_b is None:
        if obj_a == obj_b:
            return '✅ CONSENSO FATO TEXTUAL'
        return '🔀 COEXISTENCIA (TEXTOS DIFERENTES)'

    escopos_similares = (escopo_a in escopo_b) or (escopo_b in escopo_a)
    
    if escopos_similares:
        if tempo_a == tempo_b:
            if num_a != num_b:
                return '🔴 DISPUTA_DETECTADA'
            return '✅ CONSENSO NUMÉRICO'
        else:
            if num_a != num_b:
                return '🔄 ATUALIZACAO_DETECTADA'
            return '✅ CONSENSO NUMÉRICO'
    else:
        return '🔀 COEXISTENCIA (ESCOPOS DIFERENTES)'