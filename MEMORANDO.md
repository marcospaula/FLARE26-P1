# MEMORANDO DE INVENÇÃO (INVENTION DISCLOSURE)

**Projeto:** FLARE26-P1 – Web Engine Deep-Read (Motor RAG Tolerante a Falhas)
**Data de Validação:** 29 de Março de 2026

***

### 1. Título do Sistema e Método

**Sistema e Método Determinístico Tolerante a Falhas para Resolução de Conflitos Numéricos em Textos Não Estruturados Baseado em Extração Semântica Tipada.**

***

### 2. O Problema (Estado da Técnica)

Sistemas RAG (*Retrieval-Augmented Generation*) que dependem de SLMs (*Small Language Models*) sofrem de "alucinações estruturais" ao processar métricas financeiras ou dados textuais da web. Modelos enxutos (na faixa de 1.5 a 3 bilhões de parâmetros) frequentemente ignoram instruções de sistema (*System Prompts*), inventam chaves JSON não solicitadas ou falham em reconhecer a ausência de dados lógicos (como em textos bloqueados por *Paywalls* ou avisos de Cookies). Em arquiteturas padrão, essas alucinações de formatação resultam em falsos positivos na comparação de fatos, além de quebrarem a execução do pipeline de software (falhas críticas de tipagem forte).

***

### 3. A Solução (A Invenção - MVP Validado)

A presente invenção compreende uma arquitetura híbrida de "Caixa de Vidro" (*Glass Box*) que isola a "inteligência linguística" potencialmente falha de um modelo de IA de um "Motor de Julgamento Analítico" rígido. O sistema atua de ponta a ponta e é caracterizado por três pilares centrais:

- **Fallback Automático de Busca:** Chaveamento dinâmico entre APIs de notícias específicas e indexação geral da web, baseado em limiares de tolerância de *Rate Limit* (limitação de rede) e pertinência corporativa.
- **Extração com Few-Shot Prompting Embutido:** Injeção de "gabaritos lógicos e sintáticos" de sucesso e de recusa de dados diretamente no prompt do LLM, elevando artificialmente a taxa de obediência estrutural de modelos restritos computacionalmente.
- **Escudo Anti-Alucinação (Rede Elástica de Tipagem):** Um módulo de tipagem e validação (implementado via *Pydantic*) configurado passivamente para aceitar chaves opcionais e englobar um bloco de captura de erro (*try/except*). Se a IA devolver lixo sintático, o sistema descarta a alucinação e injeta silenciosamente um objeto padrão ("Não informado"), impedindo o *crash* da interface e delegando a recusa de disputa ao Motor Determinístico.

***

### 4. Elementos de Inventividade Adicionados na Fase MVP

Diferente de abordagens padrão de mercado que tentam forçar o modelo LLM a gerar um *schema* estrito no nível da API (o que causa o congelamento de modelos pequenos), este método reivindica a **absorção sistemática do erro semântico**.

Ao receber um texto ilegível (por exemplo, políticas de privacidade ao realizar a leitura profunda de sites via bibliotecas *headless*), a invenção utiliza uma 'flag' booleana requerida. Se o modelo falhar em gerar a flag por desvio de raciocínio, a arquitetura assume a indisponibilidade de dados lógicos e desarma o pipeline visual (via `st.stop()`), sem jamais interromper o *Event Loop* do Python ou comprometer o Motor M4 de comparação cruzada de tempo e espaço.

***

### 5. Reivindicações Principais (Draft de Claims)

**1. Um método implementado por computador para resolução determinística de ambiguidades em dados, caracterizado por compreender as etapas de:**
a) Acessar redes remotas com rotinas de *fallback* automáticas para captura de textos integrais (*Deep-Read*) ignorando blocos curtos (*snippets*);
b) Submeter referidos textos a um modelo de linguagem computacionalmente restrito através de instruções no formato *Few-Shot Prompting* visando a extração de métricas atômicas compostas de sujeito, objeto, grandeza temporal e escopo;
c) Interceptar erros de estruturação lógica (JSON) em tempo de execução através de um validador maleável, injetando dicionários de dados padronizados ('Não informado') que absorvem a falha do LLM sem interromper a estabilidade da interface do usuário;
d) Comparar matematicamente as extrações resultantes através de um algoritmo Python sem pesos neurais, normalizando grandezas linguísticas (mil, bilhões) e classificando as alegações de forma rastreável em: Disputas Numéricas, Atualizações Históricas ou Coexistência Divergente de Escopos.

***
*Registro técnico de arquitetura derivado do Código Original do Repositório - Commit \#1 (Blindagem de M4 e Pydantic)*