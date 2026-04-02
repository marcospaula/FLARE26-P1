REIVINDICAÇÃO INDEPENDENTE 1 (Método - Foco na Redução de CPU/RAM e bloqueio do Teste Alice)

1. Um método implementado por computador para otimização de processamento neuro-simbólico e mitigação de alucinação computacional em arquiteturas de análise documental, o método compreendendo as etapas de:
a) executar, via um processador determinístico de baixo custo computacional, um filtro heurístico de pré-processamento (M1.5) sobre um fluxo de dados de entrada, onde dito filtro varre a memória em busca de parâmetros escalares de existência predefinidos sem instanciar modelos de linguagem natural;
b) condicionar a alocação de memória RAM/VRAM para um modelo neural de extração (M2) exclusivamente aos subconjuntos de dados aprovados na etapa (a), abortando o carregamento neural e injetando um estado estruturado de "Lacuna de Evidência" diretamente em um registro de saída caso o filtro heurístico falhe, reduzindo assim o consumo de ciclos de CPU associados a inferências nulas;
c) forçar a saída do modelo neural de extração instanciado a uma estrutura de dados rigidamente tipada em formato JSON, onde as fronteiras semânticas da extração são delimitadas por esquemas de validação injetados diretamente no tensor de inferência;
d) gravar os dados tipados em uma estrutura de memória imutável chamada Ledger de Proveniência (M3), contendo mapeamentos físicos diretos (pointers) para as coordenadas originais do dado no arquivo nativo (Source ID e Timestamp).
REIVINDICAÇÃO INDEPENDENTE 2 (Sistema - Foco na Escalabilidade e Fuga dos Knowledge Graphs)

2. Um sistema de arquitetura híbrida para resolução de colisões semânticas em tempo linear sem dependência de bancos de dados vetoriais de alta dimensionalidade, o sistema compreendendo:
a) uma memória não transitória contendo a estrutura de dados imutável do Ledger de Proveniência gerada conforme a reivindicação 1;
b) um motor de colisão determinístico (M4) executado por um processador simbólico que acessa exclusivamente as chaves JSON do Ledger, configurado para calcular matrizes de colisão entre múltiplas fontes através da comparação estrita de valores escalares extraídos, contornando a necessidade de transformação computacionalmente custosa dos documentos em embeddings (vetores) para medir similaridade;
c) em que o motor de colisão categoriza os vetores de dados através de lógica booleana em instâncias de "Consenso", "Divergência" ou "Lacuna", alcançando resolução de conflitos sem invocar algoritmos de atenção estendida (cross-attention) de redes neurais, reduzindo a complexidade de tempo de $O(N^2)$ típica de comparações vetoriais densas para operações de busca de dicionário estruturado $O(1)$.
REIVINDICAÇÃO INDEPENDENTE 3 (Mídia Legível - Foco no Enclausuramento da IA Geradora)

3. Um meio de armazenamento legível por computador contendo instruções que, quando executadas por um processador, implementam um pipeline de síntese de dados restrita (Enclausuramento Neural), as instruções configuradas para:
a) receber o estado lógico final calculado pelo motor de colisão determinístico (M4) da reivindicação 2;
b) instanciar um módulo neural de síntese (M5) com um limite de janela de contexto (context window limit) restrito apenas aos metadados do Ledger de Proveniência, bloqueando o acesso físico do módulo M5 aos documentos brutos originais;
c) acionar a geração de uma cadeia de caracteres (texto) pelo módulo M5, onde a dita geração é ancorada por uma restrição de sistema que força a correspondência exata dos caracteres numéricos e monetários com o estado tipado recebido de M4;
d) renderizar em interface gráfica uma montagem acoplada que exibe a síntese gerada e o ponteiro físico para a localização exata no documento de origem (Source ID), eliminando a capacidade arquitetônica do modelo de linguagem de introduzir fatos não validados pela matriz lógica prévia.
