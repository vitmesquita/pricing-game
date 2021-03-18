# Pricing-game
### Repositório da competição 'insurance-pricing-game' de previsão de sinistro (#trackA) do AIcrowd
Nessa competição, agimos como uma companhia de seguros que constroi um modelo de precificação e compete com outros participantes pelo mercado. O mercado nessa competição é um "cheapest-wins", o que significa que cada companhia de seguro oferece um preço e o cliente escolhe a companhia com menor preço. 
Para criar o modelo de precificação foi dado aos participantes um dataset real de uma seguradora: 60k de policies por 4 anos consecutivos. Cada policy é referente a 1 veículo, seus motoristas e seu histórico. Os dados foram providenciados por uma grande seguradora europeia e é uma amostra uniforme do seu portifolio inteiro. Os participantes deveriam prever a sinistralidade de cada policy no 5° ano

1. Notebooks de sanity-check e observações sobre os dados da competição 
  * Disponível em "sanity-check", fiz algumas considerações sobre os dados da competição e notei algumas irregularidades importantes para modelagem
3. Notebooks de análises exploratórias feitas sob os dados da competição
  * Disponível em "analises-exploratorias", fiz algumas analises que conseguiram entender alguns padroes de clientes segurados e sua relação com a sinistralidade
5. Notebook de teste de modelos iniciais 
  * Disponível em "modelos", fiz alguns testes com modelos diferentes para classificação de clientes (se terão sinistro igual a zero ou não), e de regressão dos valores do sinistro
7. Modelo final no template da competição
 * Modelo treinado disponível em "trained_mode.picke"
 * Disponível em "model.py", calcula a previsão de sinistro para cada policy e o preço a cobrar pela seguradora 
