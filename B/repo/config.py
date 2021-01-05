import os

path_projeto = r'D:\Users\vicml\Documents\Projetos\pricing-game\B'

path_dados = os.path.join(path_projeto,'dados')

path_dados_entrada = os.path.join(path_dados,'entrada')
path_dados_tratados = os.path.join(path_dados,'tratados')

path_arquivo_treino = os.path.join(path_dados_entrada,'train.csv')
path_arquivo_test = os.path.join(path_dados_entrada,'test.csv')