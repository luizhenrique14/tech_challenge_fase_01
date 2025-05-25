

import pandas as pd
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# Redução de Dimensionalidade - PCA
print('***********************************************************************************************************************')
print('************************************************************ PCA ******************************************************')
print('***********************************************************************************************************************')
print('Reduzindo a dimensionalidade dos dados com PCA...')
# A seguir, iremos separar todas as colunas na lista de ‘recursos’ para uma variável ‘X’ e a variável ‘destino’ para ‘y’.
df = pd.read_csv("Dataset/simulated_insurance_100k.csv", sep=",")
# Ajustar e transformar os rótulos
print('Transformando os dados categóricos em numéricos...')
df['sexo'] = label_encoder.fit_transform(df['sexo'])
df['fumante'] = label_encoder.fit_transform(df['fumante'])
df['regiao'] = label_encoder.fit_transform(df['regiao'])

print(df.head())
features = ['idade','sexo','IMC','filhos','regiao','taxas']
X = df[features].values
y = df['fumante'].values

from sklearn.preprocessing import StandardScaler
print('Normalizando os dados utilizando o standardScaler.....')
# Normalizando os dados utilizando o standardScaler
# (Padroniza as features removendo a média e escala a variância a uma unidade.
# Isso significa que para cada feature, a média seria 0, e o Desvio Padrão seria 1)
X = StandardScaler().fit_transform(X)
#Visualizando nossos dados padronizados
df_padronizado = pd.DataFrame(data=X, columns=features)
print('df_padronizado')
print(df_padronizado.head())

# importando PCA da biblioteca sklearn
from sklearn.decomposition import PCA

pca = PCA(n_components=3)# Instanciando o pca e a quantidade de componentes que desejamos obter

principalComponents = pca.fit_transform(X) # Aplicando PCA nas nossas features
print('Criando um novo dataframe para visualizarmos como ficou nossos dados reduzidos com o PCA ....')

df_pca = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3']) # Criando um novo dataframe para visualizarmos como ficou nossos dados reduzidos com o PCA

target = pd.Series(y, name='fumante')
result_df = pd.concat([df_pca, target], axis=1)
print(result_df.head())

