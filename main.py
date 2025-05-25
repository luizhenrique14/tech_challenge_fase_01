import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Exploração de dados:
df = pd.read_csv("Dataset/simulated_insurance_100k.csv", sep=",")
print('importando o dataset...')

print(df.head())

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Ajustar e transformar os rótulos
print('***********************************************************************************************************************')
print('********************************************* Normalizacao e Padronizacao *********************************************')
print('***********************************************************************************************************************')
print('Transformando os dados categóricos em numéricos...')
df['sexo'] = label_encoder.fit_transform(df['sexo'])
df['fumante'] = label_encoder.fit_transform(df['fumante'])
df['regiao'] = label_encoder.fit_transform(df['regiao'])


print(df.head())

print('Verificando os tipos de dados...')
print(df.dtypes)



print('Analisando os Graficos...')

# # Criar o gráfico de boxplot idades
# plt.boxplot(df['idade'])
# plt.title('idade')
# plt.ylabel('valores')
# plt.show()


# # Criar o gráfico de boxplot sexo
# plt.boxplot(df['sexo'])
# plt.title('sexo')
# plt.ylabel('valores')
# plt.show()

# # Criar o gráfico de boxplot IMC
# plt.boxplot(df['IMC'])
# plt.title('IMC')
# plt.ylabel('valores')
# plt.show()


# # Criar o gráfico de boxplot IMC
# plt.boxplot(df['filhos'])
# plt.title('filhos')
# plt.ylabel('valores')
# plt.show()


# # Criar o gráfico de boxplot IMC
# plt.boxplot(df['fumante'])
# plt.title('fumante')
# plt.ylabel('valores')
# plt.show()

# # Criar o gráfico de boxplot IMC
# plt.boxplot(df['regiao'])
# plt.title('regiao')
# plt.ylabel('valores')
# plt.show()


# Separando os dados em treinos e testes
from sklearn.model_selection import train_test_split

# Estou utilziando a variavel fumante como variável alvo (target) e as demais variáveis como características (features).
X = df.drop(columns=['fumante']) # Variáveis características
y = df['fumante'] # O que eu quero prever. (Target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)


from sklearn.preprocessing import StandardScaler, MinMaxScaler

print('Treinando.......')
# Utilizando - MinMaxScaler
print('Treinando o modelo... MinMaxScaler')
scaler = MinMaxScaler() #chamando o metodo de normalização dos dados (0-1)

scaler.fit(X_train)

x_train_min_max_scaled = scaler.transform(X_train)
x_test_min_max_scaled= scaler.transform(X_test)

print(x_train_min_max_scaled)

# Utilizando - StandardScaler
print('Treinando o modelo... StandardScaler')
scaler = StandardScaler() #chamando o metodo de padronização dos dados (média e std)

scaler.fit(X_train)# qual média e std será utilizado para o escalonamento

x_train_standard_scaled = scaler.transform(X_train)
x_test_standard_scaled  = scaler.transform(X_test)


print('Testar o algoritmo sem os escalonadores e validar os resultados!')
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
print('Treinando o modelo...')
model.fit(X_train, y_train)
# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score

# Avaliar a precisão do modelo
print('Avaliando o modelo...')
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')


print('Testando com a normalização:')
model_min_max = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
model_min_max.fit(x_train_min_max_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred_min_max = model.predict(x_test_min_max_scaled)

accuracy_min_max = accuracy_score(y_test, y_pred_min_max)
print(f'Acurácia Min Max Scaler: {accuracy_min_max:.2f}')


print('Testando com a padronização:')
model_standard = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
model_standard.fit(x_train_standard_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred_standard = model.predict(x_test_standard_scaled)

accuracy_strandard = accuracy_score(y_test, y_pred_standard)
print(f'Acurácia Standart Scaler: {accuracy_strandard:.2f}')

# Meu modelo testato tanto com a padronização quanto com a normalização, ambos os métodos de escalonamento de dados, e ambos os métodos apresentaram resultados semelhantes.



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



# Arvore de Decisao
print('***********************************************************************************************************************')
print('****************************************** Arvore de Decisao - Random Forest ******************************************')
print('***********************************************************************************************************************')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

# Ajustar e transformar os rótulos
print('Random Forest- Transformando os dados categóricos em numéricos...')
df['sexo'] = label_encoder.fit_transform(df['sexo'])
df['fumante'] = label_encoder.fit_transform(df['fumante'])
df['regiao'] = label_encoder.fit_transform(df['regiao'])

print(df.head())
headers = ['idade','sexo','IMC','filhos','regiao','taxas', 'fumante']
X = df[headers].values

from sklearn.preprocessing import StandardScaler
print('Normalizando os dados utilizando o standardScaler.....')
# Normalizando os dados utilizando o standardScaler
# (Padroniza as headers removendo a média e escala a variância a uma unidade.
# Isso significa que para cada feature, a média seria 0, e o Desvio Padrão seria 1)
X = StandardScaler().fit_transform(X)
#Visualizando nossos dados padronizados
df_padronizado = pd.DataFrame(data=X, columns=headers)
print('df_padronizado')
print(df_padronizado.head())

print('Verificando os dados null...')
print(df.isnull().sum()) # Verificando se existem dados nulos no dataset
print('Ok sem dados null...')

#analisando correlações
print('Analisando as correlações entre as variáveis...')
dados_coorelacionados = df.dropna()
correlation_matrix = dados_coorelacionados.corr().round(2)
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)
plt.show()

#Separando os dados

print('Removendo a colunar Target (fumante)...')
x = df.drop(columns=['fumante'])
y = df['fumante'] #O que eu quero prever. (Target)

print('Separando os dados em treino e teste...')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=48)

print('Aplicando o algoritmo de arvore de decisão - RandomForestClassifier...')
rf = RandomForestClassifier(n_estimators=5, max_depth = 2,  random_state=7)
rf.fit(x_train, y_train)

estimator = rf.estimators_


y_predito_random_forest = rf.predict(x_test)

# Metricas de precisão, revocação, f1-score e acurácia.
print('RandomForestClassifier accuracy_score:', accuracy_score(y_test, y_predito_random_forest)) #relatório de validação das métrica de desempenho.


class_names = ['Não Fumante', 'Fumante']
label_names = ['idade', 'sexo',	'IMC',	'filhos',	'regiao',	'taxas']

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf.estimators_[0],
               feature_names = label_names,
               class_names=class_names,
               filled = True)
fig.savefig('rf_individualtree.png')



fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
for index in range(0, 5):
    tree.plot_tree(rf.estimators_[index],
                   feature_names = label_names,
                   class_names=class_names,
                   filled = True,
                   ax = axes[index])

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('rf_5trees.png')


print('Base de Treino', rf.score(x_train, y_train))
print('Base de Teste',rf.score(x_test, y_test))

