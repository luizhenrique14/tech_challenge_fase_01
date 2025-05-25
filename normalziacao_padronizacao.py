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

# Criar o gráfico de boxplot idades
plt.boxplot(df['idade'])
plt.title('idade')
plt.ylabel('valores')
plt.show()
# Criar o gráfico de boxplot sexo
plt.boxplot(df['sexo'])
plt.title('sexo')
plt.ylabel('valores')
plt.show()
# Criar o gráfico de boxplot IMC
plt.boxplot(df['IMC'])
plt.title('IMC')
plt.ylabel('valores')
plt.show()
# Criar o gráfico de boxplot IMC
plt.boxplot(df['filhos'])
plt.title('filhos')
plt.ylabel('valores')
plt.show()
# Criar o gráfico de boxplot IMC
plt.boxplot(df['fumante'])
plt.title('fumante')
plt.ylabel('valores')
plt.show()

# Criar o gráfico de boxplot taxas
plt.boxplot(df['taxas'])
plt.title('taxas')
plt.ylabel('valores')
plt.show()

# Criar o gráfico de boxplot IMC
plt.boxplot(df['regiao'])
plt.title('regiao')
plt.ylabel('valores')
plt.show()

print('Plotando o heatmap de correlação entre as colunas...')
plt.figure(figsize=(10, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap de Correlação entre as Colunas')
plt.show()

# Separando os dados em treinos e testes
from sklearn.model_selection import train_test_split

# Estou utilziando a variavel fumante como variável alvo (target) e as demais variáveis como características (features).
print('Apos a validacao entre a relacao entre as colunas fumante e taxas, removemos a coluna taxcas, pois ela tem coorelacao de 0.97 com a de fumante...')
X = df.drop(columns=['fumante']) # Variáveis características
X = df.drop(columns=['taxas']) # Variáveis características
y = df['fumante'] # O que eu quero prever. (Target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=74)


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

