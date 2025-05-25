import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print('***********************************************************************************************************************')
print('*************************************************** Regressão Linear **************************************************')
print('***********************************************************************************************************************')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Carregar o dataset e transformar variáveis categóricas em numéricas
print("\nRegressao Linear Carregando o dataset e aplicando LabelEncoder nas colunas categóricas...")
df = pd.read_csv("Dataset/simulated_insurance_100k.csv", sep=",")
label_encoder = LabelEncoder()
df['sexo'] = label_encoder.fit_transform(df['sexo'])
df['fumante'] = label_encoder.fit_transform(df['fumante'])
df['regiao'] = label_encoder.fit_transform(df['regiao'])

# 2. Definir as features e o target
# Não inclua 'fumante' nas features, pois é a variável alvo!
features = ['idade', 'sexo', 'IMC', 'filhos', 'regiao']
X = df[features]
y = df['fumante']

# 3. Normalização das features numéricas
print("\nRegressao Linear Aplicando normalização (StandardScaler) nas features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Separação em treino e teste
print("\nRegressao Linear Realizando train_test_split (80% treino, 20% teste)...")
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Treinamento do modelo de Regressão Logística
print("\nRegressao Linear Treinando o modelo LogisticRegression para prever se é fumante ou não...")
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

# 6. Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
print('\n[RESULTADO] Acurácia do modelo:', accuracy)
print('[RESULTADO] Matriz de Confusão:\n', confusion_matrix(y_test, y_pred))
print('[RESULTADO] Relatório de Classificação:\n', classification_report(y_test, y_pred))

# 7. Visualização: Matriz de confusão
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Regressão Logística')
plt.show()

# 8. Visualização: Distribuição dos valores reais e preditos
plt.figure(figsize=(8, 4))
sns.histplot(y_test, color='blue', label='Real', kde=False, stat='count', bins=2)
sns.histplot(y_pred, color='orange', label='Previsto', kde=False, stat='count', bins=2)
plt.legend()
plt.title('Distribuição dos valores reais e preditos (fumante)')
plt.xlabel('Fumante (0=Não, 1=Sim)')
plt.ylabel('Quantidade')
plt.show()

# 9. Comentários sobre o desempenho
if accuracy > 0.8:
    print("\n[COMENTÁRIO] O modelo apresentou boa acurácia! Isso indica que a Regressão Logística conseguiu aprender padrões relevantes para prever quem é fumante.")
else:
    print("\n[COMENTÁRIO] O modelo não apresentou boa acurácia. Considere revisar as features, balanceamento dos dados ou testar outros modelos.")