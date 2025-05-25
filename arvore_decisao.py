import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Árvore de Decisão - Random Forest para Classificação
print('***********************************************************************************************************************')
print('*************************************** Árvore de Decisão - Random Forest (Classificação) *****************************')
print('***********************************************************************************************************************')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Carregar e transformar apenas uma vez
df = pd.read_csv("Dataset/simulated_insurance_100k.csv", sep=",")
label_encoder = LabelEncoder()
df['sexo'] = label_encoder.fit_transform(df['sexo'])
df['fumante'] = label_encoder.fit_transform(df['fumante'])
df['regiao'] = label_encoder.fit_transform(df['regiao'])

# Features e target
features = ['idade', 'sexo', 'IMC', 'filhos', 'regiao', 'fumante']  # Inclui 'fumante' como feature
X = df[features]
# Para classificação, vamos transformar 'taxas' em uma variável categórica (exemplo: acima da mediana = 1, abaixo = 0)
mediana_taxas = df['taxas'].median()
y = (df['taxas'] >= mediana_taxas).astype(int)

# Normalização
X_scaled = StandardScaler().fit_transform(X)

df_padronizado = pd.DataFrame(data=X_scaled, columns=features)
print('df_padronizado')
print(df_padronizado.head())

# Split (faça só uma vez!)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=7)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
print('RandomForestClassifier accuracy_score:', accuracy)
print('Matriz de Confusão:\n', confusion_matrix(y_test, y_pred))
print('Relatório de Classificação:\n', classification_report(y_test, y_pred))

# Visualização: Matriz de confusão
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Random Forest Classifier (Taxas acima/abaixo da mediana)')
plt.show()