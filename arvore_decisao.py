import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Arvore de Decisao
print('***********************************************************************************************************************')
print('****************************************** Arvore de Decisao - Random Forest ******************************************')
print('***********************************************************************************************************************')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Carregar e transformar apenas uma vez
df = pd.read_csv("Dataset/simulated_insurance_100k.csv", sep=",")
label_encoder = LabelEncoder()
df['sexo'] = label_encoder.fit_transform(df['sexo'])
df['fumante'] = label_encoder.fit_transform(df['fumante'])
df['regiao'] = label_encoder.fit_transform(df['regiao'])

# Features e target
features = ['idade', 'sexo', 'IMC', 'filhos', 'regiao']
X = df[features]
y = df['fumante']

# Normalização
X_scaled = StandardScaler().fit_transform(X)

df_padronizado = pd.DataFrame(data=X_scaled, columns=features)
print('df_padronizado')
print(df_padronizado.head())

# Split (faça só uma vez!)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=7)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

print('RandomForestClassifier accuracy_score:', accuracy_score(y_test, y_pred))
print('Base de Treino', rf.score(x_train, y_train))
print('Base de Teste', rf.score(x_test, y_test))
print(df['fumante'].value_counts())