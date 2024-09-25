# Importação das Bibliotecas
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Carregamento do Dataset
url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/wisc_bc_data.csv'
df = pd.read_csv(url)

# Pré-processamento dos Dados
df.drop('id', axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Separação de Features e Variável Alvo
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Normalização dos Dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão dos Dados em Conjuntos de Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Construção e Compilação do Modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento do Modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Avaliação e Previsões
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Acurácia no conjunto de teste: {accuracy:.2f}')

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Matriz de Confusão e Relatório de Classificação
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de Confusão:')
print(conf_matrix)

print('Relatório de Classificação:')
print(classification_report(y_test, y_pred))
