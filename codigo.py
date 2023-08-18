## Importando as bibliotecas
import numpy as np
from tensorflow import keras
from tqdm import tqdm
## 1. Aquisição de dados
mnist = keras.datasets.mnist.load_data()
(X_train, y_train), (X_test, y_test) = mnist
## 2. Pré-processamento
# Criar um conj de validação de 5k e mudar a escala dos pixels de 0-255 para 0-1 (float)
X_valid, X_train = X_train[:5000] / 255., X_train[5000:] / 255.
y_valid, y_train = y_train[:5000], y_train[5000:]
X_test = X_test / 255.
## 3. Construindo a arquitetura
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
#model.summary()
## 4. Treinando a rede
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
### 4.1. Parâmetros do treino
epochs = 30
for _ in tqdm(range(epochs), desc='Treinamento'):
    history = model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid), verbose=0)

## 5. Testando o modelo
model.evaluate(X_test, y_test)
X_new = X_test[:10]
y_proba = model.predict(X_new)
y_proba.round(2)
# Pega a posição do maior valor na linha
y_pred = np.argmax(model.predict(X_new), axis=-1)
y_pred = np.argmax(model.predict(X_test), axis=-1)
from sklearn.metrics import accuracy_score, cohen_kappa_score
print('Acurácia: ', accuracy_score(y_test, y_pred))
print('Kappa: ', cohen_kappa_score(y_test, y_pred))
