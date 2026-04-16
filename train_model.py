import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Carregamento do dataset MNIST via TensorFlow
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Pré-processamento (normalização + reshape para CNN)
x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test = (x_test.astype("float32") / 255.0)[..., None]

# Construção e treinamento de um modelo de classificação baseado em CNN
model = keras.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilação do modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Treinamento do modelo
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Exibição da acurácia final no terminal
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAcurácia final no teste: {accuracy:.4f}")

# Salvamento do modelo treinado no formato Keras (.h5)
model.save("model.h5")
print("Modelo salvo como model.h5")