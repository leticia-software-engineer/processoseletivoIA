import tensorflow as tf
import os

# Carregamento do modelo treinado
model = tf.keras.models.load_model("model.h5")

# Conversão para TensorFlow Lite (.tflite)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Aplicação de técnica de otimização: Dynamic Range Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Salvar o modelo convertido
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo convertido e salvo como model.tflite")

if os.path.exists("model.tflite"):
    size = os.path.getsize("model.tflite") / 1024
    print(f"Tamanho do modelo TFLite: {size:.2f} KB")