import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.src.metrics import MeanSquaredError
from keras.src.saving import register_keras_serializable


#from tensorflow.keras.models import load_model
#from tensorflow.keras.losses import MeanSquaredError
#from tensorflow.keras.saving import register_keras_serializable




# Registrar manualmente la función de pérdida antes de cargar los modelos



@register_keras_serializable()
class CustomMSE(MeanSquaredError):
    pass

custom_objects = {"mse": CustomMSE()}


# ---------------------- CARGA DE DATOS Y MODELOS ----------------------
st.success("Cargando modelos de recomendación...")
models = {
    #"Red Neuronal Simple": keras.models.load_model("models/modelo_nn_simple.h5",custom_objects=custom_objects,  compile=False),
    "Red Profunda": keras.models.load_model("models/modelo_nn_profundo.h5",custom_objects=custom_objects,  compile=False),
    #"Autoencoder": keras.models.load_model("models/modelo_autoencoder.h5",custom_objects=custom_objects,  compile=False),
}



# Descargar archivos antes de cargar la app
#st.info("Verificando archivos y descargando si es necesario...")
#download_files()

# ---------------------- CARGA DE DATOS Y MODELOS ----------------------
# Cargar el dataset preprocesado
df = pd.read_csv("data/data_processed.csv")



# Diccionario de mapeo de productos
product_dict = dict(zip(df["productId"], df["name"]))

# # Cargar modelos
# st.success("Cargando modelos de recomendación...")
# models = {
#     "Red Neuronal Simple": keras.models.load_model("models/modelo_nn_simple.h5"),
#     "Red Profunda": keras.models.load_model("models/modelo_nn_profundo.h5"),
#     "Autoencoder": keras.models.load_model("models/modelo_autoencoder.h5"),
# }

# ---------------------- INTERFAZ EN STREAMLIT ----------------------
st.title("🛍️ Sistema de Recomendación de Productos con IA")

# Selección del modelo
selected_model = st.selectbox("Selecciona un modelo de recomendación:", list(models.keys()))

# Entrada del usuario
user_id = st.number_input("Ingrese su ID de usuario:", min_value=1, max_value=1000, step=1)

if st.button("Obtener Recomendaciones"):
    model = models[selected_model]

    # Seleccionar productos aleatorios para predecir
    productos_disponibles = np.array(df["productId"].unique())
    productos_prueba = np.random.choice(productos_disponibles, 10)

    # Hacer predicciones
    predicciones = model.predict([np.array([user_id] * len(productos_prueba)), productos_prueba])

    # Ordenar productos recomendados
    productos_recomendados = sorted(zip(productos_prueba, predicciones.flatten()), key=lambda x: -x[1])

    st.subheader(f"🎯 Productos Recomendados con {selected_model}:")
    for product_id, score in productos_recomendados[:5]:
        nombre_producto = product_dict.get(product_id, "Producto Desconocido")
        st.write(f"✅ {nombre_producto} (Predicción de rating: {score:.2f})")
