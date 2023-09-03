import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Cargar los datos desde un archivo CSV
data = pd.read_csv('datos.csv')

# Dividir los datos en características (X) y variable objetivo (y)
X = data[['Tamaño', 'Num_Habitaciones', 'Edad']]
y = data['Precio']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Crear un modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)
#joblib.dump(modelo, 'modelo_entrenado.pkl')

# Realizar predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)
#modelo_cargado = joblib.load('modelo_entrenado.pkl')
# Calcular métricas de desempeño
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir métricas de desempeño

#nuevos_datos = [[1500, 3, 5], [1200, 2, 10]]
#predicciones = modelo_cargado.predict(nuevos_datos)
#print(predicciones)
print(y)
print('Error cuadrático medio (MSE):', mse)
print('Coeficiente de determinación (R^2):', r2)
