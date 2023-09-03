# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos (reemplaza 'datos.csv' con el nombre de tu archivo de datos)
datos = {
    'Caracteristica1': [1, 2, 3, 4, 5],
    'Caracteristica2': [10, 20, 30, 40, 50],
    'Caracteristica3': [100, 200, 300, 400, 500],
    'VariableObjetivo': [300, 400, 500, 600, 700]
}

data = pd.DataFrame(datos)

# Dividir los datos en características (X) y variable de destino (y)
X = data[['Caracteristica1', 'Caracteristica2', 'Caracteristica3']]
y = data['VariableObjetivo']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Crear un modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular métricas de desempeño
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir métricas de desempeño
print('Error cuadrático medio (MSE):', mse)
print('Coeficiente de determinación (R^2):', r2)
# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos (reemplaza 'datos.csv' con el nombre de tu archivo de datos)
data = pd.read_csv('datos.csv')

# Dividir los datos en características (X) y variable de destino (y)
X = data[['Caracteristica1', 'Caracteristica2', 'Caracteristica3']]
y = data['VariableObjetivo']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Crear un modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular métricas de desempeño
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir métricas de desempeño
print('Error cuadrático medio (MSE):', mse)
print('Coeficiente de determinación (R^2):', r2)
