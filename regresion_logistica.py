import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Crear un conjunto de datos de ejemplo
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Dividir el conjunto de datos en dos clases
X_class0 = X[y == 0]
X_class1 = X[y == 1]

# Crear un modelo de regresión logística
logistic_model = LogisticRegression()
logistic_model.fit(X, y)

# Visualizar los datos y las líneas de decisión de los modelos
plt.figure(figsize=(12, 5))

# Gráfico de dispersión para los datos de la clase 0
plt.scatter(X_class0[:, 0], X_class0[:, 1], c='blue', marker='o', label='Clase 0')

# Gráfico de dispersión para los datos de la clase 1
plt.scatter(X_class1[:, 0], X_class1[:, 1], c='red', marker='x', label='Clase 1')

# Línea de decisión de regresión logística
b0, b1, b2 = logistic_model.intercept_[0], logistic_model.coef_[0, 0], logistic_model.coef_[0, 1]
plt.plot([-4, 4], [-(b0 + b1 * (-4)) / b2, -(b0 + b1 * 4) / b2], color='orange', label='Regresión Logística')

# Configuración del gráfico
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Comparación de Regresión Logística')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()
