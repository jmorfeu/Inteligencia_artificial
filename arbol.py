# Importar las bibliotecas necesarias
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Crear un conjunto de datos de frutas
data = {
    'Color': ['Rojo', 'Naranja', 'Amarillo', 'Rojo', 'Amarillo', 'Naranja', 'Naranja', 'Amarillo'],
    'Tamaño': ['Pequeño', 'Mediano', 'Grande', 'Mediano', 'Grande', 'Pequeño', 'Grande', 'Mediano'],
    'Textura': ['Suave', 'Rugosa', 'Suave', 'Suave', 'Rugosa', 'Suave', 'Rugosa', 'Rugosa'],
    'Clase': ['Manzana', 'Naranja', 'Plátano', 'Manzana', 'Plátano', 'Naranja', 'Naranja', 'Plátano']
}

# Crear un DataFrame de pandas
import pandas as pd
df = pd.DataFrame(data)

# Convertir características categóricas en numéricas
df = pd.get_dummies(df, columns=['Color', 'Tamaño', 'Textura'])

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = df.drop('Clase', axis=1)
y = df['Clase']

# Crear un modelo de árbol de decisión
modelo_arbol = DecisionTreeClassifier()

# Entrenar el modelo
modelo_arbol.fit(X, y)

# Ejemplo de predicción para una nueva fruta
nueva_fruta = pd.DataFrame({'Color_Rojo': [1], 'Color_Naranja': [1], 'Color_Amarillo': [0],
                            'Tamaño_Pequeño': [0], 'Tamaño_Mediano': [1], 'Tamaño_Grande': [0],
                            'Textura_Suave': [1], 'Textura_Rugosa': [0]})
prediccion = modelo_arbol.predict(nueva_fruta)
print("Predicción para la nueva fruta:", prediccion)
