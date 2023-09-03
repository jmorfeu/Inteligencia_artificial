import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Crear un conjunto de datos de ejemplo para la calidad de los vinos
data = {
    'fixed_acidity': [7.4, 7.8, 7.8, 11.2],
    'volatile_acidity': [0.7, 0.88, 0.76, 0.28],
    'citric_acid': [0.0, 0.0, 0.04, 0.56],
    'residual_sugar': [1.9, 2.6, 2.3, 1.9],
    'chlorides': [0.076, 0.098, 0.092, 0.075],
    'free_sulfur_dioxide': [11.0, 25.0, 15.0, 17.0],
    'total_sulfur_dioxide': [34.0, 67.0, 54.0, 60.0],
    'density': [0.9978, 0.9968, 0.9970, 0.9980],
    'pH': [3.51, 3.20, 3.26, 3.16],
    'sulphates': [0.56, 0.68, 0.65, 0.58],
    'alcohol': [9.4, 9.8, 9.8, 9.8],
    'quality': [5, 5, 5, 6]
}

# Crear un DataFrame de pandas
wine_data = pd.DataFrame(data)

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de Random Forest para clasificación
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
modelo_rf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo_rf.predict(X_test)

# Calcular la precisión del modelo
precision = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", precision)

# Mostrar un informe de clasificación detallado
print("Informe de clasificación:\n", classification_report(y_test, y_pred))
