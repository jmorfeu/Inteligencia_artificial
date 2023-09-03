import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Generar datos de ubicación ficticios
np.random.seed(0)
n_samples = 300
locations = np.random.randn(n_samples, 2) * 0.4  # Ubicaciones aleatorias alrededor del origen
locations[100:200] += [2, 2]  # Agregar un grupo aleatorio de usuarios cercanos
locations[200:250] += [3, -2]  # Otro grupo cercano
locations[250:280] += [-3, -2]  # Otro grupo cercano
locations[280:300] += [-3, 2]  # Otro grupo cercano

# Estandarizar los datos
scaler = StandardScaler()
scaled_locations = scaler.fit_transform(locations)

# Crear un modelo DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Ajustar el modelo a los datos
dbscan.fit(scaled_locations)

# Obtener etiquetas de cluster asignadas a cada punto de datos
labels = dbscan.labels_

# Número de clusters y ruido en el conjunto de datos (-1 representa ruido)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

# Visualizar los resultados
plt.scatter(scaled_locations[:, 0], scaled_locations[:, 1], c=labels, cmap='viridis')
plt.xlabel("Latitud Estandarizada")
plt.ylabel("Longitud Estandarizada")
plt.title("Clustering de Usuarios con DBSCAN")
plt.show()

# Análisis de los resultados
df = pd.DataFrame(locations, columns=['Latitud', 'Longitud'])
df['Cluster'] = labels
print("Número de clusters estimados:", n_clusters)
print("Número de puntos de ruido (ruido):", n_noise)
print(df.groupby('Cluster').mean())
