# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generar datos sintéticos
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Visualizar los datos
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title('Datos de Ejemplo para Clustering')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.show()

# Crear un modelo K-Means con 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)

# Entrenar el modelo K-Means
kmeans.fit(X)

# Obtener las etiquetas de cluster para cada punto de datos
labels = kmeans.labels_

# Obtener las coordenadas de los centroides
centroids = kmeans.cluster_centers_

# Visualizar los resultados del clustering
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, label='Centroides')
plt.title('Clustering con K-Means')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.show()
