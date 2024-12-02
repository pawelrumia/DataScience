import os
os.environ['LOKY_MAX_CPU_COUNT'] = '0'


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

np.random.seed(42)
df = pd.read_csv("marketing_campaign.csv")

missing_values = df.isnull().sum()

label_encoder = LabelEncoder()

df_categoricals = df.select_dtypes(include=["object"])
df_encoded_categoricals = df_categoricals.apply(label_encoder.fit_transform)

df[df_encoded_categoricals.columns] = df_encoded_categoricals

df_numerical = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numerical), columns=df_numerical.columns)
df[df_numerical.columns] = df_scaled

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,6))
plt.plot(range(1,11), wcss, marker="o")
plt.title("Metoda łokcia dla optymalnej liczby klastrów")
plt.xticks(np.arange(1,11,1))
plt.xlabel("Liczba klastrów")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

kl = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
optimal_clusters = kl.elbow

pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df), columns=["PC1", "PC2"])

plt.figure(figsize=(7,6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.5, c="blue")
plt.title("PCA: redukcja do dwoch wymiarów")
plt.xlabel("PC1")
plt.xlabel("PC2")
plt.show()

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_pca["KMeans_Cluster"] = kmeans.fit_predict(df_pca)

kmeans_score = silhouette_score(df_pca, kmeans.labels_)


