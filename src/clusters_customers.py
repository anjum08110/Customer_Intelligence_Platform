from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]

IN_Path= ROOT / "outputs" /"customers_behavioral.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CLUSTERS_OUT = OUT_DIR / "customer_clusters.csv"
PCA_OUT = OUT_DIR / "customer_pca.csv"

df = pd.read_csv(IN_Path)

if "CustomerID" not in df.columns:
    df= df.rename(columns={df.columns[0]: "CustomerID"})

customer_ids= df["CustomerID"].astype(str)

feature_col = [c for c in df.columns if  c != "CustomerID"]
X = df[feature_col].copy()

scaler = StandardScaler()
X_scaled= scaler.fit_transform(X)
#KMeans
scores = []
ks = range(3,9)

for k in ks:
    km= KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score= silhouette_score(X_scaled,labels)
    scores.append((k,score))

best_k, best_score = max(scores, key= lambda t: t[1])
print("Silhouette scores:", scores)
print("Best k:", best_k, "Best silhouette:", round(best_score, 4))

kmeans= KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

clusters_df = df.copy()
clusters_df["Cluster"] = cluster_labels
clusters_df.to_csv(CLUSTERS_OUT, index= False)
print("Saved:",CLUSTERS_OUT)

pca_df = pd.DataFrame({
    "CustomerID": customer_ids,
    "PC1": X_pca[:,0],
    "PC2": X_pca[:,1],
    "Cluster": cluster_labels
})

pca_df.to_csv(PCA_OUT, index= False)
print("Saved:", PCA_OUT)

print("PCA explained variance ratio:", pca.explained_variance_ratio_)
print(pca_df.head(10))