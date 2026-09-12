"""
Unsupervised Learning and Clustering Analysis - Wine Dataset
Week 3 Task: Clustering Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, silhouette_samples, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score, confusion_matrix
)
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 10

IMG = "/home/claude/clustering_project/images"
CLUSTER_COLORS = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#af7aa1"]

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
true_labels = data.target
true_names = data.target_names
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("\nFeature scale comparison (min/max):")
print(df.agg(["min", "max"]).T)

# ---------------------------------------------------------------
# 2. PREPROCESSING: STANDARDIZATION
# ---------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)
print("\nPost-scaling mean (should be ~0):\n", X_scaled_df.mean().round(2).head())
print("\nPost-scaling std (should be ~1):\n", X_scaled_df.std().round(2).head())

# ---------------------------------------------------------------
# FIGURE 1: Feature scale comparison before/after scaling (boxplots)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
df.boxplot(ax=axes[0], rot=90, grid=False)
axes[0].set_title("Before Scaling (Raw Units)", fontsize=12)
axes[0].set_ylabel("Value")
X_scaled_df.boxplot(ax=axes[1], rot=90, grid=False)
axes[1].set_title("After StandardScaler", fontsize=12)
axes[1].set_ylabel("Z-score")
fig.suptitle("Figure 1: Effect of Standardization on Feature Scales", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{IMG}/fig1_scaling_comparison.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 3. DETERMINE OPTIMAL K: ELBOW METHOD + SILHOUETTE SCORE
# ---------------------------------------------------------------
k_range = range(2, 11)
inertias = []
sil_scores = []
db_scores = []
ch_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))

print("\nK / Inertia / Silhouette / Davies-Bouldin / Calinski-Harabasz")
for k, i, s, d, c in zip(k_range, inertias, sil_scores, db_scores, ch_scores):
    print(f"{k}: {i:.1f} / {s:.3f} / {d:.3f} / {c:.1f}")

# ---------------------------------------------------------------
# FIGURE 2: Elbow plot + Silhouette score plot (side by side)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(k_range), inertias, marker="o", color="#4e79a7", linewidth=2)
axes[0].set_title("Elbow Method: Inertia vs. k", fontsize=12)
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")
axes[0].axvline(3, color="#e15759", linestyle="--", alpha=0.7, label="Chosen k=3")
axes[0].legend()

axes[1].plot(list(k_range), sil_scores, marker="o", color="#59a14f", linewidth=2)
axes[1].set_title("Silhouette Score vs. k", fontsize=12)
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Average Silhouette Score")
axes[1].axvline(3, color="#e15759", linestyle="--", alpha=0.7, label="Chosen k=3")
axes[1].legend()

fig.suptitle("Figure 2: Selecting the Optimal Number of Clusters", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{IMG}/fig2_elbow_silhouette.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 4. FINAL K-MEANS MODEL (k=3)
# ---------------------------------------------------------------
K_FINAL = 3
kmeans = KMeans(n_clusters=K_FINAL, init="k-means++", n_init=10, random_state=42)
km_labels = kmeans.fit_predict(X_scaled)
df["KMeans_Cluster"] = km_labels

km_sil = silhouette_score(X_scaled, km_labels)
km_db = davies_bouldin_score(X_scaled, km_labels)
km_ch = calinski_harabasz_score(X_scaled, km_labels)
km_ari = adjusted_rand_score(true_labels, km_labels)
print(f"\nK-Means (k=3): Silhouette={km_sil:.3f}, Davies-Bouldin={km_db:.3f}, "
      f"Calinski-Harabasz={km_ch:.1f}, ARI vs. true cultivar={km_ari:.3f}")

# ---------------------------------------------------------------
# FIGURE 3: Per-sample silhouette plot
# ---------------------------------------------------------------
sample_sil = silhouette_samples(X_scaled, km_labels)
fig, ax = plt.subplots(figsize=(7.5, 6))
y_lower = 10
for i in range(K_FINAL):
    cluster_sil = np.sort(sample_sil[km_labels == i])
    size = cluster_sil.shape[0]
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                      facecolor=CLUSTER_COLORS[i], edgecolor=CLUSTER_COLORS[i], alpha=0.8)
    ax.text(-0.05, y_lower + 0.5 * size, f"Cluster {i}", fontsize=10)
    y_lower = y_upper + 10
ax.axvline(km_sil, color="red", linestyle="--", label=f"Average score = {km_sil:.3f}")
ax.set_title("Figure 3: Silhouette Plot for K-Means (k=3)", fontsize=13, fontweight="bold")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster / Sample Index")
ax.set_yticks([])
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{IMG}/fig3_silhouette_plot.png")
plt.close()

# ---------------------------------------------------------------
# 5. PCA FOR VISUALIZATION
# ---------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_
print(f"\nPCA explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}, "
      f"Total={explained_var.sum():.3f}")

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

# ---------------------------------------------------------------
# FIGURE 4: PCA scatter colored by K-Means cluster
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 6))
for i in range(K_FINAL):
    mask = km_labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=55, alpha=0.75,
               color=CLUSTER_COLORS[i], edgecolor="black", linewidth=0.4, label=f"Cluster {i}")
centers_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], s=280, marker="X", color="black",
           edgecolor="white", linewidth=1.5, label="Centroids", zorder=5)
ax.set_title("Figure 4: K-Means Clusters Visualized via PCA", fontsize=13, fontweight="bold")
ax.set_xlabel(f"Principal Component 1 ({explained_var[0]*100:.1f}% variance)")
ax.set_ylabel(f"Principal Component 2 ({explained_var[1]*100:.1f}% variance)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{IMG}/fig4_pca_kmeans.png")
plt.close()

# ---------------------------------------------------------------
# FIGURE 5: PCA scatter colored by TRUE cultivar (for comparison)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 6))
for i, name in enumerate(true_names):
    mask = true_labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=55, alpha=0.75,
               color=CLUSTER_COLORS[i], edgecolor="black", linewidth=0.4, label=name)
ax.set_title("Figure 5: True Cultivar Labels Visualized via PCA", fontsize=13, fontweight="bold")
ax.set_xlabel(f"Principal Component 1 ({explained_var[0]*100:.1f}% variance)")
ax.set_ylabel(f"Principal Component 2 ({explained_var[1]*100:.1f}% variance)")
ax.legend(title="True Cultivar")
plt.tight_layout()
plt.savefig(f"{IMG}/fig5_pca_true_labels.png")
plt.close()

# ---------------------------------------------------------------
# 6. HIERARCHICAL CLUSTERING
# ---------------------------------------------------------------
# Dendrogram using Ward linkage
Z = linkage(X_scaled, method="ward")

fig, ax = plt.subplots(figsize=(11, 5.5))
dendrogram(Z, ax=ax, color_threshold=Z[-(K_FINAL - 1), 2], no_labels=True)
ax.axhline(Z[-(K_FINAL - 1), 2], color="#e15759", linestyle="--",
           label=f"Cut for k={K_FINAL} clusters")
ax.set_title("Figure 6: Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=13, fontweight="bold")
ax.set_xlabel("Sample Index (leaves collapsed)")
ax.set_ylabel("Ward Distance")
ax.legend()
plt.tight_layout()
plt.savefig(f"{IMG}/fig6_dendrogram.png")
plt.close()

agglo = AgglomerativeClustering(n_clusters=K_FINAL, linkage="ward")
agglo_labels = agglo.fit_predict(X_scaled)
df["Hierarchical_Cluster"] = agglo_labels

agglo_sil = silhouette_score(X_scaled, agglo_labels)
agglo_db = davies_bouldin_score(X_scaled, agglo_labels)
agglo_ari = adjusted_rand_score(true_labels, agglo_labels)
agree_with_kmeans = adjusted_rand_score(km_labels, agglo_labels)
print(f"\nHierarchical (Ward, k=3): Silhouette={agglo_sil:.3f}, Davies-Bouldin={agglo_db:.3f}, "
      f"ARI vs. true cultivar={agglo_ari:.3f}")
print(f"Agreement (ARI) between K-Means and Hierarchical labels: {agree_with_kmeans:.3f}")

# ---------------------------------------------------------------
# FIGURE 7: PCA scatter colored by Hierarchical cluster
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 6))
for i in range(K_FINAL):
    mask = agglo_labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=55, alpha=0.75,
               color=CLUSTER_COLORS[i], edgecolor="black", linewidth=0.4, label=f"Cluster {i}")
ax.set_title("Figure 7: Hierarchical Clusters Visualized via PCA", fontsize=13, fontweight="bold")
ax.set_xlabel(f"Principal Component 1 ({explained_var[0]*100:.1f}% variance)")
ax.set_ylabel(f"Principal Component 2 ({explained_var[1]*100:.1f}% variance)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{IMG}/fig7_pca_hierarchical.png")
plt.close()

# ---------------------------------------------------------------
# 7. CLUSTER VS TRUE LABEL CONTINGENCY (K-Means)
# ---------------------------------------------------------------
cm = confusion_matrix(true_labels, km_labels)
cm_df = pd.DataFrame(cm, index=[f"True: {n}" for n in true_names],
                      columns=[f"Cluster {i}" for i in range(K_FINAL)])
print("\nK-Means cluster vs. true cultivar contingency table:\n", cm_df)

fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Number of Samples"}, ax=ax)
ax.set_title("Figure 8: K-Means Clusters vs. True Cultivar Labels", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMG}/fig8_contingency_heatmap.png")
plt.close()

# ---------------------------------------------------------------
# 8. CLUSTER PROFILING: mean feature values per cluster (heatmap)
# ---------------------------------------------------------------
feature_cols = list(data.feature_names)
cluster_profile = df.groupby("KMeans_Cluster")[feature_cols].mean()
cluster_profile_z = (cluster_profile - df[feature_cols].mean()) / df[feature_cols].std()

fig, ax = plt.subplots(figsize=(11, 4.5))
sns.heatmap(cluster_profile_z, annot=cluster_profile.round(2), fmt="", cmap="coolwarm", center=0,
            linewidths=0.5, cbar_kws={"label": "Z-score relative to overall mean"}, ax=ax)
ax.set_title("Figure 9: Cluster Feature Profiles (K-Means, k=3)", fontsize=13, fontweight="bold")
ax.set_xlabel("Feature")
ax.set_ylabel("Cluster")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{IMG}/fig9_cluster_profile_heatmap.png")
plt.close()

print("\nCluster profile (raw means):\n", cluster_profile.round(2))
print("\nCluster sizes:\n", df["KMeans_Cluster"].value_counts().sort_index())

# ---------------------------------------------------------------
# FIGURE 10: Boxplots of top discriminating features by cluster
# ---------------------------------------------------------------
top_feats = ["flavanoids", "color_intensity", "proline", "alcohol",
             "od280/od315_of_diluted_wines", "malic_acid"]
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, feat in zip(axes.flat, top_feats):
    sns.boxplot(data=df, x="KMeans_Cluster", y=feat, hue="KMeans_Cluster", legend=False,
                palette=CLUSTER_COLORS[:K_FINAL], ax=ax)
    ax.set_title(feat, fontsize=11)
    ax.set_xlabel("Cluster")
fig.suptitle("Figure 10: Key Feature Distributions by Cluster", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{IMG}/fig10_boxplots_by_cluster.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# FIGURE 11: Radar chart of cluster profiles (standardized)
# ---------------------------------------------------------------
radar_feats = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "flavanoids",
               "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"]
angles = np.linspace(0, 2 * np.pi, len(radar_feats), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
for i in range(K_FINAL):
    values = cluster_profile_z.loc[i, radar_feats].tolist()
    values += values[:1]
    ax.plot(angles, values, color=CLUSTER_COLORS[i], linewidth=2, label=f"Cluster {i}")
    ax.fill(angles, values, color=CLUSTER_COLORS[i], alpha=0.15)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_feats, fontsize=9)
ax.set_title("Figure 11: Standardized Cluster Profiles (Radar Chart)", fontsize=13, fontweight="bold", y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(f"{IMG}/fig11_radar_chart.png")
plt.close()

# ---------------------------------------------------------------
# FIGURE 12: Cluster sizes bar chart
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
sizes = df["KMeans_Cluster"].value_counts().sort_index()
bars = ax.bar([f"Cluster {i}" for i in sizes.index], sizes.values,
              color=CLUSTER_COLORS[:K_FINAL], edgecolor="black", linewidth=0.6)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, str(int(b.get_height())),
            ha="center", fontsize=11, fontweight="bold")
ax.set_title("Figure 12: K-Means Cluster Sizes", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(f"{IMG}/fig12_cluster_sizes.png")
plt.close()

print("\nAll figures generated successfully.")

# Save summary CSVs for report reference
cluster_profile.to_csv("/home/claude/clustering_project/cluster_profile_means.csv")
pd.DataFrame({
    "k": list(k_range), "inertia": inertias, "silhouette": sil_scores,
    "davies_bouldin": db_scores, "calinski_harabasz": ch_scores
}).to_csv("/home/claude/clustering_project/k_selection_metrics.csv", index=False)
