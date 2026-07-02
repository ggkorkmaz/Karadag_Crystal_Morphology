#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
os.environ["OMP_NUM_THREADS"] = "4"

import warnings
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# ALL FILES
# =========================

files_all = {
    "GS11_Host_Plag": "GS11ANAKAYA plagResults.csv",
    "GS11_Host_Amph": "GS11ANAKAYA AMPHIBOLEResults.csv",
    "GS11_Enclave_Plag": "GS11ANKLAVPLAGResults.csv",
    "GS11_Enclave_Amph": "GS11ANKLAV AMPHIBOLEResults.csv",

    "GS16_Host_Plag": "GS16ANAKAYA PLAGesults.csv",
    "GS16_Host_Amph": "GS16ANAKAYA AMFIBOL esults.csv",
    "GS16_Enclave_Plag": "GS16ANKLAV PLAGesults.csv",
    "GS16_Enclave_Amph": "GS16ANKLAV AMFIBOLResults.csv",

    "GS17_Host_Plag": "gs17anakaya plagResults.csv",
    "GS17_Host_Amph": "gs17anakayamphiboleResults.csv",
    "GS17_Enclave_Plag": "gs17anklav plagResults.csv",
    "GS17_Enclave_Amph": "gs17anklavamphiboleResults.csv",
}

outdir = Path("Karadag_Pairwise_Crystal_Morphology")
outdir.mkdir(parents=True, exist_ok=True)

# =========================
# FUNCTION
# =========================

def run_pairwise_pca(sample_name, files_all):
    print(f"\n==============================")
    print(f"Running analysis for {sample_name}")
    print(f"==============================")

    selected_files = {k: v for k, v in files_all.items() if k.startswith(sample_name)}

    all_data = []

    for name, path in selected_files.items():
        if not Path(path).exists():
            print("Missing:", path)
            continue

        df = pd.read_csv(path)

        sample, domain, mineral = name.split("_", 2)

        df["Sample"] = sample
        df["Domain"] = domain
        df["Mineral"] = mineral
        df["Population"] = name

        all_data.append(df)

    if len(all_data) == 0:
        print(f"No files found for {sample_name}")
        return None

    data = pd.concat(all_data, ignore_index=True)

    # Convert numeric columns
    data["Feret_um"] = pd.to_numeric(data["Feret"], errors="coerce") * 1000
    data["Area"] = pd.to_numeric(data["Area"], errors="coerce")
    data["Circ."] = pd.to_numeric(data["Circ."], errors="coerce")

    for col in ["Major", "Minor", "MinFeret", "AR", "Round", "Solidity"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Remove problematic measurements
    data = data[(data["Feret_um"] >= 50) & (data["Feret_um"] <= 5000)]

    # Feature set
    features = ["Area", "Feret_um", "Circ."]

    for col in ["Major", "Minor", "MinFeret", "AR", "Round", "Solidity"]:
        if col in data.columns and data[col].notna().sum() > 10:
            features.append(col)

    print("Used features:", features)

    ml = data.dropna(subset=features).copy()
    ml = ml.replace([np.inf, -np.inf], np.nan).dropna(subset=features)

    X = ml[features].values
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    ml["PC1"] = pcs[:, 0]
    ml["PC2"] = pcs[:, 1]

    print("Explained variance:", pca.explained_variance_ratio_)

    # KMeans k=2-5
    silhouette_results = []

    for k in range(2, 6):
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        silhouette_results.append([k, sil])

    sil_df = pd.DataFrame(silhouette_results, columns=["k", "Silhouette"])
    print(sil_df)

    best_k = int(sil_df.loc[sil_df["Silhouette"].idxmax(), "k"])
    print("Best k:", best_k)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    ml["Cluster"] = kmeans.fit_predict(X_scaled)

    # Save output tables
    sample_outdir = outdir / sample_name
    sample_outdir.mkdir(exist_ok=True)

    ml.to_csv(sample_outdir / f"{sample_name}_crystal_morphology_PCA_KMeans.csv", index=False)
    sil_df.to_csv(sample_outdir / f"{sample_name}_silhouette_scores.csv", index=False)

    # Figure 1: Host vs Enclave
    plt.figure(figsize=(8, 6))
    for domain in ["Host", "Enclave"]:
        sub = ml[ml["Domain"] == domain]
        plt.scatter(sub["PC1"], sub["PC2"], label=domain, alpha=0.65)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(f"{sample_name}: PCA of crystal morphology - Host vs Enclave")
    plt.legend()
    plt.tight_layout()
    plt.savefig(sample_outdir / f"{sample_name}_PCA_Host_vs_Enclave.png", dpi=300)
    plt.show()

    # Figure 2: Plag vs Amph
    plt.figure(figsize=(8, 6))
    for mineral in ["Plag", "Amph"]:
        sub = ml[ml["Mineral"] == mineral]
        plt.scatter(sub["PC1"], sub["PC2"], label=mineral, alpha=0.65)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(f"{sample_name}: PCA of crystal morphology - Plagioclase vs Amphibole")
    plt.legend()
    plt.tight_layout()
    plt.savefig(sample_outdir / f"{sample_name}_PCA_Plag_vs_Amph.png", dpi=300)
    plt.show()

    # Figure 3: KMeans clusters
    plt.figure(figsize=(8, 6))
    for c in sorted(ml["Cluster"].unique()):
        sub = ml[ml["Cluster"] == c]
        plt.scatter(sub["PC1"], sub["PC2"], label=f"Cluster {c}", alpha=0.65)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(f"{sample_name}: KMeans clustering of crystal morphology, k={best_k}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(sample_outdir / f"{sample_name}_PCA_KMeans_clusters.png", dpi=300)
    plt.show()

    # Figure 4: Feret boxplot
    order = sorted(ml["Population"].unique())

    plt.figure(figsize=(10, 6))
    box_data = [ml[ml["Population"] == pop]["Feret_um"] for pop in order]
    plt.boxplot(box_data, tick_labels=order, showfliers=False)
    plt.ylabel("Feret length (µm)")
    plt.title(f"{sample_name}: Crystal size comparison among host–enclave populations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(sample_outdir / f"{sample_name}_Feret_boxplot.png", dpi=300)
    plt.show()

    # Summary table
    summary = ml.groupby(["Sample", "Domain", "Mineral"]).agg(
        N=("Feret_um", "count"),
        Feret_mean_um=("Feret_um", "mean"),
        Feret_median_um=("Feret_um", "median"),
        Feret_min_um=("Feret_um", "min"),
        Feret_max_um=("Feret_um", "max"),
        Area_mean=("Area", "mean"),
        Circ_mean=("Circ.", "mean"),
        PC1_mean=("PC1", "mean"),
        PC2_mean=("PC2", "mean")
    ).reset_index()

    summary.to_csv(sample_outdir / f"{sample_name}_crystal_morphology_summary.csv", index=False)

    print(summary)
    
    return ml, sil_df, summary

# =========================
# RUN ALL PAIRS
# =========================

results = {}

for sample in ["GS11", "GS16", "GS17"]:
    results[sample] = run_pairwise_pca(sample, files_all)

print("DONE. Outputs saved in:", outdir)

