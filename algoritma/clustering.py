import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator

def elbow(dataset_normalized):
    # Tentukan rentang jumlah klaster yang akan diuji
    wcss = []  # Within-Cluster Sum of Squares
    K = range(1, 11)  # Uji untuk klaster 1 sampai 10

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(dataset_normalized)
        wcss.append(kmeans.inertia_)

    # Menentukan elbow point secara otomatis
    knee = KneeLocator(K, wcss, curve="convex", direction="decreasing")
    optimal_k = knee.knee
    return wcss, K, optimal_k

def elbow_visualization(wcss, K, optimal_k):
    # Plot hasil WCSS untuk melihat elbow
    plt.figure(figsize=(8, 5))
    plt.plot(K, wcss, marker='o', linestyle='--', label='WCSS')
    plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal K = {optimal_k}')
    plt.title('Elbow Method untuk Menentukan Jumlah Klaster Optimal')
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(K)
    plt.legend()
    plt.grid()
    plt.show()

def kmeans_clustering(dataset_normalized, optimal_k):
    # Klastering menggunakan jumlah klaster optimal
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(dataset_normalized)
    # Tambahkan hasil klastering ke dataset
    return kmeans.labels_
    # Tampilkan hasil klastering

#visualisasi klaster
def clustering_visualization(dataset_main_fitur):
    #visualisasi dengan 2 fitur utama
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset_main_fitur.iloc[:, 0], dataset_main_fitur.iloc[:, 1],
                c=dataset_main_fitur['cluster'], cmap='viridis', s=20)
    plt.title('Hasil Klastering dengan K-Means')
    plt.xlabel('Tahun')
    plt.ylabel('Percentase Penurunan Harga')
    plt.colorbar(label='Cluster')
    plt.show()

#analisa klaster
def clustering_analysis(dataset_cluster_analyze):
    print("\n\t\tAnalisa klaster berdasarkan kepemilikan penjual")
    print(dataset_cluster_analyze.groupby('cluster')[['seller_type','owner']].value_counts())
    
    for column in dataset_cluster_analyze.columns:
        if dataset_cluster_analyze[column].dtypes != 'object':
            print(f"\n\t\tdistribusi klaster berdasarkan {column}")
            print(dataset_cluster_analyze.groupby('cluster')[column].describe())