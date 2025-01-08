# Import library
from data.data_preprocessing import *
from algoritma.clustering import *
from sklearn.metrics import silhouette_score


#path file dataset
path = "data/dataset/BIKE DETAILS.csv"
#dataset
dataset = load_dataset(path)
print("\n\t\tLoading Dataset")
print(dataset)

print("\n\t\tExploratory Data Analysis (EDA)")
#Informasi umum dataset
print("\n\tinformasi umum dataset)")
dataset.info()
#kolom kategorikal dan numerikal
print("\n\tkolom kategorical dan numerikal")
numerical_column = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_column = dataset.select_dtypes(include=['object']).columns.tolist()
print("Kolom Kategorikal : ",categorical_column)
print("Kolom Numerikal : ",numerical_column)
#missing value column
print("\n\tKolom dengan data hilang")
missing_value_column = dataset.columns[dataset.isnull().any()].tolist()
count_missing_value_column = dataset[missing_value_column].isna().sum()
print("Kolom dengan data hilang : ",missing_value_column)
print("Jumlah data hilang : \n",count_missing_value_column)
if count_missing_value_column.any():    print("Perlu penanganan data kosong! Pendekatan yang diusulkan menggunakan metode KNN Imputer")
#penanganan missing value
dataset = add_missing_value(dataset)
#Statistik deskriptif dataset
print("\n\tstatistik deskriptif dataset")
print(dataset.describe().round(2))
#inisiasi variabel distribusi dan korelasi 
# Tentukan jumlah bin yang diinginkan
num_bins = 5
#analisis distribusi data numerical
print("\n\t\tAnalisa distribusi data numerical") 
numerical_distribution_analysis(dataset, numerical_column, num_bins)
#Analisa koerelasi
print("\n\t\tAnalisa korelasi data numerical") 
numerical_correlation_analysis(dataset, numerical_column)
#kolom kategori yang diperlukan
dataset_kategorical = dataset[categorical_column].copy()
dataset_kategorical = dataset_kategorical.drop(['name'], axis=1)
categorical_column = dataset_kategorical.columns
#visualisasi dan analisa distribusi data kategorikal
print("\n\t\tAnalisa distribusi data kategorikal") 
kategorical_distribution_analysis(dataset_kategorical)
#analisa korelasi kolom kategorikal
print("\n\t\tAnalisa korelasi data numerical") 
categorical_correlation_analysis(categorical_column, dataset_kategorical)

#preprocessing data
#Data menampilkan data setelah data baru dimasukkan pada missing value
dataset_normalized = dataset.copy()
print("\n\t\tpreprocessing data")
print("\n\t\tData setelah penanganan missing value")
print(dataset_normalized)
#menghapus text owner
dataset_normalized['owner'] = dataset_normalized['owner'].str.replace(' owner', '')
print("\n\t\thapus fitur fitur name, selling price dan ex showroom price ganti dengan fitur presentase selisih")
#proses menghitung prosentase selisih selling price
dataset_normalized = percentage_column(dataset)
dataset_main_fitur = dataset_normalized[['year', 'percentage_diff_sell_price']].copy()
#menampilan data setelah setelah fitur dihilangkan dan penambahan fitur baru
print(dataset_normalized)
#normalisasi data
dataset_numerical_normalized, dataset_categorical_normalized, dataset_normalized = normalize(dataset_normalized)
#hasil normalisasi data numerik
print("\n\t\thasil normalisasi numarikal")
print(dataset_numerical_normalized)
#hasil normalisasi data kagorik
print("\n\t\thasil normalisasi kategorikal")
print(dataset_categorical_normalized)
#hasil normalisasi semua
print("\n\t\thasil normalisasi semua data")
print(dataset_normalized)

#pemilihan jumlah klaster terbaik menggunakan elbow
wcss, K, optimal_k = elbow(dataset_normalized)
#visualisasi elbow
#elbow_visualization(wcss, K, optimal_k)
print("K optimal : ",optimal_k)
#kmeans klastering
labels = kmeans_clustering(dataset_normalized, optimal_k)
dataset_normalized['cluster'] = labels
dataset_main_fitur['cluster'] = labels
#hasil klastering
print("hasil klastering")
print(dataset_normalized)
#visualisasi klastering dengan fitur utama
#clustering_visualization(dataset_main_fitur)
#Hitung Silhouette Coefficient
score = silhouette_score(dataset_normalized.drop(columns=['cluster']), labels)
print(f"Silhouette Coefficient untuk klastering dengan {optimal_k} klaster adalah: {score:.4f}")
#tahap analisa
dataset_cluster_analyze = dataset[['year', 'seller_type', 'owner', 'km_driven']].copy()
dataset_cluster_analyze = pd.concat([dataset_cluster_analyze, dataset_main_fitur['percentage_diff_sell_price'], dataset_main_fitur['cluster']], axis=1)
print(dataset_cluster_analyze)

#analisa klaster
clustering_analysis(dataset_cluster_analyze)

print("\n\t\tKesimpulan : ")
print("Klaster 0 :")
print("Pada klaster ini, harga jual kembali relatif stabil. Tampaknya hal ini disebabkan dari data penjualan sebagai beriku:")
print("- Penjual didominasi individu pemilik pertama")
print("- KM berkendara relatif lebih rendah")
print("Klaster 0 :")
print("Pada klaster ini, harga jual kembali relatif lebih rendah dari pada klaster 0. Tampaknya hal ini disebabkan dari data penjualan sebagai beriku:")
print("- Penjual didominasi individu, namun berasal tangan pemilik kedua dst")
print("- KM berkendara relatif lebih tinggi")
print("Dengan demikian penjualan dari tangan pertama dengan kilometer rendah memberikan harga jual kembali yang stabil. Namun tahun pembuatan motor tidak berpengaruh signifikan karena distribusi datanya merata")

dataset_normalized.to_csv('data/dataset/dataset.csv', index=False)  