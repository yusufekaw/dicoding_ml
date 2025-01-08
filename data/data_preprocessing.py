import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

#fungsi untuk memuat dataset
def load_dataset(path):
    #mendapatkan dataset
    dataset = pd.read_csv(path)
    return dataset

#split dataset
def split_dataset(dataset, column):
    # Memisahkan fitur (X) dan target (y)
    X = dataset.drop(columns=[column])
    y = dataset[column]
    # Split data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, y, X_train, X_test, y_train, y_test

#analisis data numerical / Analisis Distribusi dan Korelasi
#analisa distribusi data numerical
def numerical_distribution_analysis(dataset, numerical_column, num_bins):
    for column in numerical_column:
        plt.figure(figsize=(8, 6))
        # Membuat histogram
        counts, bin_edges = np.histogram(dataset[column], bins=num_bins)
        
        # Plot histogram sebagai bar chart
        plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), color='skyblue', edgecolor='black', alpha=0.7, label="Histogram")
        
        # Tambahkan garis yang menghubungkan titik-titik tengah bin
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Titik tengah bin
        plt.plot(bin_centers, counts, marker='o', color='red', linestyle='-', label="Garis penghubung")
        
        # Pengaturan plot
        plt.title(f"Distribusi data {column.capitalize()}", fontsize=14)
        plt.xlabel(column.capitalize(), fontsize=12)
        plt.ylabel("Frekuensi", fontsize=12)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # Analisa teks di console
        print(f"--- Distribusi data {column.capitalize()} ---")
        for i in range(len(counts)):
            if counts[i] > 0:  # Abaikan bin dengan 0 data
                if i>0:
                    print(f"Rentang {int(bin_edges[i]+1)}-{int(bin_edges[i + 1])}: {counts[i]} data")
                else:
                    print(f"Rentang {int(bin_edges[i])}-{int(bin_edges[i + 1])}: {counts[i]} data")
        # Menentukan rentang dengan data terbanyak dan terendah
        min_count_index = np.argmin(counts)  # Indeks dari count terkecil
        max_count_index = np.argmax(counts)  # Indeks dari count terbesar    
        # Menampilkan rentang dan nilai terendah
        print(f"Terendah pada rentang {int(bin_edges[min_count_index])}-{int(bin_edges[min_count_index+1])}, {counts[min_count_index]} data")    
        # Menampilkan rentang dan nilai tertinggi
        print(f"Tertinggi pada rentang {int(bin_edges[max_count_index])}-{int(bin_edges[max_count_index+1])}, {counts[max_count_index]} data")
        print("\n")

#analisa korelasi
def numerical_correlation_analysis(dataset, numerical_column):
    # Hitung matriks korelasi untuk kolom numerikal
    correlation_matrix = dataset[numerical_column].corr()

    # Visualisasi matriks korelasi menggunakan heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
    plt.title("Matriks Korelasi Variabel Numerikal", fontsize=16)
    plt.show()

    # Analisis Korelasi
    print("Analisis Korelasi:")
    high_corr_pairs = correlation_matrix.stack().reset_index()
    high_corr_pairs.columns = ['Var1', 'Var2', 'Correlation']
    high_corr_pairs = high_corr_pairs[(high_corr_pairs['Correlation'] != 1.0) & (high_corr_pairs['Correlation'].abs() > 0.5)]
    high_corr_pairs = high_corr_pairs.sort_values(by='Correlation', ascending=False)
    if not high_corr_pairs.empty:
        print("Pasangan variabel dengan korelasi tinggi:")
        print(high_corr_pairs)
    else:
        print("Tidak ada pasangan variabel dengan korelasi tinggi (>|0.5|).")

#analisa distribusi data kategorikal
def kategorical_distribution_analysis(dataset_kategorical):
    # Visualisasi Diagram Batang untuk Kolom Kategorikal
    for column in dataset_kategorical.columns:
        plt.figure(figsize=(8, 6))
        
        # Hitung jumlah kategori
        category_counts = dataset_kategorical[column].value_counts()
        
        # Buat bar chart dengan warna bervariasi
        colors = sns.color_palette("pastel", len(category_counts))
        sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors, hue=category_counts.index, dodge=False)
        
        # Pengaturan plot
        plt.title(f"Distribusi kategori {column.capitalize()}", fontsize=14)
        plt.xlabel(column.capitalize(), fontsize=12)
        plt.ylabel("Frekuensi", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend([], [], frameon=False)  # Hilangkan legend karena hanya satu kategori
        plt.tight_layout()
        
        # Tampilkan plot
        plt.show()
        
        # Analisa data
        print(f"\n\t--- Distribusi kategor {column.capitalize()} ---")
        print(dataset_kategorical[column].value_counts())
        most_common_category = category_counts.idxmax()
        most_common_count = category_counts.max()
        least_common_category = category_counts.idxmin()
        least_common_count = category_counts.min()
        
        print(f"Kategori terbanyak: {most_common_category} dengan {most_common_count} data.")
        print(f"Kategori paling sedikit: {least_common_category} dengan {least_common_count} data.\n")

def categorical_correlation_analysis(categorical_column, dataset_kategorical):
    # Iterasi semua pasangan kombinasi kolom kategorikal
    for i in range(len(categorical_column)):
        for j in range(i + 1, len(categorical_column)):
            col1 = categorical_column[i]
            col2 = categorical_column[j]

            # Buat crosstab antar kolom
            crosstab_data = pd.crosstab(dataset_kategorical[col1], dataset_kategorical[col2])

            # Visualisasi heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(crosstab_data, annot=True, fmt="d", cmap="Blues", cbar=True)
            plt.title(f"Heatmap Crosstab: {col1} vs {col2}", fontsize=14)
            plt.xlabel(col2)
            plt.ylabel(col1)
            plt.show()

            # Analisis hasil crosstab
            max_value = crosstab_data.values.max()
            min_value = crosstab_data[crosstab_data > 0].values.min()  # Hindari nilai 0
            max_position = crosstab_data.stack().idxmax()
            min_position = crosstab_data[crosstab_data > 0].stack().idxmin()

            print(f"--- Analisa untuk {col1} vs {col2} ---")
            print(f"Kombinasi kategori terbanyak: {max_position} dengan {max_value} data.")
            print(f"Kombinasi kategori paling sedikit (bukan 0): {min_position} dengan {min_value} data.")
#analisis data numerical / Analisis Distribusi dan Korelasi

#fungsi untuk penanganan data hilang
def add_missing_value(dataset):
    #menggunakan KNNImputer
    imputer = KNNImputer(n_neighbors=7)
    dataset[['ex_showroom_price']] = imputer.fit_transform(dataset[['ex_showroom_price']])
    return dataset

#penambahan kolom prosentase selisih harga dealer dan harga jual dan menghapus fitur selling price dan ex showroom price
def percentage_column(dataset):
    dataset['percentage_diff_sell_price'] = abs(dataset['selling_price']-dataset['ex_showroom_price'])/dataset['ex_showroom_price']
    dataset = dataset.drop(['name', 'selling_price', 'ex_showroom_price'], axis=1)
    return dataset
    
#normalisasi data
def normalize(dataset_normalized):
    #normalisasi kolom numerik dengan mimmax
    # Identifikasi kolom numerik dengan nilai > 1
    numerical_column = dataset_normalized.select_dtypes(include=['int64', 'float64']).columns
    numerical_column_to_normalize = numerical_column[dataset_normalized[numerical_column].gt(1).any()]
    # Lakukan normalisasi
    scaler = MinMaxScaler()
    dataset_normalized[numerical_column_to_normalize] = scaler.fit_transform(dataset_normalized[numerical_column_to_normalize])
    dataset_numerical_normalized = dataset_normalized[numerical_column_to_normalize].copy()
    
    #normalisasi kategorikal
    categorical_column = dataset_normalized.select_dtypes(include=['object']).columns
    # Buat encoder
    encoder = OneHotEncoder()
    # Fit dan transform data
    encoded_column = pd.DataFrame(encoder.fit_transform(dataset_normalized[categorical_column]).toarray())
    # Dapatkan nama kategori asli
    original_column_names = encoder.get_feature_names_out()
    # Ubah nama kolom sesuai format yang diinginkan
    new_column_names = []
    for col in original_column_names:
        parts = col.split('_')
        column_name = '_'.join(parts[:-1])  # Join all parts except the last one
        value = parts[-1]
        new_column_names.append(f"{column_name}_{value}")

    encoded_column.columns = new_column_names
    # Gabungkan dengan data numerik lainnya
    dataset_normalized = dataset_normalized.drop(categorical_column, axis=1)
    dataset_normalized = pd.concat([dataset_normalized, encoded_column], axis=1)
    dataset_categorical_normalized = dataset_normalized.drop(numerical_column, axis=1)

    return dataset_numerical_normalized, dataset_categorical_normalized, dataset_normalized
