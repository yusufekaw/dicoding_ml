#import library
from data.data_preprocessing import *
from algoritma.classification import *


#path file dataset
path = "data/dataset/dataset.csv"
#dataset
dataset = load_dataset(path)
print("\n\t\tLoading Dataset")
print(dataset)

#splitting dataset
X, y, X_train, X_test, y_train, y_test = split_dataset(dataset, 'cluster')
#Menampilkan splitting dataset
print("\n\t\tSplit Dataset")
print("\n\tData Training")
print(X_train)
print("\n\tData Testing")
print(X_test)

# Membuat model regresi logistik
print("\n\t\tMembangun model dengan regresi logistik biner")
print("\n\tKelas yang dihasilkan dari proses clustering berdasarkan K optimal menggunakan elbow adalah 2 sehingga menghasilkan 2 kelas bertipe biner. Maka, algoritma regresi logistik biner di pilih untuk modeling klasifikasi.")
#model
y_train_pred, y_test_pred = model(X_train, y_train, X_test)
#hasil modeling
train_set, test_set = train_test_set(X_train, X_test, y_train, y_train_pred, y_test, y_test_pred)
print("hasil training model pada data training")
print(train_set)

#evaluasi model
cm_train, accuracy_train, report_train = evaluation(y_train, y_train_pred)
#print hasil
print("\nEvaluasi pada Data Training:")
cm_visualization(cm_train)
print("\nAkurasi Model :", accuracy_train)
print("\nModeling Report:\n", report_train)

#cetak hasil prediksi klasifikasi
print("hasil prediksi klasifikasi")
print(test_set)

#evaluasi klasifikasi
cm_test, accuracy_test, report_test = evaluation(y_test, y_test_pred)
#print hasil
print("\nEvaluasi pada Data Testing:")
cm_visualization(cm_test)
print("\nAkurasi Prediksi :", accuracy_test)
print("\nClassification Report :\n", report_test)

print("\n\t\tKesimpulan")
print("Akurasi mencapai 100% (ditunjukkan dengan nilai 1 dalam hasil evaluasi), menunjukkan model telah bekerja dengan sempurna dalam mengklasifikasi data penjualan motor.")
print("\n\tRekomendasi :")
print("Dibutuhkan data lebih banyak lagi untuk mengetahui kinerja model lebih lanjut.")