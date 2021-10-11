# Laporan Proyek Machine Learning - Radya Adi Anggara
---
## Domain Proyek
Dalam menentukan rumah yang akan kita beli, kita tentu akan menentukan beberapa faktor, antara lain seperti harga rumah serta komponen atau fitur pada rumah (seperti jumlah kamar tidur dan jumlah kamar mandi). Pada dataset yang telah tersedia, ada sekitar ribuan data mengenai harga rumah beserta dengan komponennya. Karena banyaknya data, kita tidak bisa memprediksi harga suatu rumah secara langsung. Oleh karena itu, kita memerlukan sebuah solusi dimana kita dapat memprediksi sebuah harga rumah berdasarkan komponen (fasilitas) yang akan kita cari nantinya.

## Business Understanding
Kita akan mencari sebuah solusi dimana kita akan memprediksi harga sebuah rumah berdasarkan komponen atau fitur yang kita miliki menggunakan teknik _predictive modelling_.
### Problem Statements
Berikut adalah _problem statements_ yang kita miliki :
- Bagaimana hubungan antar fitur dengan harga rumah ?
- Apa fitur yang paling berpengaruh terhadap harga rumah ?
- Apa model yang cocok untuk menyelesaikan problem ini ?

### Goals
Berdasarkan _problem statements_, tujuan projek ini antara lain :
- Dapat memprediksi harga rumah berdasarkan data (fitur) yang tersedia.
- Menentukan model yang paling cocok untuk memprediksi harga rumah.

### Solution statements
Berdasarkan data yang ada, dataset tersebut termasuk dalam linear regression. Oleh karena itu, kita memiliki tiga algoritma machine learning untuk menyelesaikan problem projek ini. Algoritma machine learning yang akan kita gunakan antara lain :
- K-Nearest Neighbor, yaitu algoritma yang menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k-tetangga terdekat.
- Random Forest, yaitu algoritma yang mengkombinasikan masing - masing pohon (tree) dari model Decision Tree yang baik ke dalam satu model. Penggunaan tree yang semakin banyak akan mempengaruhi akurasi yang akan didapatkan menjadi lebih baik.
- Algoritma Boosting, yaitu algoritma yang bekerja dengan membangun model dari data latih. Kemudian algoritma ini membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan hingga data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. 

## Data Understanding
Data yang akan kita gunakan adalah dataset "House Sales in King County, USA" yang didapat dari website kaggle.
>Link : https://www.kaggle.com/harlfoxem/housesalesprediction

Di dalam dataset ini, ada berbagai macam variabel (fitur), antara lain :
| Variabel | Deskripsi |
| -------- | --------- |
| id | id rumah |
| date | Tanggal rumah dijual |
| price | Harga setiap rumah |
| bedrooms | Jumlah kamar tidur |
| bathrooms | Jumlah kamar mandi |
| sqft_living | Cuplikan persegi ruang tamu interior rumah |
| sqft_lot | Cuplikan persegi lanskap |
| floors | Jumlah lantai |
| waterfront | Lokasi dekat dengan laut atau tidak (0 untuk tidak 1 untuk ya) |
| view | Seberapa bagus pemandangan di sekitar rumah |
| condition | Kondisi rumah |
| grade | kualitas konstruksi |
| sqft_above | Ukuran rata-rata ruang interior perumahan yang berada di atas permukaan tanah |
| sqft_basement | Ukuran rata-rata ruang interior perumahan yang berada di bawah permukaan tanah |
| yr_built | Tahun rumah dibangun |
| yr_renovated | Tahun rumah direnovasi |
| zipcode | Kode pos rumah |
| lat | Garis lintang |
| long | Garis bujur |
| sqft_living15 | Cuplikan persegi ruang hidup perumahan interior untuk 15 tetangga terdekat |
| sqft_lot15 | Cuplikan persegi dari tanah kavling ruang dari 15 tetangga terdekat |

Berikut adalah visualisasi persebaran data dari setiap variabel.
![N|Solid](https://github.com/radyaadi/Predictive-Analytics/blob/main/DataVisual-House/1.png?raw=true)
![N|Solid](https://github.com/radyaadi/Predictive-Analytics/blob/main/DataVisual-House/2.png?raw=true)
![N|Solid](https://github.com/radyaadi/Predictive-Analytics/blob/main/DataVisual-House/3.png?raw=true)
![N|Solid](https://github.com/radyaadi/Predictive-Analytics/blob/main/DataVisual-House/4.png?raw=true)
![N|Solid](https://github.com/radyaadi/Predictive-Analytics/blob/main/DataVisual-House/5.png?raw=true)

Berikut adalah korelasi antar fitur dengan fitur "price"
| Variabel | Nilai Korelasi |
| -------- | --------- |
| price | 1.000000 |
| sqft_living | 0.702035 | 
| grade | 0.667434 |
| sqft_above | 0.605567 |
| sqft_living15 | 0.585379 |
| bathrooms | 0.525138 |
| view | 0.397293 |
| sqft_basement | 0.323816 |
| bedrooms | 0.308350 |
| lat | 0.307003 |
| waterfront | 0.266369 |
| floors | 0.256794 |
| yr_renovated | 0.126434 |
| vsqft_lot | 0.089661 |
| sqft_lot15 | 0.082447 |
| yr_built | 0.054012 |
| condition | 0.036362 |
| long | 0.021626 |
| id | -0.016762 |
| zipcode | -0.053203 |

Berdasarkan data di atas, variabel "sqft_living" mempunyai nilai paling tinggi terhadap "price" dibandingkan dengan variabel yang lain. Kemudian, beberapa kolom dapat dihapus, antara lain :
- Data dengan nilai korelasi dibawah 0.1 (sqft_lot, sqft_lot15, yr_built, condition, long)
- Data tidak relevan (date, id, zipcode, yr_built, yr_renovated)
- Data dengan pesebaran dominan sebelah (sqft_basement)

Berikut adalah korelasi antar data dengan heatmap (setelah dihapus)
![N|Solid](https://github.com/radyaadi/Predictive-Analytics/blob/main/DataVisual-House/correlation.png?raw=true)

## Data Preparation
Dari studi kasus data diatas, kita akan mempersiapkan terlebih dahulu sebelum digunakkan pada model. Ada dua proses _data preparation_ yang akan diterapkan, yaitu :
### Train-Test-Split
Pada proses ini, data dibagi menjadi data train dan data uji. Kita perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru, dengan menggunakan data uji ini. Data dibagi dengan rasio 0.75 untuk data tes dan 0.25 untuk data uji.
>Total # of sample in whole dataset: 20467
Total # of sample in train dataset: 15350
Total # of sample in test dataset: 5117

### Standarisasi
Pada proses ini, data akan distandarisasikan. Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.

## Modeling
Pada projek ini, kita menggunakan tiga model algoritma machine learning. Pertama kita persiapkan terlebih dahulu model yang akan dibuat.
```sh
#Siapkan daraframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'RandomForest', 'Boosting'])
```
Berikut penjelasan dari masing-masing model.

### K-Nearest Neighbor
Melalui algoritma ini, kita mencari jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k-tetangga terdekat.

```sh
from sklearn.neighbors import KNeighborsRegressor
 
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_train)
```
Kita menggunakan k = 10 tetangga dan _metric euclidean_ untuk mengukur jarak antara titik. Pada tahap ini kita hanya melatih data train dan menyimpan data uji untuk dilatih di tahap evaluasi.

### Random Forest
Algoritma ini disusun dari banyak algoritma pohon (decision tree) yang pembagian data dan fiturnya dipilih secara acak. Kita akan mengkombinasikan masing - masing pohon (tree) dari model Decision Tree yang baik ke dalam satu model.

```sh
RF = RandomForestRegressor(criterion='mae',n_estimators=150,max_depth=8,random_state=42)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)
```

Kita hanya melatih data train dan menyimpan data uji untuk dilatih di tahap evaluasi, Seperti pada tahap K-Nearest Neighbor.

### Algoritma Boosting
Pada model ini, kita menggunakan metode adaptive boosting. Pada setiap tahapan, model akan memeriksa apakah observasi yang dilakukan sudah benar atau belum. Bobot yang lebih tinggi kemudian diberikan pada model yang salah sehingga model tersebut akan dimasukkan ke dalam tahapan selanjutnya. Proses iteratif tersebut berlanjut sampai model mencapai akurasi yang diinginkan.

```sh
from sklearn.ensemble import AdaBoostRegressor
 
boosting = AdaBoostRegressor(n_estimators=50, learning_rate=0.05, random_state=55)                             
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

```

Kita hanya melatih data train dan menyimpan data uji untuk dilatih di tahap evaluasi, Seperti pada tahap K-Nearest Neighbor dan Random Forest.

Berikut hasil prediksi dari ketiga model di atas.
|  | y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting |
| - | --------- | -------- | --------- | --------- |
| 8979 | 540000.0 | 605850.0 | 572875.9 | 534155.7 |

Berdasarkan data diatas model yang memiliki hasil prediksi paling mendekati adalah Algoritma Boosting, sehingga kedepannya algoritma yang akan digunakan adalah algoritma tersebut.

## Evaluation
Projek ini termasuk dalam linear regression, sehingga kita akan mengevaluasi model dengan menggunakan metrik MSE (Mean Squared Error). Metrik ini menghitung selisih rata-rata nilai sebenarnya dengan nilai prediksi. Proses pertama adalah melakukan proses scaling fitur numerik pada data uji.

```sh
## Scale our numerical features so they have zero mean and a variance of one
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])
```

Kemudian, evaluasi ketiga model dengan metrik MSE
```sh
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
 
mse
```

Dari kode tersebut, didapatkan hasil sebagai berikut
|     | train | test |
| --- | --------- | -------- |
| KNN | 8.34553e+06 | 1.01376e+07 |
| RF | 8.03845e+06 | 9.40996e+06 |
| Boosting | 1.60496e+07 | 1.63371e+07 |

Berikut adalah plot metrik dari hasil evaluasi
![N|Solid](https://github.com/radyaadi/Predictive-Analytics/blob/main/DataVisual-House/mse.png?raw=true)

Berdasarkan plot diatas, terlihat bahwa model Algoritma Boosting memiliki nilai eror yang paling kecil. Dengan demikian, model Algoritma Boosting
