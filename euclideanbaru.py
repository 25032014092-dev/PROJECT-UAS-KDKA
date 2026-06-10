import pandas as pd
import random
from random import shuffle
import numpy as np
from collections import Counter

RANDOM_STATE = 42
K = 5
P = 2
DATA_SIZE = 1000
DATA_TRAIN = 800

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print("\nProgram Analisis Data Kepuasan Pelanggan Menggunakan K-Nearest Neighbors (euclidean)")
print("_" * 70, "\n")

df = pd.read_csv(r'E:\uaskdka\datasetkepuasan\train.csv', sep=';')
print(f"Dataset: {df.shape[0]} baris, {df.shape[1]} kolom")

print(df.isnull().sum())
print("\nMenghapus baris dengan nilai yang kosong")
df = df.dropna()
print(f"Jumlah dataset setelah data null dihapus: {df.shape[0]} baris, {df.shape[1]} kolom")

print("\nMenggunakan 5 fitur karena 21 fitur terlalu banyak untuk divisualisasikan")
print("Menghapus fitur selain : Food and drink, Checkin service, Inflight service, Baggage handling, Departure Delay in Minutes")
fitur = ['Food and drink', 'Checkin service', 'Inflight service', 'Baggage handling', 'Departure Delay in Minutes']
X = df[fitur]
y = df['satisfaction']
print("\nFitur yang digunakan:", list(X.columns))
print(f"Jumlah fitur yang berkurang menjadi: {X.shape[1]} kolom")

print("\nMebagi data menjadi (Data Test: 20%, dan Data Train: 80%)")
data = list(zip(X.values, y.values))
shuffle(data)
split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]
print(f"Jumlah data train: {len(train_data)}", f"Jumlah data test: {len(test_data)}")

if len(test_data) > DATA_SIZE:
    test_data = random.sample(test_data, k=DATA_SIZE)

if len(train_data) > DATA_TRAIN:
    train_data = random.sample(train_data, k=DATA_TRAIN)

def jarak(x1, x2, p=P):
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

def knn(train_data, test_point):
    jarak_list = []

    # hitung jarak semua data train
    for fitur, label in train_data:
        d = jarak(fitur, test_point)
        jarak_list.append((d, label))

    # urutkan jarak
    jarak_list.sort()

    # ambil K terdekat
    k_terdekat = jarak_list[:K]

    # voting label
    labels = [l for _, l in k_terdekat]
    hasil = Counter(labels).most_common(1)[0][0]

    return hasil
        
benar = 0
 
for fitur, label in test_data:
    prediksi = knn(train_data, fitur)
    
    if prediksi == label:
        benar += 1
        akurasi = benar / len(test_data)

akurasi = benar / len(test_data)

print(f"\nAkurasi KNN (euclidean)", akurasi)