from random import shuffle
import pandas as pd

print("\nProgram Analisis Data Kepuasan Pelanggan Menggunakan K-Nearest Neighbors (euclidean)")
print("_" * 70, "\n")

df = pd.read_csv(r'E:\uaskdka\uaskdka\datasetkepuasan\train.csv')
print(df.columns.tolist())
print(df.shape)
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


