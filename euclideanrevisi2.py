import numpy as np

data = [([1, 2, 1, 1], 'Lulus'),
        ([2, 1, 2, 1], 'Lulus'),
        ([3, 3, 3, 3], 'Tidak Lulus'),
        ([4, 3, 4, 4], 'Tidak Lulus'),
        ([5, 5, 5, 5], 'Tidak Lulus'),
        ([1, 1, 1, 2], 'Lulus'),
        ([4, 4, 4, 3], 'Tidak Lulus'),]
Test = (3, 2, 3, 2)
def euclide (x, y):
    return np.sqrt(np.sum((x - y) ** 2))

jarak = []

for data in data:
    d = euclide(np.array(Test), np.array(data[0]))
    jarak.append((d, data[4]))

jarak.sort()
tetangga = jarak[:3]

Lulus = 0
Tidak_Lulus = 0
for d, kelas in tetangga:
    if kelas == 'Lulus':
        Lulus += 1
    else:
        Tidak_Lulus += 1

if Lulus > Tidak_Lulus:
    print("Lulus")
else:
    print("Tidak Lulus")