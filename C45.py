
import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
import warnings

warnings.filterwarnings('ignore') # Mengabaikan peringatan untuk tampilan yang lebih bersih



try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except:
    SCRIPT_DIR = os.getcwd()


DATASET_PATH = os.path.join(SCRIPT_DIR, 'datasetkepuasan', 'train.csv')

RANDOM_STATE = 42
TEST_SIZE = 0.2   



def train_test_split_manual(X, y, test_size=0.2, random_state=42, stratify=None):
    """Membagi dataset menjadi data latih dan data uji secara manual.
    Mendukung pembagian stratifikasi untuk menjaga proporsi kelas target.
    """
    np.random.seed(random_state)
    n = len(y)
    n_test = int(n * test_size)
    
    if stratify is not None:
        train_indices = []
        test_indices = []
        # Pastikan stratify adalah numpy array
        if isinstance(stratify, (pd.Series, pd.DataFrame)):
            stratify = stratify.values

        for class_val in np.unique(stratify):
            class_indices = np.where(stratify == class_val)[0]
            np.random.shuffle(class_indices)
            n_class = len(class_indices)
            n_test_class = int(n_class * test_size)
            test_indices.extend(class_indices[:n_test_class])
            train_indices.extend(class_indices[n_test_class:])
        
        # Konversi kembali ke tipe data asli jika diperlukan
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        else:
            X_train, X_test = X[train_indices], X[test_indices]
        
        if isinstance(y, pd.Series):
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        else:
            y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test
    else:
        # Pembagian acak sederhana
        indices = np.random.permutation(n)
        
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[indices[n_test:]], X.iloc[indices[:n_test]]
        else:
            X_train, X_test = X[indices[n_test:]], X[indices[:n_test]]
        
        if isinstance(y, pd.Series):
            y_train, y_test = y.iloc[indices[n_test:]], y.iloc[indices[:n_test]]
        else:
            y_train, y_test = y[indices[n_test:]], y[indices[:n_test]]
        
        return X_train, X_test, y_train, y_test

class LabelEncoderManual:
    """Mengubah label kategorikal menjadi angka secara manual.
    Mirip dengan sklearn.preprocessing.LabelEncoder.
    """
    def __init__(self):
        self.classes_ = None # Menyimpan daftar kelas unik
        self.mapping_ = {}   # Menyimpan pemetaan dari label ke angka
    
    def fit(self, y):
        """Mempelajari kelas-kelas unik dari data."""
        self.classes_ = np.unique(y)
        self.mapping_ = {val: idx for idx, val in enumerate(self.classes_)}
        return self
    
    def transform(self, y):
        """Mengubah data kategorikal menjadi angka berdasarkan pemetaan yang dipelajari."""
        return np.array([self.mapping_[val] for val in y])
    
    def fit_transform(self, y):
        """Melakukan fit dan transform secara berurutan."""
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y):
        """Mengembalikan angka menjadi label kategorikal aslinya."""
        return np.array([self.classes_[idx] for idx in y])

def bin_arrival_delay(df, column='Arrival Delay in Minutes'):
    """Mengelompokkan kolom 'Arrival Delay in Minutes' ke dalam kategori.
    Kategori: 'No Delay', 'Short Delay', 'Medium Delay', 'Long Delay'.
    """
    if column not in df.columns:
        print(f"Kolom '{column}' tidak ditemukan untuk binning.")
        return df

    # Bersihkan format string yang bermasalah (misal: '35.00.00' -> '35.00')
    # Tangani kasus dengan multiple '.00' dengan mengambil bagian sebelum '.00' terakhir
    df[column] = df[column].astype(str).str.replace(',', '.', regex=False)
    
    # Hapus duplikat '.00' dengan regex - ganti '.00.00' dengan '.00'
    df[column] = df[column].str.replace(r'\.00\.00', '.00', regex=True)
    
    df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Hitung baris yang akan dihapus sebelum penghapusan
    nan_count_before = df[column].isna().sum()
    df = df.dropna(subset=[column]) # Hapus baris dengan NaN setelah konversi
    if nan_count_before > 0:
        print(f"  Dihapus {nan_count_before} baris dengan nilai NaN di kolom '{column}'")

    bins = [-np.inf, 0, 15, 60, np.inf] # <=0: No Delay, 0-15: Short, 15-60: Medium, >60: Long
    labels = ['No Delay', 'Short Delay', 'Medium Delay', 'Long Delay']
    df[column] = pd.cut(df[column], bins=bins, labels=labels, right=True)
    print(f"Kolom '{column}' berhasil di-binning menjadi kategori: {labels}")
    return df

# =============================================================================
# 3. FUNGSI INTI ALGORITMA C4.5 (PERHITUNGAN ENTROPY, GAIN, GAIN RATIO)
# =============================================================================

def hitung_entropy(y):
    """Menghitung Entropy dari sebuah set data.
    Entropy mengukur tingkat ketidakmurnian atau ketidakteraturan data.
    Rumus: H(S) = - sum(p_i * log2(p_i))
    """
    if len(y) == 0:
        return 0
    
    # Menghitung frekuensi setiap kelas
    counts = Counter(y)
    # Menghitung proporsi setiap kelas
    proportions = [count / len(y) for count in counts.values()]
    
    # Menghitung Entropy
    entropy = -sum(p * np.log2(p) for p in proportions if p > 0) # Hindari log(0)
    return entropy

def hitung_gain_ratio(y_parent, y_left, y_right):
    """Menghitung Gain Ratio untuk algoritma C4.5.
    Gain Ratio adalah perbaikan dari Information Gain yang mengatasi bias
    terhadap atribut dengan banyak nilai unik.
    Rumus: GainRatio(A) = Gain(A) / SplitInfo(A)
    """
    n_total = len(y_parent)
    if n_total == 0: return 0

    # 1. Hitung Information Gain
    # IG(S, A) = H(S) - sum((|Sv|/|S|) * H(Sv))
    entropy_parent = hitung_entropy(y_parent)
    
    n_left, n_right = len(y_left), len(y_right)
    
    weighted_entropy = (n_left / n_total) * hitung_entropy(y_left) + \
                       (n_right / n_total) * hitung_entropy(y_right)
    
    info_gain = entropy_parent - weighted_entropy
    
    # 2. Hitung Split Info (Intrinsic Information)
    # SplitInfo(A) = - sum((|Sv|/|S|) * log2(|Sv|/|S|))
    p_left = n_left / n_total
    p_right = n_right / n_total
    
    split_info = 0
    if p_left > 0: split_info -= p_left * np.log2(p_left)
    if p_right > 0: split_info -= p_right * np.log2(p_right)
    
    # 3. Hitung Gain Ratio
    # Jika SplitInfo nol (misal, semua data masuk ke satu cabang), hindari pembagian nol
    if split_info == 0:
        return 0
    
    return info_gain / split_info

# =============================================================================
# 4. KELAS IMPLEMENTASI DECISION TREE C4.5
# =============================================================================

class C45DecisionTree:
    """Implementasi Decision Tree C4.5 secara manual.
    Membangun pohon keputusan berdasarkan Gain Ratio.
    """
    def __init__(self, max_depth=5, min_samples_split=2, n_threshold_samples=10):
        self.max_depth = max_depth             # Kedalaman maksimum pohon
        self.min_samples_split = min_samples_split # Jumlah sampel minimum untuk melakukan split
        self.n_threshold_samples = n_threshold_samples # Jumlah sampel threshold untuk fitur numerik
        self.tree = None                       # Struktur pohon yang akan dibangun
        self.feature_types = None              # Menyimpan tipe fitur (numerik/kategorikal)

    def fit(self, X, y):
        """Melatih model Decision Tree C4.5.
        X: Fitur (DataFrame atau array numpy)
        y: Target (Series atau array numpy)
        """
        # Deteksi tipe fitur (numerik atau kategorikal)
        self.feature_types = [X.iloc[:, i].dtype for i in range(X.shape[1])]

        # Pastikan data dalam format numpy agar mudah diolah
        X_arr = np.array(X)
        y_arr = np.array(y)
        self.tree = self._bangun_pohon(X_arr, y_arr, depth=0)
        return self

    def _bangun_pohon(self, X, y, depth):
        """Fungsi rekursif untuk membangun pohon keputusan.
        Mengembalikan node pohon (berupa dictionary) atau label daun.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y)) # Jumlah kelas unik di node ini

        # --- Kriteria Penghentian (Base Cases) ---
        # 1. Jika sudah mencapai kedalaman maksimum
        # 2. Jika semua sampel di node ini memiliki kelas yang sama (murni)
        # 3. Jika jumlah sampel terlalu sedikit untuk di-split
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return self._buat_daun(y)

        # --- Mencari Split Terbaik ---
        best_feature, best_threshold, best_gain_ratio = None, None, -1
        
        # Iterasi setiap fitur untuk mencari split terbaik
        for feature_idx in range(n_features):
            current_feature_values = X[:, feature_idx]
            
            # Jika fitur numerik, gunakan sampling threshold
            if np.issubdtype(self.feature_types[feature_idx], np.number):
                # Ambil sampel threshold dari nilai unik
                unique_values = np.unique(current_feature_values)
                if len(unique_values) > self.n_threshold_samples:
                    # Ambil n_threshold_samples secara acak atau berdasarkan persentil
                    # Untuk kesederhanaan, kita ambil secara acak di sini
                    np.random.seed(RANDOM_STATE) # Pastikan konsisten
                    thresholds = np.random.choice(unique_values, self.n_threshold_samples, replace=False)
                else:
                    thresholds = unique_values
            else: # Fitur kategorikal (setelah encoding, ini akan menjadi numerik diskrit)
                thresholds = np.unique(current_feature_values)

            for threshold in thresholds:
                # Pisahkan data menjadi cabang kiri (<= threshold) dan kanan (> threshold)
                left_idx = np.where(current_feature_values <= threshold)[0]
                right_idx = np.where(current_feature_values > threshold)[0]
                
                # Jika salah satu cabang kosong, split ini tidak valid
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                
                # Hitung Gain Ratio untuk split ini
                gain_ratio = hitung_gain_ratio(y, y[left_idx], y[right_idx])
                
                # Perbarui split terbaik jika Gain Ratio lebih tinggi
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature_idx
                    best_threshold = threshold

        # Jika tidak ada split yang menghasilkan Gain Ratio positif (atau tidak ada split valid),
        # maka node ini menjadi daun.
        if best_gain_ratio <= 0 or best_feature is None:
            return self._buat_daun(y)

        # --- Membangun Cabang (Rekursi) ---
        # Masking untuk memisahkan data ke cabang kiri dan kanan
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        
        # Rekursif membangun sub-pohon untuk cabang kiri dan kanan
        return {
            'feature_idx': best_feature, # Indeks fitur yang digunakan untuk split
            'threshold': best_threshold, # Nilai threshold untuk split
            'left': self._bangun_pohon(X[left_mask], y[left_mask], depth + 1), # Sub-pohon kiri
            'right': self._bangun_pohon(X[right_mask], y[right_mask], depth + 1) # Sub-pohon kanan
        }

    def _buat_daun(self, y):
        """Menentukan label kelas mayoritas untuk node daun.
        Jika node kosong, kembalikan None atau nilai default.
        """
        if len(y) == 0: 
            return None # Atau bisa juga mengembalikan kelas mayoritas dari parent jika ada
        # Mengembalikan kelas yang paling sering muncul
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        """Melakukan prediksi untuk satu atau banyak data baru.
        X: Data fitur yang akan diprediksi (DataFrame atau array numpy)
        """
        X_arr = np.array(X)
        # Iterasi setiap baris data dan telusuri pohon untuk mendapatkan prediksi
        return np.array([self._telusuri_pohon(x, self.tree) for x in X_arr])

    def _telusuri_pohon(self, x, node):
        """Fungsi rekursif untuk menelusuri pohon keputusan untuk satu sampel data.
        x: Satu sampel data (array numpy)
        node: Node pohon saat ini
        """
        # Jika node adalah daun (bukan dictionary), kembalikan label kelasnya
        if not isinstance(node, dict):
            return node
        
        # Bandingkan nilai fitur sampel dengan threshold node
        if x[node['feature_idx']] <= node['threshold']:
            return self._telusuri_pohon(x, node['left']) # Telusuri cabang kiri
        else:
            return self._telusuri_pohon(x, node['right']) # Telusuri cabang kanan

# =============================================================================
# 5. FUNGSI EVALUASI MODEL MANUAL
# =============================================================================

def hitung_akurasi(y_true, y_pred):
    """Menghitung akurasi model secara manual.
    Akurasi = (Jumlah prediksi benar) / (Total prediksi)
    """
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix_manual(y_true, y_pred):
    """Menghitung Confusion Matrix secara manual.
    Menyediakan gambaran detail tentang kinerja model untuk setiap kelas.
    """
    # Gabungkan semua kelas unik dari y_true dan y_pred
    classes = np.unique(np.concatenate([y_true, y_pred]))
    # Inisialisasi matriks nol
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    
    # Isi confusion matrix
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            # Hitung berapa banyak sampel dengan true_class yang diprediksi sebagai pred_class
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    return cm

def hitung_precision_recall_f1(y_true, y_pred):
    """Menghitung Precision, Recall, dan F1 Score untuk setiap kelas secara manual.
    Mengembalikan dictionary dengan metrics per kelas dan rata-rata.
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix_manual(y_true, y_pred)
    
    results = {}
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []
    
    for idx, class_val in enumerate(classes):
        # TP: True Positives (diagonal confusion matrix)
        tp = cm[idx, idx]
        # FP: False Positives (column sum - diagonal)
        fp = np.sum(cm[:, idx]) - tp
        # FN: False Negatives (row sum - diagonal)
        fn = np.sum(cm[idx, :]) - tp
        # TN: True Negatives (semua elemen selain row dan column)
        tn = np.sum(cm) - tp - fp - fn
        
        # Support: jumlah sampel aktual untuk kelas ini
        support = tp + fn
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[class_val] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        support_list.append(support)
    
    # Hitung rata-rata (macro dan weighted)
    total_support = sum(support_list)
    
    # Macro average: rata-rata sederhana dari semua kelas
    macro_precision = np.mean(precision_list)
    macro_recall = np.mean(recall_list)
    macro_f1 = np.mean(f1_list)
    
    # Weighted average: rata-rata berbobot berdasarkan support
    weighted_precision = sum(p * s for p, s in zip(precision_list, support_list)) / total_support
    weighted_recall = sum(r * s for r, s in zip(recall_list, support_list)) / total_support
    weighted_f1 = sum(f * s for f, s in zip(f1_list, support_list)) / total_support
    
    results['macro_avg'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
        'support': total_support
    }
    
    results['weighted_avg'] = {
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1': weighted_f1,
        'support': total_support
    }
    
    return results

def print_classification_report(y_true, y_pred, target_names=None):
    """Menampilkan laporan klasifikasi lengkap dengan format yang rapi.
    Mirip dengan sklearn.metrics.classification_report
    """
    metrics = hitung_precision_recall_f1(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if target_names is None:
        target_names = {class_val: f"Class {class_val}" for class_val in classes}
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT (LAPORAN KLASIFIKASI RINCI)")
    print("=" * 80)
    print(f"{'Kelas':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}")
    print("-" * 80)
    
    for class_val in classes:
        class_name = target_names.get(class_val, f"Class {class_val}")
        m = metrics[class_val]
        print(f"{class_name:<20} {m['precision']:<15.4f} {m['recall']:<15.4f} {m['f1']:<15.4f} {m['support']:<10}")
    
    print("-" * 80)
    
    # Macro average
    m_macro = metrics['macro_avg']
    print(f"{'Macro Avg':<20} {m_macro['precision']:<15.4f} {m_macro['recall']:<15.4f} {m_macro['f1']:<15.4f} {m_macro['support']:<10}")
    
    # Weighted average
    m_weighted = metrics['weighted_avg']
    print(f"{'Weighted Avg':<20} {m_weighted['precision']:<15.4f} {m_weighted['recall']:<15.4f} {m_weighted['f1']:<15.4f} {m_weighted['support']:<10}")
    print("=" * 80)

# =============================================================================
# 6. MAIN PROGRAM: ALUR KERJA KLASIFIKASI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PROGRAM KLASIFIKASI KEPUASAN PELANGGAN DENGAN C4.5 MANUAL (OPTIMASI)")
    print("=" * 60)

    # --- TAHAP 1: MEMBACA DATASET ---
    print("\n" + "=" * 60)
    print("TAHAP 1: MEMBACA DATASET")
    print("=" * 60)
    
    df = None
    try:
        # Coba membaca dengan separator ";"
        df = pd.read_csv(DATASET_PATH, sep=';')
        print(f"Dataset berhasil dimuat dari '{DATASET_PATH}' dengan separator ';'")
    except FileNotFoundError:
        print(f"Error: File dataset tidak ditemukan di '{DATASET_PATH}'.")
        print("Pastikan file 'train.csv' berada di dalam folder 'datasetkepuasan' ")
        print("yang sejajar dengan script ini, atau sesuaikan DATASET_PATH.")
        sys.exit(1) # Keluar dari program jika file tidak ditemukan
    except Exception as e:
        print(f"Gagal membaca dataset dengan separator ';'. Mencoba separator ','. Error: {e}")
        try:
            # Coba membaca dengan separator ","
            df = pd.read_csv(DATASET_PATH, sep=',')
            print(f"Dataset berhasil dimuat dari '{DATASET_PATH}' dengan separator ','")
        except Exception as e_comma:
            print(f"Gagal membaca dataset dengan separator ','. Error: {e_comma}")
            print("Tidak dapat memuat dataset. Harap periksa format file CSV Anda.")
            sys.exit(1)

    print(f"Jumlah data: {df.shape[0]} baris, {df.shape[1]} kolom")
    print("5 baris pertama dataset:")
    print(df.head())

    # --- TAHAP 2: INFORMASI DATASET & PREPROCESSING AWAL ---
    print("\n" + "=" * 60)
    print("TAHAP 2: INFORMASI DATASET & PREPROCESSING AWAL")
    print("=" * 60)

    print(f"\nJumlah data awal: {len(df)}")
    print(f"Jumlah fitur: {df.shape[1]}")
    print(f"\nInformasi kolom dan tipe data:")
    df.info()

    print(f"\nJumlah data yang hilang (NaN) per kolom:")
    print(df.isnull().sum())

    # Menghapus kolom 'id' jika ada dan membersihkan baris dengan nilai NaN
    df_clean = df.copy()
    if 'id' in df_clean.columns:
        df_clean = df_clean.drop('id', axis=1)
        print("Kolom 'id' dihapus.")
    
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna()
    rows_after_na = len(df_clean)
    if initial_rows > rows_after_na:
        print(f"Menghapus {initial_rows - rows_after_na} baris dengan nilai yang hilang.")
    print(f"Jumlah data setelah pembersihan: {len(df_clean)} baris")

    # --- TAHAP 2.5: BINNING KOLOM 'Arrival Delay in Minutes' ---
    print("\n" + "=" * 60)
    print("TAHAP 2.5: BINNING KOLOM 'Arrival Delay in Minutes'")
    print("=" * 60)
    df_clean = bin_arrival_delay(df_clean, column='Arrival Delay in Minutes')

    # Memisahkan fitur (X) dan target (y)
    TARGET_COLUMN = 'satisfaction' # Sesuaikan nama kolom target jika berbeda
    if TARGET_COLUMN not in df_clean.columns:
        print(f"Error: Kolom target '{TARGET_COLUMN}' tidak ditemukan di dataset.")
        sys.exit(1)

    X = df_clean.drop(TARGET_COLUMN, axis=1)
    y = df_clean[TARGET_COLUMN]
    
    print(f"\nDistribusi kelas target ('{TARGET_COLUMN}'):")
    print(y.value_counts())

    # --- TAHAP 3: ENCODING DATA KATEGORIKAL ---
    print("\n" + "=" * 60)
    print("TAHAP 3: ENCODING DATA KATEGORIKAL")
    print("=" * 60)

    encoders = {} # Menyimpan encoder untuk setiap kolom kategorikal
    # Mengidentifikasi kolom kategorikal (tipe 'object' atau 'category' setelah binning)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        print(f"Meng-encode kolom kategorikal: {list(categorical_cols)}")
        for col in categorical_cols:
            encoder = LabelEncoderManual()
            X[col] = encoder.fit_transform(X[col])
            encoders[col] = encoder
            print(f"  - Kolom '{col}' di-encode. Kelas: {encoder.classes_.tolist()}")
    else:
        print("Tidak ada kolom kategorikal yang perlu di-encode.")

    # Meng-encode kolom target
    print(f"Meng-encode kolom target '{TARGET_COLUMN}'.")
    encoder_target = LabelEncoderManual()
    y_encoded = encoder_target.fit_transform(y)
    # Konversi y_encoded kembali ke Series dengan indeks asli agar kompatibel dengan pandas
    y_encoded = pd.Series(y_encoded, index=y.index)
    print(f"  - Kelas target asli: {encoder_target.classes_.tolist()}")
    print(f"  - Kelas target ter-encode: {np.unique(y_encoded).tolist()}")

    # --- TAHAP 4: MEMBAGI DATA LATIH DAN UJI ---
    print("\n" + "=" * 60)
    print("TAHAP 4: MEMBAGI DATA LATIH DAN UJI")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    print(f"Ukuran data latih (X_train, y_train): {len(X_train)} sampel")
    print(f"Ukuran data uji (X_test, y_test): {len(X_test)} sampel")
    print(f"Proporsi data latih: {len(X_train) / len(df_clean):.2%}")
    print(f"Proporsi data uji: {len(X_test) / len(df_clean):.2%}")

    # --- TAHAP 5: MELATIH MODEL C4.5 ---
    print("\n" + "=" * 60)
    print("TAHAP 5: MELATIH MODEL C4.5")
    print("=" * 60)
    
    # Inisialisasi model Decision Tree C4.5
    # max_depth: Batasi kedalaman pohon untuk mencegah overfitting dan membuat pohon lebih mudah dipahami
    # min_samples_split: Jumlah sampel minimum di node agar bisa di-split
    # n_threshold_samples: Jumlah sampel threshold untuk fitur numerik (misal 10 atau 20)
    model = C45DecisionTree(max_depth=5, min_samples_split=10, n_threshold_samples=20)
    print(f"Model C4.5 diinisialisasi dengan max_depth={model.max_depth}, min_samples_split={model.min_samples_split}, n_threshold_samples={model.n_threshold_samples}.")
    
    # Melatih model
    model.fit(X_train, y_train)
    print("Model C4.5 berhasil dilatih!")

    # --- TAHAP 6: PREDIKSI PADA DATA UJI ---
    print("\n" + "=" * 60)
    print("TAHAP 6: PREDIKSI PADA DATA UJI")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    print("Prediksi pada data uji selesai.")

    # --- TAHAP 7: EVALUASI MODEL ---
    print("\n" + "=" * 60)
    print("TAHAP 7: EVALUASI MODEL")
    print("=" * 60)
    
    # Pastikan y_test dalam format numpy array untuk fungsi evaluasi manual
    y_test_arr = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    accuracy = hitung_akurasi(y_test_arr, y_pred)
    cm = confusion_matrix_manual(y_test_arr, y_pred)
    
    print(f"\nAkurasi Model: {accuracy * 100:.2f}%")
    print(f"\nConfusion Matrix (Baris: Aktual, Kolom: Prediksi):\n{cm}")
    
    # Tampilkan classification report lengkap
    target_names = encoder_target.inverse_transform(np.unique(y_test_arr))
    target_names_dict = {i: name for i, name in enumerate(target_names)}
    print_classification_report(y_test_arr, y_pred, target_names=target_names_dict)

    # --- TAHAP 8: CONTOH PREDIKSI INDIVIDUAL ---
    print("\n" + "=" * 60)
    print("TAHAP 8: CONTOH PREDIKSI INDIVIDUAL")
    print("=" * 60)
    
    print(f"\n{'No':<5} {'Aktual':<15} {'Prediksi':<15} {'Status':<10}")
    print("-" * 45)
    
    # Menampilkan 10 contoh prediksi pertama dari data uji
    for i in range(min(10, len(y_test_arr))):
        actual_label = encoder_target.inverse_transform([y_test_arr[i]])[0]
        predicted_label = encoder_target.inverse_transform([y_pred[i]])[0]
        status = "BENAR" if y_test_arr[i] == y_pred[i] else "SALAH"
        print(f"{i+1:<5} {actual_label:<15} {predicted_label:<15} {status:<10}")

    # --- TAHAP 9: KESIMPULAN ---
    print("\n" + "=" * 60)
    print("TAHAP 9: KESIMPULAN")
    print("=" * 60)
    
    print(f"Akurasi keseluruhan model C4.5 manual adalah: {accuracy * 100:.2f}%")
    
    if accuracy >= 0.85:
        status_kinerja = "SANGAT BAIK - Model menunjukkan kinerja yang luar biasa."
    elif accuracy >= 0.75:
        status_kinerja = "BAIK - Model memiliki kinerja yang solid."
    elif accuracy >= 0.65:
        status_kinerja = "CUKUP - Kinerja model dapat diterima, namun ada ruang untuk peningkatan."
    else:
        status_kinerja = "PERLU DITINGKATKAN - Model memerlukan penyesuaian lebih lanjut atau data tambahan."
    
    print(f"Status Kinerja Model: {status_kinerja}")
    print("\nPROGRAM SELESAI! Semoga kode ini membantu pemahaman Anda tentang C4.5.")
