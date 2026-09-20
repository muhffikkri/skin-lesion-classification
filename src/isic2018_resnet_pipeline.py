# =========================================================================
# Klasifikasi Lesi Kulit ISIC2018 Task 3 — CNN dengan Residual Connection (ResNet)
# Mini Project Komputasi Lunak — Arsitektur 2
#
# Mengikuti struktur data resmi ISIC 2018 Challenge (bukan versi Kaggle)
# dan strategi preprocessing dari paper referensi (resize 224x224,
# normalisasi ImageNet, balancing kelas via downsampling + augmentasi).
# Sesuaikan SECTION 1 sebelum dijalankan.
# =========================================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device yang dipakai:", device)


# =========================================================================
# SECTION 1: Setup Path Dataset
# Struktur folder mengikuti rilis resmi ISIC 2018 Task 3. Sesuaikan
# DATA_DIR dengan lokasi dataset kamu. Tidak ada Validation_GroundTruth
# resmi, jadi validation set dibuat sendiri dari data training (stratified
# split), sementara Validation_Input resmi disisihkan (tanpa ground truth,
# tidak dipakai untuk training/evaluasi di sini).
# =========================================================================

# --- SESUAIKAN PATH INI ---
DATA_DIR = "/content/drive/MyDrive/ISIC2018_Task3"  # ganti sesuai lokasi kamu (Colab: mount drive dulu)

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "ISIC2018_Task3_Training_Input")
TRAIN_GT_PATH = glob.glob(os.path.join(DATA_DIR, "ISIC2018_Task3_Training_GroundTruth", "*.csv"))[0]

TEST_IMG_DIR = os.path.join(DATA_DIR, "ISIC2018_Task3_Test_Input")
TEST_GT_PATH = glob.glob(os.path.join(DATA_DIR, "ISIC2018_Task3_Test_GroundTruth", "*.csv"))[0]

VAL_IMG_DIR = os.path.join(DATA_DIR, "ISIC2018_Task3_Validation_Input")  # tanpa ground truth
# ---------------------------

train_gt_raw = pd.read_csv(TRAIN_GT_PATH)
test_gt_raw = pd.read_csv(TEST_GT_PATH)
print(train_gt_raw.head())


# =========================================================================
# SECTION 2: Konversi Ground Truth One-Hot ke Label Tunggal
# Ground truth resmi ISIC2018 berformat one-hot (kolom MEL, NV, BCC,
# AKIEC, BKL, DF, VASC, isinya 1.0 di kolom kelas yang benar).
# =========================================================================

CLASS_COLUMNS = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


def onehot_to_label(df_raw):
    df = df_raw.copy()
    df['dx'] = df[CLASS_COLUMNS].idxmax(axis=1)
    return df[['image', 'dx']]


train_df_full = onehot_to_label(train_gt_raw)
test_df = onehot_to_label(test_gt_raw)

train_df_full['path'] = train_df_full['image'].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x + ".jpg"))
test_df['path'] = test_df['image'].apply(lambda x: os.path.join(TEST_IMG_DIR, x + ".jpg"))

print("Jumlah data training (sebelum split val):", len(train_df_full))
print("Jumlah data test:", len(test_df))


# =========================================================================
# SECTION 3: EDA — Class Distribution
# Distribusi kelas di training set dan test set, dalam jumlah dan persentase.
# =========================================================================

def class_distribution_table(df, name):
    counts = df['dx'].value_counts()
    pct = df['dx'].value_counts(normalize=True) * 100
    table = pd.DataFrame({'Jumlah': counts, 'Persentase (%)': pct.round(2)})
    table.index.name = f'Kelas ({name})'
    return table


train_dist = class_distribution_table(train_df_full, "Training")
test_dist = class_distribution_table(test_df, "Test")

print("Distribusi kelas — Training set:")
print(train_dist)
print("\nDistribusi kelas — Test set:")
print(test_dist)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_df_full['dx'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title("Distribusi Kelas — Training Set")
axes[0].set_xlabel("Kelas")
axes[0].set_ylabel("Jumlah Gambar")

test_df['dx'].value_counts().plot(kind='bar', ax=axes[1], color='indianred')
axes[1].set_title("Distribusi Kelas — Test Set")
axes[1].set_xlabel("Kelas")
axes[1].set_ylabel("Jumlah Gambar")
plt.tight_layout()
plt.show()


# =========================================================================
# SECTION 4: EDA — Data Characteristics (Ukuran & Channel Gambar)
# Dihitung dari sampel acak biar cepat (naikkan SAMPLE_SIZE kalau waktu
# memungkinkan). Mengonfirmasi bahwa gambar tidak seragam ukurannya
# (resolusi asli HAM10000 bervariasi 600x450 sampai 6000x4000), sehingga
# resize wajib dilakukan sebelum masuk model.
# =========================================================================

SAMPLE_SIZE = 300
sample_paths = train_df_full['path'].sample(n=SAMPLE_SIZE, random_state=42).tolist()

image_stats = []
for p in sample_paths:
    with Image.open(p) as img:
        width, height = img.size
        mode = img.mode  # 'RGB', 'L', 'RGBA', dst
        n_channels = len(img.getbands())
        file_size_kb = os.path.getsize(p) / 1024
    image_stats.append({
        'width': width, 'height': height,
        'aspect_ratio': round(width / height, 3),
        'mode': mode, 'n_channels': n_channels,
        'file_size_kb': round(file_size_kb, 1)
    })

stats_df = pd.DataFrame(image_stats)
print(stats_df.head())

print("\n=== Ukuran Gambar ===")
print(f"Resolusi unik yang ditemukan: {stats_df[['width','height']].drop_duplicates().shape[0]} variasi dari {SAMPLE_SIZE} sampel")
print(f"Width  -> min: {stats_df['width'].min()}, max: {stats_df['width'].max()}, modus: {stats_df['width'].mode()[0]}")
print(f"Height -> min: {stats_df['height'].min()}, max: {stats_df['height'].max()}, modus: {stats_df['height'].mode()[0]}")

print("\n=== Channel Warna ===")
print(stats_df['mode'].value_counts())
print(f"Semua gambar RGB (3 channel)? {(stats_df['n_channels'] == 3).all()}")

# --- Descriptive statistics (dimensi & ukuran file) ---
print("\n=== Descriptive Statistics (width, height, aspect ratio, file size) ===")
print(stats_df[['width', 'height', 'aspect_ratio', 'file_size_kb']].describe())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(stats_df['width'], bins=20, color='steelblue', edgecolor='black')
axes[0].set_title("Distribusi Lebar Gambar (px)")
axes[0].set_xlabel("Width")

axes[1].hist(stats_df['height'], bins=20, color='seagreen', edgecolor='black')
axes[1].set_title("Distribusi Tinggi Gambar (px)")
axes[1].set_xlabel("Height")

axes[2].hist(stats_df['file_size_kb'], bins=20, color='indianred', edgecolor='black')
axes[2].set_title("Distribusi Ukuran File (KB)")
axes[2].set_xlabel("File Size (KB)")
plt.tight_layout()
plt.show()

# --- Descriptive statistics: statistik pixel per channel ---
pixel_means, pixel_stds = [], []
for p in sample_paths[:100]:  # subset lebih kecil, baca penuh piksel lebih berat
    with Image.open(p) as img:
        arr = np.array(img.convert('RGB')) / 255.0
        pixel_means.append(arr.reshape(-1, 3).mean(axis=0))
        pixel_stds.append(arr.reshape(-1, 3).std(axis=0))

pixel_means = np.array(pixel_means)
pixel_stds = np.array(pixel_stds)

pixel_stat_df = pd.DataFrame({
    'Channel': ['R', 'G', 'B'],
    'Mean': pixel_means.mean(axis=0).round(4),
    'Std': pixel_stds.mean(axis=0).round(4)
})
print("\nStatistik pixel dataset (skala 0-1):")
print(pixel_stat_df)
print("\nStatistik ImageNet (dipakai untuk normalisasi karena pretrained):")
print("Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]")


# =========================================================================
# SECTION 5: Split Train / Validation (dari Training set resmi)
# Test set resmi ISIC2018 (dengan ground truth) disisihkan penuh untuk
# evaluasi akhir, tidak disentuh selama development.
# =========================================================================

train_classes = sorted(train_df_full['dx'].unique())
class_to_idx = {c: i for i, c in enumerate(train_classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

train_df_full['label'] = train_df_full['dx'].map(class_to_idx)
test_df['label'] = test_df['dx'].map(class_to_idx)

train_df, val_df = train_test_split(
    train_df_full, test_size=0.15, stratify=train_df_full['label'], random_state=42
)

print("Mapping kelas:", class_to_idx)
print("Jumlah data train:", len(train_df))
print("Jumlah data validation:", len(val_df))
print("Jumlah data test (resmi):", len(test_df))


# =========================================================================
# SECTION 6: Data Balancing (mengikuti strategi paper, opsional)
# Threshold 500 sampel per kelas: downsampling kelas mayoritas (NV, MEL,
# BKL). Kelas minoritas (AKIEC, BCC, VASC, DF) diperkuat lewat augmentasi
# saat training (SECTION 7), bukan oversampling eksplisit di sini.
# Kalau tidak diwajibkan balancing, lewati bagian ini dan pakai train_df
# langsung di SECTION 11.
# =========================================================================

BALANCE_THRESHOLD = 500


def downsample_majority(df, threshold, seed=42):
    balanced_parts = []
    for cls, group in df.groupby('dx'):
        if len(group) > threshold:
            group = group.sample(n=threshold, random_state=seed)
        balanced_parts.append(group)
    return pd.concat(balanced_parts).reset_index(drop=True)


train_df_balanced = downsample_majority(train_df, BALANCE_THRESHOLD)
print("Distribusi kelas training SEBELUM downsampling:")
print(train_df['dx'].value_counts())
print("\nDistribusi kelas training SETELAH downsampling:")
print(train_df_balanced['dx'].value_counts())


# =========================================================================
# SECTION 7: Transform dan Augmentasi
# Resize ke 224x224 (sesuai input ResNet-18), normalisasi pakai statistik
# ImageNet karena pakai pretrained weights. Augmentasi hanya diterapkan
# ke data training.
# =========================================================================

IMG_SIZE = 224
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


# =========================================================================
# SECTION 8: Custom Dataset
# =========================================================================

class ISICDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, label


# =========================================================================
# SECTION 9: Model ResNet-18
# Memakai bobot pretrained ImageNet lalu mengganti fully connected layer
# terakhir jadi 7 kelas. dropout jadi salah satu hyperparameter yang bisa
# dieksperimenkan.
# =========================================================================

def build_model(num_classes=7, dropout=0.3, pretrained=True):
    model = torchvision.models.resnet18(
        weights='IMAGENET1K_V1' if pretrained else None
    )
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model.to(device)


# =========================================================================
# SECTION 10: Fungsi Training dan Evaluasi per Epoch
# =========================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# =========================================================================
# SECTION 11: Fungsi Utama Training
# Satu fungsi yang menerima kombinasi hyperparameter sebagai argumen,
# jadi tinggal dipanggil ulang untuk tiap eksperimen tanpa duplikasi kode.
# Default memakai train_df_balanced (hasil downsampling SECTION 6); ganti
# ke train_df kalau ingin tanpa balancing.
# =========================================================================

def run_training(lr=1e-4, batch_size=32, dropout=0.3, optimizer_name='adam',
                  weight_decay=1e-4, num_epochs=20, run_name="baseline",
                  train_data=None):

    if train_data is None:
        train_data = train_df_balanced

    train_loader = DataLoader(
        ISICDataset(train_data, train_transform),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        ISICDataset(val_df, eval_transform),
        batch_size=batch_size, shuffle=False
    )

    model = build_model(dropout=dropout)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' tidak dikenali")

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"[{run_name}] Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model, history


# =========================================================================
# SECTION 12: Jalankan Training Baseline
# Baseline dulu pakai setting default, jadi patokan sebelum tuning.
# =========================================================================

baseline_model, baseline_history = run_training(
    lr=1e-4, batch_size=32, dropout=0.3, optimizer_name='adam',
    weight_decay=1e-4, num_epochs=20, run_name="baseline"
)


# =========================================================================
# SECTION 13: Plot Kurva Loss dan Accuracy
# =========================================================================

def plot_history(history, title="Training History"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


plot_history(baseline_history, "Baseline")


# =========================================================================
# SECTION 14: Eksperimen Hyperparameter
# Minimal 3 jenis hyperparameter berbeda. Contoh di bawah: learning rate,
# batch size, dropout — ganti nilainya sesuai kebutuhan, atau tambahkan
# jenis lain (optimizer, weight decay, jumlah epoch) dengan pola yang sama.
# =========================================================================

# --- 14a. Eksperimen Learning Rate ---
learning_rates = [1e-3, 1e-4, 1e-5]
lr_results = {}

for lr in learning_rates:
    print(f"\n=== Eksperimen Learning Rate = {lr} ===")
    _, hist = run_training(
        lr=lr, batch_size=32, dropout=0.3,
        optimizer_name='adam', num_epochs=15, run_name=f"lr_{lr}"
    )
    lr_results[lr] = hist

for lr, hist in lr_results.items():
    print(f"LR={lr} -> Val Acc akhir: {hist['val_acc'][-1]:.4f}, "
          f"Val Loss akhir: {hist['val_loss'][-1]:.4f}")

# --- 14b. Eksperimen Batch Size ---
batch_sizes = [16, 32, 64]
batch_results = {}

for bs in batch_sizes:
    print(f"\n=== Eksperimen Batch Size = {bs} ===")
    _, hist = run_training(
        lr=1e-4, batch_size=bs, dropout=0.3,
        optimizer_name='adam', num_epochs=15, run_name=f"bs_{bs}"
    )
    batch_results[bs] = hist

for bs, hist in batch_results.items():
    print(f"Batch Size={bs} -> Val Acc akhir: {hist['val_acc'][-1]:.4f}, "
          f"Val Loss akhir: {hist['val_loss'][-1]:.4f}")

# --- 14c. Eksperimen Dropout Rate ---
dropout_values = [0.2, 0.3, 0.5]
dropout_results = {}

for dp in dropout_values:
    print(f"\n=== Eksperimen Dropout = {dp} ===")
    _, hist = run_training(
        lr=1e-4, batch_size=32, dropout=dp,
        optimizer_name='adam', num_epochs=15, run_name=f"dropout_{dp}"
    )
    dropout_results[dp] = hist

for dp, hist in dropout_results.items():
    print(f"Dropout={dp} -> Val Acc akhir: {hist['val_acc'][-1]:.4f}, "
          f"Val Loss akhir: {hist['val_loss'][-1]:.4f}")


# =========================================================================
# SECTION 15: Rangkum Semua Hasil Eksperimen dalam Tabel
# Langsung dipakai untuk laporan bagian Hasil dan Analisis.
# =========================================================================

summary_rows = []

for lr, hist in lr_results.items():
    summary_rows.append({"Eksperimen": "Learning Rate", "Nilai": lr,
                          "Val Acc": hist['val_acc'][-1], "Val Loss": hist['val_loss'][-1]})

for bs, hist in batch_results.items():
    summary_rows.append({"Eksperimen": "Batch Size", "Nilai": bs,
                          "Val Acc": hist['val_acc'][-1], "Val Loss": hist['val_loss'][-1]})

for dp, hist in dropout_results.items():
    summary_rows.append({"Eksperimen": "Dropout", "Nilai": dp,
                          "Val Acc": hist['val_acc'][-1], "Val Loss": hist['val_loss'][-1]})

summary_df = pd.DataFrame(summary_rows)
print(summary_df)


# =========================================================================
# SECTION 16: Evaluasi Akhir di Test Set Resmi
# Pakai model dengan kombinasi hyperparameter terbaik (ganti baseline_model
# kalau salah satu hasil eksperimen lebih bagus). Test set resmi ISIC2018
# baru disentuh sekarang, belum pernah dilihat model selama training/tuning.
# =========================================================================

test_loader = DataLoader(
    ISICDataset(test_df, eval_transform), batch_size=32, shuffle=False
)
criterion = nn.CrossEntropyLoss()

# Ganti 'baseline_model' dengan model hasil kombinasi hyperparameter terbaik
test_loss, test_acc, preds, labels = evaluate(baseline_model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=train_classes))

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=train_classes, yticklabels=train_classes, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Test Set Resmi ISIC2018")
plt.tight_layout()
plt.show()

# =========================================================================
# CATATAN:
# - Sesuaikan DATA_DIR di SECTION 1 dengan lokasi dataset kamu.
# - Validation_Input resmi tidak punya ground truth publik, jadi tidak
#   dipakai untuk evaluasi — validation set dibuat sendiri lewat
#   stratified split dari data training (SECTION 5).
# - SECTION 3-4 (EDA) menghasilkan class distribution, descriptive
#   statistics (dimensi gambar, ukuran file, statistik piksel per
#   channel), dan konfirmasi karakteristik data (ukuran bervariasi,
#   semua RGB 3 channel) — langsung bisa dipakai di laporan Metodologi.
# - Pakai ResNet-18 pretrained ImageNet (transfer learning). Kalau
#   dosen mewajibkan arsitektur dari nol, ganti build_model() dengan
#   implementasi manual residual block.
# - SECTION 6 (balancing) opsional, mengikuti strategi paper. Kalau
#   tidak dipakai, ganti argumen train_data di run_training() supaya
#   pakai train_df langsung.
# - SECTION 14 sudah mencakup 3 jenis hyperparameter (memenuhi syarat
#   minimal tugas), boleh ditambah jenis lain dengan pola yang sama.
# =========================================================================
