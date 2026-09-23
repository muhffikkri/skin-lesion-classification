# Klasifikasi Skin Lesion — ISIC 2018 Task 3

Pipeline pelatihan & evaluasi model **ResNet** untuk klasifikasi **7 kelas diagnosis lesi
kulit** (MEL, NV, BCC, AKIEC, BKL, DF, VASC) pada **ISIC 2018 Task 3** dengan PyTorch.

Notebook: [`kaggle/isic2018_resnet_pipeline.ipynb`](kaggle/isic2018_resnet_pipeline.ipynb)
(berbahasa Indonesia, siap dijalankan lokal maupun di Kaggle). Script sumber:
`src/isic2018_resnet_pipeline.py`. Pendamping analisis data awal:
[`kaggle/isic2018_eda.ipynb`](kaggle/isic2018_eda.ipynb) (7 tahap EDA, lihat
[`docs/eda.md`](docs/eda.md)).

## Fitur Utama

- **Config-first** — semua hyperparameter (path, kelas, split, balancing, normalisasi,
  training, arsitektur) ada di satu blok `CONFIG` (Section 1).
- **Model ResNet configurable** — depth (18/34/50/101), jenis residual block
  (`basic`/`bottleneck`), base channels, classifier hidden dim. Bobot pretrained ImageNet
  dipakai otomatis untuk kombinasi arsitektur standar; kombinasi lain dibangun dari nol.
- **Pencatatan run lengkap** — setiap run menyimpan skema training-test-eval + seluruh
  hyperparameter + arsitektur ke `run_<nama>.json` dan baris ringkas di `runs_log.csv`.
- **Ringkasan arsitektur ala Keras** — `model_summary` menampilkan tabel per layer (nama, tipe,
  output shape, kuota parameter, parameter trainable) — memakai `torchinfo` bila tersedia,
  fallback ke helper kustom tanpa dependency; tabel tersimpan ke
  `output/arch_summary_<experiment_name>.csv`.
  - **Eksperimen = 1 model (bukan ablation)** — hanya ada satu cell training (Section 14).
  Untuk eksperimen, ubah nilai hyperparameter langsung di `CONFIG`, ganti `experiment_name`
  di CONFIG (nama run/output), lalu jalankan ulang cell yang sama. Satu eksperimen menghasilkan
  satu model/run; perbandingan dibaca dari `runs_log.csv`.
- **Bobot loss otomatis** — `loss_weight_mode: 'inverse_frequency'` menghitung bobot
  CrossEntropy (ekivalen `class_weight='balanced'` sklearn) dari distribusi training
  **setelah** balancing; atau isi `loss_weight` manual per kelas. Bobot efektif tercatat di
  `run_<nama>.json`.
- **Evaluasi menyeluruh** — training/validation split stratified, plus evaluasi di
  **test set resmi** dan **validation set resmi** (193 label). Setiap evaluasi menyimpan
  metrik & classification report CSV, confusion matrix PNG, prediksi per gambar CSV, dan
  **sampel salah klasifikasi** (CSV + grid gambar).
- **Kaggle-ready** — autodetect `/kaggle/input`; semua output tersimpan ke `output/`.
- **EDA bawaan** — distribusi kelas + statistik deskriptif gambar untuk laporan metodologi.
- **Notebook EDA pendamping** — `kaggle/isic2018_eda.ipynb`: 7 tahap EDA
  (distribusi kelas, analisis `lesion_id` & risiko leakage, visualisasi per kelas, resolusi &
  aspect ratio, distribusi warna + sanity-check ColorJitter, duplikat/near-duplikat, dan
  ukuran split per kelas). Analisis read-only, artefak ke `output_eda/` (lihat `docs/eda.md`).

## Isi Repo

```
├── src/isic2018_resnet_pipeline.py  # script sumber pipeline
├── kaggle/isic2018_resnet_pipeline.ipynb   # notebook utama
├── kaggle/isic2018_eda.ipynb        # notebook EDA (7 tahap)
├── docs/pipeline.md                 # dokumentasi alur pipeline
├── docs/eda.md                      # dokumentasi tujuan & tahapan EDA
├── CHANGELOG.md                     # riwayat perubahan
├── dataset/                         # dataset ISIC 2018 Task 3 (lokal)
└── output/                          # hasil run
```

## Cara Menjalankan

1. **Lokal**: `python src/isic2018_resnet_pipeline.py` (perlu PyTorch + torchvision).
2. **Pipeline**: buka `kaggle/isic2018_resnet_pipeline.ipynb`, jalankan cell 1–19 berurutan.
3. **EDA (opsional)**: buka `kaggle/isic2018_eda.ipynb`, jalankan cell berurutan; hasil ke
   `output_eda/`. Analysis `lesion_id` aktif bila `HAM10000_metadata.csv` ada di `dataset/`
   atau ditemukan di `/kaggle/input`.
4. **Kaggle**: upload dataset berisi folder standar ISIC 2018 Task 3
   (`ISIC2018_Task3_Training_Input`, `..._Test_Input`, `..._Validation_Input`, dan folder
   `*_GroundTruth`). Path terdeteksi otomatis.

## Kebutuhan

- Python 3.9+; PyTorch, torchvision, pandas, numpy, matplotlib, seaborn, scikit-learn,
  Pillow.
- GPU direkomendasikan (untuk 20 epoch × batch 32 di dataset 10 ribu gambar).

## Referensi

Studi ini mengimplementasikan klasifikasi pada dataset yang diusulkan oleh ISIC
Challenge 2018 (juga terkait HAM10000):

- Codella dkk., "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted
  by the International Skin Imaging Collaboration (ISIC)", 2018.
  <https://arxiv.org/abs/1902.03368>
- Tschandl, Rosendahl & Kittler, "The HAM10000 dataset, a large collection of
  multi-source dermatoscopic images of common pigmented skin lesions", Scientific Data
  5, 180161 (2018). <https://doi.org/10.1038/sdata.2018.161>

Lihat juga `docs/pipeline.md` untuk alur lengkap dan daftar output.