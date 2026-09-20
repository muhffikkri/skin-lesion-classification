# Klasifikasi Skin Lesion — ISIC 2018 Task 3

Pipeline pelatihan & evaluasi model **ResNet** untuk klasifikasi **7 kelas diagnosis lesi
kulit** (MEL, NV, BCC, AKIEC, BKL, DF, VASC) pada **ISIC 2018 Task 3** dengan PyTorch.

Notebook utama: [`isic2018_resnet_pipeline.ipynb`](isic2018_resnet_pipeline.ipynb)
(berbahasa Indonesia, siap dijalankan lokal maupun di Kaggle).

## Fitur Utama

- **Config-first** — semua hyperparameter (path, kelas, split, balancing, normalisasi,
  training, arsitektur, nilai eksperimen) ada di satu blok `CONFIG` (Section 1).
- **Model ResNet configurable** — depth (18/34/50/101), jenis residual block
  (`basic`/`bottleneck`), base channels, classifier hidden dim. Bobot pretrained ImageNet
  dipakai otomatis untuk kombinasi arsitektur standar; kombinasi lain dibangun dari nol.
- **Pencatatan run lengkap** — setiap run menyimpan skema training-test-eval + seluruh
  hyperparameter + arsitektur ke `run_<nama>.json` dan baris ringkas di `runs_log.csv`.
- **Eksperimen nilai tunggal** — 7 jenis eksperimen (LR, batch size, dropout, depth,
  residual block, classifier hidden dim, base channels); ganti nilai di CONFIG lalu jalankan
  ulang cell yang bersangkutan.
- **Evaluasi menyeluruh** — training/validation split stratified, plus evaluasi di
  **test set resmi** dan **validation set resmi** (193 label). Setiap evaluasi menyimpan
  metrik & classification report CSV, confusion matrix PNG, prediksi per gambar CSV, dan
  **sampel salah klasifikasi** (CSV + grid gambar).
- **Kaggle-ready** — autodetect `/kaggle/input`; semua output tersimpan ke `output/`.
- **EDA bawaan** — distribusi kelas + statistik deskriptif gambar untuk laporan metodologi.

## Isi Repo

```
├── isic2018_resnet_pipeline.ipynb   # notebook utama
├── src/isic2018_resnet_pipeline.py  # script sumber pipeline
├── docs/pipeline.md                 # dokumentasi alur (flow v1)
├── CHANGELOG.md                     # riwayat perubahan
├── dataset/                         # dataset ISIC 2018 Task 3 (lokal)
├── output/                          # hasil run
└── kaggle/                          # salinan notebook untuk upload Kaggle
```

## Cara Menjalankan

1. **Lokal**: `python src/isic2018_resnet_pipeline.py` (perlu PyTorch + torchvision).
2. **Notebook**: buka `isic2018_resnet_pipeline.ipynb`, jalankan cell 1–19 berurutan.
3. **Kaggle**: upload dataset berisi folder standar ISIC 2018 Task 3
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