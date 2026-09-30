# Klasifikasi Skin Lesion — ISIC 2018 Task 3

Pipeline pelatihan & evaluasi model **ResNet** untuk klasifikasi **7 kelas diagnosis lesi
kulit** (MEL, NV, BCC, AKIEC, BKL, DF, VASC) pada **ISIC 2018 Task 3** dengan PyTorch.

Notebook: [`notebooks/isic2018_resnet_pipeline.ipynb`](notebooks/isic2018_resnet_pipeline.ipynb)
(berbahasa Indonesia, siap dijalankan lokal maupun di Kaggle) adalah sumber kebenaran alur
training; `src/isic2018_resnet_pipeline.py` adalah script versi lama (referensi).
Pendamping analisis data awal:
[`notebooks/isic2018_eda.ipynb`](notebooks/isic2018_eda.ipynb) (7 tahap EDA, lihat
[`docs/eda.md`](docs/eda.md)).

## Fitur Utama

- **Config-first** — semua hyperparameter (path, kelas, split, balancing, normalisasi,
  training, arsitektur) ada di satu blok `CONFIG` (Section 1).
- **Dua skema split anti-leakage per-lesion (`CONFIG['split_scheme']`)** — karena
  10.015 gambar training hanya berisi **7.470 lesi** (1.956 lesi multi-gambar), split
  per-gambar membocorkan lesi ke train & validation. Pilihan:
  - **`"kfold"` (default, `k_folds=5`)** — `StratifiedGroupKFold` grouped per-lesion pada
    training saja. 193 gambar **validation resmi tidak pernah dilatih** dan diperlakukan
    sebagai **public test**; 1.512 gambar **test resmi = private test**. Menghasilkan
    `k_folds` model (`<experiment_name>_f0`…`_f<k-1>`) + `kfold_summary_*.csv`.
  - **`"holdout"` (`val_ratio=0.15`)** — training di-split sekali per-lesi lalu
    **dimasukkan ke validation bersama 193 gambar validation resmi**; test resmi tetap
    private test. Tidak ada public test independen. Menghasilkan 1 model.
  - Keduanya grouped per-lesi memakai `lesion_id` dari
    `ISIC2018_Task3_Training_LesionGroupings.csv` (ditemukan otomatis di `dataset/` atau
    rekursif `/kaggle/input/**`, termasuk layout bersarang
    `/kaggle/input/datasets/<user>/<dataset>/ISIC2018_Task3_Training_Input/...`): kfold
    pakai `StratifiedGroupKFold`/`GroupKFold`, holdout pakai `train_test_split`
    stratified per-lesi.
    Ringkasan + cek overlap lesi → `split_summary.csv`.
  - **193 gambar validation resmi tidak punya `lesion_id`, jadi tidak pernah dipakai
    training** (opsi gabung ke training sengaja tidak disediakan).
- **Model ResNet configurable** — depth (18/34/50/101), jenis residual block
  (`basic`/`bottleneck`), base channels, classifier hidden dim. Bobot pretrained ImageNet
  dipakai otomatis untuk kombinasi arsitektur standar; kombinasi lain dibangun dari nol.
- **Pencatatan run lengkap** — setiap run (per fold) menyimpan blok `scheme` (skema split,
  fold, komposisi train/validation/public/private test) + seluruh
  hyperparameter + arsitektur ke `run_<nama>.json` dan baris ringkas di `runs_log.csv`.
  Kolom `split_scheme` & `fold` ditambahkan ke `runs_log.csv`.
- **Ringkasan arsitektur ala Keras** — `model_summary` menampilkan tabel per layer (nama, tipe,
  output shape, kuota parameter, parameter trainable) — memakai `torchinfo` bila tersedia,
  fallback ke helper kustom tanpa dependency; tabel tersimpan ke
  `output/arch_summary_<experiment_name>.csv`.
  - **Eksperimen = 1 model per fold (bukan ablation)** — hanya ada satu cell training
  (Section 14) yang looping semua split job. Untuk eksperimen, ubah nilai hyperparameter
  langsung di `CONFIG`, ganti `experiment_name` di CONFIG (nama run/output), lalu jalankan
  ulang cell yang sama. Satu eksperimen menghasilkan satu model per fold; perbandingan
  dibaca dari `runs_log.csv`.
- **Model k-fold ditahan di Section 14** — model, history, dan run record tiap fold
  disimpan di `RUN_RESULTS` (selama sesi kernel yang sama) untuk dipakai Section 18–19.
- **Bobot loss otomatis** — `loss_weight_mode: 'inverse_frequency'` menghitung bobot
  CrossEntropy (ekivalen `class_weight='balanced'` sklearn) dari distribusi training
  **setelah** balancing; atau isi `loss_weight` manual per kelas. Bobot efektif tercatat di
  `run_<nama>.json`.
- **Oversample kelas minoritas via augmentasi acak** — sampel minoritas direplikasi hingga
  `oversample_target` sample per kelas; tiap salinan di-augmentasi acak **hanya saat training**:
  rotasi (`aug_rotation_range`, default 20°), pergeseran H/V (`aug_width/height_shift_range`,
  default 0.2), flip horizontal (`aug_horizontal_flip`, default true). Target 0 = replikasi
  dilewati (kelas minoritas tetap di-augmentasi dari sampel asli). Aktif via `oversample_augment`.
- **Resize sesuai rasio data** — `resize_mode`: `'stretch'` (tekan ke persegi, default) |
  `'center_crop'` | `'random_crop'` (pertahankan aspek; random crop hanya saat training,
  evaluasi selalu center-crop agar deterministik) — menyikapi temuan EDA resolusi 600×450
  (4:3).
- **Validation objective** — `val_objective` memilih metrik untuk memilih & memonitor model
  terbaik selama training: `'accuracy' | 'balanced_accuracy' | 'macro_f1'` (dihitung tanpa
  dependency sklearn; kurva objective ikut di-plot dan dicatat di run).
- **LR scheduler `ReduceLROnPlateau` + early stopping** — keduanya memantau metrik
  `val_monitor` (default `'macro_f1'` = *validation_macro_f1*) dengan `scheduler_mode: 'max'`
  (metrik naik = lebih baik). Scheduler memangkas LR setelah `scheduler_patience` epoch stagnan
  (`scheduler_factor`, `scheduler_min_lr`); training dihentikan setelah
  `early_stopping_patience` (default `10`) epoch tanpa perbaikan (`early_stopping_min_delta`).
  Kurva learning rate, `epochs_ran`, `best_epoch`, dan `stopped_early` dicatat di history,
  `run_<nama>.json`, `runs_log.csv`, dan di-plot.
- **Evaluasi menyeluruh** — validation split selalu grouped per-lesion, plus evaluasi
  **private test** (test resmi, 1.512 label) untuk setiap model/fold dan **public test**
  (validation resmi, 193 label — hanya pada skema k-fold). Setiap evaluasi menyimpan
  metrik & classification report CSV, confusion matrix PNG, prediksi per gambar CSV, dan
  **sampel salah klasifikasi** (CSV + grid gambar); rata-rata antar fold disimpan di
  `test_private_metrics_all.csv` dan `test_public_metrics_all.csv`.
- **Kaggle-ready** — autodetect `/kaggle/input`; semua output tersimpan ke `output/`.
- **EDA bawaan** — distribusi kelas + statistik deskriptif gambar untuk laporan metodologi.
- **Notebook EDA pendamping** — `notebooks/isic2018_eda.ipynb`: 7 tahap EDA
  (distribusi kelas, analisis `lesion_id` & risiko leakage, visualisasi per kelas, resolusi &
  aspect ratio, distribusi warna + sanity-check ColorJitter, duplikat/near-duplikat, dan
  perbandingan 4 skenario split per kelas). Sumber `lesion_id`:
  `ISIC2018_Task3_Training_LesionGroupings.csv` (fallback `HAM10000_metadata.csv`).
  Analisis read-only, artefak ke `output_eda/` (lihat `docs/eda.md`).

## Isi Repo

```
├── src/isic2018_resnet_pipeline.py  # script versi lama (referensi)
├── notebooks/isic2018_resnet_pipeline.ipynb  # notebook utama
├── notebooks/isic2018_eda.ipynb               # notebook EDA (7 tahap)
├── docs/pipeline.md                 # dokumentasi alur pipeline
├── docs/eda.md                      # dokumentasi tujuan & tahapan EDA
├── docs/eksperimen.md               # rencana eksperimen & ablation (roadmap)
├── CHANGELOG.md                     # riwayat perubahan
├── dataset/                         # dataset ISIC 2018 Task 3 (lokal)
└── output/                          # hasil run
```

## Cara Menjalankan

1. **Pipeline**: buka `notebooks/isic2018_resnet_pipeline.ipynb`, jalankan Section 1–19
   berurutan. Section 6 mencetak skema split yang dipakai; Section 14 melatih semua fold.
2. **Skema split**: set `CONFIG['split_scheme']` = `"kfold"` (default) atau `"holdout"`
   di Section 1. `k_folds` hanya dipakai pada `"kfold"`; `val_ratio` hanya dipakai pada
   `"holdout"`.
3. **EDA (opsional)**: buka `notebooks/isic2018_eda.ipynb`, jalankan cell berurutan; hasil ke
   `output_eda/`. Analisis `lesion_id` aktif bila
   `ISIC2018_Task3_Training_LesionGroupings.csv` ada di `dataset/` atau ditemukan di
   `/kaggle/input`.
4. **Kaggle**: upload dataset berisi folder standar ISIC 2018 Task 3
   (`ISIC2018_Task3_Training_Input`, `..._Test_Input`, `..._Validation_Input`, folder
   `*_GroundTruth`, dan `ISIC2018_Task3_Training_LesionGroupings.csv`). Path terdeteksi
   otomatis — baik layout datar maupun bersarang
   (`/kaggle/input/datasets/<user>/<dataset>/ISIC2018_Task3_Training_Input/...`).
5. **Script (opsional)**: `src/isic2018_resnet_pipeline.py` — versi lama, belum mendukung
   `split_scheme`.

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

Lihat juga `docs/pipeline.md` untuk alur lengkap dan daftar output, serta
`docs/eksperimen.md` untuk **rencana eksperimen** berikutnya (perbandingan baseline,
attention squeeze-and-excitation, strategi fine-tuning backbone, dan ablation).