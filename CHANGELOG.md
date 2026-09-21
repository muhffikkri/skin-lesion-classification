# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Ringkasan arsitektur model ala Keras** (Section 10.1): `model_summary` menampilkan tabel
  per layer — nama, tipe, output shape (dari satu forward pass dummy), jumlah parameter, dan
  parameter trainable. Memakai `torchinfo` bila tersedia, fallback ke helper kustom tanpa
  dependency; tabel disimpan ke `output/arch_summary_<experiment_name>.csv`.

### Changed
- Nama eksperimen kini dikontrol melalui CONFIG (`experiment_name`, Section 1) — dipakai
  untuk seluruh output run (`model_<nama>_best.pt`, `run_<nama>.json`, file plot,
  `runs_log.csv`). Tidak ada lagi `run_name` hardcoded di cell training; eksperimen cukup
  mengubah `experiment_name` + nilai hyperparameter di CONFIG lalu menjalankan ulang
  Section 14. Variabel hasil training diubah `baseline_*` → `trained_*`.

## [v1.5] - 2026-09-21

### Changed
- **Eksperimen = 1 model, bukan ablation**: hapus 7 cell eksperimen terpisah (15a–15g) dan
  `exp_results`/`hyperparameter_summary.csv`. Kini hanya ada satu cell training (Section 14);
  eksperimen dilakukan dengan mengubah nilai hyperparameter **langsung di `CONFIG`** (Section 1)
  lalu mengganti **`run_name` hardcoded** pada pemanggilan `run_training`, dan menjalankan
  ulang cell yang sama. Setiap eksperimen = 1 run = 1 model (`model_<run_name>_best.pt`).
- Hapus key `*_experiment` dan `experiment_epochs` dari `CONFIG`.
- Section 15 jadi panduan eksperimen + print konfigurasi efektif; Section 16 membaca
  ringkasan dari `runs_log.csv`.
- Generator notebook kini menulis langsung ke `kaggle/isic2018_resnet_pipeline.ipynb`.

### Added
- **Bobot loss otomatis relatif**: key `loss_weight` (manual per kelas) dan
  `loss_weight_mode` (`'none'` / `'inverse_frequency'`). Mode inverse_frequency menghitung
  bobot `total/(n_kelas×frekuensi)` dari distribusi **training asli** (sebelum balancing);
  `build_criterion()` dipakai konsisten untuk training & semua evaluasi. Bobot efektif
  tercatat di `run_<nama>.json` (`loss.loss_weight_applied`).

## [v1.4] - 2026-09-21

### Added
- **Tuning arsitektur ResNet**: depth (18/34/50/101), jenis residual block
  (`basic`/`bottleneck`), base channels, dan classifier hidden dim — nilai tunggal
  diubah manual via `CONFIG` (`depth`, `residual_blocks`, `base_channels`,
  `classifier_hidden_dim`) atau melalui 4 eksperimen baru di Section 15
  (15d depth, 15e residual blocks, 15f classifier hidden, 15g base channels).
- **Model configurable** (`build_model` + `ResNetCustom`): bobot pretrained ImageNet
  dipakai otomatis hanya untuk kombinasi arsitektur standar; kombinasi lain dibangun
  dari nol. `arch_label` + jumlah parameter dicetak saat sanity check.
- **Pencatatan run lengkap**: `run_training` kembali `(model, history, run_record)` dan
  menyimpan skema training-test-eval + seluruh hyperparameter + arsitektur + snapshot
  `CONFIG` ke `run_<nama>.json`; baris ringkas direkam di `runs_log.csv` (re-run
  menggantikan baris yang sama).
- **Evaluasi bersama** (`evaluate_and_report`): dipakai untuk test resmi dan validation
  resmi — menyimpan metrics CSV, classification report CSV, confusion matrix PNG,
  prediksi per-gambar CSV, serta **sampel salah klasifikasi** (CSV + grid gambar).
- Dokumen: `docs/pipeline.md` (flow v1), `README.md`, `CHANGELOG.md`.
- Referensi [1] (ISIC 2018 challenge, arXiv:1902.03368) dan [2] (HAM10000,
  doi:10.1038/sdata.2018.161) ditambahkan di notebook.

## [v1.3] - 2026-09-21

### Added
- **Validation set resmi** (193 label) dimuat dan dipakai untuk evaluasi final terpisah
  (Section 18): metrics CSV, classification report CSV, confusion matrix PNG, prediksi
  per-gambar CSV. `val_gt_dir`/`VAL_GT_PATH` otomatis dicari di `*_GroundTruth`.
- EDA mencantumkan distribusi kelas validation resmi.

### Changed
- Bersih-bersih komentar: komentar instruksional/meta (AI-style) dihapus; komentar
  penjelas dipertahankan.
- `val_df_official` diberi kolom `label` dari `class_to_idx` agar evaluasi val resmi
  konsisten.

## [v1.2] - 2026-09-21

### Added
- Kaggle-ready: `_resolve_data_root` autodetect `/kaggle/input`; CONFIG memakai key
  subfolder (`train_img_dir`, `test_img_dir`, dst.) + `output_dir`.
- Output tersimpan pereksperimen: checkpoint `model_<run>_best.pt`, history JSON,
  plot kurva PNG, `hyperparameter_summary.csv`, dan hasil evaluasi test set (`test_*`).

## [v1.1] - 2026-09-21

### Changed
- Eksperimen hyperparameter dikonversi dari array/loop menjadi **nilai tunggal**
  (`lr_experiment`, `batch_size_experiment`, `dropout_experiment`); hasil re-run
  menggantikan baris sebelumnya di tabel ringkasan (Section 16).
- `exp_results` menambah kolom Best Val Acc.

## [v1.0] - 2026-09-20

### Added
- Notifikasi: pipeline versi 1 (ResNet-18 pretrained ImageNet) untuk ISIC 2018 Task 3.
- Section 1–19 berbasis script (`src/isic2018_resnet_pipeline.py`): CONFIG, EDA,
  split stratified, balancing opsional, augmentasi, dataset, training baseline,
  eksperimen hyperparameter, ringkasan, evaluasi test resmi.
- Generator notebook (`build_resnet_notebook.py`) sebagai source of truth notebook;
  validasi syntax otomatis.