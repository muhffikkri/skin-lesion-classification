# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Pemilihan skema split di pipeline (`CONFIG['split_scheme']`)** — dua skema, keduanya
  **grouped per-lesion** dan keduanya **tidak pernah memakai 193 gambar validation resmi
  untuk training**:
  - **`"kfold"` (default, `k_folds=5`)** — `StratifiedGroupKFold` pada training saja
    (`shuffle=True`, `random_state` dari CONFIG, fallback `GroupKFold`). 193 gambar
    validation resmi = **public test**; 1.512 gambar test resmi = **private test**;
    menghasilkan `k_folds` model (`<experiment_name>_f0`…`_f<k-1>`).
  - **`"holdout"` (`val_ratio=0.15`)** — training di-split sekali per-lesi lalu
    **dimasukkan ke validation bersama 193 gambar validation resmi**; test resmi tetap
    private test; tidak ada public test independen.
  - Struktur baru di Section 5–12: `SPLIT_JOBS` (satu dict per fold/job berisi `train`,
    `val`, `fold`, `train_balanced`), `TEST_PUBLIC_DF` (kfold), `TEST_PRIVATE_DF`,
    balancing/oversampling **per split job**, `run_training(train_data, val_data,
    train_balanced, fold)`, serta `RUN_RESULTS` (model per fold untuk Section 18–19).
  - Evaluasi per fold + agregat: `test_private_metrics_all.csv` (private test) dan
    `test_public_metrics_all.csv` (public test, hanya kfold); prefix `public_*` menggantikan
    `validation_*`. Output baru: `split_summary.csv`, `kfold_summary_<experiment_name>.csv`.
    `runs_log.csv` dapat kolom baru `split_scheme` & `fold`; `run_<run>.json` dapat blok
    `scheme`.
  - Key CONFIG baru: `split_scheme`, `k_folds`, `lesion_groupings`,
    `lesion_groupings_file`, `ham10000_metadata` (resolver sama seperti di EDA: path
    eksplisit → `dataset/` → rekursif `/kaggle/input/**`).
- **Skenario split lesion-aware di EDA Tahap 7** — perbandingan 4 skenario dengan jumlah
  **per kelas**: (1) stratified split per-image (skema pipeline lama), (2) stratified
  split per-lesion, (3) **k-fold grouped per-lesion** pada training saja via
  `StratifiedGroupKFold`, dan (4) **holdout**: split `val_ratio` + validation resmi
  digabung (meniru `split_scheme="holdout"`). Dilaporkan jumlah gambar per kelas tiap fold
  (min..max) untuk mengecek keseimbangan. Key EDA_CFG baru: **`k_folds`** (default `5`) dan
  **`combine_train_val`** (default `false`, dokumentasi saja — tidak memengaruhi pipeline).
  Output baru: `eda_split_lesion_counts.csv`, `eda_split_leak_summary.csv`,
  `eda_split_holdout_counts.csv`, `eda_split_compare.png`, `eda_cv_folds.csv`.
- **Sumber `lesion_id` resmi** — EDA kini memakai
  `ISIC2018_Task3_Training_LesionGroupings.csv` (rilis resmi Task 3) sebagai sumber utama,
  dengan `HAM10000_metadata.csv` hanya sebagai fallback. Pencarian otomatis: path eksplisit
  (`EDA_CFG['lesion_groupings']`) → `dataset/` → rekursif `/kaggle/input/**` (menangani path
  bersarang `/kaggle/input/datasets/<user>/<dataset>/`). Key EDA_CFG baru:
  `lesion_groupings`, `lesion_groupings_file`, `ham10000_metadata`.
- **Distribusi kelas dua skenario (Tahap 1)** — Skenario A memakai partisi rilis asli
  (train / validation resmi / test resmi, `eda_class_dist.csv`), Skenario B memakai training
  yang dipecah jadi train+val (`eda_class_dist_split.csv`), plus plot dua panel
  `eda_class_dist.png`.

### Fixed
- **Resolusi folder training yang duplikat** (EDA + pipeline) — di Kaggle, container dan
  folder gambar keduanya bernama `ISIC2018_Task3_Training_Input`, sehingga resolver memilih
  container yang kosong (`Train gambar: 0`, sampel grid tidak muncul, `KeyError: 'width'`
  di EDA Tahap 4). Kini `_find_by_name` memilih kandidat yang **benar-benar berisi file
  gambar** (`_dir_score`, kandidat langsung kosong diabaikan) → `TRAIN_IMG_DIR` menunjuk folder
  gambar sebenarnya dan `DATA_DIR` = induknya.
- **Guard folder gambar kosong** (EDA + pipeline) — Section 1 mencetak jumlah gambar per
  partisi + peringatan bila 0; EDA Tahap 3/4/5/5b/6 dan EDA Data Characteristics pipeline
  melewatkan analisis dengan pesan jelas (bukan crash) saat folder kosong; DataFrame diberi
  kolom eksplisit (`["image","width","height"]`, `["image","split","md5"]`) dan samsel
  pipeline di-clamp ke jumlah data + difilter hanya file yang ada.
- **Resolusi path dataset di Section 1 (EDA + pipeline)** — sebelumnya folder test/validation/
  ground truth hanya dicari sebagai anak langsung `data_dir`, sehingga gagal pada dataset
  Kaggle yang tersarang
  `/kaggle/input/datasets/<user>/<dataset>/ISIC2018_Task3_Training_Input/` + semua folder
  (`AssertionError: Test: folder tidak ada -> .../ISIC2018_Task3_Test_Input`). Kini setiap
  folder & CSV dicari **berdasarkan nama**: anak langsung `data_dir` → satu level di atasnya
  → `/kaggle/input` → penelusuran rekursif (folder penuh gambar dipangkas agar cepat).
  `data_dir` boleh path relatif (`dataset`), path lengkap Kaggle, atau kosong; pesan error
  menyebut folder yang tidak ditemukan. Pencarian `LesionGroupings`/`HAM10000_metadata`
  memakai resolver yang sama dan memaafkan typo nama file (mis.
  `...LesionGroupings.csv.csv`) maupun path penuh di `lesion_groupings_file`.
- **Redaksi README soal algoritma split** — holdout memakai `train_test_split` stratified
  per-lesi, bukan `StratifiedGroupKFold`/`GroupKFold` (hanya kfold yang memakainya).

### Changed
- `docs/pipeline.md` — Section 1/6/7/12/14/16/18/19 ditulis ulang untuk skema split;
  ditambah tabel perbandingan kfold vs holdout, tabel output baru, dan catatan evaluasi
  (193 gambar = public test hanya pada kfold). Klaim generator `build_resnet_notebook.py`
  dihapus: notebook adalah sumber kebenaran, `src/isic2018_resnet_pipeline.py` script lama.
- `README.md` — fitur skema split per-lesion, alur Section 14 yang looping fold, dan
  petunjuk menjalankan pipeline/EDA.
- `docs/eda.md` — Tahap 1 & 2 & 7 ditulis ulang mengikuti kode baru; snapshot diisi angka
  asli: 10.015 gambar = **7.470 lesi** (1.956 lesi multi-gambar, maks. 6, 4.501 gambar
  terdampak, 0 konflik label), **leakage 589 lesi / 1.420 gambar (14,2%)** pada split
  per-image vs **0** pada split per-lesion, tabel fold k-fold 5, serta tabel Skenario 4
  (holdout: 8.518 train / 1.497 + 193 val).

### Known limitations
- `LesionGroupings` hanya mencakup training. Gambar validation resmi (193) dan test resmi
  (1.512) tidak punya `lesion_id`, sehingga lesi lintas training↔validation/test **tidak
  bisa disingkirkan** untuk 193 gambar tersebut. Karena itu **193 gambar tidak pernah
  dipakai training** di kedua skema split; pada skema holdout risikonya tetap ada pada
  validation set dan tidak ada public test independen.
- Belum ada agregasi OOF (out-of-fold) pada skema k-fold: `kfold_summary_*.csv` berisi
  rata-rata ± std metrik tiap fold, bukan metrik OOF gabungan.

- **LR scheduler `ReduceLROnPlateau` + early stopping** — keduanya memantau metrik
  `val_monitor` (default `'macro_f1'` = *validation_macro_f1*) dengan `scheduler_mode: 'max'`
  (nilai metrik naik = lebih baik). Key CONFIG baru: `val_monitor`, `lr_scheduler`
  (`"ReduceLROnPlateau" | "none"`), `scheduler_mode`, `scheduler_factor` (default `0.1`),
  `scheduler_patience` (default `5`), `scheduler_min_lr` (default `1e-6`), `early_stopping`
  (default `true`), **`early_stopping_patience` (default `10`)**, dan
  `early_stopping_min_delta` (default `0.0`). `scheduler.step(monitor)` dipanggil sekali per
  epoch setelah evaluasi validation; training di-`break` saat metrik stagnan melewati
  `early_stopping_patience`. `history` menambah `val_monitor` & `val_lr`;
  `plot_history` jadi grid 2×2 (loss, accuracy+objective, kurva monitor, kurva LR log);
  `run_<nama>.json` menambah blok `monitoring` + `epochs_ran`/`best_epoch`/`stopped_early`;
  `runs_log.csv` menambah `epochs_ran`, `best_epoch`, `stopped_early`, `val_monitor`,
  `scheduler`, `scheduler_patience`, `es_patience`, `final_lr`. Checkpoint terbaik tetap
  dipilih berdasarkan `val_objective`.
- **Oversample kelas minoritas via augmentasi acak** — `oversample_augment` (default
  `true`) mereplikasi sampel kelas minoritas hingga **`oversample_target` sample per kelas**;
  tiap salinan di-augmentasi acak **hanya saat training** melalui CONFIG:
  `aug_rotation_range` (default `20` derajat), `aug_width_shift_range`/`aug_height_shift_range`
  (default `0.2`), `aug_horizontal_flip` (default `true`). `oversample_target: 0`
  melewati replikasi (minoritas tetap di-augmentasi dari sampel asli). Diterapkan di
  Section 7 (`oversample_minority`) sebelum transform Section 8.
- **Resize mode** — `resize_mode: 'stretch' | 'center_crop' | 'random_crop'`. Mode crop
  mempertahankan aspek (resize sisi terpendek lalu potong); `random_crop` hanya aktif saat
  training, evaluasi selalu `CenterCrop` agar deterministik — menyikapi temuan EDA
  resolusi 600×450 (4:3).
- **Validation objective** — `val_objective: 'accuracy' | 'balanced_accuracy' | 'macro_f1'`
  dipakai untuk memilih & memonitor model terbaik (`model_*_best.pt`), di-plot di
  `plot_history`, dan dicatat di `run_*.json` / `runs_log.csv`. Dihitung via `compute_metric`
  tanpa dependency sklearn. Evaluasi final juga melaporkan balanced accuracy & macro F1
  (`test_metrics.csv` / `validation_metrics.csv`).
- **Roadmap eksperimen** — `docs/eksperimen.md` berisi rencana perbandingan baseline
  (CNN / ResNet / Transformer), eksperimen attention squeeze-and-excitation dengan 4
  strategi fine-tuning (`layer4`, `layer3+layer4`, full, frozen backbone), dan ablation
  (CNN → CNN+ResNet → CNN+ResNet+CrossEntropy).

## [v1.6.0] - 2026-09-24

### Added
- **Ringkasan arsitektur model ala Keras** (Section 10.1): `model_summary` menampilkan tabel
  per layer — nama, tipe, output shape (dari satu forward pass dummy), jumlah parameter, dan
  parameter trainable. Memakai `torchinfo` bila tersedia, fallback ke helper kustom tanpa
  dependency; tabel disimpan ke `output/arch_summary_<experiment_name>.csv`.
- **Notebook EDA pendamping** — `kaggle/isic2018_eda.ipynb` (22 cell, read-only, artefak ke
  `output_eda/`): 7 tahap EDA —
  1) distribusi kelas (termasuk rasio tiap kelas vs kelas terkecil & terbesar), 2) analisis
  `lesion_id` (jumlah image / unique lesion / images per
  lesion + risiko data leakage saat split, aktif bila `HAM10000_metadata.csv` tersedia),
  3) visualisasi 3 gambar per kelas, 4) resolusi & aspect ratio + diskusi resize 224×224
  (cost, detail, receptive field, memori, batch), 5) distribusi warna (RGB histogram &
  brightness/contrast/saturation) + sanity-check ColorJitter via `PIL.ImageEnhance`,
  6) duplikat eksak (md5) & near-duplikat (dhash) antar partition, 7) ukuran split per kelas
  + opsi split berbasis `lesion_id`. Dependensi: pandas/numpy/matplotlib/PIL (+
  scikit-learn untuk tahap 7). Deskripsi lengkap di `docs/eda.md`.
- **Snapshot hasil EDA** di `docs/eda.md` — distribusi & rasio kelas (NV:DF 58.3:1),
  resolusi seragam 600×450 (4:3 → center-crop/pad), warna (R dominan, saturation ~0.31),
  sanity-check ColorJitter ±0.2 (wajar), dan ukuran split per kelas (DF: 17/val, 44/test).

### Changed
- Nama eksperimen kini dikontrol melalui CONFIG (`experiment_name`, Section 1) — dipakai
  untuk seluruh output run (`model_<nama>_best.pt`, `run_<nama>.json`, file plot,
  `runs_log.csv`). Tidak ada lagi `run_name` hardcoded di cell training; eksperimen cukup
  mengubah `experiment_name` + nilai hyperparameter di CONFIG lalu menjalankan ulang
  Section 14. Variabel hasil training diubah `baseline_*` → `trained_*`.
- `loss_weight_mode: 'inverse_frequency'` kini menghitung bobot dari distribusi training
  **setelah balancing** (`train_df_balanced`), bukan training asli — bobot relatif terhadap
  set yang benar-benar dilatih. `run_<nama>.json` mencatat sumbernya
  (`loss.loss_weight_distribution`).
- Opsi optimizer bertambah: `'adamw'` (`optim.AdamW`) selain `'adam' | 'sgd' | 'rmsprop'`.

### Fixed
- `downsample_majority` dengan `balance_threshold: 0` (balancing nonaktif) sebelumnya
  mengosongkan seluruh data; kini threshold `<= 0` mengembalikan data apa adanya.

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