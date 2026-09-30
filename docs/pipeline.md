# Pipeline: Klasifikasi Skin Lesion (ISIC 2018 Task 3)

Dokumentasi alur terkini (v1.8) dari notebook `notebooks/isic2018_resnet_pipeline.ipynb`.
Notebook adalah **sumber kebenaran** alur training; `src/isic2018_resnet_pipeline.py` adalah
script lama yang dipakai sebagai referensi (bukan generator notebook).

Rujukan tugas: EMED (7 kelas diagnosis) pada ISIC 2018 Task 3
(diusulkan oleh ISIC Challenge 2018, lihat referensi [1] dan [2] di notebook).

## Struktur Repo

```
skin-lesion-classification/
├── notebooks/isic2018_resnet_pipeline.ipynb  # notebook utama (output akhir)
├── notebooks/isic2018_eda.ipynb               # notebook EDA (acuan keputusan split)
├── src/isic2018_resnet_pipeline.py        # script referensi (versi lama, tidak di-generate)
├── docs/pipeline.md                       # dokumen ini
├── README.md
├── CHANGELOG.md
├── dataset/                               # dataset lokal (folder standar ISIC 2018 Task 3)
└── output/                                # hasil run (model, JSON, CSV, PNG)
```

## Alur (1–19, mengikuti section di notebook)

1. **Configuration (CONFIG)** — satu tempat untuk semua hyperparameter: path dataset,
   path ground truth, output, kelas & ukuran gambar, split ratio, balancing, normalisasi
   ImageNet, training (batch/lr/dropout/optimizer/epoch), **validation objective**
   (`val_objective`), **LR scheduler & early stopping** (`lr_scheduler`,
   `scheduler_mode`/`factor`/`patience`/`min_lr`, `early_stopping`,
   `early_stopping_patience`/`min_delta`, metrik yang dipantau `val_monitor`),
   **resize mode** (`resize_mode`), **oversample augmentasi minoritas**
   (`oversample_augment`, `oversample_target`, `aug_*`), arsitektur model (depth, residual block, channel,
   classifier hidden dim), dan **bobot loss** (`loss_weight` manual per kelas /
   `loss_weight_mode` otomatis inverse-frequency). Path ground truth dicari otomatis
   (folder `*_GroundTruth`), dan `data_dir` autodetect `/kaggle/input`. `OUTPUT_DIR`
   dibuat saat config dimuat.
   - **Resolusi path tahan layout Kaggle bersarang** — dataset Kaggle bisa tersusun
     `/kaggle/input/datasets/<user>/<dataset>/ISIC2018_Task3_Training_Input/`
     `ISIC2018_Task3_Test_Input/ ...`, yaitu semua folder **satu level di dalam**
     `ISIC2018_Task3_Training_Input`. Karena itu setiap folder & CSV dicari **berdasarkan
     nama** (`_search_roots` → `_find_by_name` → `_resolve_dir`/`_resolve_csv`), urutan:
anak langsung `data_dir` → satu level di atasnya → `/kaggle/input` → penelusuran
      rekursif (folder penuh gambar dipangkas agar cepat). `data_dir` boleh `dataset` (lokal),
      path lengkap Kaggle, atau dikosongkan. Bila ada dua folder bernama sama (container
      vs folder gambar), kandidat yang **benar-benar berisi file gambar** dipilih
      (`_dir_score`), jadi `TRAIN_IMG_DIR` menunjuk ke folder gambar sebenarnya.
      Error jatuh sudah menyebut folder mana yang tidak ditemukan; folder gambar kosong
      memicu peringatan (bukan crash) dan tahap pembaca gambar dilewati.
   - **Pemilihan skema split** — `split_scheme` (`"kfold"` default / `"holdout"`),
     `k_folds`, `val_ratio`, dan sumber `lesion_id` (`lesion_groupings` eksplisit /
     `lesion_groupings_file` (nama file) / fallback `HAM10000_metadata.csv`).
2. **Imports** — torch, torchvision, pandas, matplotlib/seaborn, sklearn.
3. **Muat Ground Truth** — training (10015), test (1512), validation resmi (193) dibaca
   dari CSV one-hot lalu dipetakan ke label tunggal (`dx`).
4. **EDA – Distribusi Kelas** — distribusi 7 kelas di tiap set (diagram batang).
5. **EDA – Karakteristik Data** — statistik deskriptif (dimensi, ukuran file, mean/std
   piksel per channel) pada sampel acak `sample_size`.
6. **Split Data sesuai `CONFIG['split_scheme']` & Mapping Label** — memilih salah satu dari
   dua skema, keduanya **grouped per-lesion** (stratifikasi per kelas, grouping per
   `lesion_id` dari `LesionGroupings`) sehingga tidak ada lesi sama di train & validation:
   - **`split_scheme="kfold"` (default, `k_folds=5`)** — `StratifiedGroupKFold(shuffle=True,
     random_state=CONFIG)` (fallback `GroupKFold` bila tidak tersedia) pada data training
     saja. Training = 10.015 gambar dibagi `k_folds` job; tiap job punya training &
     validation sendiri. **193 gambar validation resmi tidak pernah dipakai training** dan
     diperlakukan sebagai **public test**; 1.512 gambar test resmi = **private test**.
   - **`split_scheme="holdout"` (`val_ratio=0.15`)** — training di-split sekali secara
     stratified per-lesi, lalu **dimasukkan ke validation bersama 193 gambar validation
     resmi**; test resmi tetap private test. Tidak ada public test, karena 193 gambar
     sudah berada di validation set.
   - Hasil split disimpan sebagai **`SPLIT_JOBS`** (list dict per fold / satu job holdout:
     `train`, `val`, `fold`, `train_balanced`), plus `TEST_PUBLIC_DF` (kfold) dan
     `TEST_PRIVATE_DF`. Ringkasan per-job + cek overlap lesi ditulis ke
     **`split_summary.csv`**. `class_to_idx` dipakai juga untuk test/validation resmi.
   - Constraint: 193 gambar validation resmi **tidak punya `lesion_id`**, sehingga pada
     mode holdout lesi silang terhadap training tidak bisa disingkirkan (pipeline
     memberi peringatan eksplisit).
7. **Balancing + Oversample (opsional)** — dijalankan **per split job** (per fold pada
   skema k-fold). Downsampling kelas mayoritas di atas
   `balance_threshold` mengikuti strategi rujukan (NV di-training penuh, dsb.), lalu
   **oversample kelas minoritas** (`oversample_augment`): sampel minoritas direplikasi
   hingga **`oversample_target` sample per kelas** (0 = replikasi dilewati); tiap salinan
   di-augmentasi acak saat training. Hasil akhir disimpan pada `job["train_balanced"]` dan
   bobot loss dihitung dari data tersebut.
8. **Transform & Augmentasi** — resize mengikuti `resize_mode`: `'stretch'` (tekan ke
   persegi), `'center_crop'` (pertahankan aspek + potong tengah), `'random_crop'`
   (pertahankan aspek + potong acak, **hanya training**; evaluasi selalu `CenterCrop`
   agar deterministik). Augmentasi acak ter-konfigurasi (`aug_rotation_range`,
   `aug_width/height_shift_range`, `aug_horizontal_flip`) + ColorJitter ringan **hanya
   untuk training**; evaluasi tanpa augmentasi. Normalisasi memakai mean/std ImageNet.
9. **Custom Dataset** — `ISICDataset` membaca path dan mentransform gambar.
10. **Model ResNet (configurable)** — `build_model` dengan knob: `depth`
    (18/34/50/101), `residual_blocks` (`basic`/`bottleneck`), `base_channels`,
    `classifier_hidden_dim`, `use_pretrained`. Bobot pretrained ImageNet dipakai **otomatis
    hanya untuk kombinasi standar** (18/34=basic, 50/101=bottleneck, base=64); kombinasi
    lain dibangun dari nol (`ResNetCustom`). Label arsitektur (`arch_label`) + jumlah
    parameter dicetak.
    - **10.1 Ringkasan Arsitektur (tabel per layer)** — `model_summary` menampilkan arsitektur
      ala Keras: tabel per layer (nama, tipe, output shape dari satu forward pass dummy, jumlah
      params, params trainable). Memakai **`torchinfo`** bila tersedia, fallback ke helper kustom
      **tanpa dependency**. Tabel disimpan ke `output/arch_summary_<experiment_name>.csv`.
11. **Fungsi Training & Evaluasi per Epoch** — `train_one_epoch`, `evaluate`, serta
    `build_criterion()` yang membuat CrossEntropyLoss dengan bobot per kelas (manual atau
    otomatis inverse-frequency); **`compute_metric`** menghitung `val_objective`
    (`'accuracy' | 'balanced_accuracy' | 'macro_f1'`, tanpa dependency sklearn);
    menyimpan metrik loss/accuracy/objective tiap epoch.
12. **Fungsi Utama Training (`run_training`)** — menerima seluruh hyperparameter sebagai
    argumen (default dari `CONFIG`). Memasang **`ReduceLROnPlateau`** dan **early stopping**
    sesuai CONFIG: keduanya memantau `val_monitor` (default `macro_f1`) dengan mode
    `scheduler_mode` (default `max`); `scheduler.step(monitor)` dipanggil sekali per epoch
    setelah evaluasi, dan loop berhenti setelah `early_stopping_patience` epoch tanpa
    perbaikan. Mengembalikan `(model, history, run_record)` dan menulis:
     - checkpoint `model_<run>_best.pt` (state_dict terbaik menurut **`val_objective`**),
     - **`run_<run>.json`** berisi skema lengkap training-test-eval dalam blok **`scheme`**
       (`split_scheme`, `fold`, `k_folds`, `val_ratio`, sumber `lesion_id`, jumlah sampel
       train/validation, komposisi validation, jumlah public/private test, balancing,
       oversample, seed, device), seluruh hyperparameter, blok **`monitoring`**
       (scheduler + early stopping),
       arsitektur, snapshot `CONFIG`, history (termasuk kurva `val_metric` + `val_objective`,
       `val_monitor`, dan `val_lr`), dan metrik akhir (`epochs_ran`, `best_epoch`,
       `stopped_early`),
     - baris ringkas di **`runs_log.csv`** (re-run menggantikan baris yang sama), kini
       memuat kolom `split_scheme` & `fold`.
     - run menerima data training/validation secara eksplisit
       (`train_data`, `val_data`, `train_balanced`, `fold`) sehingga k-fold dan holdout
       memakai satu fungsi yang sama; nama run pada k-fold diberi suffix `_f<fold>`.
13. **Plot Kurva** — grid 2×2: loss, accuracy + `val_objective`, kurva metrik monitor, dan
    kurva learning rate (sumbu log) (PNG + tampilan).
14. **Jalankan Training (satu cell untuk semua split job)** — SATU-SATUNYA cell yang
    menjalankan training. Cell ini ** looping `SPLIT_JOBS`**: k-fold melatih
    `k_folds` model (`<experiment_name>_f0` … `_f<k-1>`), holdout melatih 1 model
    (`<experiment_name>`), lalu menaruh model, history, dan run record ke
    **`RUN_RESULTS`** (selama sesi kernel yang sama) untuk dipakai Section 18–19.
    Plot kurva dilapis per fold, dan `kfold_summary_<experiment_name>.csv`
    (rata-rata ± std metrik tiap fold) ditulis bila skema k-fold.
    Untuk **eksperimen**: ubah nilai hyperparameter langsung di `CONFIG` (Section 1) dan
    ganti **`experiment_name`** di CONFIG (nama run/output, mis. `"lr_1e-3"`), lalu jalankan
    ulang cell ini — **1 eksperimen = 1 model per fold** (bukan ablation).
15. **Eksperimen Hyperparameter (1 model per eksperimen per fold)** — panduan nilai yang
    bisa dicoba
    (LR, batch size, dropout, optimizer/weight decay, depth, residual block, base channels,
    classifier hidden dim, `resize_mode`, `val_objective`, `val_monitor`, parameter
    scheduler/early stopping, oversample augmentasi, `split_scheme`/`k_folds`/`val_ratio`) +
    contoh
    `experiment_name`. Tidak ada cell eksperimen terpisah;
    cukup cell Section 14. Cell pembantu `current_run_cfg` mencetak konfigurasi efektif yang
    akan dipakai (termasuk `experiment_name`).
16. **Ringkasan Hasil Eksperimen** — tabel dibaca dari `runs_log.csv` (satu baris per run
    per fold; re-run dengan `experiment_name` sama menggantikan baris lama).
17. **Fungsi Evaluasi Lengkap** — helper `evaluate_and_report` yang dipakai bersama:
    metrik & classification report (CSV), **balanced accuracy & macro F1** (CSV),
    confusion matrix (PNG), prediksi per-gambar (CSV), dan **sampel salah klasifikasi**
    (CSV `misclassified_<tag>.csv` + grid PNG). `tag` = `test`/`public` + suffix `_f<fold>`
    bila k-fold.
18. **Evaluasi Akhir di Test Set Resmi (PRIVATE test)** — 1.512 gambar test resmi, tidak
    pernah disentuh selama training/tuning; dievaluasi untuk **setiap model** (per fold)
    lalu dirata-ratakan ke `test_private_metrics_all.csv`. Output ber-prefix `test_*`.
19. **Evaluasi di Validation Set Resmi (PUBLIC test, hanya skema k-fold)** — 193 label
    resmi; output ber-prefix `public_*` (bukan `validation_*`) karena perannya sebagai
    public test. Pada skema **holdout** section ini dilewati: 193 gambar sudah menjadi bagian
    validation set, jadi tidak ada public test independen. Rata-rata per fold ditulis ke
    `test_public_metrics_all.csv`.

## Output (folder `CONFIG['output_dir']`, default `output/`)

| Berkas | Penjelasan |
|---|---|
| `split_summary.csv` | per fold/job: jumlah train/val, jumlah lesi, overlap lesi, komposisi validation, status public/private test |
| `model_<run>_best.pt` | checkpoint state_dict terbaik per run (`<run>` = `<experiment_name>` atau `<experiment_name>_f<fold>`) |
| `run_<run>.json` | blok `scheme` + seluruh konfigurasi + history + metrik |
| `runs_log.csv` | ringkasan semua run (1 baris/run/fold; ringkasan eksperimen) |
| `history_<run>.png` (+ kurva training) | plot loss/accuracy/monitor/lr |
| `kfold_summary_<experiment_name>.csv` | rata-rata ± std metrik tiap fold (hanya k-fold) |
| `test_metrics.csv`, `test_classification_report.csv`, `test_predictions.csv` | evaluasi private test |
| `confusion_matrix_test.png`, `misclassified_test.csv`, `misclassified_test.png` | evaluasi private test |
| `test_private_metrics_all.csv` | rata-rata metrik private test seluruh model/fold |
| `public_metrics.csv`, `public_predictions.csv`, `confusion_matrix_public.png` | evaluasi public test (= 193 gambar validation resmi, hanya k-fold) |
| `test_public_metrics_all.csv` | rata-rata metrik public test seluruh model/fold (k-fold) |
| suffix `_f<fold>` | seluruh artefak per fold diberi suffix pada skema k-fold |

---

## Skema Split

| | `split_scheme="kfold"` (default) | `split_scheme="holdout"` |
|---|---|---|
| Training | 10.015 gambar dibagi `k_folds` job (grouped per-lesion) | 8.518 gambar (85% per-lesi) |
| Validation | ~2.003 gambar/fold (grouped per-lesion) | ~1.497 gambar (split 15%) **+ 193 gambar validation resmi** |
| Public test | 193 gambar validation resmi | **tidak ada** |
| Private test | 1.512 gambar test resmi | 1.512 gambar test resmi |
| Jumlah model | `k_folds` | 1 |
| Overlap lesi train↔val | 0 (dijamin) | 0 untuk val-split; 193 gambar resmi tidak punya `lesion_id` |
| 193 gambar dipakai training? | tidak | tidak |

**193 gambar validation resmi tidak pernah dipakai training** di kedua skema, karena
`LesionGroupings` tidak memuat `lesion_id` untuk partition tersebut sehingga lesi silang
tidak bisa disingkirkan. Pipeline sengaja tidak menyediakan opsi menggabungkannya ke
training.

## Cara Pakai

- **Notebook (utama)**: jalankan cell berurutan Section 1–19 di atas. Section 6 mencetak
  skema split yang dipakai; Section 14 melatih semua split job. Untuk eksperimen: ubah nilai
  CONFIG di Section 1 (termasuk `experiment_name`, `split_scheme`) lalu jalankan ulang
  Section 14 (1 eksperimen = 1 model per fold).
- **Script**: `src/isic2018_resnet_pipeline.py` adalah versi lama (referensi), **belum**
  mendukung `split_scheme`; pipeline=klasifikasi yang berjalan adalah notebook.
- **Kaggle**: upload `notebooks/isic2018_resnet_pipeline.ipynb` + dataset dengan folder
  standar ISIC 2018 Task 3 (`..._Input`, folder `*_GroundTruth`, dan
  `ISIC2018_Task3_Training_LesionGroupings.csv`). Path `data_dir` dan file metadata
  terdeteksi otomatis, termasuk layout bersarang
  `/kaggle/input/datasets/<user>/<dataset>/ISIC2018_Task3_Training_Input/...`.
- **EDA**: jalankan `notebooks/isic2018_eda.ipynb` terlebih dahulu bila ingin memahami data
  sebelum training (distribusi kelas, `lesion_id`, resolusi, warna, duplikat, ukuran split).
  Lihat [`docs/eda.md`](eda.md) untuk tujuan & penjelasan tiap tahapan.

## Catatan Evaluasi

- **Validation saat training** selalu grouped per-lesion (k-fold atau holdout), sehingga
  pembanding eksperimen konsisten dan bebas leakage lesi.
- **Test resmi (1.512)** hanya disentuh di Section 18 (private test).
- **Validation resmi (193)** berperan sebagai public test (Section 19) **hanya** pada skema
  k-fold. Pada skema holdout 193 gambar itu sudah masuk validation set, sehingga tidak ada
  public test independen dan metrik validation tidak boleh dibandingkan langsung dengan
  skema k-fold.
- Angka di public test (n=193, `DF`=1, `VASC`=3) tidak stabil; pakai private test dan
  rata-rata antar fold sebagai acuan.

## Roadmap Eksperimen

Rencana perbandingan baseline, attention (squeeze-and-excitation), strategi fine-tuning
backbone (frozen / partial / full), dan ablation — lihat
[`docs/eksperimen.md`](eksperimen.md).