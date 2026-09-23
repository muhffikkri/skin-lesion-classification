# Pipeline: Klasifikasi Skin Lesion (ISIC 2018 Task 3)

Dokumentasi alur terkini (v1.5) dari notebook `kaggle/isic2018_resnet_pipeline.ipynb`.
Notebook dibangun dari `src/isic2018_resnet_pipeline.py` melalui generator
(`build_resnet_notebook.py`); **seluruh perubahan alur dilakukan di generator lalu
notebook diregenerasi ulang**.

Rujukan tugas: EMED (7 kelas diagnosis) pada ISIC 2018 Task 3
(diusulkan oleh ISIC Challenge 2018, lihat referensi [1] dan [2] di notebook).

## Struktur Repo

```
skin-lesion-classification/
├── kaggle/isic2018_resnet_pipeline.ipynb  # notebook utama (output akhir)
├── src/isic2018_resnet_pipeline.py        # script sumber pipeline
├── docs/pipeline.md                       # dokumen ini
├── README.md
├── CHANGELOG.md
├── dataset/                               # dataset lokal (folder standar ISIC 2018 Task 3)
└── output/                                # hasil run (model, JSON, CSV, PNG)
```

## Alur (1–19, mengikuti section di notebook)

1. **Configuration (CONFIG)** — satu tempat untuk semua hyperparameter: path dataset,
   path ground truth, output, kelas & ukuran gambar, split ratio, balancing, normalisasi
   ImageNet, training (batch/lr/dropout/optimizer/epoch), arsitektur model (depth,
   residual block, channel, classifier hidden dim), dan **bobot loss** (`loss_weight`
   manual per kelas / `loss_weight_mode` otomatis inverse-frequency). Path ground truth
   dicari otomatis (folder `*_GroundTruth`), dan `data_dir` autodetect `/kaggle/input`.
   `OUTPUT_DIR` dibuat saat config dimuat.
2. **Imports** — torch, torchvision, pandas, matplotlib/seaborn, sklearn.
3. **Muat Ground Truth** — training (10015), test (1512), validation resmi (193) dibaca
   dari CSV one-hot lalu dipetakan ke label tunggal (`dx`).
4. **EDA – Distribusi Kelas** — distribusi 7 kelas di tiap set (diagram batang).
5. **EDA – Karakteristik Data** — statistik deskriptif (dimensi, ukuran file, mean/std
   piksel per channel) pada sampel acak `sample_size`.
6. **Split & Mapping Label** — stratified split (default `val_ratio=0.15`) dari training
   set sebagai validation training; `class_to_idx` dipakai juga untuk validation resmi.
7. **Balancing (opsional)** — downsampling kelas mayoritas di atas `balance_threshold`
   mengikuti strategi rujukan (NV di-training penuh, dsb.).
8. **Transform & Augmentasi** — resize 224x224, augmentasi ringan (flip/rotasi) untuk
   training, normalisasi memakai mean/std ImageNet.
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
    otomatis inverse-frequency); menyimpan metrik loss/accuracy tiap epoch.
12. **Fungsi Utama Training (`run_training`)** — menerima seluruh hyperparameter sebagai
    argumen (default dari `CONFIG`). Mengembalikan `(model, history, run_record)` dan
    menulis:
    - checkpoint `model_<run>_best.pt` (state_dict val_acc terbaik),
    - **`run_<run>.json`** berisi skema lengkap training-test-eval (jumlah sampel
      train/val-split/test-resmi/val-resmi, rasio split, balancing, seed, device),
      seluruh hyperparameter, arsitektur, snapshot `CONFIG`, history, dan metrik akhir;
    - baris ringkas di **`runs_log.csv`** (re-run menggantikan baris yang sama).
13. **Plot Kurva** — loss & accuracy training vs validation (PNG + tampilan).
14. **Satu Cell Training (baseline / eksperimen)** — SATU-SATUNYA cell yang menjalankan
    `run_training`. Baseline memakai default `CONFIG`. Untuk **eksperimen**: ubah nilai
    hyperparameter langsung di `CONFIG` (Section 1) dan ganti **`experiment_name`** di
    CONFIG (nama run/output, mis. `"lr_1e-3"`), lalu jalankan ulang cell ini —
    **1 eksperimen = 1 model = 1 run** (bukan ablation).
15. **Eksperimen Hyperparameter (1 model per eksperimen)** — panduan nilai yang bisa dicoba
    (LR, batch size, dropout, optimizer/weight decay, depth, residual block, base channels,
    classifier hidden dim) + contoh `experiment_name`. Tidak ada cell eksperimen terpisah;
    cukup cell Section 14. Cell pembantu `current_run_cfg` mencetak konfigurasi efektif yang
    akan dipakai (termasuk `experiment_name`).
16. **Ringkasan Hasil Eksperimen** — tabel dibaca dari `runs_log.csv` (satu baris per run;
    re-run dengan `experiment_name` sama menggantikan baris lama).
17. **Fungsi Evaluasi Lengkap** — helper `evaluate_and_report` yang dipakai bersama:
    metrik & classification report (CSV), confusion matrix (PNG), prediksi per-gambar
    (CSV), dan **sampel salah klasifikasi** (CSV `misclassified_<tag>.csv` + grid PNG).
18. **Evaluasi Akhir di Test Set Resmi** — tidak pernah disentuh selama training/tuning;
    output ber-prefix `test_*` (`confusion_matrix_test.png`, `test_predictions.csv`, dsb.).
19. **Evaluasi di Validation Set Resmi** — ground truth resmi (193 label); output
    ber-prefix `validation_*` (`confusion_matrix_validation.png`, dsb.).

## Output (folder `CONFIG['output_dir']`, default `output/`)

| Berkas | Penjelasan |
|---|---|
| `model_<run>_best.pt` | checkpoint state_dict terbaik per run |
| `run_<run>.json` | skema training-test-eval + seluruh konfigurasi + history + metrik |
| `runs_log.csv` | ringkasan semua run (1 baris/run; ringkasan eksperimen) |
| `history_baseline.png` (+ kurva training) | plot loss/accuracy |
| `test_metrics.csv`, `test_classification_report.csv`, `test_predictions.csv` | evaluasi test |
| `confusion_matrix_test.png`, `misclassified_test.csv`, `misclassified_test.png` | evaluasi test |
| `validation_metrics.csv`, `validation_classification_report.csv`, `validation_predictions.csv` | evaluasi val resmi |
| `confusion_matrix_validation.png`, `misclassified_validation.csv`, `misclassified_validation.png` | evaluasi val resmi |

## Cara Pakai

- **Lokal**: `python src/isic2018_resnet_pipeline.py` (butuh torch di lingkungan).
- **Notebook**: jalankan cell berurutan 1–19 di atas. Untuk eksperimen: ubah nilai CONFIG
  di Section 1 (termasuk `experiment_name`) lalu jalankan ulang cell Section 14
  (1 eksperimen = 1 model).
- **Kaggle**: upload `kaggle/isic2018_resnet_pipeline.ipynb` + dataset dengan folder
  standar ISIC 2018 Task 3. Path terdeteksi otomatis.
- **EDA**: jalankan `kaggle/isic2018_eda.ipynb` terlebih dahulu bila ingin memahami data
  sebelum training (distribusi kelas, `lesion_id`, resolusi, warna, duplikat, ukuran split).
  Lihat [`docs/eda.md`](eda.md) untuk tujuan & penjelasan tiap tahapan.

## Catatan Evaluasi

Validation selama training memakai stratified split (pembanding eksperimen konsisten);
validation set resmi dipakai untuk evaluasi final terpisah (Section 19). Test set resmi
hanya disentuh di Section 18.