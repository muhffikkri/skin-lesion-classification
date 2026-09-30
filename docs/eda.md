# Eksplorasi Data (EDA) — Tujuan & Penjelasan Tahapan

Notebook: [`notebooks/isic2018_eda.ipynb`](../notebooks/isic2018_eda.ipynb)

EDA (Exploratory Data Analysis) adalah langkah **diagnosis data terlebih dahulu sebelum
training**: memahami distribusi, struktur `lesion_id`, karakteristik visual, resolusi, warna,
duplikasi, dan skenario ukuran split. Hasil EDA menjadi dasar keputusan desain pipeline
(lihat [pipeline.md](pipeline.md)) seperti `loss_weight`, balancing, augmentasi, resize, dan
interpretasi metrik.

Semua analisis **read-only**; artefak (CSV + PNG) disimpan ke `output_eda/`. Di Kaggle,
dataset terdeteksi otomatis dari `/kaggle/input`; di lokal dari folder `dataset/`.
Folder dicari **berdasarkan nama**, sehingga layout bersarang Kaggle
(`/kaggle/input/datasets/<user>/<dataset>/ISIC2018_Task3_Training_Input/`
`ISIC2018_Task3_Test_Input/ ...`) juga terbaca; bila ada dua folder bernama sama,
folder yang berisi file gambar langsung yang dipilih. Folder gambar kosong memicu
peringatan dan tahap analisis gambar dilewati (bukan crash).

---

## 1. EDA Tahap 1 — Distribusi Kelas

**Tujuan**: menghitung jumlah, persentase, dan **rasio setiap kelas** (terhadap kelas
terkecil dan terbesar — bukan hanya kelas minoritas) pada dua skenario sekaligus:

- **Skenario A — data sebagai rilis asli**: training / validation resmi (193) / test resmi
  (1512). Acuan skema competition.
- **Skenario B — training dipecah** jadi `train + val` (stratified, `val_ratio=0.15`)
  seperti skema pipeline saat ini; test tetap test resmi.

**Mengapa**: distribusi ISIC 2018 Task 3 sangat timpang — `NV` (~67%) jauh di atas `DF`
(~1%) dan `VASC` (~1%). Dengan CrossEntropy, kelas mayoritas mendominasi gradien sehingga
kelas minoritas berisiko tidak terpelajari. Angka ini menjadi masukan langsung untuk
penentuan bobot loss (`loss_weight_mode: 'inverse_frequency'`) dan `balance_threshold` di
pipeline. Kedua skenario disimpan terpisah agar tidak tercampur saat interpretasi.

**Output**: `eda_class_dist.csv` (skenario A), `eda_class_dist_split.csv` (skenario B),
`eda_class_dist.png`, `eda_class_ratio.csv`.

## 2. EDA Tahap 2 — Analisis lesion_id

**Tujuan**: memeriksa asumsi "1 image = 1 lesi independen". Sebuah lesi bisa difoto lebih
dari satu kali sehingga beberapa gambar berada di bawah satu `lesion_id`:

```
lesion_id
   |--- image A
   |--- image B
   `--- image C
```

Dihitung: jumlah image, jumlah *unique* `lesion_id`, distribusi **images per lesion**, dan
cek lesi yang muncul di train-split **dan** val-split pada split stratified per-image.

**Mengapa (data leakage)**: jika `image A -> train` dan `image B -> validation`, model dapat
mengintip lesi yang sama di dua partition sekaligus → estimasi performa terlalu optimistis.
Implementasi HAM10000 modern melakukan split berbasis `lesion_id` untuk mencegahnya.

**Sumber metadata** (auto-deteksi, berurutan):
1. `ISIC2018_Task3_Training_LesionGroupings.csv` — rilis resmi Task 3 (**sumber utama**),
   dicari di `EDA_CFG['lesion_groupings']`, `dataset/`, lalu rekursif `/kaggle/input/**`
   (menangani path bersarang seperti `/kaggle/input/datasets/<user>/<dataset>/`).
2. `HAM10000_metadata.csv` — fallback bila file di atas tidak ada.
3. Tanpa keduanya, notebook tetap menampilkan penjelasan konsep + cara melengkapi data,
   dan Tahap 7 otomatis melewati skenario 2–4.

**Keterbatasan data**: `LesionGroupings` **hanya mencakup training**. Gambar validation
resmi (193) dan test resmi (1512) tidak punya `lesion_id`, sehingga lesi lintas
training↔validation/test tidak bisa disingkirkan sepenuhnya.

**Output**: `eda_lesion_counts.csv` (+ `eda_lesion_split_leak.csv` bila ada lesi lintas split).

**Rekomendasi**: gunakan **split berbasis `lesion_id`** (dijumlahkan di EDA Tahap 7), bukan
split acak per gambar.

## 3. EDA Tahap 3 — Visualisasi Setiap Kelas

Grid **3 gambar per kelas** untuk memahami dasar pembeda antar kelas: **shape, color,
texture, border, size, global structure**. Observasi ini membantu memilih augmentasi
yang relevan dan memahami mengapa kelas tertentu sulit dibedakan (mis. `MEL` vs `BKL`).

**Output**: `eda_class_samples_grid.png`.

## 4. EDA Tahap 4 — Resolusi & Aspect Ratio

**Tujuan**: distribusi lebar/tinggi/aspect ratio (mean, median, std, min, max) + histogram.

**Mengapa berpengaruh langsung ke keputusan resize 224×224**:
- **Computational cost & memori** — aktivasi & minibatch tumbuh kuadrat terhadap ukuran input
  (tabel estimasi memori RGB float32 per resolusi disertakan).
- **Detail lesi** — mengecilkan gambar bisa menghilangkan tekstur halus / border bergerigi.
- **Receptive field** — struktur ResNet dirancang untuk input 224; pola field efektif berubah
  bila resolusi diubah.
- **Batch size** — input besar memaksa batch kecil, mengubah stabilitas optimasi.

**Output**: `eda_resolution_stats.csv`, `eda_resolution_hists.png`.

## 5. EDA Tahap 5 — Distribusi Warna

**Tujuan**: histogram RGB, brightness, contrast, saturation untuk memahami seberapa jauh
warna dapat menjadi fitur diskriminatif (mis. `VASC` merah, `DF` cokelat).

**Mengapa (augmentasi warna)**: ColorJitter **tidak boleh ditentukan sembarangan**. Jika
jitter terlalu kuat:

```
lesion asli --augmentasi--> warna berubah ekstrem
model belajar distribusi warna yang tidak realistis
```

Sub-tahap **5.1** mensimulasikan `ColorJitter(brightness/saturation/contrast ±0.2)` memakai
`PIL.ImageEnhance` dan membandingkan statistik asli vs setelah jitter, agar bisa menilai
apakah jitter memperluas distribusi secara wajar.

**Output**: `eda_color_stats.csv`, `eda_color_hists.png`, `eda_jitter_shift.csv`,
`eda_jitter_shift.png`.

## 6. EDA Tahap 6 — Duplikat / Near-Duplikat

**Tujuan**: mencari gambar **sama persis** (`md5`) dan **hampir sama** (`dhash` 64-bit dengan
ambang Hamming) antar partition (training, test resmi, validation resmi).

**Mengapa**: `train ~= validation` membuat hasil evaluasi **bias**. Laporan berupa daftar
pasangan gambar duplikat dan hitungan duplikat eksak lintas partition.

> Catatan: `full_hash=True` (default) menghitung md5 seluruh gambar — bisa butuh 1–2 menit.

**Output**: `eda_dup_exact.csv`, `eda_dup_near.csv`.

## 7. EDA Tahap 7 — Perbandingan Skenario Split

**Tujuan**: membandingkan **empat skenario split** (jumlah **per kelas**, bukan hanya total)
yang menjadi bahan keputusan untuk pipeline utama:

| Skenario | Cara split | Lesi bocor | Catatan |
|---|---|---|---|
| 1 | stratified per-**image** (`val_ratio=0.15`) | 589 lesi | skema pipeline lama |
| 2 | stratified per-**lesion** (`val_ratio=0.15`) | 0 | holdout murni |
| 3 | **k-fold** grouped per-lesion (`k_folds`) | 0 | tiap gambar validasi tepat 1× → default pipeline |
| 4 | **holdout**: split `val_ratio` + **validation resmi** | tidak terverifikasi | `split_scheme="holdout"`; 193 gambar val resmi tanpa `lesion_id` |

Skenario 3 memakai `StratifiedGroupKFold(shuffle=True, random_state=42)` (fallback
`GroupKFold` bila tidak tersedia) dan melaporkan jumlah gambar per kelas di tiap fold
(min..max) untuk mengecek keseimbangan. Skenario 4 memakai split per-lesi yang sama dengan
Skenario 2, lalu menggabungkan 193 gambar validation resmi ke validation set.

Skenario 3 dan 4 adalah **dua skema split yang dipilih lewat `CONFIG['split_scheme']` di
pipeline**, bukan dua tahap berurutan. Keduanya tidak pernah memakai 193 gambar validation
resmi untuk training.

**Mengapa**: `DF` dan `VASC` yang sangat sedikit memengaruhi:
- **class weight** (bobot CrossEntropy — pipeline memakai `inverse_frequency`),
- **sampling** (balancing / oversampling),
- **augmentation** (berapa kuat untuk kelas minoritas),
- **interpretasi F1** (accuracy boleh tinggi meski F1 per kelas rendah),
- **reliabilitas metrik per kelas** (n kecil → interval kepercayaan lebar).

Skema evaluasi akhir:

- `split_scheme="kfold"` (default) → training di-split k-fold grouped per-lesi; validation
  resmi (193) = **public test** (tidak dilatih, tidak dipisah lagi); test resmi (1.512) =
  **private test**. Skema competition ISIC 2018.
- `split_scheme="holdout"` → training di-split sekali `val_ratio`, digabung dengan validation
  resmi (193) menjadi validation set; test resmi = **private test**. Tidak ada public test,
  karena 193 gambar sudah masuk validation sehingga angkanya tidak independen dari training.
- **Tidak ada** opsi menggabungkan 193 gambar ke training (`combine_train_val`): file
  `LesionGroupings` tidak memuat `lesion_id` untuk partition tersebut, sehingga lesi silang
  tidak bisa disingkirkan. Flag tersebut hanya dokumentasi di EDA (default `False`) dan
  tidak memengaruhi pipeline.

**Output**: `eda_split_counts.csv`, `eda_split_counts.png`, `eda_split_lesion_counts.csv`,
`eda_split_leak_summary.csv`, `eda_split_holdout_counts.csv`, `eda_split_compare.png`,
`eda_cv_folds.csv`.

---

## Hasil EDA ISIC 2018 Task 3 (snapshot)

Snapshot hasil eksekusi notebook EDA (konfigurasi: `sample_size=300`,
`color_sample_size=200`, `val_ratio=0.15`, `k_folds=5`, `random_state=42`,
`combine_train_val=False`). Sumber `lesion_id`:
`ISIC2018_Task3_Training_LesionGroupings.csv` (rilis resmi).

### Ukuran dataset

```
Train gambar  : 10015
Test gambar   : 1512
Validation    : 193
```

### Tahap 1 — Distribusi & rasio kelas

| class  | train_count | train_pct | test_count | test_pct | val_count | val_pct |
|--------|------------:|----------:|-----------:|---------:|----------:|--------:|
| MEL    |        1113 |     11.11 |        171 |    11.31 |        21 |   10.88 |
| NV     |        6705 |     66.95 |        909 |    60.12 |       123 |   63.73 |
| BCC    |         514 |      5.13 |         93 |     6.15 |        15 |    7.77 |
| AKIEC  |         327 |      3.27 |         43 |     2.84 |         8 |    4.15 |
| BKL    |        1099 |     10.97 |        217 |    14.35 |        22 |   11.40 |
| DF     |         115 |      1.15 |         44 |     2.91 |         1 |    0.52 |
| VASC   |         142 |      1.42 |         35 |     2.31 |         3 |    1.55 |

Rasio **NV : DF = 58.3 : 1**.

| class  | train_count | train_pct | rasio_vs_terkecil | rasio_vs_terbesar |
|--------|------------:|----------:|------------------:|------------------:|
| MEL    |        1113 |     11.11 |               9.68 |              6.02 |
| NV     |        6705 |     66.95 |              58.30 |              1.00 |
| BCC    |         514 |      5.13 |               4.47 |             13.04 |
| AKIEC  |         327 |      3.27 |               2.84 |             20.50 |
| BKL    |        1099 |     10.97 |               9.56 |              6.10 |
| DF     |         115 |      1.15 |               1.00 |             58.30 |
| VASC   |         142 |      1.42 |               1.23 |             47.22 |

**Implikasi**: ketimpangan ekstrem (NV 58× DF). Bobot loss `inverse_frequency` +
balancing wajib dipakai; akurasi bukan metrik utama — F1 per kelas lebih informatif
(pipeline mendukung `val_objective: 'balanced_accuracy' | 'macro_f1'`).

### Tahap 4 — Resolusi & aspect ratio (n=300)

|        | width | height | aspect_ratio |
|--------|------:|-------:|-------------:|
| mean   | 600.0 |  450.0 |         1.33 |
| median | 600.0 |  450.0 |         1.33 |
| std    |   0.0 |    0.0 |         0.00 |
| min    | 600.0 |  450.0 |         1.33 |
| max    | 600.0 |  450.0 |         1.33 |

Semua gambar ISIC 2018 Task 3 berukuran seragam **600×450 (4:3)**. **Implikasi**: resize
langsung ke 224×224 akan mendistorsi; opsi pipeline yang tepat adalah resize mempertahankan
rasio lalu **center-crop/pad** ke persegi — sudah didukung oleh
`CONFIG['resize_mode']: 'center_crop' | 'random_crop'` di pipeline.

### Tahap 5 — Distribusi warna (n=200)

| stat   | mean_R | mean_G | mean_B | brightness | contrast | saturation |
|--------|-------:|-------:|-------:|-----------:|---------:|-----------:|
| mean   | 193.82 | 136.86 | 142.92 |     154.58 |    27.60 |       0.31 |
| median | 199.47 | 134.29 | 141.79 |     153.18 |    26.09 |       0.30 |
| std    |  25.99 |  21.67 |  24.61 |      19.19 |    10.70 |       0.12 |

Channel merah rata-rata dominan (193.8 vs G 136.9 / B 142.9) — konsisten dengan lesi
berpigmen; saturation ~0.31.

### Tahap 5.1 — ColorJitter sanity-check (±0.2, n=200)

| stat      | orig_mean | jitter_mean | orig_median | jitter_median | orig_std | jitter_std |
|-----------|----------:|------------:|------------:|--------------:|---------:|-----------:|
| brightness |   154.582 |     153.547 |     153.185 |        151.930 |   19.185 |     26.860 |
| contrast   |    27.597 |      27.396 |      26.088 |         25.484 |   10.700 |     11.522 |
| saturation |     0.312 |       0.309 |       0.298 |          0.288 |    0.123 |      0.133 |

**Implikasi**: jitter ±0.2 memperluas distribusi (brightness std 19.2 → 26.9, saturation
0.123 → 0.133) tanpa menggeser median secara ekstrem — kekuatan ini **wajar**, tidak merusak
distribusi warna asli.

### Tahap 7 — Split per kelas (val_ratio=0.15)

Setelah stratified split: train 8512, val 1503 (total 10015) + test resmi 1512.

| class | train | val | test | train_% | val_% | test_% |
|-------|------:|----:|-----:|--------:|------:|-------:|
| MEL   |   946 | 167 |  171 |    11.1 |  11.1 |   11.3 |
| NV    |  5699 |1006 |  909 |    67.0 |  66.9 |   60.1 |
| BCC   |   437 |  77 |   93 |     5.1 |   5.1 |    6.2 |
| AKIEC |   278 |  49 |   43 |     3.3 |   3.3 |    2.8 |
| BKL   |   934 | 165 |  217 |    11.0 |  11.0 |   14.4 |
| DF    |    98 |  17 |   44 |     1.2 |   1.1 |    2.9 |
| VASC  |   120 |  22 |   35 |     1.4 |   1.5 |    2.3 |

**Implikasi**: DF hanya 17 & VASC 22 di validation split; DF 44 / VASC 35 di test resmi.
Metrik per kelas untuk DF/VASC punya variabilitas tinggi & CI lebar; interpretasi F1 harus
hati-hati. Distribusi proporsi train/val konsisten (stratified) — pembanding eksperimen fair
— dan test resmi sedikit berbeda (minoritas relatif lebih besar).

### Tahap 2 — Struktur `lesion_id` (LesionGroupings)

| metrik | nilai |
|---|---:|
| gambar training | 10.015 |
| lesi unik | 7.470 |
| lesi dengan >1 gambar | 1.956 |
| gambar dari lesi multi-gambar | 4.501 |
| maksimum gambar per lesi | 6 |
| lesi dengan label konflik | 0 |

Distribusi gambar per lesi: 1→5.514, 2→1.423, 3→490, 4→34, 5→5, 6→4.

**Bukti leakage pada split per-image sekarang** (stratified 15%, `random_state=42`):

| skenario | lesi bocor | gambar terlibat | % gambar training |
|---|---:|---:|---:|
| 1. split per-image (skema lama) | **589** | **1.420** | 14,2% |
| 2. split per-lesion | 0 | 0 | 0,0% |

Rincian lesi bocor: 370 lesi (2 gambar), 198 (3), 19 (4), 2 (5).

### Tahap 7 — Perbandingan skenario split

**Skenario 2 — per-lesion holdout** (6.349 lesi train / 1.121 lesi val, overlap 0):

| class | train | val | class | train | val |
|---|---:|---:|---|---:|---:|
| MEL   |  945 | 168 | BKL   |  942 | 157 |
| NV    | 5705 |1000 | DF    |   93 |  22 |
| BCC   |  437 |  77 | VASC  |  120 |  22 |
| AKIEC |  276 |  51 | | | |

**Skenario 3 — k-fold=5 grouped per-lesion (training saja, 10.015 gambar)**:

| fold | n_train | n_val | n_lesion_val | MEL | NV | BCC | AKIEC | BKL | DF | VASC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8011 | 2004 | 1495 | 222 | 1341 | 103 | 66 | 220 | 23 | 29 |
| 1 | 8012 | 2003 | 1494 | 222 | 1341 | 103 | 66 | 220 | 23 | 28 |
| 2 | 8012 | 2003 | 1494 | 223 | 1341 | 102 | 65 | 220 | 23 | 29 |
| 3 | 8012 | 2003 | 1494 | 223 | 1341 | 103 | 65 | 220 | 23 | 28 |
| 4 | 8013 | 2002 | 1493 | 223 | 1341 | 103 | 65 | 219 | 23 | 28 |

**Skenario 4 — holdout: split `val_ratio` + validation resmi** (skema `split_scheme="holdout"`):

| partisi | gambar | lesi |
|---|---:|---:|
| train (split 85% per-lesi) | 8.518 | 6.349 |
| val = split 15% per-lesi | 1.497 | 1.121 |
| val = validation resmi | 193 | tidak diketahui (`lesion_id` tidak tersedia) |
| **val total** | **1.690** | — |
| test resmi (private test) | 1.512 | — |

Overlap lesi train vs val-split = 0. Tidak ada public test: 193 gambar sudah menjadi bagian
validation set. Risiko yang tersisa: lesi silang antara training dan 193 gambar validation
resmi **tidak bisa disingkirkan** karena `lesion_id`-nya tidak ada.

**Rekomendasi**: pakai **k-fold grouped per-lesion** sebagai skema utama
(`CONFIG['split_scheme'] = "kfold"`), sehingga 193 gambar validation resmi menjadi
**public test** dan tidak pernah dilatih. Gunakan `"holdout"` hanya bila ingin satu split besar
+ 193 gambar di validation untuk iterasi cepat.

---

## Ringkasan

Setiap tahap menutup dengan angka dan observasi yang mengarah ke keputusan pipeline, antara
lain: penggunaan bobot loss / balancing (Tahap 1 & 7), **split berbasis `lesion_id` dan
k-fold grouped** (Tahap 2 & 7), pemilihan augmentasi warna yang tidak merusak distribusi asli
(Tahap 5), keputusan resize 224×224 (Tahap 4), dan ekslusi duplikat dari split (Tahap 6).
`docs/pipeline.md` menjelaskan bagaimana keputusan ini diimplementasikan di pipeline
training (`notebooks/isic2018_resnet_pipeline.ipynb`).