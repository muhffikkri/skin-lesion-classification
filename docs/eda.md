# Eksplorasi Data (EDA) — Tujuan & Penjelasan Tahapan

Notebook: [`kaggle/isic2018_eda.ipynb`](../kaggle/isic2018_eda.ipynb) · Generator: `build_eda_notebook.py`

EDA (Exploratory Data Analysis) adalah langkah **diagnosis data terlebih dahulu sebelum
training**: memahami distribusi, struktur `lesion_id`, karakteristik visual, resolusi, warna,
duplikasi, dan ukuran split. Hasil EDA menjadi dasar keputusan desain pipeline
(lihat [pipeline.md](pipeline.md)) seperti `loss_weight`, balancing, augmentasi, resize, dan
interpretasi metrik.

Semua analisis **read-only**; artefak (CSV + PNG) disimpan ke `output_eda/`. Di Kaggle,
dataset terdeteksi otomatis dari `/kaggle/input`; di lokal dari folder `dataset/`.

---

## 1. EDA Tahap 1 — Distribusi Kelas

**Tujuan**: menghitung jumlah, persentase, dan **rasio setiap kelas** (terhadap kelas
terkecil dan terbesar — bukan hanya kelas minoritas) pada training, test resmi, dan
validation resmi (193 label).

**Mengapa**: distribusi ISIC 2018 Task 3 sangat timpang — `NV` (~67%) jauh di atas `DF`
(~1%) dan `VASC` (~1%). Dengan CrossEntropy, kelas mayoritas mendominasi gradien sehingga
kelas minoritas berisiko tidak terpelajari. Angka ini menjadi masukan langsung untuk
penentuan bobot loss (`loss_weight_mode: 'inverse_frequency'`) dan `balance_threshold` di
pipeline.

**Output**: `eda_class_dist.csv`, `eda_class_dist.png`, `eda_class_ratio.csv`.

## 2. EDA Tahap 2 — Analisis lesion_id

**Tujuan**: memeriksa asumsi "1 image = 1 lesi independen". Pada HAM10000 sebuah lesi bisa
difoto lebih dari satu kali sehingga beberapa `image_id` berada di bawah satu `lesion_id`:

```
lesion_id
   |--- image A
   |--- image B
   `--- image C
```

Dihitung: jumlah image, jumlah *unique* `lesion_id`, dan distribusi **images per lesion**.

**Mengapa (data leakage)**: jika `image A -> train` dan `image B -> validation`, model dapat
mengintip lesi yang sama di dua partition sekaligus → estimasi performa terlalu optimistis.
Implementasi HAM10000 modern melakukan split berbasis `lesion_id` untuk mencegahnya.

**Keterbatasan data**: ground truth resmi ISIC 2018 tidak menyertakan `lesion_id`. Analisis
numerik berjalan bila `HAM10000_metadata.csv` tersedia (otomatis dicari di `/kaggle/input`
dan `dataset/`). Tanpa file itu, notebook tetap menampilkan penjelasan konsep + cara
melengkapi data.

**Output**: `eda_lesion_counts.csv` (+ `eda_lesion_split_leak.csv` bila ada lesi lintas split).

**Rekomendasi**: bila metadata tersedia, periksa berapa banyak lesi yang nyangkut di
train-split dan val-split pada split acak, dan gunakan **split berbasis `lesion_id`**
(disediakan demo di EDA Tahap 7).

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

## 7. EDA Tahap 7 — Ukuran Dataset Setelah Split

**Tujuan**: setelah split (sama dengan pipeline utama: stratified `train`/`val` dari training
set resmi, `test` = test set resmi), lihat jumlah **per kelas** — bukan hanya total:

```
       train  val  test
akiec    ...   ...   ...
bcc      ...   ...   ...
bkl      ...   ...   ...
df       ...   ...   ...   <- jauh lebih sedikit
mel      ...   ...   ...
nv       ...   ...   ...
vasc     ...   ...   ...   <- jauh lebih sedikit
```

**Mengapa**: `DF` dan `VASC` yang sangat sedikit memengaruhi:
- **class weight** (bobot CrossEntropy — pipeline memakai `inverse_frequency`),
- **sampling** (balancing / oversampling),
- **augmentation** (berapa kuat untuk kelas minoritas),
- **interpretasi F1** (accuracy boleh tinggi meski F1 per kelas rendah),
- **reliabilitas metrik per kelas** (n kecil → interval kepercayaan lebar).

Bila `HAM10000_metadata` tersedia, bagian kedua membandingkan **split biasa vs split berbasis
`lesion_id`** (anti-leakage) dengan cek overlap antar partition (harusnya 0).

**Output**: `eda_split_counts.csv`, `eda_split_counts.png`.

---

## Hasil EDA ISIC 2018 Task 3 (snapshot)

Snapshot hasil eksekusi notebook EDA (konfigurasi: `sample_size=300`,
`color_sample_size=200`, `val_ratio=0.15`, `random_state=42`). `HAM10000_metadata.csv`
tidak tersedia sehingga tahap 2 (analisis `lesion_id`) hanya berisi penjelasan konsep.

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
balancing wajib dipakai; akurasi bukan metrik utama — F1 per kelas lebih informatif.

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
rasio lalu **center-crop/pad** ke persegi.

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

---

## Ringkasan

Setiap tahap menutup dengan angka dan observasi yang mengarah ke keputusan pipeline, antara
lain: penggunaan bobot loss / balancing (Tahap 1 & 7), split berbasis lesion bila perlu
(Tahap 2), pemilihan augmentasi warna yang tidak merusak distribusi asli (Tahap 5),
keputusan resize 224×224 (Tahap 4), dan ekslusi duplikat dari split (Tahap 6).
`docs/pipeline.md` menjelaskan bagaimana keputusan ini diimplementasikan di pipeline
training (`kaggle/isic2018_resnet_pipeline.ipynb`).