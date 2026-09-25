# Rencana Eksperimen (Roadmap)

Rencana eksperimen & ablation untuk pipeline `kaggle/isic2018_resnet_pipeline.ipynb`.
Status setiap item harus diperbarui saat sudah dijalankan (tandai dengan `[x]` + tautan
run `run_<nama>.json` bila ada).

Semua eksperimen dijalankan dengan **1 model = 1 run** melalui `CONFIG['experiment_name']`
(Section 1) + cell Section 14; perbandingan dibaca dari `runs_log.csv` (Section 16).

---

## 1. Perbandingan dengan Baseline

Pertama-tama tetapkan **baseline** (setting default `CONFIG`), lalu bandingkan dengan
model/pendekatan lain pada metrik yang sama di test set resmi.

| # | Model | Keterangan | Status |
|---|-------|------------|--------|
| 1.1 | **CNN** | CNN konvolusional sederhana (dari nol) | [ ] |
| 1.2 | **ResNet** | ResNet configurable (baseline pipeline ini) | [ ] |
| 1.3 | **Transformer** | Arsitektur berbasis transformer (mis. ViT) | [ ] |

Metrik pembanding: accuracy, balanced accuracy, macro F1, confusion matrix, per kelas di
test set resmi (`test_metrics.csv`).

---

## 2. Eksperimen Attention Mechanism: Squeeze-and-Excitation (SE)

Menambahkan blok **Squeeze-and-Excitation** (calibrasi channel, Hu dkk. 2018) ke
backbone/baseline, lalu menguji **strategi fine-tuning** bobot pretrained:

| # | Percobaan | Backbone | Lapisan trainable | Status |
|---|-----------|----------|-------------------|--------|
| 2.1 | **Percobaan 1** | frozen | classifier saja | [ ] |
| 2.2 | **Percobaan 2** | pretrained | `layer4` + classifier | [ ] |
| 2.3 | **Percobaan 3** | pretrained | `layer3` + `layer4` + classifier | [ ] |
| 2.4 | **Percobaan 4** | pretrained | full fine-tuning (semua) | [ ] |

> **Prasyarat implementasi**: opsi pembekuan/trainable lapisan per layer
> (`requires_grad`) di `build_model`/`run_training` — saat ini pipeline melakukan
> full fine-tuning untuk semua parameter.

---

## 3. Ablation Study

Ablasi pengaruh tiap komponen secara bertahap untuk mengukur kontribusinya:

| # | Kombinasi | Komponen baru | Status |
|---|-----------|---------------|--------|
| 3.1 | **CNN** | CNN dasar | [ ] |
| 3.2 | **CNN + ResNet** | tambah backbone ResNet (residual/skip) | [ ] |
| 3.3 | **CNN + ResNet + CrossEntropy** | tambah loss CrossEntropy (baseline lengkap) | [ ] |

Setiap langkah ablation mempertahankan komponen sebelumnya dan menambahkan satu komponen
baru agar kontribusinya terukur.