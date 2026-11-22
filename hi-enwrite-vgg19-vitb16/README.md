# HiEnWrite — VGG19 vs ViT-B16

Repository penelitian prediksi lima skor kepribadian Big Five (OCEAN) dari citra tulisan tangan
menggunakan dataset HiEnWrite. Notebook Kaggle asli tetap disimpan, dan alur kodenya juga
disediakan dalam dua skrip Python berdasarkan tahapan penelitian.

## Isi repository

```text
hi-enwrite-vgg19-vitb16/
├── notebooks/
│   ├── 00_full_kaggle_experiment.ipynb
│   └── 01_check_dataset.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── model_experiment.py
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── data/
│   └── README.md
├── results/
├── requirements.txt
└── .gitignore
```

## Pembagian kode

1. **`notebooks/00_full_kaggle_experiment.ipynb`** — notebook lengkap sesuai file Kaggle yang diberikan.
2. **`src/data_pipeline.py`** — Bagian A1–A11: pemuatan metadata, audit label dan citra,
   seleksi manual, pemeriksaan duplikat, eksklusi penulis, split 75:25 tingkat penulis,
   5-Fold grouped, dan prapemrosesan citra 224×224.
3. **`src/model_experiment.py`** — Bagian B1–B11: pemuatan artefak, baseline, VGG19,
   ViT-B16, optimizer Adam/AdamW/SGD, pelatihan, 5-Fold CV, pemilihan konfigurasi,
   pengujian akhir, RMSE, dan ΔRMSE.

Folder `src/preprocessing`, `src/models`, `src/training`, dan `src/evaluation` disiapkan
sebagai batas modul agar pemisahan lebih rinci dapat dilakukan tanpa mengubah alur eksperimen
yang sekarang. Untuk tahap ini, implementasi lengkap dipertahankan dalam dua skrip berurutan
agar dependensi variabel dari notebook tidak terputus.

## Menjalankan di Kaggle

Cara yang paling aman untuk mereproduksi eksperimen saat ini adalah membuka
`00_full_kaggle_experiment.ipynb` di Kaggle, menambahkan dataset HiEnWrite melalui **Add Input**,
memilih GPU, lalu mengikuti instruksi pada bagian awal notebook.

- Smoke test: `SMOKE_TEST = True` (uji alur singkat).
- Eksperimen penuh: setelah konfigurasi dan keputusan data diverifikasi, ubah
  `SMOKE_TEST = False`.
- Notebook mengharapkan `data/merged_data.csv` dan folder `images/`.
- Artefak Bagian A diperlukan untuk Bagian B. Tambahkan output/artefak tahap data sebagai Input
  jika menjalankan Bagian B secara terpisah.
- Untuk bobot pretrained tanpa internet, ikuti petunjuk lokasi bobot pada notebook.

Skrip `src/data_pipeline.py` dan `src/model_experiment.py` adalah pemisahan kode berdasarkan
urutan cell notebook, bukan paket CLI mandiri. Keduanya mempertahankan variabel bersama dari
notebook. Gunakan notebook lengkap sebagai jalur utama eksekusi/reproduksi sampai refaktor
menjadi fungsi-fungsi modular mandiri selesai diuji.

## Lingkup eksperimen yang tercermin pada notebook

- Split data tingkat penulis: 75% train dan 25% test.
- 5-Fold grouped cross-validation.
- Arsitektur: VGG19 dan ViT-B16.
- Optimizer: Adam, AdamW, dan SGD.
- Target regresi: lima dimensi OCEAN.
- Metrik utama: RMSE.

## Dataset, bobot, dan hasil

Dataset HiEnWrite, bobot pretrained, checkpoint, cache, dan hasil eksperimen berukuran besar
tidak dimasukkan ke GitHub. Ikuti ketentuan lisensi/sumber dataset dan bobot masing-masing.
Simpan tabel CSV dan gambar hasil yang memang diperlukan di `results/`.

## Catatan sebelum publikasi

Periksa kembali keputusan manual pada notebook, termasuk `OPENNESS_FIX`, `MANUAL_ROTATE`,
dan `MANUAL_REMOVE`, agar sesuai dengan keputusan final penelitian. Notebook sendiri mencatat
bahwa daftar seleksi manual belum sepenuhnya dapat direproduksi dari file yang tersedia.
