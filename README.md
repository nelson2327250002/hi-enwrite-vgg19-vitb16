# HiEnWrite — VGG19 vs ViT-B16

Repository penelitian prediksi lima skor kepribadian Big Five (OCEAN) dari citra tulisan tangan menggunakan dataset HiEnWrite. Notebook Kaggle asli disimpan di `notebooks/` beserta output yang terekam pada file yang diunggah. Kode juga dipisahkan ke dua skrip Python berdasarkan tahapan penelitian.

## Struktur repository

```text
hi-enwrite-vgg19-vitb16/
├── notebooks/
│   └── 00_full_kaggle_experiment.ipynb
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── model_experiment.py
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── data/
│   └── README.md
├── results/
│   └── .gitkeep
├── .gitignore
├── README.md
└── requirements.txt
```

## Pembagian kode

1. **`notebooks/00_full_kaggle_experiment.ipynb`** — notebook lengkap dari Kaggle, termasuk output yang tersimpan pada file notebook yang diunggah.
2. **`src/data_pipeline.py`** — Bagian A1–A11: pemuatan metadata, audit label dan citra, seleksi manual, pemeriksaan duplikat, eksklusi penulis, pembagian data 75:25 pada tingkat penulis, 5-Fold grouped, dan preprocessing citra 224×224.
3. **`src/model_experiment.py`** — Bagian B1–B11: pemuatan artefak, baseline, VGG19, ViT-B16, optimizer Adam/AdamW/SGD, pelatihan, cross-validation, pemilihan konfigurasi, pengujian akhir, RMSE, dan ΔRMSE.

Folder `src/preprocessing/`, `src/models/`, `src/training/`, dan `src/evaluation/` disediakan sebagai batas modul. Implementasi lengkap saat ini tetap berada di dua skrip utama agar urutan cell dan dependensi variabel dari notebook tidak terputus. Kedua skrip bukan CLI mandiri; gunakan notebook sebagai jalur utama eksekusi.

## Konfigurasi seleksi manual

Konfigurasi pada notebook dan `src/data_pipeline.py` diselaraskan sebagai berikut:

```python
APPLY_MANUAL = True
MANUAL_ROTATE = {}
MANUAL_REMOVE = {}
```

Empat kandidat orientasi yang diperiksa secara visual (38e.jpeg, 38h.jpeg, 195e.jpeg, dan 195h.jpeg) tampak sudah tegak, sehingga tidak diberi rotasi. Tidak ada penghapusan manual yang ditetapkan dari pemeriksaan yang tersedia. Kandidat ketidaksesuaian aksara dari classifier tetap harus diperiksa secara visual; probabilitas classifier saja bukan alasan penghapusan.

## Menjalankan di Kaggle

1. Buka `notebooks/00_full_kaggle_experiment.ipynb` di Kaggle.
2. Tambahkan dataset HiEnWrite melalui **Add Input**. Dataset harus memuat `data/merged_data.csv` dan folder `images/`. Jika pencarian otomatis gagal, atur `DATA_ROOT` pada bagian A1.
3. Pilih **GPU** pada Session options. Jika bobot pretrained perlu diunduh selama sesi, aktifkan Internet dan atur `ALLOW_WEIGHT_DOWNLOADS = True`; jika tidak, tambahkan bobot sebagai Input Kaggle.
4. Default `SMOKE_TEST = True` adalah uji cepat (fold 0, maksimum 2 epoch). Setelah uji alur berhasil, ubah `SMOKE_TEST = False` untuk eksperimen penuh. Nilai default pada notebook dan `src/model_experiment.py` sudah disamakan.
5. Bagian B membutuhkan artefak hasil Bagian A di folder `hienwrite_clean/`. Jika dijalankan pada sesi terpisah, tambahkan output Bagian A sebagai Input atau isi `ARTIFACT_DIR`.

## Catatan output notebook

Output yang tersimpan di notebook dipertahankan dari file Kaggle yang diunggah. Konfigurasi seleksi manual kemudian diperbarui setelah pemeriksaan visual empat kandidat orientasi. Karena output tersimpan berasal dari eksekusi sebelumnya, jalankan ulang Bagian A dari awal agar artefak dan output baru konsisten dengan konfigurasi manual terbaru sebelum mengutipnya sebagai hasil final. Jangan menganggap output lama telah dihitung ulang oleh perubahan kode di repository ini.

## Dataset, bobot, dan hasil

Dataset HiEnWrite, bobot pretrained, checkpoint model, cache, dan artefak berukuran besar tidak dimasukkan ke GitHub. Ikuti ketentuan lisensi dan atribusi dataset serta bobot yang digunakan. Folder `data/` menjelaskan penempatan dataset; folder `results/` disediakan untuk tabel CSV dan grafik hasil yang ingin dibagikan. `.gitignore` mencegah file dataset besar, bobot, checkpoint, cache, dan file sementara masuk ke repository secara tidak sengaja.

## Dependensi

Daftar pustaka Python terdapat pada `requirements.txt`. Untuk lingkungan Kaggle, beberapa pustaka (terutama PyTorch/torchvision) mungkin sudah tersedia; sesuaikan instalasi dengan versi CUDA dan runtime yang digunakan.
