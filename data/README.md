# Dataset HiEnWrite

Dataset tidak disertakan di repository karena ukuran dan ketentuan distribusinya.
Unduh/akses dataset dari sumber resmi atau Kaggle sesuai izin penggunaannya, lalu tambahkan
sebagai Input pada Kaggle.

Struktur yang diharapkan oleh notebook:
```text
HiEnWrite-Dataset-main/
├── data/
│   └── merged_data.csv
└── images/
    ├── ...
```

Notebook Bagian A mencari `merged_data.csv` secara otomatis. Jika tidak ditemukan,
atur `DATA_ROOT` pada cell A1 ke folder root dataset.
