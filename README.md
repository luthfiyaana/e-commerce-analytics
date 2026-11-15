
# E-Commerce Analytics

## Deskripsi
Proyek ini menganalisis Brazilian E-Commerce Public Dataset untuk mengetahui perkembangan penjualan, karakteristik pelanggan, persebaran pelanggan berdasarkan wilayah, serta distribusi nilai pesanan.

Analisis dilakukan menggunakan Python dan Pandas, sedangkan hasil analisis disajikan melalui dashboard interaktif menggunakan Streamlit.

## Business Questions
1. Bagaimana perkembangan jumlah pesanan dan total nilai penjualan produk setiap bulan selama periode transaksi yang tersedia, serta bulan mana yang mencatat performa tertinggi?
2. Bagaimana karakteristik pelanggan berdasarkan Recency, Frequency, dan Monetary selama periode pengamatan, serta segmen pelanggan mana yang perlu diprioritaskan dalam strategi retensi?
3. Bagaimana persebaran pelanggan berdasarkan negara bagian di Brazil, dan wilayah mana yang memberikan kontribusi jumlah pesanan serta nilai penjualan terbesar selama periode pengamatan?

## Fitur Dashboard
- Sales Overview: Perkembangan jumlah pesanan dan nilai penjualan bulanan.
- Customer RFM: Segmentasi pelanggan berdasarkan Recency, Frequency, dan Monetary.
- Regional Analysis: Persebaran pelanggan dan kontribusi penjualan berdasarkan negara bagian.
- Order Value Analysis: Distribusi nilai pesanan berdasarkan kategori transaksi.

## Struktur Proyek

```text
Submission/
├── dashboard/
│   └── dashboard.py
├── data/
│   └── (9 file dataset CSV)
├── README.md
├── notebook.ipynb
├── requirements.txt
├── url.txt
└── E-commerce-public-dataset.zip
```

## Instalasi Library

```bash
pip install -r requirements.txt
```

## Menjalankan Dashboard

```bash
streamlit run dashboard/dashboard.py
```

## Dataset
Dataset yang digunakan dalam proyek ini merupakan Brazilian E-Commerce Public Dataset yang telah disediakan oleh Dicoding sebagai bagian dari instruksi proyek akhir.

Dataset diunduh dalam format ZIP dan digunakan untuk proses analisis data serta pengembangan dashboard.
