# Aya Notulen App

Aya Notulen App adalah aplikasi berbasis Python untuk merekam, mentranskripsi, dan merangkum notulen rapat secara otomatis menggunakan teknologi speech-to-text dan summarization berbasis AI. Aplikasi ini dirancang untuk mempermudah pembuatan notulen dalam bahasa Indonesia dengan akurasi tinggi.

## Fitur
- **Transkripsi Audio**: Mengubah rekaman rapat menjadi teks menggunakan model `faster-whisper`.
- **Summarization**: Merangkum isi rapat menggunakan model `t5-indonesian-summarization`.
- **Pengenalan Pembicara**: Mengidentifikasi pembicara dalam rapat dengan `speechbrain` (ECAPA-VoxCeleb).
- **Database Lokal**: Menyimpan notulen dalam database SQLite (`notulen.db`).
- **Ekspor PDF**: Menghasilkan laporan notulen dalam format PDF menggunakan `reportlab`.
- **Antarmuka Konsol**: Mudah digunakan melalui command line (dengan rencana GUI di masa depan).

## Prasyarat
Untuk menjalankan Aya Notulen App, kamu perlu:
- Python 3.8–3.11
- Virtual environment (disarankan)
- Sistem operasi: Windows, Linux, atau macOS
- Ruang penyimpanan cukup untuk model AI (minimal 5 GB)
- (Opsional) GPU dengan CUDA untuk performa lebih cepat

## Instalasi
Ikuti langkah-langkah berikut untuk mengatur proyek:

1. **Clone Repository**:
   ```bash
   git clone https://github.com/username/aya-notulen-app.git
   cd aya-notulen-app
   ```

2. **Buat Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```

3. **Instal Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Unduh Model AI**:
   - Model seperti `ecapa_voxceleb`, `whisper_medium`, dan `t5-indonesian-summarization` akan diunduh otomatis saat pertama kali menjalankan aplikasi, atau kamu bisa menyalinnya ke folder `models/` secara manual.

## Cara Penggunaan
1. **Jalankan Aplikasi**:
   ```bash
   python notulen_app.py
   ```
   - Ikuti prompt di konsol untuk merekam audio atau memasukkan file audio.
   - Aplikasi akan menghasilkan transkrip, ringkasan, dan menyimpannya ke `notulen.db`.

2. **Bangun Executable** (opsional):
   - Untuk membuat file executable yang bisa dijalankan tanpa Python:
     ```bash
     pyinstaller build.spec
     ```
   - Executable akan ada di folder `dist/Aya_NotulenApp`.

## Struktur Proyek
```
aya-notulen-app/
├── models/                     # Folder untuk model AI
│   ├── ecapa_voxceleb/        # Model pengenalan pembicara
│   ├── whisper_medium/        # Model transkripsi
│   └── t5-indonesian-summarization/  # Model summarization
├── notulen.db                 # Database SQLite untuk menyimpan notulen
├── helper.py                  # Fungsi bantu
├── AYA.ico                    # Ikon aplikasi
├── notulen_app.py             # Script utama
├── requirements.txt           # Daftar dependensi
├── build.spec                 # File konfigurasi PyInstaller
└── README.md                  # Dokumentasi proyek
```

## Kontribusi
Kami menyambut kontribusi! Jika ingin berkontribusi:
1. Fork repository ini.
2. Buat branch baru: `git checkout -b fitur-baru`.
3. Commit perubahan: `git commit -m "Menambahkan fitur baru"`.
4. Push ke branch: `git push origin fitur-baru`.
5. Buat Pull Request di GitHub.

## Lisensi
Dilisensikan di bawah [MIT License](LICENSE).

## Kontak
- **GitHub Issues**: Laporkan bug atau saran di [Issues](https://github.com/username/aya-notulen-app/issues).
- **Email**: hubungi kami di example@email.com.