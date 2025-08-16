# Aya Notulen App

Aya Notulen App is a Python-based application designed to automatically record, transcribe, and summarize meeting notes using AI-powered speech-to-text and summarization technologies. It is tailored for creating accurate meeting minutes in Indonesian with additional speaker identification capabilities.


## Features
- **Audio Transcription**: Converts meeting recordings to text using the `faster-whisper` model.
- **Summarization**: Generates concise summaries of meetings with the `t5-indonesian-summarization` model.
- **Speaker Identification**: Recognizes speakers using the `speechbrain` ECAPA-VoxCeleb model.
- **Local Database**: Stores meeting notes in a SQLite database (`notulen.db`).
- **PDF Export**: Exports meeting minutes as PDF files using `reportlab`.
- ** Word Export**: Export meeting minuts as Docx files using  `python-docx`.
- **Console Interface**: Easy-to-use command-line interface (GUI planned for future releases).

## Prerequisites
- Python 3.8–3.11
- Virtual environment (recommended)
- Operating system: Windows, Linux, or macOS
- At least 5 GB of storage for AI models
- (Optional) GPU with CUDA for faster performance

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/etriowidodo/aya-notulen-app.git
   cd aya-notulen-app
   ```
2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download AI Models**:
   - Models (`ecapa_voxceleb`, `whisper_medium`, `t5-indonesian-summarization`) will be downloaded automatically on first run, or manually place them in the `models/` directory.

## Download
Get the latest release: [Download AyaNotulenApp.rar](https://www.mediafire.com/file/edw9cdztvri04vt/AyaNotulenApp.rar/file)

## Usage
Run the application:
```bash
python notulen_app.py
```
- Follow the console prompts to record audio or input an audio file.
- The app will generate transcripts, summaries, and store them in `notulen.db`.

To build an executable:
```bash
pyinstaller build.spec
```
- The executable will be in the `dist/Aya_NotulenApp` directory.

## Project Structure
```
aya-notulen-app/
├── models/                     # AI model files
│   ├── ecapa_voxceleb/        # Speaker identification model
│   ├── whisper_medium/        # Transcription model
│   └── t5-indonesian-summarization/  # Summarization model
├── notulen.db                 # SQLite database for notes
├── helper.py                  # Helper functions
├── AYA.ico                    # Application icon
├── notulen_app.py             # Main script
├── requirements.txt           # Dependencies
├── build.spec                 # PyInstaller configuration
└── README.md                  # Project documentation
```

## Contributing
Contributions are welcome! To contribute:
1. Fork this repository.
2. Create a new branch: `git checkout -b new-feature`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Push to the branch: `git push origin new-feature`.
5. Create a Pull Request on GitHub.

## License
[MIT License](LICENSE)

## Contact
- **GitHub Issues**: Report bugs or suggestions at [Issues](https://github.com/etriowidodo/aya-notulen-app/issues).
- **Email**: Contact us at example@email.com.
