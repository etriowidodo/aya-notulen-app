# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
import os

# Tentukan direktori dasar proyek
base_dir = r"D:\KERJAAN\experiment\notulen_app"  # Pindahkan proyek ke path lebih pendek

block_cipher = None

# 1. Kumpulkan resource speechbrain
speechbrain_datas, speechbrain_binaries, speechbrain_hiddenimports = collect_all('speechbrain')

# 2. File model dan tambahan
model_files = [
    ('models/ecapa_voxceleb', 'models/ecapa_voxceleb'),  # Gunakan path relatif tanpa wildcard
    ('models/whisper_medium', 'models/whisper_medium'),
    ('models/t5-indonesian-summarization', 'models/t5-indonesian-summarization'),
    ('notulen.db', '.'),
    ('helper.py', '.'),
    ('AYA.ico', '.'),
    ('venv/Lib/site-packages/faster_whisper/assets', 'faster_whisper/assets'),  # Path relatif
]

# 3. Gabungkan semua datas
all_datas = model_files + speechbrain_datas

# 4. Hidden imports (kurangi yang tidak diperlukan)
hidden_imports = [
    'speechbrain',
    'speechbrain.inference.speaker',
    'torchaudio._backend',
    'faster_whisper',
    'webrtcvad',
    'reportlab',
    'numpy',
    'torch',
    'spacy',
    'flair',
] + speechbrain_hiddenimports

# 5. Analysis
a = Analysis(
    ['notulen_app.py'],
    pathex=[base_dir],  # Gunakan base_dir
    binaries=speechbrain_binaries,
    datas=all_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=['tensorflow', 'tensorboard', 'k2'],  # Exclude modul besar yang mungkin tidak diperlukan
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # Pindahkan binaries ke COLLECT
    name='Aya_NotulenApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Nonaktifkan UPX untuk menghindari masalah
    runtime_tmpdir=None,
    console=False,  # Ubah ke False jika GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(base_dir, 'AYA.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,  # Nonaktifkan UPX
    upx_exclude=[],
    name='Aya_NotulenApp'
)