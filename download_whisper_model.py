import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import login  # Untuk model privat/gated

# 1. Setup model directory
model_dir = "./models/t5-indonesian-summarization"  # Ubah nama direktori agar spesifik untuk T5
os.makedirs(model_dir, exist_ok=True)

# 2. Model configuration (TRY THESE ALTERNATIVES IF MAIN MODEL FAILS)
model_options = [
    "cahya/t5-base-indonesian-summarization-cased",  # Opsi utama untuk summarisasi bahasa Indonesia
    "cahya/bart-base-indonesian-522M",              # Alternatif BART (jika T5 gagal)
    "facebook/bart-large-cnn"                       # Fallback ke model English
]

# 3. Download with error handling
for model_name in model_options:
    try:
        print(f"Attempting to download: {model_name}")

        # Uncomment jika perlu autentikasi (untuk model privat)
        # login(token="your_hf_token_here")  # Ganti dengan token Anda dari Hugging Face

        # Download model dan tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Simpan secara lokal
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

        print(f"Success! Model saved to: {model_dir}")
        print(f"Files created: {os.listdir(model_dir)}")
        break  # Keluar dari loop jika berhasil

    except Exception as e:
        print(f"Failed to download {model_name}: {str(e)}")
        continue

# 4. Verify download
if not os.listdir(model_dir):
    print("\nAll download attempts failed. Possible solutions:")
    print("1. Check model names are correct")
    print("2. Ensure internet connection")
    print("3. For private models: use hf_auth_token from https://huggingface.co/settings/tokens")
    print("4. Try simpler model like 'facebook/bart-large-cnn'")
else:
    print("\nModel ready for local use!")
    print(f"Use with: AutoTokenizer.from_pretrained('{model_dir}')")
    print(f"Use with: AutoModelForSeq2SeqLM.from_pretrained('{model_dir}')")