import logging
import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline


class MeetingAnalyzer:
    def __init__(self, model_path="models/t5-indonesian-summarization"):
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                framework="pt"
            )
            self.logger.info(f"MeetingAnalyzer initialized on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to initialize MeetingAnalyzer: {str(e)}")
            raise

    def _clean_text(self, text):
        """Enhanced cleaning for Indonesian meeting transcripts"""
        # Remove speaker labels and metadata
        text = re.sub(r'Pembicara \d+(-varian)?:', '', text)
        text = re.sub(r'===.*?===', '', text)
        text = re.sub(r'---.*?---', '', text)

        # Remove common Indonesian slang and filler words
        slang = ['bang', 'iya', 'gitu', 'kan', 'lah', 'sih', 'deh', 'nih', 'dong', 'cuy', 'anjir', 'weh']
        for word in slang:
            text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)

        # Remove extra whitespace and empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = ' '.join(lines)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _is_text_too_short(self, text, min_words=50):
        """Check if text is too short to summarize"""
        return len(text.split()) < min_words

    def _chunk_text(self, text, max_tokens=400):
        """Token-aware text chunking"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            return_overflowing_tokens=True,
            max_length=max_tokens,
            stride=50
        )
        chunks = [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in inputs["input_ids"]]
        return chunks

    def _prepare_prompt(self, text):
        """Create an effective prompt for economic-focused meeting summarization"""
        return (
            "Ringkas percakapan berikut dalam Bahasa Indonesia dengan format berikut:\n"
            "- Topik Utama: [Daftar topik utama]\n"
            "- Fakta Penting: [Poin-poin fakta ekonomi atau peluang]\n"
            "- Mitos vs Fakta: [Mitos] → [Fakta]\n"
            "- Rekomendasi: [Saran praktis]\n"
            "- Tindak Lanjut: [Langkah konkret]\n"
            "Contoh:\n"
            "- Topik Utama: Peluang side hustle dan literasi keuangan.\n"
            "- Fakta Penting: Side hustle seperti content creator bisa menghasilkan 2-7,5 juta per proyek.\n"
            "- Mitos vs Fakta: Menabung tidak bisa kaya → Menabung adalah langkah awal membangun modal.\n"
            "- Rekomendasi: Manfaatkan media sosial untuk side hustle.\n"
            "- Tindak Lanjut: Bangun koneksi dengan komunitas pengusaha.\n"
            f"Teks: {text}"
        )

    def generate_summary(self, meeting_text):
        """Generate comprehensive meeting summary"""
        if not meeting_text.strip():
            return "Tidak ada transkrip yang dapat diringkas"

        try:
            cleaned_text = self._clean_text(meeting_text)

            if self._is_text_too_short(cleaned_text):
                prompt = f"Jelaskan inti dari teks berikut dalam 2-3 kalimat: {cleaned_text}"
                result = self.summarizer(
                    prompt,
                    max_new_tokens=80,
                    min_length=20,
                    do_sample=False,
                    num_beams=4
                )
                return "=== INTISARI ===\n" + result[0]['summary_text']

            # Chunk the text for longer inputs
            chunks = self._chunk_text(cleaned_text)
            chunk_summaries = []
            for chunk in chunks:
                prompt = self._prepare_prompt(chunk)
                summary = self.summarizer(
                    prompt,
                    max_new_tokens=150,
                    min_length=50,
                    do_sample=False,
                    num_beams=6,
                    no_repeat_ngram_size=3
                )[0]['summary_text']
                chunk_summaries.append(summary)

            # Combine chunk summaries
            combined_summary = " ".join(chunk_summaries)
            final_prompt = self._prepare_prompt(combined_summary)
            final_summary = self.summarizer(
                final_prompt,
                max_new_tokens=200,
                min_length=80,
                do_sample=False,
                num_beams=6,
                no_repeat_ngram_size=3
            )[0]['summary_text']

            # Post-process the summary
            final_summary = re.sub(r'ringkas.*?:', '', final_summary)
            final_summary = re.sub(r'^\s*[\d\.\)]\s*', '', final_summary, flags=re.MULTILINE)
            final_summary = re.sub(r'\s+', ' ', final_summary).strip()

            # Ensure structured output
            lines = final_summary.split('\n')
            structured_summary = []
            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith(('- Topik Utama:', '- Fakta Penting:', '- Mitos vs Fakta:', '- Rekomendasi:',
                                    '- Tindak Lanjut:')):
                    current_section = line
                    structured_summary.append(line)
                elif current_section and line:
                    structured_summary.append(f"  {line}")

            return "=== RINGKASAN RAPAT ===\n" + '\n'.join(structured_summary)

        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return "Gagal membuat ringkasan"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = MeetingAnalyzer()

    # Use the provided sample text
    with open("meeting_transcript.txt", "r", encoding="utf-8") as f:
        sample_text = f.read()

    print("\n" + "=" * 50)
    summary = analyzer.generate_summary(sample_text)
    print(summary)