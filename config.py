import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Workspace & Paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    DB_PATH = str(DATA_DIR / "assistant.db")
    AUDIO_LOG_DIR = DATA_DIR / "audio_logs"
    AUDIO_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # API Keys
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
    # Default to Gemini if key is provided or system key exists
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # STT Settings
    STT_MODEL = os.getenv("STT_MODEL", "parthiv11/indic_whisper_nodcil")
    STT_DEVICE = os.getenv("STT_DEVICE", "cpu")  # default to cpu if no torch cuda config is set
    STT_LANGUAGE = os.getenv("STT_LANGUAGE", "hi")  # default Whisper language

    # VAD Settings
    VAD_SAMPLING_RATE = 16000
    VAD_FRAME_SIZE = 512  # ms frame sizes depend on VAD (for silero, 512 samples at 16khz = 32ms)
    VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))

    # LLM Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "transformers-local")  # options: gemini, openai, ollama, openai-compatible, transformers-local, mockup
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-1.7b")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # URL for local/custom endpoints (e.g. Ollama, LM Studio)

    # TTS Settings
    TTS_MODEL = os.getenv("TTS_MODEL", "ai4bharat/indic-parler-tts")
    TTS_SPEAKER = os.getenv("TTS_SPEAKER", "Rohit")
    TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "hi")
    TTS_DEVICE = os.getenv("TTS_DEVICE", "cpu")

    @classmethod
    def validate(cls):
        # We need HuggingFace token for Indic Whisper nodcil
        if not cls.HUGGINGFACE_TOKEN:
            print("[WARN] HUGGINGFACE_TOKEN not set in environment variables.")
