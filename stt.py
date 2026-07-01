import os
import transformers
from pathlib import Path
import numpy as np
import torch
from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor, pipeline
from indic_assistant.utils.config import Config
from indic_assistant.utils.logger import logger

INDIC_WHISPER_LARGE_DIMS = {
    "d_model": 1280,
    "encoder_layers": 32,
    "decoder_layers": 32,
    "num_hidden_layers": 32,
    "encoder_ffn_dim": 5120,
    "decoder_ffn_dim": 5120,
    "encoder_attention_heads": 20,
    "decoder_attention_heads": 20,
}

class SpeechToText:
    """Handles Speech-to-Text using AI4Bharat Indic Whisper model."""

    def __init__(self):
        self.model_name = Config.STT_MODEL
        self.device = Config.STT_DEVICE
        self.hf_token = Config.HUGGINGFACE_TOKEN

        # Configure HF token if provided
        if self.hf_token:
            os.environ["HUGGINGFACE_TOKEN"] = self.hf_token
            os.environ["HF_TOKEN"] = self.hf_token

        logger.info(f"Initializing SpeechToText with model: {self.model_name} on device: {self.device}")
        
        # Decide torch type
        self.torch_dtype = torch.float32
        if self.device.startswith("cuda"):
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Try to find local cached model or download
        model_source = self._find_local_model() or self.model_name
        logger.info(f"Loading Whisper model from: {model_source}")
        
        try:
            config = WhisperConfig.from_pretrained(str(model_source), token=self.hf_token)
            # Apply Indic WhisperLarge dimensions if needed
            for key, value in INDIC_WHISPER_LARGE_DIMS.items():
                setattr(config, key, value)
            
            # Load model in correct device and dtype
            self.model = WhisperForConditionalGeneration.from_pretrained(
                str(model_source), 
                config=config,
                torch_dtype=self.torch_dtype,
                token=self.hf_token
            ).to(self.device)
            
            self._patch_whisper_generation_config(self.model)
            
            self.processor = WhisperProcessor.from_pretrained(str(model_source), token=self.hf_token)
            
            # Build automatic-speech-recognition pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=0 if self.device.startswith("cuda") else -1,
                torch_dtype=self.torch_dtype
            )
            logger.info("SpeechToText pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Error loading STT pipeline: {e}")
            raise

    def _find_local_model(self) -> Path | None:
        """Looks for model in Hugging Face local cache."""
        local_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--parthiv11--indic_whisper_nodcil"
        if not local_cache.exists():
            return None
        
        main_ref = local_cache / "refs" / "main"
        if main_ref.exists():
            try:
                ref_hash = main_ref.read_text(encoding="utf-8").strip()
                snapshot = local_cache / "snapshots" / ref_hash
                if snapshot.is_dir() and (snapshot / "pytorch_model.bin").exists():
                    return snapshot
            except Exception:
                pass
                
        for snapshot in sorted(local_cache.glob("snapshots/*"), reverse=True):
            if snapshot.is_dir() and (snapshot / "pytorch_model.bin").exists():
                return snapshot
        return None

    def _patch_whisper_generation_config(self, model: WhisperForConditionalGeneration) -> None:
        """IndicWhisper ships without language/task mappings in generation_config.
        We load the default whisper-large-v2 generation config to provide standard settings.
        """
        try:
            model.generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v2")
            logger.info("Successfully patched model.generation_config from openai/whisper-large-v2")
        except Exception as e:
            logger.warning(f"Could not patch generation config: {e}. Standard defaults will be used.")

    def _whisper_generate_kwargs(self, language: str, task: str = "transcribe") -> dict:
        """Build generate kwargs compatible with transformers, using beam search for higher accuracy."""
        # Use beam search with 5 beams to prioritize accuracy over speed
        kwargs = {
            "num_beams": 5,
            "task": task
        }
        
        # Determine language/task forced tokens
        try:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)
            kwargs["forced_decoder_ids"] = forced_decoder_ids
        except Exception as e:
            logger.warning(f"Could not get forced_decoder_ids: {e}. Falling back to setting language argument.")
            kwargs["language"] = language
            
        return kwargs

    def transcribe(self, audio_array: np.ndarray, language: str) -> str:
        """
        Transcribes a 16kHz float32 numpy array directly using Whisper pipeline.
        Normalizes and prepares audio first to maximize transcription accuracy.
        """
        if audio_array.size == 0:
            return ""

        # Peak normalization to prevent low gain / quiet recording issues and clipping
        max_amplitude = np.max(np.abs(audio_array))
        if max_amplitude > 0.0:
            audio_array = (audio_array / max_amplitude) * 0.95
            
        logger.info(f"Transcribing {len(audio_array)/16000:.2f}s of normalized audio...")
        try:
            # We construct a dictionary input for pipeline
            inputs = {
                "raw": audio_array,
                "sampling_rate": 16000
            }
            
            gen_kwargs = self._whisper_generate_kwargs(language)
            result = self.pipe(inputs, generate_kwargs=gen_kwargs)
            
            transcribed_text = result.get("text", "")
            logger.info(f"Transcription result: '{transcribed_text}'")
            return transcribed_text.strip()
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return ""
