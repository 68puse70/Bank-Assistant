import nltk
import torch
import numpy as np
from transformers import AutoTokenizer, AutoFeatureExtractor
from parler_tts import ParlerTTSForConditionalGeneration
from indic_assistant.utils.config import Config
from indic_assistant.utils.logger import logger

nltk.download("punkt_tab", quiet=True)

class TTSEngine:
    """Manages synthesising speech using ai4bharat/indic-parler-tts."""

    def __init__(self):
        self.model_name = Config.TTS_MODEL
        self.device = Config.TTS_DEVICE
        self.speaker = Config.TTS_SPEAKER
        self.language = Config.TTS_LANGUAGE
        
        # Determine device settings
        if self.device == "cuda":
            if torch.cuda.is_available():
                logger.info("CUDA detected – running TTS on GPU.")
            else:
                logger.warning("CUDA requested but not available – falling back to CPU.")
                self.device = "cpu"
        
        # Determine torch dtype – use fp16 on GPU for speed, fp32 on CPU
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info(f"Loading TTS model {self.model_name} on device: {self.device}...")
        
        try:
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.model_name,
                attn_implementation="eager",
                torch_dtype=self.torch_dtype
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.description_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.sampling_rate = self.model.audio_encoder.config.sampling_rate
            logger.info("TTS engine initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing ParlerTTS: {e}")
            raise

    def chunk_text(self, text: str, chunk_size: int = 25) -> list[str]:
        """Splits text into sentence-aware chunks of <= chunk_size words."""
        sentences = nltk.sent_tokenize(text)
        chunks, current = [], ""
        for sent in sentences:
            candidate = f"{current} {sent}".strip()
            if len(candidate.split()) >= chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks or [text]

    def generate_streaming(self, text: str, speaker_desc: str, callback) -> None:
        """
        Synthesises text chunks sequentially, and immediately executes the callback 
        with the generated numpy audio chunk to stream playback.
        """
        logger.info(f"Generating TTS for: '{text}'")
        
        desc_inputs = self.description_tokenizer(speaker_desc, return_tensors="pt").to(self.device)
        chunks = self.chunk_text(text)
        logger.info(f"Text split into {len(chunks)} chunks: {chunks}")

        for idx, chunk in enumerate(chunks):
            logger.info(f"Synthesising chunk {idx+1}/{len(chunks)}: '{chunk}'")
            try:
                prompt_inputs = self.tokenizer(chunk, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    generation = self.model.generate(
                        input_ids=desc_inputs.input_ids,
                        attention_mask=desc_inputs.attention_mask,
                        prompt_input_ids=prompt_inputs.input_ids,
                        prompt_attention_mask=prompt_inputs.attention_mask,
                        do_sample=True,
                        return_dict_in_generate=True,
                    )
                
                if hasattr(generation, "sequences") and hasattr(generation, "audios_length"):
                    audio = generation.sequences[0, : generation.audios_length[0]]
                    audio_np = audio.to(torch.float32).cpu().numpy().squeeze()
                    if audio_np.ndim > 1:
                        audio_np = audio_np.flatten()
                    
                    # Yield audio segment to callback for real-time play
                    logger.info(f"Chunk {idx+1} synthesized ({len(audio_np)/self.sampling_rate:.2f}s). Triggering playback.")
                    callback(audio_np)
            except Exception as e:
                logger.error(f"Error generating speech for chunk '{chunk}': {e}")
