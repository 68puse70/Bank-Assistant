import numpy as np
import torch
from indic_assistant.utils.config import Config
from indic_assistant.utils.logger import logger

class VADManager:
    """Manages Voice Activity Detection (VAD) using Silero VAD."""
    
    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.device = torch.device("cpu")  # Silero VAD runs extremely fast on CPU
        
        logger.info("Loading Silero VAD model...")
        try:
            # Load Silero VAD from PyTorch Hub
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                trust_repo=True,
                verbose=False
            )
            logger.info("Silero VAD model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model from PyTorch Hub: {e}")
            raise RuntimeError(f"Could not load VAD: {e}")

        # Buffer for incoming audio bytes
        self.buffer = bytearray()
        
        # Audio configuration constants
        self.sample_width = 2  # 16-bit PCM (2 bytes per sample)
        self.chunk_samples = 512  # Silero works with 512, 1024, or 1536 samples
        self.chunk_bytes = self.chunk_samples * self.sample_width

        # State tracking
        self.is_speaking = False
        self.speech_frames = []
        self.silent_chunks = 0
        
        # Silence threshold: 1.0s of silence = stop speaking
        # At 16000 Hz, 512 samples is 32ms. 1000ms / 32ms = ~31 chunks of silence.
        self.max_silence_chunks = int(0.8 / (self.chunk_samples / self.sampling_rate))
        
        # Min speech duration: 0.4s of speech to trigger speaking state
        # At 16000 Hz, 512 samples is 32ms. 400ms / 32ms = ~12 chunks of speech.
        self.min_speech_chunks = int(0.4 / (self.chunk_samples / self.sampling_rate))
        self.speech_chunks_count = 0

    def process_chunk(self, raw_bytes: bytes) -> np.ndarray | None:
        """
        Appends raw audio bytes to buffer, runs VAD on 512-sample chunks,
        and returns a complete speech segment (as a numpy array) when the user stops speaking.
        """
        self.buffer.extend(raw_bytes)
        
        # Process in chunks of 512 samples (1024 bytes)
        while len(self.buffer) >= self.chunk_bytes:
            chunk = self.buffer[:self.chunk_bytes]
            del self.buffer[:self.chunk_bytes]
            
            # Convert PCM16 bytes -> Float32 tensor normalized to [-1.0, 1.0]
            audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            tensor_chunk = torch.from_numpy(audio_data).to(self.device)
            
            # Get probability of speech
            with torch.no_grad():
                speech_prob = self.model(tensor_chunk, self.sampling_rate).item()
            
            if speech_prob >= self.threshold:
                self.silent_chunks = 0
                self.speech_frames.append(audio_data)
                
                if not self.is_speaking:
                    self.speech_chunks_count += 1
                    if self.speech_chunks_count >= self.min_speech_chunks:
                        self.is_speaking = True
                        logger.info("User started speaking...")
            else:
                if self.is_speaking:
                    # Do NOT append silent audio; just count silent chunks
                    self.silent_chunks += 1
                    
                    if self.silent_chunks >= self.max_silence_chunks:
                        # Silence detected, finalize speech segment (speech_frames contain only speech)
                        logger.info("Silence detected. Processing speech...")
                        speech_segment = np.concatenate(self.speech_frames)
                        self.reset()
                        return speech_segment
                else:
                    # Reset speech chunk count if it was a false trigger
                    self.speech_chunks_count = 0
        
        return None

    def reset(self):
        """Resets the state of the VAD manager."""
        self.is_speaking = False
        self.speech_frames = []
        self.silent_chunks = 0
        self.speech_chunks_count = 0
        self.buffer.clear()
