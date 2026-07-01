import time
import uuid
import sys
import numpy as np
import traceback
import torch
from typing import Optional

from indic_assistant.utils.config import Config
from indic_assistant.utils.logger import logger
from indic_assistant.speech.microphone import MicrophoneStream
from indic_assistant.speech.vad import VADManager
from indic_assistant.speech.stt import SpeechToText
from indic_assistant.translation.translator import Translator
from indic_assistant.llm.client import LLMClient
from indic_assistant.memory.manager import MemoryManager
from indic_assistant.storage.db import DatabaseManager
from indic_assistant.utils.audio import AudioPlayer
from indic_assistant.tts.engine import TTSEngine
from indic_assistant.utils.summary import SummaryGenerator

# System description for speaker template
DESCRIPTION_TEMPLATE = (
    "{speaker} speaks at a moderate pace with a clear, neutral tone. "
    "The recording is very high quality with no background noise."
)

class MultilingualAssistant:
    """Core orchestrator for the Multilingual AI Voice Assistant."""

    def __init__(self):
        Config.validate()
        
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.db = DatabaseManager()
        self.memory = MemoryManager()
        self.translator = Translator()
        
        # Audio Player (Playback)
        self.player = AudioPlayer(channels=1, sample_rate=44100) # ParlerTTS standard is 44100Hz
        
        # TTS Engine
        self.tts = TTSEngine()
        
        # LLM client
        self.llm = LLMClient()
        self.llm.warmup()
        
        # Summary helper
        self.summary_gen = SummaryGenerator(self.llm)
        
        # STT Engine
        self.stt = SpeechToText()
        
        # VAD Manager
        self.vad = VADManager(threshold=Config.VAD_THRESHOLD)
        
        # Microphone input stream (16kHz for VAD/STT)
        self.mic = MicrophoneStream(rate=Config.VAD_SAMPLING_RATE)
        
        # Cache speaker details
        self.speaker_desc = DESCRIPTION_TEMPLATE.format(speaker=Config.TTS_SPEAKER)
        self.target_lang = Config.TTS_LANGUAGE
        self.source_lang = Config.STT_LANGUAGE

    def run(self):
        """Starts the main assistant orchestrator loop."""
        logger.info(f"=== Starting Multilingual Voice Assistant [Session ID: {self.session_id}] ===")
        print("\n=======================================================")
        print("Multilingual AI Voice Assistant is Ready.")
        print(f"Target Language: {self.target_lang.upper()} (Speaker: {Config.TTS_SPEAKER})")
        print("Speak into your microphone. Press Ctrl+C or type 'exit' to quit.")
        print("=======================================================\n")
        
        self.db.start_session(self.session_id)
        self.mic.start()
        
        try:
            # We iterate over microphone audio chunks in a loop
            for chunk in self.mic.get_audio_chunks():
                # 1. Skip VAD checks and discard mic input while assistant is speaking to prevent echo feedback loop
                if self.player.is_playing:
                    self.vad.reset()
                    continue

                # 2. Regular VAD processing when assistant is silent
                speech_segment = self.vad.process_chunk(chunk)
                if speech_segment is not None:
                    # User finished a sentence, process it
                    self.process_turn(speech_segment)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        except Exception as e:
            logger.error(f"Unexpected error in main orchestrator loop: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.shutdown()

    def check_barge_in_chunk(self, chunk: bytes) -> bool:
        """Lightweight check to see if incoming audio chunk indicates speech during playback."""
        # Standard format is PCM16 (2 bytes per sample)
        # We perform simple energy check first (faster than running full VAD model on every single small chunk during playback)
        audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        energy = np.sqrt(np.mean(audio_data**2))
        
        # If energy is high, confirm with Silero VAD
        if energy > 0.03: 
            tensor_chunk = torch.from_numpy(audio_data).to(self.vad.device)
            with torch.no_grad():
                prob = self.vad.model(tensor_chunk, self.vad.sampling_rate).item()
            return prob > self.vad.threshold
        return False

    def process_turn(self, speech_segment: np.ndarray):
        """Processes one complete speech turn from user input to assistant speech response."""
        start_time = time.time()
        errors = None
        retries = 0
        
        # Fallback values
        original_text = ""
        english_translation = ""
        llm_response = ""
        translated_response = ""

        try:
            # 1. Speech to Text
            original_text = self.stt.transcribe(speech_segment, self.source_lang)
            if not original_text.strip():
                logger.info("Transcription was empty. Ignoring turn.")
                return

            print(f"\nUser [{self.source_lang}]: {original_text}")

            # 2. Translate to English
            english_translation = self.translator.translate_to_english(original_text, self.source_lang)
            logger.info(f"English translation: '{english_translation}'")

            # 3. Query LLM
            # Inject long-term context/facts if present
            system_injection = self.memory.get_long_term_summary()
            llm_prompt = f"{system_injection}User: {english_translation}"
            
            # Retrieve history & query
            history = self.memory.get_context()
            llm_response = self.llm.query(llm_prompt, history)
            
            # Save turns to short-term memory
            self.memory.add_interaction("user", english_translation)
            self.memory.add_interaction("assistant", llm_response)

            print(f"Assistant [en]: {llm_response}")

            # 4. Translate Response back to target Indian language
            translated_response = self.translator.translate_from_english(llm_response, self.target_lang)
            print(f"Assistant [{self.target_lang}]: {translated_response}")

            # 5. Playback: TTS Engine synthesis (streams chunks to play callback)
            # This callback gets executed whenever a chunk of audio is ready
            def play_callback(audio_chunk):
                self.player.play(audio_chunk)

            self.tts.generate_streaming(translated_response, self.speaker_desc, play_callback)

        except Exception as e:
            errors = str(e)
            logger.error(f"Error processing interaction turn: {e}")
            logger.error(traceback.format_exc())
        finally:
            latency_ms = (time.time() - start_time) * 1000
            # Log turn incrementally in SQLite database
            self.db.log_interaction(
                session_id=self.session_id,
                original_text=original_text,
                english_translation=english_translation,
                llm_prompt=llm_prompt if 'llm_prompt' in locals() else english_translation,
                llm_response=llm_response,
                translated_response=translated_response,
                latency_ms=latency_ms,
                errors=errors,
                retries=retries
            )

    def shutdown(self):
        """Gracefully shuts down microphone, playback, and saves session summary on exit."""
        logger.info("Shutting down Assistant...")
        print("\nExiting Assistant...")
        
        # Stop recording
        try:
            self.mic.close()
        except Exception:
            pass

        # Stop playback
        try:
            self.player.close()
        except Exception:
            pass

        # Generate exit summary
        try:
            history = self.memory.get_context()
            summary = self.summary_gen.generate_summary(history)
            
            print("\n================ Session Summary ================")
            print(summary)
            print("=================================================")
            
            self.db.end_session(self.session_id, summary)
        except Exception as e:
            logger.error(f"Failed during exit summary generation: {e}")

        logger.info("Assistant shutdown complete.")

if __name__ == "__main__":
    assistant = MultilingualAssistant()
    assistant.run()
