import pyaudio
import queue
import threading
import numpy as np
from indic_assistant.utils.logger import logger

class AudioPlayer:
    """Handles asynchronous audio playback using PyAudio with barge-in support."""

    def __init__(self, channels: int = 1, sample_rate: int = 44100):
        self.channels = channels
        self.sample_rate = sample_rate
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        self.playback_queue = queue.Queue()
        self.is_playing = False
        self.play_thread = None
        self._stop_event = threading.Event()

    def _play_loop(self):
        """Worker thread to consume and play audio chunks from the queue."""
        logger.info("Playback thread started.")
        while not self._stop_event.is_set():
            try:
                # Use short timeout to check stop event periodically
                audio_chunk = self.playback_queue.get(timeout=0.1)
                
                # Make sure the stream is open
                if not self.stream:
                    self._open_stream()
                
                # Write audio chunk to device
                if self.stream and not self._stop_event.is_set():
                    self.stream.write(audio_chunk.tobytes())
                
                self.playback_queue.task_done()
            except queue.Empty:
                # If queue is empty, we can close the stream after a brief delay of inactivity to release output device
                if self.is_playing and self.playback_queue.empty():
                    # Playback finished
                    self.is_playing = False
                    logger.info("Playback completed.")
                continue
            except Exception as e:
                logger.error(f"Error during audio write: {e}")
                break

    def _open_stream(self):
        """Opens PyAudio stream for output."""
        try:
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,  # ParlerTTS output is float32
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )
        except Exception as e:
            logger.error(f"Failed to open playback stream: {e}")

    def play(self, audio_data: np.ndarray):
        """Queues a numpy float32 audio chunk for playback."""
        # Ensure it is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Start thread if not running
        if not self.play_thread or not self.play_thread.is_alive():
            self._stop_event.clear()
            self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.play_thread.start()

        self.is_playing = True
        self.playback_queue.put(audio_data)

    def stop(self):
        """Immediately interrupts playback and clears the queue (barge-in)."""
        if not self.is_playing and self.playback_queue.empty():
            return
            
        logger.info("Interrupting playback (barge-in triggered)...")
        self._stop_event.set()
        
        # Clear the queue
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
                self.playback_queue.task_done()
            except queue.Empty:
                break
        
        # Close stream
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
            
        self.is_playing = False
        
        # Wait for thread to finish
        if self.play_thread:
            self.play_thread.join(timeout=0.5)
            self.play_thread = None
            
        logger.info("Playback interrupted and cleared.")

    def close(self):
        """Releases PyAudio playback resources."""
        self.stop()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            logger.info("PyAudio playback terminated.")
