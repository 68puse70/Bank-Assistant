import pyaudio
import queue
import threading
import time
from typing import Generator
from indic_assistant.utils.logger import logger

class MicrophoneStream:
    """Streams audio from the microphone in a non-blocking background thread."""
    
    def __init__(self, rate: int = 16000, chunk_size: int = 512):
        self.rate = rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        self.thread = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Pushes raw audio data chunks onto the queue."""
        if status:
            logger.warning(f"PyAudio status warning: {status}")
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def start(self):
        """Starts recording audio from the microphone."""
        if self.is_running:
            return
        
        logger.info("Initializing microphone stream...")
        try:
            # Find default input device
            try:
                default_device = self.pyaudio_instance.get_default_input_device_info()
                logger.info(f"Using default input device: {default_device['name']}")
            except IOError:
                logger.error("No input device (microphone) found!")
                raise RuntimeError("No microphone found. Please connect a microphone.")

            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.is_running = True
            self.stream.start_stream()
            logger.info("Microphone stream started successfully.")
        except Exception as e:
            logger.error(f"Failed to start microphone stream: {e}")
            self.close()
            raise

    def stop(self):
        """Stops the audio recording stream."""
        if not self.is_running:
            return
        logger.info("Stopping microphone stream...")
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing stream: {e}")
            self.stream = None
        logger.info("Microphone stream stopped.")

    def get_audio_chunks(self) -> Generator[bytes, None, None]:
        """Generates audio chunks from the queue as they become available."""
        while self.is_running or not self.audio_queue.empty():
            try:
                # Use a small timeout so the generator check is responsive to loop termination
                chunk = self.audio_queue.get(timeout=0.1)
                yield chunk
            except queue.Empty:
                continue

    def close(self):
        """Releases PyAudio resources."""
        self.stop()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            logger.info("PyAudio terminated.")

if __name__ == "__main__":
    # Test script to verify microphone input works
    import time
    stream = MicrophoneStream()
    try:
        stream.start()
        print("Recording test... Speak into the microphone. Press Ctrl+C to stop.")
        start_time = time.time()
        for idx, chunk in enumerate(stream.get_audio_chunks()):
            print(f"\rCaptured chunk {idx+1} (size: {len(chunk)} bytes)", end="", flush=True)
            # Stop after 5 seconds of testing
            if time.time() - start_time > 5.0:
                print("\nTest completed successfully!")
                break
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    except Exception as e:
        print(f"\nTest failed: {e}")
    finally:
        stream.close()
