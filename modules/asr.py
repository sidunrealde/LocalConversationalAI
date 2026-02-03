"""
Automatic Speech Recognition (ASR) module using faster-whisper.
Handles audio input and transcription with caching and error handling.
"""

import os
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
from .logger import setup_logger

logger = setup_logger(__name__)


class ASREngine:
    """Automatic Speech Recognition engine using faster-whisper."""
    
    # Model options: tiny, base, small, medium, large
    MODELS = {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
        "medium": "medium",
        "large": "large-v3",
    }
    
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        model_dir: str = ".cache/whisper"
    ):
        """
        Initialize ASR engine.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: Device to use (cuda, cpu)
            compute_type: Computation type (float32, float16, int8)
            model_dir: Directory to cache models
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model_dir = Path(model_dir)
        
        self.model = None
        
        logger.info(
            f"ASREngine initialized: model={model_size}, device={device}, "
            f"compute_type={compute_type}"
        )
    
    def load_model(self):
        """Load the Whisper model into memory."""
        if self.model is not None:
            logger.debug("ASR model already loaded")
            return
        
        if self.model_size not in self.MODELS:
            available = list(self.MODELS.keys())
            raise ValueError(
                f"Unknown model size: {self.model_size}. "
                f"Available: {available}"
            )
        
        logger.info(f"Loading Whisper model: {self.model_size}")
        
        try:
            self.model = WhisperModel(
                self.MODELS[self.model_size],
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.model_dir),
            )
            logger.debug("ASR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            raise
    
    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Numpy array of audio samples
            sample_rate: Sample rate of audio (typically 16000 Hz)
            language: Language code (e.g., 'en' for English)
        
        Returns:
            Transcribed text
        """
        if self.model is None:
            self.load_model()
        
        logger.debug(f"Transcribing audio: {len(audio_data)} samples at {sample_rate} Hz")
        
        try:
            # Ensure audio is float32 and normalized to [-1, 1]
            audio_float = audio_data.astype(np.float32)
            if audio_float.max() > 1.0 or audio_float.min() < -1.0:
                audio_float = audio_float / 32768.0
            
            # Transcribe with VAD filter to prevent hallucinations on silent audio
            segments, info = self.model.transcribe(
                audio_float,
                language=language,
                beam_size=5,
                best_of=5,
                vad_filter=True,  # Use Silero VAD to filter out silence
                vad_parameters={"min_silence_duration_ms": 500},
                no_speech_threshold=0.6,  # Higher threshold = stricter no-speech detection
                log_prob_threshold=-1.0,  # Filter low confidence
                condition_on_previous_text=False,  # Prevent hallucination continuation
            )
            
            # Collect all text, filtering out likely hallucinations
            text_parts = []
            for segment in segments:
                # Skip segments with high no_speech probability
                if hasattr(segment, 'no_speech_prob') and segment.no_speech_prob > 0.5:
                    logger.debug(f"Skipping segment with high no_speech_prob: {segment.no_speech_prob:.2f}")
                    continue
                text_parts.append(segment.text)
            text = "".join(text_parts).strip()
            
            logger.debug(f"Transcription complete: '{text}'")
            logger.debug(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
            
            return text
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    def transcribe_file(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code
        
        Returns:
            Transcribed text
        """
        logger.info(f"Transcribing file: {audio_path}")
        
        try:
            import librosa
        except ImportError:
            logger.error("librosa not installed. Install with: pip install librosa")
            raise
        
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            return self.transcribe(audio, sample_rate=sr, language=language)
        except Exception as e:
            logger.error(f"Failed to transcribe file: {e}")
            raise
    
    def set_model_size(self, model_size: str):
        """Switch to a different model size."""
        if model_size not in self.MODELS:
            logger.warning(f"Unknown model size: {model_size}")
            return False
        
        logger.info(f"Switching ASR model from {self.model_size} to {model_size}")
        self.model_size = model_size
        self.model = None  # Reset cached model
        return True
    
    def get_available_models(self) -> list:
        """Get list of available model sizes."""
        return list(self.MODELS.keys())
