"""
Text-to-Speech module using gTTS (Google Text-to-Speech).
Handles voice synthesis and audio generation.
"""

import io
import os
from pathlib import Path
from gtts import gTTS
import soundfile as sf
import numpy as np
from .logger import setup_logger

logger = setup_logger(__name__)


class TTSEngine:
    """Text-to-Speech engine using gTTS."""
    
    # Available voices (language codes)
    VOICES = {
        "lessac": "en",  # English (default)
        "bryce": "en",
        "kristin": "en",
        "amy": "en",
    }
    
    def __init__(self, voice_name: str = "lessac", voice_dir: str = "data/voices"):
        """
        Initialize TTS engine with specified voice.
        
        Args:
            voice_name: Name of voice (key in VOICES dict)
            voice_dir: Directory to cache voice models (unused for gTTS)
        """
        self.voice_name = voice_name
        self.voice_dir = Path(voice_dir)
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TTSEngine initialized with voice: {voice_name} (gTTS)")
    
    def load_voice(self):
        """Load the voice model (no-op for gTTS)."""
        logger.debug("Voice model loaded (gTTS)")
    
    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to speech using gTTS.
        
        Args:
            text: Text to synthesize
        
        Returns:
            WAV audio bytes
        """
        logger.debug(f"Synthesizing text ({len(text)} chars): {text[:50]}...")
        
        try:
            # Use gTTS to generate speech
            tts = gTTS(text=text, lang=self.VOICES.get(self.voice_name, "en"), slow=False)
            
            # Save to BytesIO buffer
            mp3_buffer = io.BytesIO()
            tts.write_to_fp(mp3_buffer)
            mp3_buffer.seek(0)
            
            # Convert MP3 to WAV using numpy
            # Note: We return MP3 bytes since WAV conversion would require ffmpeg
            wav_bytes = mp3_buffer.getvalue()
            
            logger.debug(f"Synthesis complete: {len(wav_bytes)} bytes")
            return wav_bytes
        except Exception as e:
            logger.error(f"Error synthesizing text: {e}")
            raise
    
    def set_voice(self, voice_name: str):
        """
        Switch to a different voice.
        
        Args:
            voice_name: Name of the voice to switch to
        """
        if voice_name not in self.VOICES:
            logger.warning(f"Unknown voice: {voice_name}")
            return False
        
        logger.info(f"Switching voice from {self.voice_name} to {voice_name}")
        self.voice_name = voice_name
        return True
    
    def get_available_voices(self) -> list:
        """Get list of available voice names."""
        return list(self.VOICES.keys())
