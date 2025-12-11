"""
Audio I/O module for loading audio files.
Provides a clean separation between file loading and audio processing.
"""

import numpy as np
import soundfile as sf


def load_audio(audio_path):
    """
    Load audio file from disk.
    
    Args:
        audio_path: Path to audio file (WAV or other formats supported by soundfile)
        
    Returns:
        Tuple of (waveform, sample_rate):
        - waveform: numpy array of audio samples (float32)
        - sample_rate: integer sample rate in Hz
        
    Raises:
        Exception if file cannot be loaded
    """
    audio, sr = sf.read(audio_path, dtype='float32')
    return audio, sr


def load_audio_safe(audio_path):
    """
    Load audio file with error handling.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (waveform, sample_rate, error):
        - waveform: numpy array or None if error
        - sample_rate: integer or None if error
        - error: error message string or None if success
    """
    try:
        audio, sr = load_audio(audio_path)
        return audio, sr, None
    except Exception as e:
        return None, None, str(e)
