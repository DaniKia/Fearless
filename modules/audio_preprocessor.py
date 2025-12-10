"""
Audio preprocessing module with configurable processing steps.
Provides a modular pipeline for audio preprocessing that can be
enabled/disabled for ablation studies.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import librosa
from scipy.signal import butter, sosfilt


@dataclass
class PreprocessConfig:
    """Configuration for audio preprocessing pipeline."""
    
    enable_mono: bool = True
    enable_resample: bool = True
    enable_dc_removal: bool = True
    enable_bandpass: bool = False
    enable_rms_normalization: bool = True
    enable_trim: bool = True
    
    target_sr: int = 16000
    
    target_rms: float = 0.04
    target_rms_db: float = -20.0
    rms_min: float = 0.005
    rms_max: float = 0.2
    
    trim_db: float = 25.0
    
    highpass_cutoff: float = 80.0
    lowpass_cutoff: float = 7500.0
    
    min_duration: float = 0.2
    
    @classmethod
    def disabled(cls):
        """Return a config with all preprocessing disabled."""
        return cls(
            enable_mono=False,
            enable_resample=False,
            enable_dc_removal=False,
            enable_bandpass=False,
            enable_rms_normalization=False,
            enable_trim=False
        )
    
    @classmethod
    def default(cls):
        """Return default preprocessing config."""
        return cls()


@dataclass
class PreprocessResult:
    """Result of audio preprocessing."""
    waveform: np.ndarray
    sample_rate: int
    is_valid: bool = True
    duration: float = 0.0
    applied_gain: float = 1.0
    steps_applied: list = field(default_factory=list)
    error: Optional[str] = None


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono by averaging channels.
    
    Args:
        audio: Audio waveform (1D or 2D array)
        
    Returns:
        Mono audio waveform (1D array)
    """
    if len(audio.shape) > 1:
        return audio.mean(axis=1)
    return audio


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio waveform
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio waveform
    """
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """
    Remove DC offset by subtracting the mean.
    
    Args:
        audio: Audio waveform
        
    Returns:
        Audio with DC offset removed
    """
    return audio - np.mean(audio)


def apply_bandpass_filter(audio: np.ndarray, sample_rate: int, 
                          lowcut: float = 80.0, highcut: float = 7500.0,
                          order: int = 5) -> np.ndarray:
    """
    Apply bandpass filter to audio.
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate in Hz
        lowcut: High-pass cutoff frequency (removes below this)
        highcut: Low-pass cutoff frequency (removes above this)
        order: Filter order
        
    Returns:
        Filtered audio waveform
    """
    nyquist = sample_rate / 2.0
    low = lowcut / nyquist
    high = min(highcut / nyquist, 0.99)
    
    if low >= high:
        return audio
    
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def normalize_rms(audio: np.ndarray, target_rms: float = 0.04,
                  rms_min: float = 0.005, rms_max: float = 0.2) -> Tuple[np.ndarray, float]:
    """
    Normalize audio to target RMS level.
    
    Args:
        audio: Audio waveform
        target_rms: Target RMS level
        rms_min: Minimum RMS threshold (below this, scale to target)
        rms_max: Maximum RMS threshold (above this, scale to target)
        
    Returns:
        Tuple of (normalized audio, applied gain factor)
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    
    if current_rms < 1e-10:
        return audio, 1.0
    
    if current_rms < rms_min or current_rms > rms_max:
        gain = target_rms / current_rms
        return audio * gain, gain
    
    return audio, 1.0


def trim_silence(audio: np.ndarray, sample_rate: int, 
                 top_db: float = 25.0) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate in Hz
        top_db: Threshold in dB below peak to consider as silence
        
    Returns:
        Trimmed audio waveform
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return np.asarray(trimmed)


def check_duration(audio: np.ndarray, sample_rate: int, 
                   min_duration: float = 0.2) -> Tuple[bool, float]:
    """
    Check if audio meets minimum duration requirement.
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate in Hz
        min_duration: Minimum valid duration in seconds
        
    Returns:
        Tuple of (is_valid, duration_in_seconds)
    """
    duration = len(audio) / sample_rate
    is_valid = duration >= min_duration
    return is_valid, duration


def preprocess_audio(audio: np.ndarray, sample_rate: int, 
                     config: Optional[PreprocessConfig] = None) -> PreprocessResult:
    """
    Apply preprocessing pipeline to audio.
    
    Args:
        audio: Raw audio waveform from audio_io.load_audio()
        sample_rate: Sample rate of the audio
        config: PreprocessConfig object (uses default if None)
        
    Returns:
        PreprocessResult with processed waveform and metadata
    """
    if config is None:
        config = PreprocessConfig.default()
    
    result = PreprocessResult(
        waveform=audio.copy(),
        sample_rate=sample_rate,
        steps_applied=[]
    )
    
    try:
        if config.enable_mono:
            result.waveform = convert_to_mono(result.waveform)
            result.steps_applied.append('mono')
        
        if config.enable_resample:
            result.waveform = resample_audio(
                result.waveform, 
                sample_rate, 
                config.target_sr
            )
            result.sample_rate = config.target_sr
            result.steps_applied.append(f'resample_{config.target_sr}')
        
        if config.enable_dc_removal:
            result.waveform = remove_dc_offset(result.waveform)
            result.steps_applied.append('dc_removal')
        
        if config.enable_bandpass:
            result.waveform = apply_bandpass_filter(
                result.waveform,
                result.sample_rate,
                lowcut=config.highpass_cutoff,
                highcut=config.lowpass_cutoff
            )
            result.steps_applied.append('bandpass')
        
        if config.enable_rms_normalization:
            result.waveform, result.applied_gain = normalize_rms(
                result.waveform,
                target_rms=config.target_rms,
                rms_min=config.rms_min,
                rms_max=config.rms_max
            )
            result.steps_applied.append('rms_norm')
        
        if config.enable_trim:
            result.waveform = trim_silence(
                result.waveform,
                result.sample_rate,
                top_db=config.trim_db
            )
            result.steps_applied.append('trim')
        
        result.is_valid, result.duration = check_duration(
            result.waveform,
            result.sample_rate,
            min_duration=config.min_duration
        )
        
    except Exception as e:
        result.error = str(e)
        result.is_valid = False
    
    return result


def basic_preprocess_file(audio_path: str, 
                          config: Optional[PreprocessConfig] = None) -> PreprocessResult:
    """
    Load and preprocess an audio file in one step.
    
    This is a convenience function that combines audio_io.load_audio()
    with preprocess_audio().
    
    Args:
        audio_path: Path to audio file
        config: PreprocessConfig object (uses default if None)
        
    Returns:
        PreprocessResult with processed waveform and metadata
    """
    from modules.audio_io import load_audio
    
    try:
        audio, sr = load_audio(audio_path)
        return preprocess_audio(audio, sr, config)
    except Exception as e:
        return PreprocessResult(
            waveform=np.array([]),
            sample_rate=0,
            is_valid=False,
            error=str(e)
        )
