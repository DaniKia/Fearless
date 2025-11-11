"""
Configuration file for ASR pipeline.
Supports both Replit and Google Colab environments.
"""

import os

PHASE = "Phase3"
DATASET = "Dev"

def is_colab():
    """Check if running in Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_root_path():
    """Get the root path for the dataset based on environment."""
    if is_colab():
        return f"/content/drive/MyDrive/FS/{PHASE}"
    else:
        return None

def get_audio_path(dataset=DATASET):
    """Get path to audio segments for ASR_track2."""
    root = get_root_path()
    if root:
        return f"{root}/Audio/Segments/ASR_track2/{dataset}"
    return None

def get_transcript_path(dataset=DATASET):
    """Get path to reference transcripts for ASR_track2."""
    root = get_root_path()
    if root:
        return f"{root}/Transcripts/ASR_track2/{dataset}"
    return None

def get_output_path():
    """Get path to save ASR outputs."""
    root = get_root_path()
    if root:
        output_dir = f"{root}/ASR_outputs"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    return "./outputs"

WHISPER_MODEL = "tiny.en"
