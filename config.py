"""
Configuration file for ASR pipeline.
Supports both Replit and Google Colab environments.
"""

import os

PHASE = "Phase3"
DATASET = "Dev"
DRIVE_ROOT_FOLDER = "Fearless_Steps_Challenge_Phase3"

def is_colab():
    """Check if running in Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_replit():
    """Check if running in Replit environment."""
    return 'REPL_ID' in os.environ

def get_local_cache_dir():
    """Get local cache directory for Replit."""
    cache_dir = os.path.join(os.getcwd(), ".asr_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_root_path():
    """Get the root path for the dataset based on environment."""
    if is_colab():
        return f"/content/drive/MyDrive/{DRIVE_ROOT_FOLDER}"
    elif is_replit():
        return get_local_cache_dir()
    else:
        return get_local_cache_dir()

def get_drive_folder_path(dataset=DATASET):
    """
    Get the Google Drive folder structure path for accessing via API.
    Returns dict with audio and transcript folder names.
    """
    return {
        'audio': f"{DRIVE_ROOT_FOLDER}/FSC_P3_Train_Dev/Audio/Segments/ASR_track2/{dataset}",
        'transcript': f"{DRIVE_ROOT_FOLDER}/FSC_P3_Train_Dev/Transcripts/ASR_track2/{dataset}"
    }

def get_audio_path(dataset=DATASET):
    """Get path to audio segments for ASR_track2."""
    root = get_root_path()
    if is_colab():
        return f"{root}/FSC_P3_Train_Dev/Audio/Segments/ASR_track2/{dataset}"
    else:
        audio_dir = os.path.join(root, "audio", dataset)
        os.makedirs(audio_dir, exist_ok=True)
        return audio_dir

def get_transcript_path(dataset=DATASET):
    """Get path to reference transcripts for ASR_track2."""
    root = get_root_path()
    if is_colab():
        return f"{root}/FSC_P3_Train_Dev/Transcripts/ASR_track2/{dataset}"
    else:
        transcript_dir = os.path.join(root, "transcripts", dataset)
        os.makedirs(transcript_dir, exist_ok=True)
        return transcript_dir

def get_output_path():
    """Get path to save ASR outputs."""
    root = get_root_path()
    output_dir = os.path.join(root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_sid_audio_path(dataset=DATASET):
    """Get path to audio segments for SID."""
    root = get_root_path()
    if is_colab():
        return f"{root}/FSC_P3_Train_Dev/Audio/Segments/SID/{dataset}"
    else:
        audio_dir = os.path.join(root, "sid_audio", dataset)
        os.makedirs(audio_dir, exist_ok=True)
        return audio_dir

def get_sid_label_path(dataset=DATASET):
    """Get path to SID speaker labels."""
    root = get_root_path()
    if is_colab():
        return f"{root}/FSC_P3_Train_Dev/Transcripts/SID"
    else:
        label_dir = os.path.join(root, "sid_labels")
        os.makedirs(label_dir, exist_ok=True)
        return label_dir

def get_speaker_database_path():
    """Get path to speaker database file."""
    root = get_root_path()
    return os.path.join(root, "speaker_database.pkl")

WHISPER_MODEL = "tiny.en"
SPEAKER_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
