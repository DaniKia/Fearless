"""
Configuration file for ASR pipeline.
Supports both Replit and Google Colab environments.
"""

import os

PHASE = "Phase3"
DATASET = "Dev"
DRIVE_ROOT_FOLDER = "Fearless_Steps_Challenge_Phase3"

_SID_LABEL_SUFFIXES = ("", ".txt", ".TXT")

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

def get_sid_label_basename(dataset=DATASET):
    """Return the base filename for SID speaker labels without extension."""
    return f"fsc_p3_SID_uttID2spkID_{dataset}"

def find_sid_label_file(label_dir=None, dataset=DATASET):
    """
    Locate the SID label file, accounting for optional extensions used in the dataset.

    Args:
        label_dir: Directory containing SID label files. Defaults to configured path.
        dataset: Dataset name (Dev, Train, or Eval).

    Returns:
        Full path to the SID label file. If no candidate exists on disk, returns the
        path using the canonical filename without extension.
    """
    if label_dir is None:
        label_dir = get_sid_label_path(dataset)

    base_name = get_sid_label_basename(dataset)

    for suffix in _SID_LABEL_SUFFIXES:
        candidate = os.path.join(label_dir, base_name + suffix)
        if os.path.exists(candidate):
            return candidate

    return os.path.join(label_dir, base_name)

def get_asr_track2_speaker_label_path(dataset=DATASET):
    """Get path to ASR_track2 speaker labels (same directory as transcripts)."""
    return get_transcript_path(dataset)

def get_asr_track2_speaker_label_basename(dataset=DATASET):
    """Return the base filename for ASR_track2 speaker labels without extension."""
    return f"fsc_p3_ASR_track2_uttID2spkID_{dataset}"

def find_asr_track2_speaker_label_file(label_dir=None, dataset=DATASET):
    """
    Locate the ASR_track2 speaker label file, accounting for optional extensions.

    Args:
        label_dir: Directory containing ASR_track2 label files. Defaults to configured path.
        dataset: Dataset name (Dev, Train, or Eval).

    Returns:
        Full path to the ASR_track2 speaker label file.
    """
    if label_dir is None:
        label_dir = get_asr_track2_speaker_label_path(dataset)

    base_name = get_asr_track2_speaker_label_basename(dataset)

    for suffix in (".text", ".txt", ".TXT", ""):
        candidate = os.path.join(label_dir, base_name + suffix)
        if os.path.exists(candidate):
            return candidate

    return os.path.join(label_dir, base_name + ".text")

def detect_dataset_info_from_path(audio_path):
    """
    Detect dataset type (ASR_track2 or SID) and split (Dev/Train/Eval) from audio path.
    Works with both Colab paths and local cache paths.
    
    Args:
        audio_path: Full path to audio directory or file
        
    Returns:
        Tuple of (dataset_type, dataset_split)
        dataset_type: 'ASR_track2' or 'SID' (defaults to 'ASR_track2' if ambiguous)
        dataset_split: 'Dev', 'Train', or 'Eval' (defaults to 'Dev' if not found)
    """
    path_str = str(audio_path)
    
    # Detect dataset type from path patterns
    dataset_type = 'ASR_track2'  # Default to ASR_track2
    
    # Check for SID indicators first (more specific)
    if '/SID/' in path_str or '\\SID\\' in path_str or '/sid_audio/' in path_str or '\\sid_audio\\' in path_str:
        dataset_type = 'SID'
    # Check for ASR_track2 indicators
    elif '/ASR_track2/' in path_str or '\\ASR_track2\\' in path_str:
        dataset_type = 'ASR_track2'
    # Local cache pattern: /audio/ (not sid_audio) defaults to ASR_track2
    elif '/audio/' in path_str or '\\audio\\' in path_str:
        # Make sure it's not sid_audio
        if 'sid_audio' not in path_str.lower():
            dataset_type = 'ASR_track2'
    
    # Detect dataset split
    dataset_split = 'Dev'  # Default to Dev
    for split in ['Dev', 'Train', 'Eval']:
        if f'/{split}/' in path_str or f'\\{split}\\' in path_str or path_str.endswith(split):
            dataset_split = split
            break
    
    return dataset_type, dataset_split

def get_speaker_database_path():
    """
    Get path to speaker database file.
    Returns a single global database path (not dataset-specific).
    The database contains all enrolled speakers from Train set,
    used for identification across all datasets (Dev, Test, Eval).
    """
    root = get_root_path()
    return os.path.join(root, "speaker_database.pkl")

def get_track_audio_path(track, dataset=DATASET):
    """
    Get path to audio segments for any track (generic function).
    
    Args:
        track: Track name (e.g., 'SID', 'ASR_track2', 'SD_track2', 'SD_track1', 'SAD', etc.)
        dataset: Dataset name (Dev, Train, or Eval)
        
    Returns:
        Full path to audio directory for the specified track
    """
    root = get_root_path()
    if is_colab():
        return f"{root}/FSC_P3_Train_Dev/Audio/Segments/{track}/{dataset}"
    else:
        # Local cache: maintain backwards compatibility for SID and ASR_track2
        if track == 'SID':
            audio_dir = os.path.join(root, "sid_audio", dataset)
        elif track == 'ASR_track2':
            audio_dir = os.path.join(root, "audio", dataset)
        else:
            # For custom tracks, create directory based on track name
            safe_track_name = track.lower().replace('_', '').replace('-', '')
            audio_dir = os.path.join(root, f"{safe_track_name}_audio", dataset)
        os.makedirs(audio_dir, exist_ok=True)
        return audio_dir

def get_track_label_path(track, dataset=DATASET):
    """
    Get path to labels/transcripts for any track (generic function).
    
    Args:
        track: Track name (e.g., 'SID', 'ASR_track2', 'SD_track2', 'SD_track1', 'SAD', etc.)
        dataset: Dataset name (Dev, Train, or Eval)
        
    Returns:
        Full path to label/transcript directory for the specified track
    """
    root = get_root_path()
    if is_colab():
        # For SID, labels are in Transcripts/SID (no dataset subfolder)
        # For others like ASR_track2, labels are in Transcripts/ASR_track2/{dataset}
        if track == 'SID':
            return f"{root}/FSC_P3_Train_Dev/Transcripts/SID"
        else:
            return f"{root}/FSC_P3_Train_Dev/Transcripts/{track}/{dataset}"
    else:
        # Local cache: maintain backwards compatibility for SID and ASR_track2
        if track == 'SID':
            label_dir = os.path.join(root, "sid_labels")
        elif track == 'ASR_track2':
            label_dir = os.path.join(root, "transcripts", dataset)
        else:
            # For custom tracks, create directory based on track name
            safe_track_name = track.lower().replace('_', '').replace('-', '')
            label_dir = os.path.join(root, f"{safe_track_name}_labels", dataset)
        os.makedirs(label_dir, exist_ok=True)
        return label_dir

WHISPER_MODEL = "tiny.en"
SPEAKER_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
