"""
Data loader for ASR audio segments and reference transcripts.
"""

import os
import glob
from pathlib import Path
from modules.drive_connector import is_colab, list_files_in_folder

_transcript_cache = {}
_sid_label_cache = {}

def load_audio_file_list(audio_dir):
    """
    List all WAV files in the audio directory.
    
    Args:
        audio_dir: Path to directory containing audio files
        
    Returns:
        List of audio file paths
    """
    if is_colab():
        if not os.path.exists(audio_dir):
            print(f"Audio directory not found: {audio_dir}")
            return []
        wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
        return sorted(wav_files)
    else:
        files = list_files_in_folder(audio_dir)
        wav_files = [f for f in files if f.endswith('.wav')]
        return sorted(wav_files)

def load_all_transcripts(transcript_dir, dataset='Dev'):
    """
    Load all transcripts from the consolidated transcript file.
    
    Args:
        transcript_dir: Directory containing transcript file
        dataset: Dataset name (Dev, Train, or Eval)
        
    Returns:
        Dictionary mapping audio_id to transcript text
    """
    cache_key = f"{transcript_dir}_{dataset}"
    
    if cache_key in _transcript_cache:
        return _transcript_cache[cache_key]
    
    transcript_file = os.path.join(
        transcript_dir, 
        f"fsc_p3_ASR_track2_transcriptions_{dataset}.text"
    )
    
    transcripts = {}
    
    if not os.path.exists(transcript_file):
        print(f"Warning: Transcript file not found: {transcript_file}")
        return transcripts
    
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    audio_id, transcript = parts
                    transcripts[audio_id] = transcript
                elif len(parts) == 1:
                    transcripts[parts[0]] = ""
        
        _transcript_cache[cache_key] = transcripts
        print(f"Loaded {len(transcripts)} transcripts from {dataset} dataset")
        
    except Exception as e:
        print(f"Error loading transcripts: {e}")
    
    return transcripts

def load_reference_transcript(transcript_path, audio_filename, dataset='Dev'):
    """
    Load the reference transcript for a given audio file.
    
    Args:
        transcript_path: Directory containing transcript files
        audio_filename: Name of the audio file (to find matching transcript)
        dataset: Dataset name (Dev, Train, or Eval)
        
    Returns:
        String containing the reference transcript, or None if not found
    """
    base_name = Path(audio_filename).stem
    
    transcripts = load_all_transcripts(transcript_path, dataset)
    
    if base_name in transcripts:
        return transcripts[base_name]
    
    print(f"Warning: No transcript found for {audio_filename} (ID: {base_name})")
    return None

def get_audio_files_with_transcripts(audio_dir, transcript_dir, limit=None, dataset='Dev'):
    """
    Get list of audio files that have corresponding transcripts.
    
    Args:
        audio_dir: Directory with audio files
        transcript_dir: Directory with transcript files
        limit: Maximum number of files to return (None for all)
        dataset: Dataset name (Dev, Train, or Eval)
        
    Returns:
        List of tuples: (audio_path, reference_transcript)
    """
    audio_files = load_audio_file_list(audio_dir)
    
    if not audio_files:
        print("No audio files found")
        return []
    
    transcripts = load_all_transcripts(transcript_dir, dataset)
    
    pairs = []
    for audio_path in audio_files:
        audio_filename = os.path.basename(audio_path)
        base_name = Path(audio_filename).stem
        
        if base_name in transcripts:
            pairs.append((audio_path, transcripts[base_name]))
        
        if limit and len(pairs) >= limit:
            break
    
    return pairs

def load_all_sid_labels(label_dir, dataset='Dev'):
    """
    Load all SID speaker labels from the uttID2spkID file.
    
    Args:
        label_dir: Directory containing SID label files
        dataset: Dataset name (Dev, Train, or Eval)
        
    Returns:
        Dictionary mapping utterance_id to speaker_id
    """
    cache_key = f"{label_dir}_{dataset}"
    
    if cache_key in _sid_label_cache:
        return _sid_label_cache[cache_key]
    
    label_file = os.path.join(
        label_dir,
        f"fsc_p3_SID_uttID2spkID_{dataset}"
    )
    
    labels = {}
    
    if not os.path.exists(label_file):
        print(f"Warning: SID label file not found: {label_file}")
        return labels
    
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    utt_id = parts[0]
                    speaker_id = parts[1]
                    labels[utt_id] = speaker_id
        
        _sid_label_cache[cache_key] = labels
        print(f"Loaded {len(labels)} SID labels from {dataset} dataset")
        
    except Exception as e:
        print(f"Error loading SID labels: {e}")
    
    return labels

def load_sid_label(label_dir, audio_filename, dataset='Dev'):
    """
    Load the speaker label for a given audio file.
    
    Args:
        label_dir: Directory containing SID label files
        audio_filename: Name of the audio file
        dataset: Dataset name (Dev, Train, or Eval)
        
    Returns:
        Speaker ID string, or None if not found
    """
    base_name = Path(audio_filename).stem
    
    labels = load_all_sid_labels(label_dir, dataset)
    
    if base_name in labels:
        return labels[base_name]
    
    print(f"Warning: No SID label found for {audio_filename} (ID: {base_name})")
    return None

def get_sid_files_with_labels(audio_dir, label_dir, limit=None, dataset='Dev'):
    """
    Get list of SID audio files with their speaker labels.
    
    Args:
        audio_dir: Directory with audio files
        label_dir: Directory with label files
        limit: Maximum number of files to return (None for all)
        dataset: Dataset name (Dev, Train, or Eval)
        
    Returns:
        List of tuples: (audio_path, speaker_label)
    """
    audio_files = load_audio_file_list(audio_dir)
    
    if not audio_files:
        print("No audio files found")
        return []
    
    labels = load_all_sid_labels(label_dir, dataset)
    
    pairs = []
    for audio_path in audio_files:
        audio_filename = os.path.basename(audio_path)
        base_name = Path(audio_filename).stem
        
        if base_name in labels:
            pairs.append((audio_path, labels[base_name]))
        
        if limit and len(pairs) >= limit:
            break
    
    return pairs

def group_audio_by_speaker(audio_dir, label_dir, dataset='Dev'):
    """
    Group audio files by speaker for enrollment.
    
    Args:
        audio_dir: Directory with audio files
        label_dir: Directory with label files
        dataset: Dataset name (Dev, Train, or Eval)
        
    Returns:
        Dictionary mapping speaker_id to list of audio file paths
    """
    pairs = get_sid_files_with_labels(audio_dir, label_dir, limit=None, dataset=dataset)
    
    speaker_files = {}
    for audio_path, speaker_id in pairs:
        if speaker_id not in speaker_files:
            speaker_files[speaker_id] = []
        speaker_files[speaker_id].append(audio_path)
    
    return speaker_files
