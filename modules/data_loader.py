"""
Data loader for ASR audio segments and reference transcripts.
"""

import os
import glob
from pathlib import Path
from modules.drive_connector import is_colab, list_files_in_folder

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

def load_reference_transcript(transcript_path, audio_filename):
    """
    Load the reference transcript for a given audio file.
    
    Args:
        transcript_path: Directory containing transcript files
        audio_filename: Name of the audio file (to find matching transcript)
        
    Returns:
        String containing the reference transcript, or None if not found
    """
    base_name = Path(audio_filename).stem
    
    txt_file = os.path.join(transcript_path, f"{base_name}.txt")
    json_file = os.path.join(transcript_path, f"{base_name}.json")
    
    if os.path.exists(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    elif os.path.exists(json_file):
        import json
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data.get('text', data.get('transcript', str(data)))
            return str(data)
    
    print(f"Warning: No transcript found for {audio_filename}")
    return None

def get_audio_files_with_transcripts(audio_dir, transcript_dir, limit=None):
    """
    Get list of audio files that have corresponding transcripts.
    
    Args:
        audio_dir: Directory with audio files
        transcript_dir: Directory with transcript files
        limit: Maximum number of files to return (None for all)
        
    Returns:
        List of tuples: (audio_path, reference_transcript)
    """
    audio_files = load_audio_file_list(audio_dir)
    
    if not audio_files:
        print("No audio files found")
        return []
    
    pairs = []
    for audio_path in audio_files:
        audio_filename = os.path.basename(audio_path)
        transcript = load_reference_transcript(transcript_dir, audio_filename)
        
        if transcript:
            pairs.append((audio_path, transcript))
        
        if limit and len(pairs) >= limit:
            break
    
    return pairs
