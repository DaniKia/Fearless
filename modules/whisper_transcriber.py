"""
Whisper ASR transcriber module.
"""

import torch
import whisper

_model_cache = {}

def load_whisper_model(model_name="tiny.en"):
    """
    Load Whisper model with caching.
    
    Args:
        model_name: Name of Whisper model to load
        
    Returns:
        Loaded Whisper model
    """
    global _model_cache

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = (model_name, device)

    if cache_key not in _model_cache:
        precision = "FP16" if device == "cuda" else "FP32"
        print(f"Loading Whisper model: {model_name} ({precision} on {device})...")
        _model_cache[cache_key] = whisper.load_model(model_name, device=device)
        print("Model loaded successfully")

    return _model_cache[cache_key]

def transcribe_audio(audio_path, model_name="tiny.en", include_timestamps=True):
    """
    Transcribe audio file using Whisper.
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model to use
        include_timestamps: Whether to include word-level timestamps
        
    Returns:
        Dictionary with transcription results:
        - text: Transcribed text
        - segments: List of segments with timestamps
        - language: Detected language
    """
    model = load_whisper_model(model_name)
    
    print(f"Transcribing: {audio_path}")

    use_fp16 = model.device.type != "cpu"

    result = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=include_timestamps,
        verbose=False,
        fp16=use_fp16
    )
    
    return {
        'text': result['text'].strip(),
        'segments': result.get('segments', []),
        'language': result.get('language', 'en')
    }

def format_transcript_with_timestamps(result):
    """
    Format transcript with timestamps for display.
    
    Args:
        result: Whisper transcription result
        
    Returns:
        Formatted string with timestamps
    """
    if not result.get('segments'):
        return result.get('text', '')
    
    formatted = []
    for segment in result['segments']:
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        text = segment.get('text', '').strip()
        formatted.append(f"[{start:.2f}s - {end:.2f}s] {text}")
    
    return '\n'.join(formatted)
