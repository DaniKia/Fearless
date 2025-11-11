"""
Evaluation module for comparing transcripts and calculating metrics.
"""

import jiwer

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate between reference and hypothesis.
    
    Args:
        reference: Ground truth transcript
        hypothesis: Predicted transcript
        
    Returns:
        WER as a percentage
    """
    if not reference or not hypothesis:
        return 100.0
    
    try:
        wer = jiwer.wer(reference, hypothesis) * 100
        return wer
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return 100.0

def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate between reference and hypothesis.
    
    Args:
        reference: Ground truth transcript
        hypothesis: Predicted transcript
        
    Returns:
        CER as a percentage
    """
    if not reference or not hypothesis:
        return 100.0
    
    try:
        cer = jiwer.cer(reference, hypothesis) * 100
        return cer
    except Exception as e:
        print(f"Error calculating CER: {e}")
        return 100.0

def get_alignment_details(reference, hypothesis):
    """
    Get detailed alignment between reference and hypothesis.
    Shows insertions, deletions, and substitutions.
    
    Args:
        reference: Ground truth transcript
        hypothesis: Predicted transcript
        
    Returns:
        Dictionary with alignment details
    """
    try:
        measures = jiwer.compute_measures(reference, hypothesis)
        return {
            'substitutions': measures['substitutions'],
            'deletions': measures['deletions'],
            'insertions': measures['insertions'],
            'hits': measures['hits']
        }
    except Exception as e:
        print(f"Error computing alignment: {e}")
        return None

def display_comparison(audio_filename, reference, hypothesis, show_timestamps=False):
    """
    Display side-by-side comparison of reference and hypothesis transcripts.
    
    Args:
        audio_filename: Name of the audio file
        reference: Reference transcript
        hypothesis: Hypothesis transcript or Whisper result dict
        show_timestamps: Whether to show timestamp info
    """
    print("\n" + "="*80)
    print(f"File: {audio_filename}")
    print("="*80)
    
    hypothesis_text = hypothesis
    if isinstance(hypothesis, dict):
        hypothesis_text = hypothesis.get('text', '')
    
    print("\nREFERENCE (Ground Truth):")
    print("-" * 80)
    print(reference)
    print()
    
    print("WHISPER (Hypothesis):")
    print("-" * 80)
    print(hypothesis_text)
    print()
    
    if show_timestamps and isinstance(hypothesis, dict) and hypothesis.get('segments'):
        print("WHISPER (With Timestamps):")
        print("-" * 80)
        from modules.whisper_transcriber import format_transcript_with_timestamps
        print(format_transcript_with_timestamps(hypothesis))
        print()
    
    wer = calculate_wer(reference, hypothesis_text)
    cer = calculate_cer(reference, hypothesis_text)
    
    print("METRICS:")
    print("-" * 80)
    print(f"Word Error Rate (WER): {wer:.2f}%")
    print(f"Character Error Rate (CER): {cer:.2f}%")
    
    alignment = get_alignment_details(reference, hypothesis_text)
    if alignment:
        print(f"\nError Breakdown:")
        print(f"  Correct words: {alignment['hits']}")
        print(f"  Substitutions: {alignment['substitutions']}")
        print(f"  Deletions: {alignment['deletions']}")
        print(f"  Insertions: {alignment['insertions']}")
    
    print("="*80)
    
    return {
        'wer': wer,
        'cer': cer,
        'alignment': alignment
    }
