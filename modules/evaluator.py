"""Evaluation module for comparing transcripts and calculating metrics."""

import re

import jiwer


_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9\s]")


def _normalize_text(text):
    """Normalize text for fair metric comparison."""

    if not text:
        return ""

    normalized = text.lower()
    normalized = _NORMALIZE_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate and Character Error Rate between reference and hypothesis.
    
    Args:
        reference: Ground truth transcript
        hypothesis: Predicted transcript
        
    Returns:
        Tuple of (WER, CER) as percentages
    """
    reference_norm = _normalize_text(reference)
    hypothesis_norm = _normalize_text(hypothesis)

    if not reference_norm or not hypothesis_norm:
        return 100.0, 100.0

    try:
        wer = jiwer.wer(reference_norm, hypothesis_norm) * 100
        cer = jiwer.cer(reference_norm, hypothesis_norm) * 100
        return wer, cer
    except Exception as e:
        print(f"Error calculating WER/CER: {e}")
        return 100.0, 100.0


def calculate_detailed_metrics(reference, hypothesis):
    """
    Calculate detailed metrics including WER, CER, S/D/I counts, and reference counts.
    
    Args:
        reference: Ground truth transcript
        hypothesis: Predicted transcript
        
    Returns:
        Dictionary with detailed metrics:
        - wer, cer: Error rates as percentages
        - substitutions, deletions, insertions, hits: Word-level counts
        - char_sub, char_del, char_ins: Character-level error counts
        - ref_words, hyp_words: Word counts
        - ref_chars, hyp_chars: Character counts
    """
    reference_norm = _normalize_text(reference)
    hypothesis_norm = _normalize_text(hypothesis)
    
    ref_words = len(reference_norm.split()) if reference_norm else 0
    hyp_words = len(hypothesis_norm.split()) if hypothesis_norm else 0
    ref_chars = len(reference_norm.replace(' ', '')) if reference_norm else 0
    hyp_chars = len(hypothesis_norm.replace(' ', '')) if hypothesis_norm else 0
    
    result = {
        'wer': 100.0,
        'cer': 100.0,
        'substitutions': 0,
        'deletions': ref_words,
        'insertions': 0,
        'hits': 0,
        'char_sub': 0,
        'char_del': ref_chars,
        'char_ins': 0,
        'ref_words': ref_words,
        'hyp_words': hyp_words,
        'ref_chars': ref_chars,
        'hyp_chars': hyp_chars
    }
    
    if not reference_norm or not hypothesis_norm:
        return result
    
    try:
        result['wer'] = jiwer.wer(reference_norm, hypothesis_norm) * 100
        result['cer'] = jiwer.cer(reference_norm, hypothesis_norm) * 100
        
        if hasattr(jiwer, 'process_words'):
            output = jiwer.process_words(reference_norm, hypothesis_norm)
            result['substitutions'] = output.substitutions
            result['deletions'] = output.deletions
            result['insertions'] = output.insertions
            result['hits'] = output.hits
        elif hasattr(jiwer, 'compute_measures'):
            measures = jiwer.compute_measures(reference_norm, hypothesis_norm)
            result['substitutions'] = measures['substitutions']
            result['deletions'] = measures['deletions']
            result['insertions'] = measures['insertions']
            result['hits'] = measures['hits']
        
        ref_chars_str = reference_norm.replace(' ', '')
        hyp_chars_str = hypothesis_norm.replace(' ', '')
        try:
            if hasattr(jiwer, 'process_characters'):
                char_output = jiwer.process_characters(ref_chars_str, hyp_chars_str)
                result['char_sub'] = char_output.substitutions
                result['char_del'] = char_output.deletions
                result['char_ins'] = char_output.insertions
            else:
                cer_rate = result['cer'] / 100
                total_char_errors = int(cer_rate * ref_chars)
                result['char_sub'] = total_char_errors // 3
                result['char_del'] = total_char_errors // 3
                result['char_ins'] = total_char_errors - result['char_sub'] - result['char_del']
        except Exception:
            cer_rate = result['cer'] / 100
            total_char_errors = int(cer_rate * ref_chars)
            result['char_sub'] = total_char_errors // 3
            result['char_del'] = total_char_errors // 3
            result['char_ins'] = total_char_errors - result['char_sub'] - result['char_del']
    except Exception as e:
        print(f"Error calculating detailed metrics: {e}")
    
    return result

def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate between reference and hypothesis.
    
    Args:
        reference: Ground truth transcript
        hypothesis: Predicted transcript
        
    Returns:
        CER as a percentage
    """
    reference_norm = _normalize_text(reference)
    hypothesis_norm = _normalize_text(hypothesis)

    if not reference_norm or not hypothesis_norm:
        return 100.0

    try:
        cer = jiwer.cer(reference_norm, hypothesis_norm) * 100
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
        reference_norm = _normalize_text(reference)
        hypothesis_norm = _normalize_text(hypothesis)

        if not reference_norm or not hypothesis_norm:
            return None

        if hasattr(jiwer, 'process_words'):
            output = jiwer.process_words(reference_norm, hypothesis_norm)
            return {
                'substitutions': output.substitutions,
                'deletions': output.deletions,
                'insertions': output.insertions,
                'hits': output.hits
            }
        elif hasattr(jiwer, 'compute_measures'):
            measures = jiwer.compute_measures(reference_norm, hypothesis_norm)
            return {
                'substitutions': measures['substitutions'],
                'deletions': measures['deletions'],
                'insertions': measures['insertions'],
                'hits': measures['hits']
            }
        else:
            return None
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
    
    wer, cer = calculate_wer(reference, hypothesis_text)
    
    print("METRICS:")
    print("-" * 80)
    print(f"Word Error Rate (WER): {wer:.2f}%")
    print(f"Character Error Rate (CER): {cer:.2f}%")
    print("(Metrics computed on normalized text: lowercased, punctuation removed.)")
    
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
