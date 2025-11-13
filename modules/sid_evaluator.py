"""
Evaluator for Speaker Identification (SID) performance.
"""

import numpy as np
from collections import defaultdict

def calculate_accuracy(predictions, references):
    """
    Calculate speaker identification accuracy.
    
    Args:
        predictions: List of predicted speaker IDs
        references: List of reference speaker IDs
        
    Returns:
        Dictionary with accuracy metrics
    """
    if len(predictions) != len(references):
        print("Warning: Number of predictions and references don't match")
        return None
    
    total = len(predictions)
    correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    
    return {
        'total': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': accuracy
    }

def display_comparison(audio_filename, reference_speaker, predicted_speaker, similarity_score=None):
    """
    Display side-by-side comparison of reference and predicted speakers.
    
    Args:
        audio_filename: Name of the audio file
        reference_speaker: Reference speaker ID
        predicted_speaker: Predicted speaker ID
        similarity_score: Confidence score (optional)
    """
    is_correct = (reference_speaker == predicted_speaker)
    status_symbol = "✓" if is_correct else "✗"
    status_text = "CORRECT" if is_correct else "INCORRECT"
    
    print("\n" + "="*80)
    print(f"File: {audio_filename}")
    print("="*80)
    print(f"Reference Speaker : {reference_speaker}")
    print(f"Predicted Speaker : {predicted_speaker} {status_symbol}")
    
    if similarity_score is not None:
        print(f"Similarity Score  : {similarity_score:.4f}")
    
    print(f"\nStatus: {status_text}")
    print("="*80)

def display_batch_summary(results):
    """
    Display summary statistics for batch processing.
    
    Args:
        results: List of dictionaries with 'reference', 'predicted', 'correct' keys
    """
    if not results:
        print("No results to display")
        return
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = (correct / total) * 100
    
    per_speaker_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for r in results:
        ref = r['reference']
        per_speaker_stats[ref]['total'] += 1
        if r['correct']:
            per_speaker_stats[ref]['correct'] += 1
    
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Total Files Processed: {total}")
    print(f"Correct Predictions  : {correct}")
    print(f"Incorrect Predictions: {total - correct}")
    print(f"Overall Accuracy     : {accuracy:.2f}%")
    print("="*80)
    
    print("\nPer-Speaker Accuracy:")
    print("-"*80)
    for speaker_id in sorted(per_speaker_stats.keys()):
        stats = per_speaker_stats[speaker_id]
        speaker_acc = (stats['correct'] / stats['total']) * 100
        print(f"{speaker_id:20s}: {stats['correct']:3d}/{stats['total']:3d} ({speaker_acc:5.1f}%)")
    print("="*80)

def create_confusion_matrix(predictions, references):
    """
    Create confusion matrix for speaker identification.
    
    Args:
        predictions: List of predicted speaker IDs
        references: List of reference speaker IDs
        
    Returns:
        Dictionary with confusion matrix data
    """
    unique_speakers = sorted(set(references + predictions))
    speaker_to_idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
    
    n_speakers = len(unique_speakers)
    confusion_matrix = np.zeros((n_speakers, n_speakers), dtype=int)
    
    for pred, ref in zip(predictions, references):
        pred_idx = speaker_to_idx[pred]
        ref_idx = speaker_to_idx[ref]
        confusion_matrix[ref_idx, pred_idx] += 1
    
    return {
        'matrix': confusion_matrix,
        'speakers': unique_speakers,
        'speaker_to_idx': speaker_to_idx
    }

def display_top_confusions(confusion_data, top_n=10):
    """
    Display most common confusion pairs.
    
    Args:
        confusion_data: Dictionary from create_confusion_matrix
        top_n: Number of top confusions to show
    """
    matrix = confusion_data['matrix']
    speakers = confusion_data['speakers']
    
    confusions = []
    for i in range(len(speakers)):
        for j in range(len(speakers)):
            if i != j and matrix[i, j] > 0:
                confusions.append((speakers[i], speakers[j], matrix[i, j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nTop {top_n} Speaker Confusions:")
    print("-"*80)
    print(f"{'Reference':20s} {'Predicted As':20s} {'Count':>10s}")
    print("-"*80)
    
    for ref, pred, count in confusions[:top_n]:
        print(f"{ref:20s} {pred:20s} {count:10d}")
    
    print("="*80)
