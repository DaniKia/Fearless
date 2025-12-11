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

def display_top_confusions(confusion_data, top_n=10, return_output=False):
    """
    Display most common confusion pairs.
    
    Args:
        confusion_data: Dictionary from create_confusion_matrix
        top_n: Number of top confusions to show
        return_output: If True, return lines as list instead of just printing
        
    Returns:
        List of output lines if return_output=True, else None
    """
    output_lines = []
    
    def out(text=""):
        print(text)
        output_lines.append(text)
    
    matrix = confusion_data['matrix']
    speakers = confusion_data['speakers']
    
    confusions = []
    for i in range(len(speakers)):
        for j in range(len(speakers)):
            if i != j and matrix[i, j] > 0:
                confusions.append((speakers[i], speakers[j], matrix[i, j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    out(f"\nTop {top_n} Speaker Confusions:")
    out("-"*80)
    out(f"{'Reference':20s} {'Predicted As':20s} {'Count':>10s}")
    out("-"*80)
    
    for ref, pred, count in confusions[:top_n]:
        out(f"{ref:20s} {pred:20s} {count:10d}")
    
    out("="*80)
    
    return output_lines if return_output else None


def calculate_top_k_accuracy(results, k_values=[1, 3, 5]):
    """
    Calculate Top-K accuracy metrics.
    
    Args:
        results: List of dictionaries with 'reference' and 'top_k' keys
        k_values: List of K values to compute
        
    Returns:
        Dictionary mapping K to accuracy percentage
    """
    if not results:
        return {}
    
    total = len(results)
    top_k_accuracy = {}
    
    for k in k_values:
        correct = sum(1 for r in results if r['reference'] in r['top_k'][:k])
        top_k_accuracy[k] = (correct / total) * 100
    
    return top_k_accuracy


def calculate_precision_per_speaker(results):
    """
    Calculate precision for each predicted speaker.
    Precision = how often a predicted speaker is actually correct.
    
    Args:
        results: List of dictionaries with 'reference', 'predicted', 'correct' keys
        
    Returns:
        Dictionary mapping speaker_id to {'predicted': count, 'correct': count, 'precision': float}
    """
    precision_stats = defaultdict(lambda: {'predicted': 0, 'correct': 0})
    
    for r in results:
        pred = r['predicted']
        precision_stats[pred]['predicted'] += 1
        if r['correct']:
            precision_stats[pred]['correct'] += 1
    
    for speaker_id in precision_stats:
        stats = precision_stats[speaker_id]
        stats['precision'] = (stats['correct'] / stats['predicted']) * 100 if stats['predicted'] > 0 else 0.0
    
    return dict(precision_stats)


def display_batch_summary_extended(results, show_confusion_matrix=False, return_output=False):
    """
    Display extended summary with Top-K accuracy and precision metrics.
    
    Args:
        results: List of dictionaries with 'reference', 'predicted', 'correct', 'top_k' keys
        show_confusion_matrix: Whether to display confusion matrix analysis
        return_output: If True, return lines as list instead of just printing
        
    Returns:
        List of output lines if return_output=True, else None
    """
    output_lines = []
    
    def out(text=""):
        print(text)
        output_lines.append(text)
    
    if not results:
        out("No results to display")
        return output_lines if return_output else None
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = (correct / total) * 100
    
    top_k_acc = calculate_top_k_accuracy(results)
    
    out("\n" + "="*80)
    out("BATCH PROCESSING SUMMARY")
    out("="*80)
    out(f"Total Files Processed: {total}")
    out(f"Correct Predictions  : {correct}")
    out(f"Incorrect Predictions: {total - correct}")
    out("="*80)
    
    out("\nAccuracy Metrics:")
    out("-"*80)
    out(f"Top-1 Accuracy       : {accuracy:.2f}%")
    for k in [3, 5]:
        if k in top_k_acc:
            out(f"Top-{k} Accuracy       : {top_k_acc[k]:.2f}%")
    out("="*80)
    
    per_speaker_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        ref = r['reference']
        per_speaker_stats[ref]['total'] += 1
        if r['correct']:
            per_speaker_stats[ref]['correct'] += 1
    
    out("\nPer-Speaker Recall (accuracy per reference speaker):")
    out("-"*80)
    for speaker_id in sorted(per_speaker_stats.keys()):
        stats = per_speaker_stats[speaker_id]
        speaker_acc = (stats['correct'] / stats['total']) * 100
        out(f"{speaker_id:20s}: {stats['correct']:3d}/{stats['total']:3d} ({speaker_acc:5.1f}%)")
    out("="*80)
    
    precision_stats = calculate_precision_per_speaker(results)
    
    out("\nPer-Speaker Precision (how often each predicted speaker is correct):")
    out("-"*80)
    out(f"{'Speaker':20s} {'Predicted':>10s} {'Correct':>10s} {'Precision':>12s}")
    out("-"*80)
    for speaker_id in sorted(precision_stats.keys()):
        stats = precision_stats[speaker_id]
        out(f"{speaker_id:20s} {stats['predicted']:10d} {stats['correct']:10d} {stats['precision']:11.1f}%")
    out("="*80)
    
    sorted_by_predictions = sorted(precision_stats.items(), key=lambda x: x[1]['predicted'], reverse=True)
    low_precision_speakers = [(spk, stats) for spk, stats in sorted_by_predictions 
                               if stats['precision'] < 50.0 and stats['predicted'] >= 10]
    
    if low_precision_speakers:
        out("\n'Sink' Speakers (predicted 10+ times with <50% precision):")
        out("-"*80)
        out(f"{'Speaker':20s} {'Predicted':>10s} {'Correct':>10s} {'Precision':>12s}")
        out("-"*80)
        for speaker_id, stats in low_precision_speakers[:15]:
            out(f"{speaker_id:20s} {stats['predicted']:10d} {stats['correct']:10d} {stats['precision']:11.1f}%")
        out("="*80)
    
    if show_confusion_matrix:
        predictions = [r['predicted'] for r in results]
        references = [r['reference'] for r in results]
        
        confusion_data = create_confusion_matrix(predictions, references)
        confusion_lines = display_top_confusions(confusion_data, top_n=20, return_output=return_output)
        if confusion_lines:
            output_lines.extend(confusion_lines)
    
    return output_lines if return_output else None
