"""
Utility functions for the ASR and SID pipelines.
"""

import os
import sys
import argparse
import soundfile as sf
import numpy as np
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import config

def count_speakers(dataset='Dev'):
    """
    Count unique speakers in the SID dataset.
    
    Args:
        dataset: Dataset name (Dev, Train, or Eval)
        
    Returns:
        Dictionary with speaker count and list of unique speakers
    """
    label_file = config.find_sid_label_file(dataset=dataset)

    if not os.path.exists(label_file):
        print(f"Error: Label file not found: {label_file}")
        return None
    
    speakers = set()
    total_utterances = 0
    
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    speaker = parts[1]
                    speakers.add(speaker)
                    total_utterances += 1
        
        result = {
            'dataset': dataset,
            'unique_speakers': len(speakers),
            'total_utterances': total_utterances,
            'speakers': sorted(list(speakers))
        }
        
        return result
        
    except Exception as e:
        print(f"Error reading label file: {e}")
        return None

def display_speaker_stats(stats):
    """Display speaker statistics in a formatted way."""
    if not stats:
        return
    
    print("\n" + "="*60)
    print(f"Speaker Statistics for {stats['dataset']} Dataset")
    print("="*60)
    print(f"Total Utterances: {stats['total_utterances']}")
    print(f"Unique Speakers: {stats['unique_speakers']}")
    print("\nSpeaker List:")
    print("-"*60)
    
    for i, speaker in enumerate(stats['speakers'], 1):
        print(f"{i:3d}. {speaker}")
    
    print("="*60)


def check_sample_rates(folder='SID', dataset='Dev', limit=None):
    """
    Check sample rates of audio files in a dataset.
    
    Args:
        folder: Folder name (SID, ASR_track2, SD_track2, etc.)
        dataset: Dataset name (Dev, Train, or Eval)
        limit: Maximum number of files to check (None for all)
        
    Returns:
        Dictionary with sample rate statistics
    """
    audio_dir = config.get_folder_audio_path(folder, dataset)
    
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory not found: {audio_dir}")
        return None
    
    sample_rates = Counter()
    file_examples = {}
    total_files = 0
    errors = 0
    
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    if limit:
        audio_files = audio_files[:limit]
    
    print(f"\nScanning {len(audio_files)} audio files in {folder}/{dataset}...")
    print(f"Directory: {audio_dir}\n")
    
    for filename in audio_files:
        filepath = os.path.join(audio_dir, filename)
        try:
            info = sf.info(filepath)
            sr = info.samplerate
            sample_rates[sr] += 1
            total_files += 1
            
            if sr not in file_examples:
                file_examples[sr] = {
                    'file': filename,
                    'duration': info.duration,
                    'channels': info.channels
                }
        except Exception as e:
            errors += 1
            print(f"Error reading {filename}: {e}")
    
    result = {
        'folder': folder,
        'dataset': dataset,
        'total_files': total_files,
        'errors': errors,
        'sample_rates': dict(sample_rates),
        'examples': file_examples
    }
    
    return result


def display_sample_rate_stats(stats):
    """Display sample rate statistics in a formatted way."""
    if not stats:
        return
    
    print("="*60)
    print(f"Sample Rate Analysis: {stats['folder']}/{stats['dataset']}")
    print("="*60)
    print(f"Total files scanned: {stats['total_files']}")
    if stats['errors'] > 0:
        print(f"Errors: {stats['errors']}")
    
    print("\nSample Rates Found:")
    print("-"*60)
    
    for sr, count in sorted(stats['sample_rates'].items()):
        pct = (count / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
        example = stats['examples'].get(sr, {})
        print(f"  {sr:>6} Hz: {count:>6} files ({pct:5.1f}%)")
        if example:
            print(f"           Example: {example['file']}")
            print(f"           Duration: {example['duration']:.2f}s, Channels: {example['channels']}")
    
    print("-"*60)
    
    if len(stats['sample_rates']) == 1:
        sr = list(stats['sample_rates'].keys())[0]
        if sr == 16000:
            print("All files are 16kHz - optimal for ECAPA model!")
        else:
            print(f"All files are {sr}Hz - consider resampling to 16kHz for ECAPA model.")
    else:
        print("Mixed sample rates detected - consider standardizing to 16kHz.")
    
    print("="*60)


class ReportWriter:
    """Captures output and writes to both console and file."""
    
    def __init__(self, report_path=None):
        self.report_path = report_path
        self.lines = []
    
    def print(self, text=""):
        """Print to console and store for report."""
        print(text)
        self.lines.append(text)
    
    def save(self):
        """Save captured output to file."""
        if self.report_path:
            with open(self.report_path, 'w') as f:
                f.write('\n'.join(self.lines))
            print(f"\nReport saved to: {self.report_path}")


def compute_file_metrics(audio_path):
    """
    Compute audio quality metrics for a single file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with metrics or None if error:
        - duration_sec: total length in seconds
        - rms_db: overall loudness in dBFS
        - peak_amp: maximum absolute sample value
        - clip_ratio: fraction of samples near peak (clipping indicator)
    """
    try:
        audio, sr = sf.read(audio_path, dtype='float32')
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        duration_sec = len(audio) / sr
        
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            rms_db = 20 * np.log10(rms)
        else:
            rms_db = -np.inf
        
        peak_amp = np.max(np.abs(audio))
        
        if peak_amp > 0 and len(audio) > 0:
            clip_threshold = 0.99 * peak_amp
            clipped_samples = np.sum(np.abs(audio) >= clip_threshold)
            clip_ratio = clipped_samples / len(audio)
        else:
            clip_ratio = 0.0
        
        return {
            'duration_sec': duration_sec,
            'rms_db': rms_db,
            'peak_amp': peak_amp,
            'clip_ratio': clip_ratio
        }
        
    except Exception as e:
        return None


def compute_dataset_metrics(folder='SID', dataset='Dev', limit=None):
    """
    Compute audio quality metrics for all files in a dataset.
    
    Args:
        folder: Folder name (SID, ASR_track2, SD_track2, etc.)
        dataset: Dataset name (Dev, Train, or Eval)
        limit: Maximum number of files to process (None for all)
        
    Returns:
        Dictionary with per-file metrics and metadata
    """
    audio_dir = config.get_folder_audio_path(folder, dataset)
    
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory not found: {audio_dir}")
        return None
    
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    if limit:
        audio_files = audio_files[:limit]
    
    file_metrics = []
    errors = 0
    
    print(f"\nComputing audio quality metrics for {len(audio_files)} files...")
    print(f"Directory: {audio_dir}\n")
    
    for filename in tqdm(audio_files, desc="Analyzing files", unit="file"):
        filepath = os.path.join(audio_dir, filename)
        metrics = compute_file_metrics(filepath)
        
        if metrics:
            metrics['filename'] = filename
            file_metrics.append(metrics)
        else:
            errors += 1
    
    return {
        'folder': folder,
        'dataset': dataset,
        'audio_dir': audio_dir,
        'total_files': len(audio_files),
        'processed_files': len(file_metrics),
        'errors': errors,
        'file_metrics': file_metrics
    }


def aggregate_metrics(file_metrics):
    """
    Compute aggregate statistics across all files.
    
    Args:
        file_metrics: List of per-file metric dictionaries
        
    Returns:
        Dictionary with aggregate statistics for each metric
    """
    if not file_metrics:
        return None
    
    metric_names = ['duration_sec', 'rms_db', 'peak_amp', 'clip_ratio']
    aggregates = {}
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in file_metrics if m[metric_name] is not None and not np.isinf(m[metric_name])]
        
        if not values:
            aggregates[metric_name] = None
            continue
        
        values = np.array(values)
        
        aggregates[metric_name] = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'p5': float(np.percentile(values, 5)),
            'p25': float(np.percentile(values, 25)),
            'p75': float(np.percentile(values, 75)),
            'p95': float(np.percentile(values, 95)),
            'count': len(values)
        }
    
    return aggregates


def display_audio_quality_stats(dataset_result, report=None):
    """
    Display audio quality statistics in a formatted way.
    
    Args:
        dataset_result: Result from compute_dataset_metrics()
        report: Optional ReportWriter for saving to file
    """
    if not dataset_result:
        return
    
    out = report if report else ReportWriter()
    
    aggregates = aggregate_metrics(dataset_result['file_metrics'])
    
    out.print("=" * 70)
    out.print("Audio Quality Analysis Report")
    out.print("=" * 70)
    out.print(f"Dataset: {dataset_result['folder']}/{dataset_result['dataset']}")
    out.print(f"Directory: {dataset_result['audio_dir']}")
    out.print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.print(f"Total Files: {dataset_result['total_files']}")
    out.print(f"Processed: {dataset_result['processed_files']}")
    if dataset_result['errors'] > 0:
        out.print(f"Errors: {dataset_result['errors']}")
    out.print("=" * 70)
    
    if not aggregates:
        out.print("No valid metrics computed.")
        if report:
            report.save()
        return
    
    metric_labels = {
        'duration_sec': ('Duration (seconds)', '{:.3f}'),
        'rms_db': ('RMS Loudness (dBFS)', '{:.2f}'),
        'peak_amp': ('Peak Amplitude', '{:.4f}'),
        'clip_ratio': ('Clip Ratio', '{:.6f}')
    }
    
    for metric_name, (label, fmt) in metric_labels.items():
        stats = aggregates.get(metric_name)
        if not stats:
            out.print(f"\n{label}: No valid data")
            continue
        
        out.print(f"\n{label}")
        out.print("-" * 70)
        out.print(f"  Count:  {stats['count']}")
        out.print(f"  Mean:   {fmt.format(stats['mean'])}")
        out.print(f"  Median: {fmt.format(stats['median'])}")
        out.print(f"  Std:    {fmt.format(stats['std'])}")
        out.print(f"  Min:    {fmt.format(stats['min'])}")
        out.print(f"  Max:    {fmt.format(stats['max'])}")
        out.print(f"  P5:     {fmt.format(stats['p5'])}")
        out.print(f"  P25:    {fmt.format(stats['p25'])}")
        out.print(f"  P75:    {fmt.format(stats['p75'])}")
        out.print(f"  P95:    {fmt.format(stats['p95'])}")
    
    out.print("\n" + "=" * 70)
    out.print("Summary")
    out.print("-" * 70)
    
    clip_stats = aggregates.get('clip_ratio')
    if clip_stats and clip_stats['mean'] > 0.001:
        out.print(f"  WARNING: Potential clipping detected (mean clip_ratio: {clip_stats['mean']:.4f})")
    else:
        out.print("  No significant clipping detected.")
    
    rms_stats = aggregates.get('rms_db')
    if rms_stats:
        if rms_stats['mean'] < -30:
            out.print(f"  Note: Audio is relatively quiet (mean RMS: {rms_stats['mean']:.1f} dBFS)")
        elif rms_stats['mean'] > -10:
            out.print(f"  Note: Audio is relatively loud (mean RMS: {rms_stats['mean']:.1f} dBFS)")
        else:
            out.print(f"  Audio levels are within normal range (mean RMS: {rms_stats['mean']:.1f} dBFS)")
    
    out.print("=" * 70)
    
    if report:
        report.save()


def main():
    parser = argparse.ArgumentParser(
        description='Utility functions for ASR and SID pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check sample rates for SID Train dataset
  python utils.py --sample-rate --folder SID --dataset Train

  # Check sample rates for Dev dataset (default)
  python utils.py --sample-rate --folder SID

  # Check only first 100 files
  python utils.py --sample-rate --folder SID --dataset Train --limit 100

  # Count unique speakers
  python utils.py --count-speakers --dataset Dev

  # Compute audio quality metrics for Dev dataset
  python utils.py --audio-quality --folder SID --dataset Dev

  # Compute audio quality metrics and save report to file
  python utils.py --audio-quality --folder SID --dataset Dev --report /path/to/audio_quality_report.txt
        """
    )
    
    parser.add_argument(
        '--count-speakers',
        action='store_true',
        help='Count unique speakers in the dataset'
    )
    
    parser.add_argument(
        '--sample-rate',
        action='store_true',
        help='Check sample rates of audio files'
    )
    
    parser.add_argument(
        '--audio-quality',
        action='store_true',
        help='Compute audio quality metrics (duration, RMS, peak amplitude, clip ratio)'
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        default='SID',
        help='Folder name (SID, ASR_track2, SD_track2, etc.) - default: SID'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='Dev',
        choices=['Dev', 'Train', 'Eval'],
        help='Dataset to analyze (default: Dev)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to check (default: all)'
    )
    
    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Path to save report file (e.g., /path/to/audio_quality_report.txt)'
    )
    
    args = parser.parse_args()
    
    if args.count_speakers:
        stats = count_speakers(dataset=args.dataset)
        if stats:
            display_speaker_stats(stats)
    elif args.sample_rate:
        stats = check_sample_rates(
            folder=args.folder,
            dataset=args.dataset,
            limit=args.limit
        )
        if stats:
            display_sample_rate_stats(stats)
    elif args.audio_quality:
        result = compute_dataset_metrics(
            folder=args.folder,
            dataset=args.dataset,
            limit=args.limit
        )
        if result:
            report = ReportWriter(args.report) if args.report else None
            display_audio_quality_stats(result, report=report)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
