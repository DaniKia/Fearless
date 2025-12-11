"""
Utility functions for the ASR and SID pipelines.
"""

import os
import sys
import argparse
import soundfile as sf
from collections import Counter
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
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
