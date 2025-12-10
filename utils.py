"""
Utility functions for the ASR and SID pipelines.
"""

import os
import sys
import argparse
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

def main():
    parser = argparse.ArgumentParser(
        description='Utility functions for ASR and SID pipelines'
    )
    
    parser.add_argument(
        '--count-speakers',
        action='store_true',
        help='Count unique speakers in the dataset'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='Dev',
        choices=['Dev', 'Train', 'Eval'],
        help='Dataset to analyze (default: Dev)'
    )
    
    args = parser.parse_args()
    
    if args.count_speakers:
        stats = count_speakers(dataset=args.dataset)
        if stats:
            display_speaker_stats(stats)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
