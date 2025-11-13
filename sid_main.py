"""
Main entry point for Speaker Identification (SID) pipeline.
Supports speaker enrollment and identification.
"""

import os
import sys
import argparse
import config
from modules.data_loader import group_audio_by_speaker, get_sid_files_with_labels, load_sid_label
from modules.speaker_identifier import SpeakerIdentifier
from modules.sid_evaluator import display_comparison, display_batch_summary

def enroll_speakers(dataset='Train', batch_size=16):
    """
    Enroll speakers from the training dataset.
    
    Args:
        dataset: Dataset to use for enrollment (default: Train)
        batch_size: Number of files to process per GPU batch (default: 16)
    """
    print(f"\n{'='*60}")
    print(f"Speaker Enrollment - {dataset} Dataset")
    print(f"{'='*60}\n")
    
    audio_dir = config.get_sid_audio_path(dataset)
    label_dir = config.get_sid_label_path(dataset)
    database_path = config.get_speaker_database_path()
    
    print(f"Audio directory: {audio_dir}")
    print(f"Label directory: {label_dir}")
    print(f"Database will be saved to: {database_path}\n")
    
    speaker_files = group_audio_by_speaker(audio_dir, label_dir, dataset=dataset)
    
    if not speaker_files:
        print("Error: No speaker files found for enrollment")
        return
    
    print(f"Found {len(speaker_files)} unique speakers")
    total_files = sum(len(files) for files in speaker_files.values())
    print(f"Total audio files: {total_files}\n")
    
    identifier = SpeakerIdentifier(batch_size=batch_size)
    identifier.enroll_speakers(speaker_files, save_path=database_path)
    
    print(f"\nEnrollment complete! Database saved to: {database_path}")

def identify_single_file(audio_path, label_dir, dataset='Dev'):
    """
    Identify speaker from a single audio file.
    
    Args:
        audio_path: Path to audio file
        label_dir: Directory containing labels
        dataset: Dataset name
    """
    audio_filename = os.path.basename(audio_path)
    
    reference_speaker = load_sid_label(label_dir, audio_filename, dataset=dataset)
    
    if not reference_speaker:
        print(f"Error: No reference label found for {audio_filename}")
        return
    
    database_path = config.get_speaker_database_path()
    if not os.path.exists(database_path):
        print(f"Error: Speaker database not found at {database_path}")
        print("Please run enrollment first: python sid_main.py --enroll")
        return
    
    identifier = SpeakerIdentifier()
    if not identifier.load_database(database_path):
        return
    
    result = identifier.identify_speaker(audio_path, top_k=1)
    
    if result:
        predicted_speaker, similarity = result
        display_comparison(audio_filename, reference_speaker, predicted_speaker, similarity)

def identify_batch(audio_dir, label_dir, limit=5, dataset='Dev'):
    """
    Identify speakers for multiple audio files.
    
    Args:
        audio_dir: Directory with audio files
        label_dir: Directory with labels
        limit: Maximum number of files to process
        dataset: Dataset name
    """
    database_path = config.get_speaker_database_path()
    if not os.path.exists(database_path):
        print(f"Error: Speaker database not found at {database_path}")
        print("Please run enrollment first: python sid_main.py --enroll")
        return
    
    pairs = get_sid_files_with_labels(audio_dir, label_dir, limit=limit, dataset=dataset)
    
    if not pairs:
        print("No audio files with labels found")
        return
    
    print(f"\nProcessing {len(pairs)} files...\n")
    
    identifier = SpeakerIdentifier()
    if not identifier.load_database(database_path):
        return
    
    results = []
    
    for audio_path, reference_speaker in pairs:
        audio_filename = os.path.basename(audio_path)
        
        result = identifier.identify_speaker(audio_path, top_k=1)
        
        if result:
            predicted_speaker, similarity = result
            is_correct = (predicted_speaker == reference_speaker)
            
            display_comparison(audio_filename, reference_speaker, predicted_speaker, similarity)
            
            results.append({
                'filename': audio_filename,
                'reference': reference_speaker,
                'predicted': predicted_speaker,
                'similarity': similarity,
                'correct': is_correct
            })
    
    if results:
        display_batch_summary(results)

def main():
    parser = argparse.ArgumentParser(
        description='Speaker Identification (SID) Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll speakers from Train dataset (one-time setup)
  python sid_main.py --enroll --dataset Train
  
  # Enroll with custom GPU batch size (tune based on GPU memory)
  python sid_main.py --enroll --dataset Train --batch-size 32
  
  # Identify speaker from single file
  python sid_main.py --file fsc_p3_SID_dev_0010.wav
  
  # Test identification on 5 files from Dev set
  python sid_main.py --batch 5
  
  # Test identification on 10 files from Train set
  python sid_main.py --dataset Train --batch 10
        """
    )
    
    parser.add_argument(
        '--enroll',
        action='store_true',
        help='Enroll speakers from the dataset (required before identification)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Process a single audio file'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=5,
        help='Number of files to identify for testing (default: 5). Used without --enroll flag.'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='Dev',
        choices=['Dev', 'Train', 'Eval'],
        help='Dataset to use (default: Dev)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='GPU batch size for enrollment: how many audio files to process simultaneously (default: 16). Only used with --enroll flag. Increase for more GPU memory, decrease if you get OOM errors.'
    )
    
    args = parser.parse_args()
    
    if args.enroll:
        enroll_speakers(dataset=args.dataset, batch_size=args.batch_size)
    elif args.file:
        audio_dir = config.get_sid_audio_path(args.dataset)
        label_dir = config.get_sid_label_path(args.dataset)
        audio_path = os.path.join(audio_dir, args.file)
        
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)
        
        identify_single_file(audio_path, label_dir, dataset=args.dataset)
    else:
        audio_dir = config.get_sid_audio_path(args.dataset)
        label_dir = config.get_sid_label_path(args.dataset)
        
        print(f"\nAudio directory: {audio_dir}")
        print(f"Label directory: {label_dir}")
        
        identify_batch(audio_dir, label_dir, limit=args.batch, dataset=args.dataset)

if __name__ == "__main__":
    main()
