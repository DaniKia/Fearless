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
from modules.sid_evaluator import display_comparison, display_batch_summary, create_confusion_matrix, display_top_confusions

def enroll_speakers(folder='SID', dataset='Train', batch_size=16):
    """
    Enroll speakers from the training dataset.
    
    Args:
        folder: Folder name (SID, ASR_track2, SD_track2, etc.)
        dataset: Dataset to use for enrollment (default: Train)
        batch_size: Number of files to process per GPU batch (default: 16)
    """
    print(f"\n{'='*60}")
    print(f"Speaker Enrollment - {folder}/{dataset} Dataset")
    print(f"{'='*60}\n")
    
    audio_dir = config.get_folder_audio_path(folder, dataset)
    label_dir = config.get_folder_label_path(folder, dataset)
    database_path = config.get_speaker_database_path()
    
    print(f"Folder: {folder}")
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

def identify_single_file(audio_path, label_dir, dataset='Dev', folder='SID'):
    """
    Identify speaker from a single audio file.
    
    Args:
        audio_path: Path to audio file
        label_dir: Directory containing labels
        dataset: Dataset name
        folder: Folder name (for display purposes)
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

def identify_batch(audio_dir, label_dir, limit=None, dataset='Dev', show_confusion_matrix=False, folder='SID', verbose=False):
    """
    Identify speakers for multiple audio files.
    
    Args:
        audio_dir: Directory with audio files
        label_dir: Directory with labels
        limit: Maximum number of files to process
        dataset: Dataset name
        show_confusion_matrix: Whether to display confusion matrix analysis
        folder: Folder name (for display purposes)
        verbose: Whether to show per-file status
    """
    from modules.sid_evaluator import display_batch_summary_extended
    
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
        
        top_k_results = identifier.identify_speaker(audio_path, top_k=5)
        
        if top_k_results:
            predicted_speaker, similarity = top_k_results[0]
            is_correct = (predicted_speaker == reference_speaker)
            
            top_k_speakers = [spk for spk, _ in top_k_results]
            
            if verbose:
                display_comparison(audio_filename, reference_speaker, predicted_speaker, similarity)
            
            results.append({
                'filename': audio_filename,
                'reference': reference_speaker,
                'predicted': predicted_speaker,
                'similarity': similarity,
                'correct': is_correct,
                'top_k': top_k_speakers
            })
    
    if results:
        display_batch_summary_extended(results, show_confusion_matrix=show_confusion_matrix)

def main():
    parser = argparse.ArgumentParser(
        description='Speaker Identification (SID) Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll speakers from SID folder Train dataset (one-time setup)
  python sid_main.py --enroll --folder SID --dataset Train
  
  # Enroll from ASR_track2
  python sid_main.py --enroll --folder ASR_track2 --dataset Train
  
  # Enroll with custom GPU batch size (tune based on GPU memory)
  python sid_main.py --enroll --folder SID --dataset Train --batch-size 32
  
  # Identify speaker from single file (default folder: SID)
  python sid_main.py --file fsc_p3_SID_dev_0010.wav
  
  # Process all files in Dev set from SID folder
  python sid_main.py --folder SID --dataset Dev
  
  # Process all files from ASR_track2 with confusion matrix
  python sid_main.py --folder ASR_track2 --dataset Dev --confusion-matrix
  
  # Process custom folders (SD, SAD, etc.)
  python sid_main.py --folder SD_track2 --dataset Train
  python sid_main.py --folder SAD --dataset Dev --confusion-matrix
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
        default=None,
        help='Number of files to identify for testing. If not specified, processes all files in dataset. Used without --enroll flag.'
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        default='SID',
        help='Folder to use: SID, ASR_track2, SD_track2, SD_track1, SAD, etc. (default: SID)'
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
    
    parser.add_argument(
        '--confusion-matrix',
        action='store_true',
        help='Display confusion matrix showing which speakers are most commonly misidentified. Only used in batch mode.'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show per-file status during batch processing. Default is summary only.'
    )
    
    args = parser.parse_args()
    
    if args.enroll:
        enroll_speakers(folder=args.folder, dataset=args.dataset, batch_size=args.batch_size)
    elif args.file:
        audio_dir = config.get_folder_audio_path(args.folder, args.dataset)
        label_dir = config.get_folder_label_path(args.folder, args.dataset)
        audio_path = os.path.join(audio_dir, args.file)
        
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)
        
        identify_single_file(audio_path, label_dir, dataset=args.dataset, folder=args.folder)
    else:
        audio_dir = config.get_folder_audio_path(args.folder, args.dataset)
        label_dir = config.get_folder_label_path(args.folder, args.dataset)
        
        print(f"\nFolder: {args.folder}")
        print(f"Audio directory: {audio_dir}")
        print(f"Label directory: {label_dir}")
        
        identify_batch(audio_dir, label_dir, limit=args.batch, dataset=args.dataset, show_confusion_matrix=args.confusion_matrix, folder=args.folder, verbose=args.verbose)

if __name__ == "__main__":
    main()
