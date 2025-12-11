"""
Main script for ASR pipeline.
Transcribe audio segments using Whisper and compare with reference transcripts.
"""

import sys
import os
import argparse
from pathlib import Path

from modules.drive_connector import setup_drive_access, is_colab
from modules.data_loader import get_audio_files_with_transcripts
from modules.whisper_transcriber import transcribe_audio
from modules.evaluator import display_comparison
import config

def run_single_file(audio_path, transcript_dir, dataset='Dev', show_timestamps=True, model_name=None, folder='ASR_track2'):
    """
    Process a single audio file.
    
    Args:
        audio_path: Path to audio file
        transcript_dir: Directory containing reference transcripts
        dataset: Dataset name (Dev, Train, or Eval)
        show_timestamps: Whether to show timestamps in output
        model_name: Whisper model identifier to use for transcription
        folder: Folder name (for display purposes)
    """
    audio_filename = os.path.basename(audio_path)
    
    from modules.data_loader import load_reference_transcript
    reference = load_reference_transcript(transcript_dir, audio_filename, dataset=dataset)
    
    if not reference:
        print(f"Error: No reference transcript found for {audio_filename}")
        return
    
    hypothesis = transcribe_audio(audio_path, model_name=model_name or config.WHISPER_MODEL)
    
    display_comparison(audio_filename, reference, hypothesis, show_timestamps=show_timestamps)

def run_batch(audio_dir, transcript_dir, limit=5, dataset='Dev', show_timestamps=True, model_name=None, folder='ASR_track2'):
    """
    Process multiple audio files in batch.
    
    Args:
        audio_dir: Directory containing audio files
        transcript_dir: Directory containing reference transcripts
        limit: Maximum number of files to process
        dataset: Dataset name (Dev, Train, or Eval)
        show_timestamps: Whether to show timestamps in output
        model_name: Whisper model identifier to use for transcription
        folder: Folder name (for display purposes)
    """
    pairs = get_audio_files_with_transcripts(audio_dir, transcript_dir, limit=limit, dataset=dataset)
    
    if not pairs:
        print("No audio files with transcripts found")
        return
    
    print(f"\nFound {len(pairs)} audio files with transcripts")
    print(f"Processing {min(limit, len(pairs))} files...\n")
    
    results = []
    
    for i, (audio_path, reference) in enumerate(pairs, 1):
        audio_filename = os.path.basename(audio_path)
        print(f"\n[{i}/{len(pairs)}] Processing: {audio_filename}")
        
        hypothesis = transcribe_audio(audio_path, model_name=model_name or config.WHISPER_MODEL)
        
        metrics = display_comparison(
            audio_filename, 
            reference, 
            hypothesis, 
            show_timestamps=show_timestamps
        )
        
        results.append({
            'filename': audio_filename,
            'wer': metrics['wer'],
            'cer': metrics['cer']
        })
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    avg_wer = sum(r['wer'] for r in results) / len(results) if results else 0
    avg_cer = sum(r['cer'] for r in results) / len(results) if results else 0
    
    print(f"Files processed: {len(results)}")
    print(f"Average WER: {avg_wer:.2f}%")
    print(f"Average CER: {avg_cer:.2f}%")
    print("="*80)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='ASR Pipeline with Whisper')
    parser.add_argument('--folder', type=str, default='ASR_track2',
                       help='Folder to use: ASR_track2, SID, SD_track2, SD_track1, SAD, etc. (default: ASR_track2)')
    parser.add_argument('--dataset', type=str, default='Dev', 
                       help='Dataset to use: Dev, Train, or Eval')
    parser.add_argument('--file', type=str, default=None,
                       help='Specific audio file to process')
    parser.add_argument('--batch', type=int, default=5,
                       help='Number of files to process in batch mode')
    parser.add_argument('--no-timestamps', action='store_true',
                       help='Disable timestamp display')
    parser.add_argument('--whisper-model', type=str, default=None,
                       help='Override the configured Whisper model (e.g., tiny.en, base, small.en)')

    args = parser.parse_args()

    model_name = args.whisper_model or config.WHISPER_MODEL

    print("="*80)
    print("ASR Pipeline with Whisper")
    print("="*80)
    print(f"Phase: {config.PHASE}")
    print(f"Folder: {args.folder}")
    print(f"Dataset: {args.dataset}")
    print(f"Whisper Model: {model_name}")
    print("="*80)
    
    if not is_colab():
        print("\nNOTE: For best results, run this pipeline in Google Colab where your")
        print("      Phase 3 dataset is stored in Google Drive.")
        print("\nTo use in Google Colab:")
        print("  1. Clone this repository to Colab")
        print("  2. Mount your Google Drive")
        print("  3. Run: python main.py --dataset Dev --batch 5")
        print("\nIn Replit, this code is primarily for development and review.")
        print("="*80)
    
    if not setup_drive_access():
        print("\nError: Could not set up Google Drive access")
        if not is_colab():
            print("For Replit: This tool is designed to run in Google Colab")
            print("For Colab: Make sure Google Drive is mounted")
        sys.exit(1)
    
    audio_dir = config.get_folder_audio_path(args.folder, args.dataset)
    transcript_dir = config.get_folder_label_path(args.folder, args.dataset)
    
    if not audio_dir or not transcript_dir:
        print("\nError: Could not determine data paths")
        print("This pipeline is designed to run in Google Colab with mounted Drive")
        sys.exit(1)
    
    print(f"\nAudio directory: {audio_dir}")
    print(f"Transcript directory: {transcript_dir}")
    
    show_timestamps = not args.no_timestamps
    
    if args.file:
        audio_path = os.path.join(audio_dir, args.file)
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)
        run_single_file(
            audio_path,
            transcript_dir,
            dataset=args.dataset,
            show_timestamps=show_timestamps,
            model_name=model_name,
            folder=args.folder
        )
    else:
        run_batch(
            audio_dir,
            transcript_dir,
            limit=args.batch,
            dataset=args.dataset,
            show_timestamps=show_timestamps,
            model_name=model_name,
            folder=args.folder
        )

if __name__ == "__main__":
    main()
