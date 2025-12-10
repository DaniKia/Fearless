"""
Main entry point for Speaker Identification (SID) pipeline.
Supports speaker enrollment and identification.
"""

import os
import sys
import argparse
import config
from datetime import datetime
from tqdm import tqdm
from modules.data_loader import group_audio_by_speaker, get_sid_files_with_labels, load_sid_label
from modules.speaker_identifier import SpeakerIdentifier
from modules.sid_evaluator import display_comparison, display_batch_summary, create_confusion_matrix, display_top_confusions
from modules.audio_preprocessor import PreprocessConfig


def get_preprocess_config(enable_preprocessing):
    """
    Get preprocessing configuration based on flag.
    
    Args:
        enable_preprocessing: Whether to enable preprocessing
        
    Returns:
        PreprocessConfig or None
    """
    if enable_preprocessing:
        return PreprocessConfig.default()
    return None


def format_preprocess_settings(preprocess_config):
    """
    Format preprocessing settings for report output.
    
    Args:
        preprocess_config: PreprocessConfig or None
        
    Returns:
        List of formatted setting strings
    """
    if preprocess_config is None:
        return ["Preprocessing: Disabled (raw audio)"]
    
    lines = ["Preprocessing Settings:"]
    
    if preprocess_config.enable_mono:
        lines.append("  Mono Conversion:     Enabled")
    else:
        lines.append("  Mono Conversion:     Disabled")
    
    if preprocess_config.enable_resample:
        lines.append(f"  Resampling:          Enabled (target_sr={preprocess_config.target_sr})")
    else:
        lines.append("  Resampling:          Disabled")
    
    if preprocess_config.enable_dc_removal:
        lines.append("  DC Removal:          Enabled")
    else:
        lines.append("  DC Removal:          Disabled")
    
    if preprocess_config.enable_bandpass:
        lines.append(f"  Bandpass Filter:     Enabled ({preprocess_config.highpass_cutoff}-{preprocess_config.lowpass_cutoff} Hz)")
    else:
        lines.append("  Bandpass Filter:     Disabled")
    
    if preprocess_config.enable_rms_normalization:
        lines.append(f"  RMS Normalization:   Enabled (target={preprocess_config.target_rms_db} dB)")
    else:
        lines.append("  RMS Normalization:   Disabled")
    
    if preprocess_config.enable_trim:
        lines.append(f"  Silence Trimming:    Enabled (threshold={preprocess_config.trim_db} dB)")
    else:
        lines.append("  Silence Trimming:    Disabled")
    
    return lines


def generate_report_filename(embedding_path):
    """
    Generate report filename from embedding pkl path.
    
    Args:
        embedding_path: Path to the embedding pkl file
        
    Returns:
        Report filename (e.g., 'model_v1_sid_report.txt')
    """
    base_name = os.path.basename(embedding_path)
    name_without_ext = os.path.splitext(base_name)[0]
    return f"{name_without_ext}_sid_report.txt"


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

def enroll_speakers(folder='SID', dataset='Train', batch_size=16, preprocess_config=None):
    """
    Enroll speakers from the training dataset.
    
    Args:
        folder: Folder name (SID, ASR_track2, SD_track2, etc.)
        dataset: Dataset to use for enrollment (default: Train)
        batch_size: Number of files to process per GPU batch (default: 16)
        preprocess_config: Optional PreprocessConfig for audio preprocessing
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
    print(f"Database will be saved to: {database_path}")
    print(f"Preprocessing: {'Enabled' if preprocess_config else 'Disabled'}\n")
    
    speaker_files = group_audio_by_speaker(audio_dir, label_dir, dataset=dataset)
    
    if not speaker_files:
        print("Error: No speaker files found for enrollment")
        return
    
    print(f"Found {len(speaker_files)} unique speakers")
    total_files = sum(len(files) for files in speaker_files.values())
    print(f"Total audio files: {total_files}\n")
    
    identifier = SpeakerIdentifier(batch_size=batch_size)
    identifier.enroll_speakers(speaker_files, save_path=database_path, preprocess_config=preprocess_config)
    
    print(f"\nEnrollment complete! Database saved to: {database_path}")

def identify_single_file(audio_path, label_dir, dataset='Dev', folder='SID', preprocess_config=None):
    """
    Identify speaker from a single audio file.
    
    Args:
        audio_path: Path to audio file
        label_dir: Directory containing labels
        dataset: Dataset name
        folder: Folder name (for display purposes)
        preprocess_config: Optional PreprocessConfig for audio preprocessing
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
    
    result = identifier.identify_speaker(audio_path, top_k=1, preprocess_config=preprocess_config)
    
    if result:
        predicted_speaker, similarity = result
        display_comparison(audio_filename, reference_speaker, predicted_speaker, similarity)

def identify_batch(audio_dir, label_dir, limit=None, dataset='Dev', show_confusion_matrix=False, 
                   folder='SID', verbose=False, preprocess_config=None, embedding_path=None, 
                   report_path=None):
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
        preprocess_config: Optional PreprocessConfig for audio preprocessing
        embedding_path: Custom path to embedding pkl file
        report_path: Path to save report file
    """
    from modules.sid_evaluator import display_batch_summary_extended
    
    database_path = embedding_path or config.get_speaker_database_path()
    if not os.path.exists(database_path):
        print(f"Error: Speaker database not found at {database_path}")
        print("Please run enrollment first: python sid_main.py --enroll")
        return
    
    pairs = get_sid_files_with_labels(audio_dir, label_dir, limit=limit, dataset=dataset)
    
    if not pairs:
        print("No audio files with labels found")
        return
    
    report = ReportWriter(report_path)
    
    report.print("=" * 60)
    report.print("Speaker Identification Report")
    report.print("=" * 60)
    report.print(f"Embedding File: {os.path.basename(database_path)}")
    report.print(f"Dataset: {folder}/{dataset}")
    report.print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.print("")
    for line in format_preprocess_settings(preprocess_config):
        report.print(line)
    report.print("=" * 60)
    report.print("")
    report.print(f"Processing {len(pairs)} files...")
    
    identifier = SpeakerIdentifier()
    if not identifier.load_database(database_path):
        return
    
    results = []
    correct_count = 0
    
    for audio_path, reference_speaker in tqdm(pairs, desc="Identifying speakers", unit="file"):
        audio_filename = os.path.basename(audio_path)
        
        top_k_results = identifier.identify_speaker(audio_path, top_k=5, preprocess_config=preprocess_config)
        
        if top_k_results:
            predicted_speaker, similarity = top_k_results[0]
            is_correct = (predicted_speaker == reference_speaker)
            if is_correct:
                correct_count += 1
            
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
        summary_output = display_batch_summary_extended(results, show_confusion_matrix=show_confusion_matrix, return_output=True)
        if summary_output:
            for line in summary_output:
                report.lines.append(line)
        report.save()

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
  
  # Use a specific embedding file (creates model_v1_sid_report.txt)
  python sid_main.py --folder SID --dataset Dev --embedding model_v1.pkl
  
  # Compare multiple embedding files (creates separate reports for each)
  python sid_main.py --folder SID --dataset Dev --embedding model_v1.pkl model_v2.pkl
  
  # Custom report filename
  python sid_main.py --folder SID --dataset Dev --embedding model_v1.pkl --report my_results.txt
  
  # With preprocessing enabled
  python sid_main.py --folder SID --dataset Dev --embedding model_v1.pkl --preprocess
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
    
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Enable audio preprocessing (mono conversion, resampling, DC removal, RMS normalization, trimming)'
    )
    
    parser.add_argument(
        '--embedding',
        type=str,
        nargs='+',
        help='Path to embedding pkl file(s). Can specify one or more files. If multiple files provided, runs identification for each and creates separate reports.'
    )
    
    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Custom report filename (only used with single embedding file). By default, report is named after the embedding file.'
    )
    
    args = parser.parse_args()
    
    preprocess_config = get_preprocess_config(args.preprocess)
    
    if args.enroll:
        enroll_speakers(folder=args.folder, dataset=args.dataset, batch_size=args.batch_size, preprocess_config=preprocess_config)
    elif args.file:
        audio_dir = config.get_folder_audio_path(args.folder, args.dataset)
        label_dir = config.get_folder_label_path(args.folder, args.dataset)
        audio_path = os.path.join(audio_dir, args.file)
        
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)
        
        identify_single_file(audio_path, label_dir, dataset=args.dataset, folder=args.folder, preprocess_config=preprocess_config)
    else:
        audio_dir = config.get_folder_audio_path(args.folder, args.dataset)
        label_dir = config.get_folder_label_path(args.folder, args.dataset)
        
        print(f"\nFolder: {args.folder}")
        print(f"Audio directory: {audio_dir}")
        print(f"Label directory: {label_dir}")
        
        embedding_files = args.embedding if args.embedding else [None]
        
        for i, embedding_path in enumerate(embedding_files):
            if embedding_path and not os.path.exists(embedding_path):
                print(f"Error: Embedding file not found: {embedding_path}")
                continue
            
            if len(embedding_files) > 1:
                print(f"\n{'#'*60}")
                print(f"# Processing embedding {i+1}/{len(embedding_files)}: {os.path.basename(embedding_path) if embedding_path else 'default'}")
                print(f"{'#'*60}")
            
            if args.report and len(embedding_files) == 1:
                report_path = args.report
            elif embedding_path:
                report_path = generate_report_filename(embedding_path)
            else:
                report_path = None
            
            identify_batch(
                audio_dir, label_dir, 
                limit=args.batch, 
                dataset=args.dataset, 
                show_confusion_matrix=args.confusion_matrix, 
                folder=args.folder, 
                verbose=args.verbose, 
                preprocess_config=preprocess_config,
                embedding_path=embedding_path,
                report_path=report_path
            )

if __name__ == "__main__":
    main()
