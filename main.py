"""
Main script for ASR pipeline.
Transcribe audio segments using Whisper and compare with reference transcripts.
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

from tqdm import tqdm
from modules.drive_connector import setup_drive_access, is_colab
from modules.data_loader import get_audio_files_with_transcripts
from modules.whisper_transcriber import transcribe_audio
from modules.evaluator import display_comparison, calculate_detailed_metrics
from modules.audio_preprocessor import PreprocessConfig
import config


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


def create_preprocess_config(args):
    """
    Create PreprocessConfig from CLI arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        PreprocessConfig or None if preprocessing disabled
    """
    if not args.preprocess:
        return None
    
    return PreprocessConfig(
        enable_mono=args.mono,
        enable_resample=args.resample,
        target_sr=args.target_sr,
        enable_dc_removal=args.dc_removal,
        enable_bandpass=args.bandpass,
        highpass_cutoff=args.highpass,
        lowpass_cutoff=args.lowpass,
        enable_rms_normalization=args.rms_norm,
        target_rms_db=args.rms_db,
        enable_trim=args.trim,
        trim_db=args.trim_db,
        min_duration=0.1
    )

def run_single_file(audio_path, transcript_dir, dataset='Dev', show_timestamps=True, model_name=None, folder='ASR_track2', preprocess_config=None):
    """
    Process a single audio file.
    
    Args:
        audio_path: Path to audio file
        transcript_dir: Directory containing reference transcripts
        dataset: Dataset name (Dev, Train, or Eval)
        show_timestamps: Whether to show timestamps in output
        model_name: Whisper model identifier to use for transcription
        folder: Folder name (for display purposes)
        preprocess_config: Optional PreprocessConfig for audio preprocessing
    """
    audio_filename = os.path.basename(audio_path)
    
    from modules.data_loader import load_reference_transcript
    reference = load_reference_transcript(transcript_dir, audio_filename, dataset=dataset)
    
    if not reference:
        print(f"Error: No reference transcript found for {audio_filename}")
        return
    
    hypothesis = transcribe_audio(audio_path, model_name=model_name or config.WHISPER_MODEL, preprocess_config=preprocess_config)
    
    display_comparison(audio_filename, reference, hypothesis, show_timestamps=show_timestamps)

def format_preprocess_config(preprocess_config):
    """Format preprocessing config for report display."""
    if not preprocess_config:
        return ["Preprocessing: DISABLED (baseline)"]
    
    lines = ["Preprocessing: ENABLED"]
    steps = []
    if preprocess_config.enable_mono:
        steps.append("mono")
    if preprocess_config.enable_resample:
        steps.append(f"resample({preprocess_config.target_sr}Hz)")
    if preprocess_config.enable_dc_removal:
        steps.append("dc-removal")
    if preprocess_config.enable_bandpass:
        steps.append(f"bandpass({preprocess_config.highpass_cutoff}-{preprocess_config.lowpass_cutoff}Hz)")
    if preprocess_config.enable_rms_normalization:
        steps.append(f"rms-norm({preprocess_config.target_rms_db}dB)")
    if preprocess_config.enable_trim:
        steps.append(f"trim({preprocess_config.trim_db}dB)")
    lines.append(f"  Steps: {', '.join(steps) if steps else 'none'}")
    return lines


def run_batch(audio_dir, transcript_dir, limit=None, dataset='Dev', model_name=None, 
              folder='ASR_track2', preprocess_config=None, report_path=None, verbose=False):
    """
    Process multiple audio files in batch.
    
    Args:
        audio_dir: Directory containing audio files
        transcript_dir: Directory containing reference transcripts
        limit: Maximum number of files to process (None = all files)
        dataset: Dataset name (Dev, Train, or Eval)
        model_name: Whisper model identifier to use for transcription
        folder: Folder name (for display purposes)
        preprocess_config: Optional PreprocessConfig for audio preprocessing
        report_path: Optional path to save report file
        verbose: Whether to show per-utterance details
    """
    pairs = get_audio_files_with_transcripts(audio_dir, transcript_dir, limit=limit, dataset=dataset)
    
    if not pairs:
        print("No audio files with transcripts found")
        return
    
    report = ReportWriter(report_path)
    
    report.print("=" * 60)
    report.print("ASR Transcription Report")
    report.print("=" * 60)
    report.print(f"Dataset: {folder}/{dataset}")
    report.print(f"Whisper Model: {model_name or config.WHISPER_MODEL}")
    report.print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.print("")
    for line in format_preprocess_config(preprocess_config):
        report.print(line)
    report.print("=" * 60)
    report.print("")
    report.print(f"Processing {len(pairs)} files...")
    report.print("")
    
    results = []
    
    from modules.audio_io import load_audio_safe
    
    for audio_path, reference in tqdm(pairs, desc="Transcribing", unit="file"):
        audio_filename = os.path.basename(audio_path)
        
        audio, sr, error = load_audio_safe(audio_path)
        duration = len(audio) / sr if audio is not None and sr else 0
        
        hypothesis = transcribe_audio(audio_path, model_name=model_name or config.WHISPER_MODEL, preprocess_config=preprocess_config)
        
        metrics = calculate_detailed_metrics(reference, hypothesis.get('text', ''))
        
        if verbose:
            report.print(f"\n--- {audio_filename} ---")
            report.print(f"Reference:  {reference}")
            report.print(f"Hypothesis: {hypothesis.get('text', '')}")
            report.print(f"WER: {metrics['wer']:.2f}%  |  CER: {metrics['cer']:.2f}%")
            report.print(f"S: {metrics['substitutions']} | D: {metrics['deletions']} | I: {metrics['insertions']}")
        
        results.append({
            'filename': audio_filename,
            'reference': reference,
            'hypothesis': hypothesis.get('text', ''),
            'duration': duration,
            **metrics
        })
    
    if results:
        total_duration = sum(r['duration'] for r in results)
        total_ref_words = sum(r['ref_words'] for r in results)
        total_ref_chars = sum(r['ref_chars'] for r in results)
        total_sub = sum(r['substitutions'] for r in results)
        total_del = sum(r['deletions'] for r in results)
        total_ins = sum(r['insertions'] for r in results)
        total_char_sub = sum(r.get('char_sub', 0) for r in results)
        total_char_del = sum(r.get('char_del', 0) for r in results)
        total_char_ins = sum(r.get('char_ins', 0) for r in results)
        
        total_word_errors = total_sub + total_del + total_ins
        total_char_errors = total_char_sub + total_char_del + total_char_ins
        corpus_wer = (total_word_errors / total_ref_words * 100) if total_ref_words > 0 else 0
        corpus_cer = (total_char_errors / total_ref_chars * 100) if total_ref_chars > 0 else 0
        
        avg_wer = sum(r['wer'] for r in results) / len(results)
        avg_cer = sum(r['cer'] for r in results) / len(results)
        min_wer = min(r['wer'] for r in results)
        max_wer = max(r['wer'] for r in results)
        
        hours = total_duration / 3600
        
        report.print("")
        report.print("=" * 60)
        report.print("CORPUS-LEVEL STATISTICS")
        report.print("=" * 60)
        report.print(f"Utterances: {len(results):,}")
        report.print(f"Total audio: {hours:.1f} h ({total_duration:.0f} sec)")
        report.print(f"Ref words: {total_ref_words:,}")
        report.print(f"Ref chars: {total_ref_chars:,}")
        report.print("")
        report.print(f"WER: {corpus_wer:.1f}%  |  CER: {corpus_cer:.1f}%")
        report.print(f"S: {total_sub:,}  |  D: {total_del:,}  |  I: {total_ins:,}")
        report.print("")
        report.print("=" * 60)
        report.print("PER-UTTERANCE STATISTICS")
        report.print("=" * 60)
        report.print(f"Avg WER: {avg_wer:.2f}%  (Min: {min_wer:.2f}%, Max: {max_wer:.2f}%)")
        report.print(f"Avg CER: {avg_cer:.2f}%")
        report.print("=" * 60)
        
        report.save()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='ASR Pipeline with Whisper')
    parser.add_argument('--folder', type=str, default='ASR_track2',
                       help='Folder to use: ASR_track2, SID, SD_track2, SD_track1, SAD, etc. (default: ASR_track2)')
    parser.add_argument('--dataset', type=str, default='Dev', 
                       help='Dataset to use: Dev, Train, or Eval')
    parser.add_argument('--file', type=str, default=None,
                       help='Specific audio file to process')
    parser.add_argument('--batch', type=int, default=None,
                       help='Limit number of files to process (default: all files)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show per-utterance comparison details')
    parser.add_argument('--report', type=str, default=None,
                       help='Save report to specified file path')
    parser.add_argument('--whisper-model', type=str, default=None,
                       help='Override the configured Whisper model (e.g., tiny.en, base, small.en)')

    preprocess_group = parser.add_argument_group('Preprocessing Options')
    preprocess_group.add_argument('--preprocess', action='store_true',
                                   help='Enable audio preprocessing')
    preprocess_group.add_argument('--mono', action='store_true',
                                   help='Enable mono conversion')
    preprocess_group.add_argument('--resample', action='store_true',
                                   help='Enable resampling to target sample rate')
    preprocess_group.add_argument('--target-sr', type=int, default=16000,
                                   help='Target sample rate (default: 16000)')
    preprocess_group.add_argument('--dc-removal', action='store_true',
                                   help='Enable DC offset removal')
    preprocess_group.add_argument('--bandpass', action='store_true',
                                   help='Enable bandpass filter')
    preprocess_group.add_argument('--highpass', type=int, default=80,
                                   help='Highpass cutoff Hz (default: 80)')
    preprocess_group.add_argument('--lowpass', type=int, default=7500,
                                   help='Lowpass cutoff Hz (default: 7500)')
    preprocess_group.add_argument('--rms-norm', action='store_true',
                                   help='Enable RMS normalization')
    preprocess_group.add_argument('--rms-db', type=float, default=-20.0,
                                   help='Target RMS level in dB (default: -20)')
    preprocess_group.add_argument('--trim', action='store_true',
                                   help='Enable silence trimming')
    preprocess_group.add_argument('--trim-db', type=float, default=30.0,
                                   help='Trim threshold in dB (default: 30)')

    args = parser.parse_args()

    model_name = args.whisper_model or config.WHISPER_MODEL
    preprocess_config = create_preprocess_config(args)

    print("="*80)
    print("ASR Pipeline with Whisper")
    print("="*80)
    print(f"Phase: {config.PHASE}")
    print(f"Folder: {args.folder}")
    print(f"Dataset: {args.dataset}")
    print(f"Whisper Model: {model_name}")
    if preprocess_config:
        print(f"Preprocessing: ENABLED")
        steps = []
        if preprocess_config.enable_mono:
            steps.append("mono")
        if preprocess_config.enable_resample:
            steps.append(f"resample({preprocess_config.target_sr}Hz)")
        if preprocess_config.enable_dc_removal:
            steps.append("dc-removal")
        if preprocess_config.enable_bandpass:
            steps.append(f"bandpass({preprocess_config.highpass_cutoff}-{preprocess_config.lowpass_cutoff}Hz)")
        if preprocess_config.enable_rms_normalization:
            steps.append(f"rms-norm({preprocess_config.target_rms_db}dB)")
        if preprocess_config.enable_trim:
            steps.append(f"trim({preprocess_config.trim_db}dB)")
        print(f"  Steps: {', '.join(steps) if steps else 'none'}")
    else:
        print(f"Preprocessing: DISABLED (baseline)")
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
    
    if args.file:
        audio_path = os.path.join(audio_dir, args.file)
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)
        run_single_file(
            audio_path,
            transcript_dir,
            dataset=args.dataset,
            show_timestamps=True,
            model_name=model_name,
            folder=args.folder,
            preprocess_config=preprocess_config
        )
    else:
        run_batch(
            audio_dir,
            transcript_dir,
            limit=args.batch,
            dataset=args.dataset,
            model_name=model_name,
            folder=args.folder,
            preprocess_config=preprocess_config,
            report_path=args.report,
            verbose=args.verbose
        )

if __name__ == "__main__":
    main()
