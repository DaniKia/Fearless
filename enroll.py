"""
Standalone speaker enrollment script.
Generates pkl files with speaker embeddings and metadata.
"""

import os
import argparse
from datetime import datetime
from modules.data_loader import group_audio_by_speaker
from modules.speaker_identifier import SpeakerIdentifier, compute_znorm_stats
from modules.audio_preprocessor import PreprocessConfig
import config


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


def config_to_dict(preprocess_config):
    """
    Convert PreprocessConfig to dictionary for metadata storage.
    
    Args:
        preprocess_config: PreprocessConfig or None
        
    Returns:
        Dictionary with preprocessing settings
    """
    if preprocess_config is None:
        return {'enabled': False}
    
    return {
        'enabled': True,
        'enable_mono': preprocess_config.enable_mono,
        'enable_resample': preprocess_config.enable_resample,
        'target_sr': preprocess_config.target_sr,
        'enable_dc_removal': preprocess_config.enable_dc_removal,
        'enable_bandpass': preprocess_config.enable_bandpass,
        'highpass_cutoff': preprocess_config.highpass_cutoff,
        'lowpass_cutoff': preprocess_config.lowpass_cutoff,
        'enable_rms_normalization': preprocess_config.enable_rms_normalization,
        'target_rms_db': preprocess_config.target_rms_db,
        'enable_trim': preprocess_config.enable_trim,
        'trim_db': preprocess_config.trim_db,
        'min_duration': preprocess_config.min_duration
    }


def enroll_speakers(folder, dataset, output_path, batch_size=16, preprocess_config=None, normalize_method=None):
    """
    Enroll speakers from the training dataset.
    
    Args:
        folder: Folder name (SID, ASR_track2, SD_track2, etc.)
        dataset: Dataset to use for enrollment (default: Train)
        output_path: Path to save the output pkl file
        batch_size: Number of files to process per GPU batch
        preprocess_config: Optional PreprocessConfig for audio preprocessing
        normalize_method: Embedding normalization method ('l2', 'l2-centered', or None)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Speaker Enrollment - {folder}/{dataset} Dataset")
    print(f"{'='*60}\n")
    
    audio_dir = config.get_folder_audio_path(folder, dataset)
    label_dir = config.get_folder_label_path(folder, dataset)
    
    print(f"Folder: {folder}")
    print(f"Audio directory: {audio_dir}")
    print(f"Label directory: {label_dir}")
    print(f"Output file: {output_path}")
    print(f"Preprocessing: {'Enabled' if preprocess_config else 'Disabled'}")
    print(f"Normalization: {normalize_method or 'None'}\n")
    
    speaker_files = group_audio_by_speaker(audio_dir, label_dir, dataset=dataset)
    
    if not speaker_files:
        print("Error: No speaker files found for enrollment")
        return False
    
    print(f"Found {len(speaker_files)} unique speakers")
    total_files = sum(len(files) for files in speaker_files.values())
    print(f"Total audio files: {total_files}\n")
    
    metadata = {
        'folder': folder,
        'dataset': dataset,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_speakers': len(speaker_files),
        'num_files': total_files,
        'batch_size': batch_size,
        'preprocessing': config_to_dict(preprocess_config),
        'normalize_method': normalize_method
    }
    
    identifier = SpeakerIdentifier(batch_size=batch_size, normalize_method=normalize_method)
    embeddings = identifier.enroll_speakers(
        speaker_files, 
        save_path=output_path, 
        preprocess_config=preprocess_config,
        metadata=metadata
    )
    
    if embeddings:
        print(f"\nComputing z-norm statistics for score normalization...")
        znorm_stats = compute_znorm_stats(embeddings)
        identifier.znorm_stats = znorm_stats
        identifier.save_database(output_path, include_znorm=True)
        print(f"Z-norm stats computed for {len(znorm_stats)} speakers")
        
        print(f"\nEnrollment complete! Database saved to: {output_path}")
        return True
    else:
        print("\nEnrollment failed!")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Speaker Enrollment - Generate speaker embedding pkl files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic enrollment (no preprocessing, no normalization)
  python enroll.py --output baseline.pkl

  # Enrollment with preprocessing and L2 normalization (recommended)
  python enroll.py --preprocess --resample --normalize l2 --output enrolled.pkl

  # Custom preprocessing settings
  python enroll.py --preprocess --bandpass --highpass 100 --lowpass 7000 --normalize l2 --output custom.pkl

  # Multiple output files
  python enroll.py --preprocess --normalize l2 --output config_a.pkl config_b.pkl

  # Different dataset
  python enroll.py --folder SID --dataset Train --normalize l2 --output sid_train.pkl
        """
    )
    
    parser.add_argument('--folder', type=str, default='SID',
                        help='Folder name (SID, ASR_track2, SD_track2, etc.)')
    parser.add_argument('--dataset', type=str, default='Train',
                        help='Dataset to use (Train, Dev, Eval)')
    parser.add_argument('--output', '-o', type=str, nargs='+', required=True,
                        help='Output pkl file path(s). Can specify multiple files.')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='GPU batch size (default: 16)')
    
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
    
    embedding_group = parser.add_argument_group('Embedding Options')
    embedding_group.add_argument('--normalize', type=str, default=None,
                                  choices=['l2'],
                                  help='Embedding normalization method (l2 recommended)')
    
    args = parser.parse_args()
    
    preprocess_config = create_preprocess_config(args)
    
    output_files = args.output
    all_success = True
    
    for i, output_path in enumerate(output_files):
        if len(output_files) > 1:
            print(f"\n{'#'*60}")
            print(f"# Creating PKL {i+1}/{len(output_files)}: {output_path}")
            print(f"{'#'*60}")
        
        success = enroll_speakers(
            folder=args.folder,
            dataset=args.dataset,
            output_path=output_path,
            batch_size=args.batch_size,
            preprocess_config=preprocess_config,
            normalize_method=args.normalize
        )
        
        if not success:
            all_success = False
    
    return 0 if all_success else 1


if __name__ == '__main__':
    exit(main())
