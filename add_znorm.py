"""
Utility script to add z-norm statistics to existing PKL files.
Updates PKL files in place or saves to a new file.
"""

import os
import argparse
import pickle
from modules.speaker_identifier import compute_znorm_stats


def add_znorm_to_pkl(input_path, output_path=None, sigma_floor=0.03, cohort_size=None):
    """
    Add z-norm statistics to a PKL file.
    
    Args:
        input_path: Path to the source PKL file
        output_path: Path to save output (None = update in place)
        sigma_floor: Minimum reliable sigma value (default 0.03)
        cohort_size: Number of top impostors to use (None = all)
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return False
    
    save_path = output_path if output_path else input_path
    
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return False
    
    if isinstance(data, dict) and 'embeddings' in data:
        embeddings = data['embeddings']
        metadata = data.get('metadata')
    else:
        embeddings = data
        metadata = None
        data = {'embeddings': embeddings, 'metadata': metadata}
    
    if 'znorm_stats' in data and data['znorm_stats'] and not output_path:
        print(f"  Warning: {input_path} already has z-norm stats ({len(data['znorm_stats'])} speakers)")
        print(f"  Recomputing and overwriting...")
    
    print(f"  Computing z-norm statistics for {len(embeddings)} speakers...")
    znorm_stats = compute_znorm_stats(embeddings, sigma_floor=sigma_floor, cohort_size=cohort_size)
    
    if not znorm_stats:
        print(f"  Error: Failed to compute z-norm stats")
        return False
    
    reliable_count = sum(1 for s in znorm_stats.values() if s.get('reliable', True))
    unreliable_count = len(znorm_stats) - reliable_count
    
    data['znorm_stats'] = znorm_stats
    
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"\n  Z-norm stats added successfully:")
        print(f"    Total speakers: {len(znorm_stats)}")
        print(f"    Reliable (will use z-norm): {reliable_count}")
        print(f"    Unreliable (will use raw scores): {unreliable_count}")
        if output_path:
            print(f"  Saved to: {save_path}")
        return True
    except Exception as e:
        print(f"  Error saving {save_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Add z-norm statistics to existing PKL files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With cohort scoring (RECOMMENDED) - use top-50 hard negatives
  python add_znorm.py baseline.pkl --cohort-size 50 --output znorm_cohort50.pkl

  # Update a single PKL file in place (all impostors)
  python add_znorm.py model.pkl

  # Save to a new file (keeps original unchanged)
  python add_znorm.py baseline.pkl --output znorm_only.pkl

  # Update multiple PKL files in place
  python add_znorm.py baseline.pkl preprocessed.pkl experiment_v2.pkl

What this does:
  - Loads existing speaker embeddings from the PKL file
  - Computes impostor statistics (mean, std) for each speaker
  - With --cohort-size N: uses only top-N most similar impostors (hard negatives)
  - Marks speakers with sigma < sigma_floor as "unreliable"
  - Saves the z-norm stats to the output file (or updates in place)

Cohort Scoring (--cohort-size):
  Instead of using all impostors, only uses the top-N most similar speakers.
  This is the standard practice that:
  - Produces larger, more stable sigma values
  - Focuses statistics on confusing/similar speakers
  - Avoids tiny sigma from including easy rejects
  Recommended values: 50-100 for datasets with 100+ speakers.
  
  At inference time (sid_main.py --score-norm znorm):
  - Reliable speakers: z-normalized scores = (raw - mu) / sigma
  - Unreliable speakers: raw cosine similarity (avoids score explosions)
  
  The sigma_floor (default 0.03) prevents tiny sigma values from
  causing score explosions that turn speakers into "sinks".
        """
    )
    
    parser.add_argument('pkl_files', type=str, nargs='+',
                        help='PKL file(s) to process')
    parser.add_argument('--output', '-o', type=str, nargs='*', default=None,
                        help='Output file path(s). If not specified, updates in place. If specified, must match number of input files.')
    parser.add_argument('--sigma-floor', type=float, default=0.03,
                        help='Minimum reliable sigma value (default: 0.03). Speakers with sigma below this use raw scores at inference.')
    parser.add_argument('--cohort-size', type=int, default=None,
                        help='Number of top impostors (hard negatives) to use for computing stats. Default: None (use all). Recommended: 50-100.')
    
    args = parser.parse_args()
    
    output_files = args.output if args.output else [None] * len(args.pkl_files)
    
    if args.output and len(args.output) != len(args.pkl_files):
        print(f"Error: Number of output files ({len(args.output)}) must match number of input files ({len(args.pkl_files)})")
        return 1
    
    print("=" * 60)
    print("Adding Z-Norm Statistics to PKL Files")
    print(f"Sigma floor: {args.sigma_floor}")
    if args.cohort_size:
        print(f"Cohort size: {args.cohort_size} (top-N hard negatives)")
    else:
        print("Cohort size: all impostors")
    print("=" * 60)
    
    success_count = 0
    for input_path, output_path in zip(args.pkl_files, output_files):
        print(f"\nProcessing: {input_path}")
        if add_znorm_to_pkl(input_path, output_path, sigma_floor=args.sigma_floor, cohort_size=args.cohort_size):
            success_count += 1
    
    print(f"\n{'=' * 60}")
    print(f"Complete: {success_count}/{len(args.pkl_files)} files processed")
    print("=" * 60)
    
    return 0 if success_count == len(args.pkl_files) else 1


if __name__ == '__main__':
    exit(main())
