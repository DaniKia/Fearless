"""
Utility script to add z-norm statistics to existing PKL files.
Updates PKL files in place or saves to a new file.
"""

import os
import argparse
import pickle
from modules.speaker_identifier import compute_znorm_stats


def add_znorm_to_pkl(input_path, output_path=None):
    """
    Add z-norm statistics to a PKL file.
    
    Args:
        input_path: Path to the source PKL file
        output_path: Path to save output (None = update in place)
        
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
    znorm_stats = compute_znorm_stats(embeddings)
    
    if not znorm_stats:
        print(f"  Error: Failed to compute z-norm stats")
        return False
    
    data['znorm_stats'] = znorm_stats
    
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Z-norm stats added successfully ({len(znorm_stats)} speakers)")
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
  # Update a single PKL file in place
  python add_znorm.py model.pkl

  # Save to a new file (keeps original unchanged)
  python add_znorm.py baseline.pkl --output znorm_only.pkl

  # Update multiple PKL files in place
  python add_znorm.py baseline.pkl preprocessed.pkl experiment_v2.pkl

What this does:
  - Loads existing speaker embeddings from the PKL file
  - Computes impostor statistics (mean, std) for each speaker
  - Saves the z-norm stats to the output file (or updates in place)
  
  The z-norm stats enable --score-norm znorm in sid_main.py, which
  normalizes similarity scores to handle "easy" vs "hard" speakers.
        """
    )
    
    parser.add_argument('pkl_files', type=str, nargs='+',
                        help='PKL file(s) to process')
    parser.add_argument('--output', '-o', type=str, nargs='*', default=None,
                        help='Output file path(s). If not specified, updates in place. If specified, must match number of input files.')
    
    args = parser.parse_args()
    
    output_files = args.output if args.output else [None] * len(args.pkl_files)
    
    if args.output and len(args.output) != len(args.pkl_files):
        print(f"Error: Number of output files ({len(args.output)}) must match number of input files ({len(args.pkl_files)})")
        return 1
    
    print("=" * 60)
    print("Adding Z-Norm Statistics to PKL Files")
    print("=" * 60)
    
    success_count = 0
    for input_path, output_path in zip(args.pkl_files, output_files):
        print(f"\nProcessing: {input_path}")
        if add_znorm_to_pkl(input_path, output_path):
            success_count += 1
    
    print(f"\n{'=' * 60}")
    print(f"Complete: {success_count}/{len(args.pkl_files)} files processed")
    print("=" * 60)
    
    return 0 if success_count == len(args.pkl_files) else 1


if __name__ == '__main__':
    exit(main())
