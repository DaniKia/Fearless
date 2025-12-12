"""
Utility script to add z-norm statistics to existing PKL files.
Updates PKL files in place with per-speaker impostor statistics.
"""

import os
import argparse
import pickle
from modules.speaker_identifier import compute_znorm_stats


def add_znorm_to_pkl(pkl_path):
    """
    Add z-norm statistics to an existing PKL file.
    
    Args:
        pkl_path: Path to the PKL file to update
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(pkl_path):
        print(f"Error: File not found: {pkl_path}")
        return False
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return False
    
    if isinstance(data, dict) and 'embeddings' in data:
        embeddings = data['embeddings']
        metadata = data.get('metadata')
    else:
        embeddings = data
        metadata = None
        data = {'embeddings': embeddings, 'metadata': metadata}
    
    if 'znorm_stats' in data and data['znorm_stats']:
        print(f"  Warning: {pkl_path} already has z-norm stats ({len(data['znorm_stats'])} speakers)")
        print(f"  Recomputing and overwriting...")
    
    print(f"  Computing z-norm statistics for {len(embeddings)} speakers...")
    znorm_stats = compute_znorm_stats(embeddings)
    
    if not znorm_stats:
        print(f"  Error: Failed to compute z-norm stats")
        return False
    
    data['znorm_stats'] = znorm_stats
    
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Z-norm stats added successfully ({len(znorm_stats)} speakers)")
        return True
    except Exception as e:
        print(f"  Error saving {pkl_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Add z-norm statistics to existing PKL files (updates in place)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add z-norm stats to a single PKL file
  python add_znorm.py model.pkl

  # Add z-norm stats to multiple PKL files
  python add_znorm.py baseline.pkl preprocessed.pkl experiment_v2.pkl

What this does:
  - Loads existing speaker embeddings from the PKL file
  - Computes impostor statistics (mean, std) for each speaker
  - Saves the z-norm stats back to the same PKL file
  
  The z-norm stats enable --score-norm znorm in sid_main.py, which
  normalizes similarity scores to handle "easy" vs "hard" speakers.
        """
    )
    
    parser.add_argument('pkl_files', type=str, nargs='+',
                        help='PKL file(s) to update with z-norm statistics')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Adding Z-Norm Statistics to PKL Files")
    print("=" * 60)
    
    success_count = 0
    for pkl_path in args.pkl_files:
        print(f"\nProcessing: {pkl_path}")
        if add_znorm_to_pkl(pkl_path):
            success_count += 1
    
    print(f"\n{'=' * 60}")
    print(f"Complete: {success_count}/{len(args.pkl_files)} files updated")
    print("=" * 60)
    
    return 0 if success_count == len(args.pkl_files) else 1


if __name__ == '__main__':
    exit(main())
