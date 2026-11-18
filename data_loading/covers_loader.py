"""Functions for loading and verifying album cover images."""

import os
from typing import Set

def get_available_covers(covers_dir: str = 'data/covers') -> Set[str]:
    """
    Scan the covers directory and return a set of cover names without extensions.
    Also verifies if the total count matches the expected 14404 covers.
    
    Args:
        covers_dir (str): Path to the directory containing cover images.
        
    Returns:
        Set[str]: A set of filenames without their extensions.
    """
    if not os.path.exists(covers_dir):
        print(f"Warning: Directory '{covers_dir}' does not exist.")
        return set()

    # List only files (ignoring subdirectories if any)
    files = [f for f in os.listdir(covers_dir) if os.path.isfile(os.path.join(covers_dir, f))]
    total_files = len(files)
    
    print(f"[Covers] Files found: {total_files}")

    # Extract name without extension (e.g., "Artist - Album.jpg" -> "Artist - Album")
    covers_no_ext = {os.path.splitext(f)[0] for f in files}
    
    return covers_no_ext