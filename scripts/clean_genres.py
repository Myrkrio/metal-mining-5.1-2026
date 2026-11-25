#!/usr/bin/env python3
"""
CLI script to clean and reduce Metal Archives genres from bands_genres.json.
"""

import argparse
import os
import sys
import json

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.genres_processing import clean_genres_file

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and reduce genres from bands_genres.json"
    )
    parser.add_argument(
        "-i", "--input",
        default=os.path.join(PROJECT_ROOT, "data", "bands_genres.json"),
        help="Input JSON file (default: data/bands_genres.json)",
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(PROJECT_ROOT, "data", "bands_genres_cleaned.json"),
        help="Output JSON file (default: data/bands_genres_cleaned.json)",
    )
    args = parser.parse_args()

    print(f"Cleaning genres from {args.input}...")
    cleaned_data = clean_genres_file(args.input, args.output)
    print(f"Cleaned genres saved to {args.output}")

    with open(args.input, "r", encoding="utf-8") as f:
        print(f"\nBands before processing: {len(json.load(f))}")

    unique_genres = sorted({g for genres in cleaned_data.values() for g in genres})
    print(f"\nBands kept: {len(cleaned_data)}")
    print(f"Unique genres found ({len(unique_genres)}): {unique_genres}")


if __name__ == "__main__":
    main()