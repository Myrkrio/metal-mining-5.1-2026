#!/usr/bin/env python3
"""
Compute genre statistics and generate visualizations from data/dataset_ml.json.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.genres_processing import compute_genres_stats, load_ml_dataset
from visualization.distribution import (
    plot_genres_cooccurrence,
    plot_genres_distribution,
    plot_genres_per_cover,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze genre distribution and co-occurrence on dataset_ml.json"
    )
    parser.add_argument(
        "-i", "--input",
        default=os.path.join(PROJECT_ROOT, "data", "dataset_ml.json"),
        help="Input ML dataset JSON (default: data/dataset_ml.json)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "genres"),
        help="Output directory for PNG plots (default: outputs/genres)",
    )
    args = parser.parse_args()

    dataset = load_ml_dataset(args.input)
    stats = compute_genres_stats(dataset)

    total = stats["total_covers"]
    genre_counts = stats["genre_counts"]
    genres_per_cover = stats["genres_per_cover"]
    avg_genres = sum(k * v for k, v in genres_per_cover.items()) / total if total > 0 else 0.0

    print(f"Total covers          : {total}")
    print(f"Unique genres         : {len(genre_counts)}")
    print(f"Avg genres per cover  : {avg_genres:.2f}\n")

    print("Multi-label distribution:")
    for nb_genres, count in genres_per_cover.items():
        print(f"  {nb_genres} genre(s): {count:5d} ({count / total * 100:5.1f}%)")

    print("\nGenre counts:")
    for genre, count in genre_counts.items():
        print(f"  {genre:<15}: {count:5d} ({count / total * 100:5.1f}%)")

    os.makedirs(args.output_dir, exist_ok=True)

    dist_path = os.path.join(args.output_dir, "genres_distribution.png")
    multi_path = os.path.join(args.output_dir, "genres_per_cover.png")
    cooc_pct_path = os.path.join(args.output_dir, "genres_cooccurrence_pct.png")

    plot_genres_distribution(genre_counts, total, dist_path)
    plot_genres_per_cover(genres_per_cover, total, multi_path)
    plot_genres_cooccurrence(stats["cooccurrence_pct"], stats["sorted_genres"], cooc_pct_path)

    print(f"\nGraphiques sauvegardés dans {args.output_dir}/ :")
    print(f"  - {os.path.basename(dist_path)}")
    print(f"  - {os.path.basename(multi_path)}")
    print(f"  - {os.path.basename(cooc_pct_path)}")


if __name__ == "__main__":
    main()