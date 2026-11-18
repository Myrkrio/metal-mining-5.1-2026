#!/usr/bin/env python3
"""
Covers distribution analysis.

Analyzes the distribution of available cover images across metal genres
and identifies class imbalances.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loading.dataset_loader import load_music_data, load_genres_data
from data_loading.covers_loader import get_available_covers
from utils.covers_processing import calculate_covers_distribution

def main():
    """Run covers distribution analysis."""
    print("Loading data...")
    df_songs, _ = load_music_data('data/dataset.json')
    genres_dict = load_genres_data('data/bands_genres_cleaned.json')
    available_covers = get_available_covers('data/covers')

    print(f"Total songs in dataset: {len(df_songs)}")


    print("\nProcessing cover matching...")

    unique_albums = df_songs[['artist', 'album']].drop_duplicates()
    print(f"Total des albums uniques dans dataset.json : {len(unique_albums)}")
    
    albums_with_genre = [
        row['album'] for _, row in unique_albums.iterrows() 
        if genres_dict.get(row['artist'])
    ]
    print(f"Albums conservés (artistes avec un genre valide) : {len(albums_with_genre)}")
    print(f"Albums ignorés (genres null ou artiste absent) : {len(unique_albums) - len(albums_with_genre)}")

    df_stats = calculate_covers_distribution(df_songs, genres_dict, available_covers)
    print(len(df_stats), "genres found in the dataset.")
    print(df_stats['Expected'].sum(), "expected covers in total.")
    print(df_stats['Found'].sum(), "covers found in total.")
    print(df_stats)



if __name__ == "__main__":
    main()
