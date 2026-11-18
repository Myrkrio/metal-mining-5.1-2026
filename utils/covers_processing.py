"""Utility functions for processing album covers data."""

import pandas as pd
from collections import defaultdict
from typing import Set, Dict

def calculate_covers_distribution(df_songs: pd.DataFrame, genres_dict: Dict[str, list], available_covers: Set[str]) -> pd.DataFrame:
    """
    Cross-reference the dataset albums, their genres, and the available cover images.
    
    Args:
        df_songs (pd.DataFrame): DataFrame containing 'artist' and 'album' columns.
        genres_dict (Dict[str, list]): Mapping of artists to their genres.
        available_covers (Set[str]): Set of available cover filenames (without extensions).
        
    Returns:
        pd.DataFrame: DataFrame containing expected vs found covers per genre.
    """
    expected_counts = defaultdict(int)
    actual_counts = defaultdict(int)

    unique_albums = df_songs[['artist', 'album']].drop_duplicates()
    available_covers_norm = {"".join(c for c in cover if c.isalnum()).lower() for cover in available_covers}

    for _, row in unique_albums.iterrows():
        artist = row['artist']
        album = row['album']
        
        genres = genres_dict.get(artist)
        if not genres:
            continue
            
        expected_norm = "".join(c for c in f"{artist}{album}" if c.isalnum()).lower()
        is_present = expected_norm in available_covers_norm
        
        for genre in genres:
            expected_counts[genre] += 1
            if is_present:
                actual_counts[genre] += 1

    results = []
    for genre, expected in expected_counts.items():
        actual = actual_counts.get(genre, 0)
        ratio = (actual / expected) * 100 if expected > 0 else 0
        results.append({
            'Genre': genre,
            'Expected': expected,
            'Found': actual,
            'Percentage': round(ratio, 2)
        })

    df = pd.DataFrame(results)
    
    if not df.empty:
        df = df.sort_values(by='Percentage', ascending=True).reset_index(drop=True)
        
    return df