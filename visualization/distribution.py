"""Visualization functions for dataset distribution analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import seaborn as sns


def analyze_lyrics_distribution(df_songs: pd.DataFrame, df_albums: pd.DataFrame):
    """
    Analyze and visualize the distribution of music with and without lyrics.
    
    Creates multiple visualizations:
    - Pie chart: With vs Without Lyrics
    - Pie chart: Publication Types
    - Bar chart: Top 10 bands by song count
    - Bar chart: Top 10 bands by album count
    - Bar chart: Distribution of songs over the years
    - Pie chart: Language distribution (Top 4 + Other)
    - Table: Top 20 languages
    """
    # With vs Without Lyrics
    plt.figure(figsize=(8, 6))
    lyrics_counts = df_songs['lyrics_status'].value_counts()
    colors = ['#ff9999', '#66b3ff']
    total = lyrics_counts.sum()
    percentages = [(status, f"{status} ({(count/total*100):.1f}%)") for status, count in lyrics_counts.items()]
    labels = [label for _, label in percentages]
    plt.pie(lyrics_counts.values, colors=colors, startangle=90)
    plt.legend(labels, title="Lyrics Status", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title('Distribution of Songs: With vs Without Lyrics', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Publication types pie chart
    plt.figure(figsize=(8, 6))
    type_counts = df_albums['album_type'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#cc99ff']
    total = type_counts.sum()
    percentages = [(type_, f"{type_} ({(count/total*100):.1f}%)") for type_, count in type_counts.items()]
    labels = [label for _, label in percentages]
    plt.pie(type_counts.values, colors=colors[:len(type_counts)], startangle=90)
    plt.legend(labels, title="Publication Types", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title('Distribution of Publication Types', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Top 10 bands by song count
    plt.figure(figsize=(10, 6))
    top_bands = df_songs['artist'].value_counts().head(10)
    plt.barh(top_bands.index[::-1], top_bands.values[::-1])
    plt.title('Top 10 Bands by Song Count', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Songs')
    plt.ylabel('Bands')
    plt.tight_layout()
    plt.show()
    
    # Top 10 bands by album count
    plt.figure(figsize=(10, 6))
    top_bands_albums = df_songs.groupby('artist')['album'].nunique().sort_values(ascending=False).head(10)
    plt.barh(top_bands_albums.index[::-1], top_bands_albums.values[::-1])
    plt.title('Top 10 Bands by Album Count', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Albums')
    plt.ylabel('Bands')
    plt.tight_layout()
    plt.show()

    # Distribution of songs over the years
    plt.figure(figsize=(12, 6))
    year_data = df_songs[df_songs['release_year'] != 'Unknown'].copy()
    year_data['release_year'] = pd.to_numeric(year_data['release_year'], errors='coerce')
    year_data = year_data.dropna(subset=['release_year'])
    year_counts = year_data['release_year'].value_counts().sort_index()
    
    plt.bar(year_counts.index, year_counts.values)
    plt.title('Distribution of Songs Over the Years', fontsize=12, fontweight='bold')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Songs')
    plt.xticks(year_counts.index, rotation=45)
    plt.tight_layout()
    plt.show()

    # Language distribution for songs with lyrics (Top 4 + Other)
    plt.figure(figsize=(8, 6))
    language_counts = df_songs[df_songs['has_lyrics']]['language'].value_counts()
    top_languages = language_counts.head(4)
    other_count = language_counts[4:].sum() if len(language_counts) > 4 else 0
    if other_count > 0:
        top_languages['Other Languages'] = other_count
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#cc99ff']
    total = top_languages.sum()
    percentages = [(lang, f"{lang} ({(count/total*100):.1f}%)") for lang, count in top_languages.items()]
    labels = [label for _, label in percentages]
    plt.pie(top_languages.values, colors=colors[:len(top_languages)], startangle=90)
    plt.legend(labels, title="Languages", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title('Distribution of Song Languages (Top 4 + Other)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Language distribution table (Top 20)
    plt.figure(figsize=(8, 10))
    plt.title('Top 20 Languages in Songs', fontsize=12, fontweight='bold', pad=20)
    language_counts = df_songs[df_songs['has_lyrics']]['language'].value_counts()
    top_20_languages = language_counts.head(20)
    language_table = pd.DataFrame({
        'Language': top_20_languages.index,
        'Song Count': top_20_languages.values,
        'Percentage': (top_20_languages / top_20_languages.sum() * 100).round(1)
    })
    table = plt.table(cellText=language_table.values,
                    colLabels=language_table.columns,
                    loc='center',
                    cellLoc='center',
                    colColours=['#f0f0f0'] * 3)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return df_songs, df_albums


def plot_genres_distribution(
    genre_counts: Dict[str, int],
    total_covers: int,
    output_path: str,
) -> None:
    genres = list(genre_counts.keys())
    counts = list(genre_counts.values())

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(genres, counts, color="#4C72B0")
    ax.invert_yaxis()

    max_count = max(counts) if counts else 1
    ax.set_xlim(0, max_count * 1.18)

    for bar, count in zip(bars, counts):
        pct = (count / total_covers * 100) if total_covers > 0 else 0.0
        ax.text(
            bar.get_width() + max_count * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Nombre de covers")
    ax.set_title(f"Distribution des genres sur le dataset ({total_covers} covers)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_genres_per_cover(
    genres_per_cover: Dict[int, int],
    total_covers: int,
    output_path: str,
) -> None:
    labels = [str(k) for k in genres_per_cover.keys()]
    counts = list(genres_per_cover.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color="#55A868")

    max_count = max(counts) if counts else 1
    ax.set_ylim(0, max_count * 1.18)

    for bar, count in zip(bars, counts):
        pct = (count / total_covers * 100) if total_covers > 0 else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_count * 0.015,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Nombre de genres par cover")
    ax.set_ylabel("Nombre de covers")
    ax.set_title(f"Répartition multi-label ({total_covers} covers)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_genres_cooccurrence(
    matrix: List[List[float]],
    labels: List[str],
    output_path: str,
) -> None:
    data = np.array(matrix)
    mask = np.eye(len(labels), dtype=bool)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        data,
        mask=mask,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".0f",
        annot_kws={"size": 8},
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        linewidths=0.3,
        cbar_kws={"label": "% des covers du genre en ligne"},
        ax=ax,
    )

    ax.set_title("Co-occurrence des genres (% par ligne, diagonale masquée)", fontsize=13, pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
