import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

EXACT_MAPPINGS = {
    "nwobhm": "Heavy",
    "djent": "Progressive",
    "crossover": "Thrash",
    "death 'n' roll": "Death",
}

STANDALONE_METAL_GENRES = {
    "metalcore": "Metalcore",
    "deathcore": "Deathcore",
    "grindcore": "Grindcore",
    "goregrind": "Goregrind",
}

NON_METAL_GENRES = {
    "various", "rock", "hard rock", "punk", "hardcore", "ambient", "dark ambient",
    "darkwave", "neofolk", "folk", "shred", "noise", "noisecore", "mathcore",
    "powerviolence", "crust", "shoegaze", "trance", "electronic", "electronica",
    "ebm", "grunge", "acoustic", "classical", "jazz", "funk", "pop", "trip hop",
    "dungeon synth", "rac", "oi!", "aor", "synthpop", "new wave", "film score",
    "a cappella", "experimental", "industrial", "neoclassical", "fusion",
}

PURE_DESCRIPTORS = {
    "atmospheric", "blackened", "brutal", "depressive", "epic",
    "melodic", "raw", "slam", "technical",
}

def _propagate_metal_suffix(slash_parts: List[str]) -> List[str]:
    """
    Propagate the 'Metal' suffix right-to-left across '/' separated genres.
    Example: ['Heavy', 'Speed', 'Power Metal'] -> ['Heavy Metal', 'Speed Metal', 'Power Metal']
    """
    result = []
    propagate_metal = False

    for part in reversed(slash_parts):
        part_clean = part.strip()
        part_lower = part_clean.lower()

        if not part_clean:
            continue

        if re.search(r"[-\s]metal$", part_lower):
            propagate_metal = True
            result.append(part_clean)
        elif (
            part_lower in STANDALONE_METAL_GENRES
            or part_lower in EXACT_MAPPINGS
            or part_lower in NON_METAL_GENRES
            or re.search(r"\b(rock|punk|core|grind|wave|folk|ambient|jazz|pop)\b$", part_lower)
        ):
            propagate_metal = False
            result.append(part_clean)
        elif propagate_metal:
            result.append(f"{part_clean} Metal")
        else:
            result.append(part_clean)

    return list(reversed(result))


def _reduce_genre(raw_genre: str) -> Optional[str]:
    """
    Filter non-Metal genres and reduce valid Metal genres to their root form.
    Returns None if the genre should be excluded.
    """
    genre = raw_genre.strip()
    genre_lower = genre.lower()

    if not genre or genre_lower == "various":
        return None

    # 1. Exact mappings (NWOBHM, Djent, Crossover, Death 'n' Roll)
    for key, mapped_value in EXACT_MAPPINGS.items():
        if genre_lower == key or genre_lower.endswith(f" {key}"):
            return mapped_value

    # 2. Standalone -core / -grind genres (strip leading descriptors if any)
    for key, canonical_value in STANDALONE_METAL_GENRES.items():
        if genre_lower == key or genre_lower.endswith(f" {key}"):
            return canonical_value

    # 3. Standard Metal genres (must end with ' Metal' or '-Metal')
    if not re.search(r"[-\s]metal$", genre_lower):
        return None

    # Handle hyphenated roots directly preceding '-Metal' (e.g., Post-Metal, Progressive Post-Metal, Nu-Metal)
    hyphen_match = re.search(r"\b(post|nu)-metal$", genre_lower)
    if hyphen_match:
        return hyphen_match.group(1).capitalize()

    # Strip trailing ' Metal' safely
    without_metal = re.sub(r"\s+metal$", "", genre, flags=re.IGNORECASE).strip()
    if not without_metal:
        return None

    # Keep only the root word (last token), then strip 'Post-' prefix if attached to a root
    root = without_metal.split()[-1]
    root = re.sub(r"^post-", "", root, flags=re.IGNORECASE)

    if not root or root.lower() == "various" or root.lower() in PURE_DESCRIPTORS:
        return None

    return root.capitalize() if root.islower() else root


def clean_genre_string(genre_str: str) -> List[str]:
    """
    Clean a raw Metal Archives genre string into a deduplicated list of root Metal genres.
    """
    if not genre_str:
        return []

    all_genres: List[str] = []

    # Step 1: Split by ';' (time periods)
    for period_part in genre_str.split(";"):
        # Step 2: Remove temporal markers in parentheses and 'with ...' clauses
        cleaned = re.sub(r"\s*\([^)]*\)", "", period_part)
        cleaned = re.sub(r"\s+with\s+.*$", "", cleaned, flags=re.IGNORECASE).strip()

        if not cleaned:
            continue

        # Step 3: Split by ',' (distinct genres)
        for comma_part in cleaned.split(","):
            comma_part = comma_part.strip()
            if not comma_part:
                continue

            # Step 4: Split by '/' and propagate 'Metal' suffix
            slash_parts = [p.strip() for p in comma_part.split("/") if p.strip()]
            expanded_parts = _propagate_metal_suffix(slash_parts)

            # Step 5: Filter non-Metal genres and reduce to roots
            for part in expanded_parts:
                reduced = _reduce_genre(part)
                if reduced:
                    all_genres.append(reduced)

    # Step 6: Deduplicate while preserving order
    deduplicated: List[str] = []
    seen = set()
    for genre in all_genres:
        key = genre.lower()
        if key not in seen:
            deduplicated.append(genre)
            seen.add(key)

    return deduplicated


def clean_genres_file(input_file: str, output_file: str) -> Dict[str, List[str]]:
    """
    Load raw genres JSON, clean all entries, drop artists with no valid Metal genres,
    and save the result to output_file.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data: Dict[str, List[str]] = {}

    for band_name, genre_value in data.items():
        if not genre_value:
            continue
        cleaned_genres = clean_genre_string(genre_value)
        if cleaned_genres:
            cleaned_data[band_name] = cleaned_genres

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    return cleaned_data

def load_ml_dataset(dataset_path: str) -> List[Dict]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_genres_stats(dataset: List[Dict]) -> Dict:
    total_covers = len(dataset)
    genre_counter: Counter = Counter()
    genres_per_cover_counter: Counter = Counter()

    for entry in dataset:
        genres = entry.get("genres") or []
        genres_per_cover_counter[len(genres)] += 1
        genre_counter.update(genres)

    sorted_genres = [g for g, _ in genre_counter.most_common()]
    genre_counts = {g: genre_counter[g] for g in sorted_genres}
    genres_per_cover = dict(sorted(genres_per_cover_counter.items()))

    index = {g: i for i, g in enumerate(sorted_genres)}
    n = len(sorted_genres)
    cooccurrence_counts = [[0] * n for _ in range(n)]

    for entry in dataset:
        genres = entry.get("genres") or []
        for g1 in genres:
            i = index[g1]
            for g2 in genres:
                j = index[g2]
                cooccurrence_counts[i][j] += 1

    cooccurrence_pct = [
        [
            (cooccurrence_counts[i][j] / genre_counts[sorted_genres[i]]) * 100.0
            if genre_counts[sorted_genres[i]] > 0 else 0.0
            for j in range(n)
        ]
        for i in range(n)
    ]

    return {
        "total_covers": total_covers,
        "sorted_genres": sorted_genres,
        "genre_counts": genre_counts,
        "genres_per_cover": genres_per_cover,
        "cooccurrence_counts": cooccurrence_counts,
        "cooccurrence_pct": cooccurrence_pct,
    }