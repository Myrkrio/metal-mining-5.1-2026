import json
import re
from pathlib import Path
from typing import Dict, List, Optional

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

    # Handle hyphenated roots directly preceding '-Metal' (e.g., Post-Metal, Nu-Metal)
    if re.fullmatch(r"(post|nu)-metal", genre_lower):
        return genre.split("-")[0].capitalize()

    # Strip trailing ' Metal' safely
    without_metal = re.sub(r"\s+metal$", "", genre, flags=re.IGNORECASE).strip()
    if not without_metal:
        return None

    # Keep only the root word (last token), then strip 'Post-' prefix if attached to a root
    root = without_metal.split()[-1]
    root = re.sub(r"^post-", "", root, flags=re.IGNORECASE)

    if not root or root.lower() == "various":
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