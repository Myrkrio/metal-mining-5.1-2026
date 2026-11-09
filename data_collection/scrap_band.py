#!/usr/bin/env python3
"""
Script to scrape genres from Metal Archives for all bands and save them 
in a flat JSON format: {"Band Name": "Genre", ...}
"""

import sys
import os
import json
import time
import argparse
import urllib.parse
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    try:
        from webdriver_manager.firefox import GeckoDriverManager
        GECKO_MANAGER_AVAILABLE = True
    except ImportError:
        GECKO_MANAGER_AVAILABLE = False
except ImportError:
    print("Error: Missing required packages. Install with:")
    print("pip install selenium webdriver-manager")
    sys.exit(1)

BASE_URL = "https://www.metal-archives.com/bands/"
DEFAULT_OUTPUT = "./data/bands_genres.json"
DELAY = 1  # Delay between requests in seconds

def init_selenium_driver(headless=True):
    """
    Initialise Firefox to bypass Cloudflare protection.
    """
    firefox_options = Options()
    if headless:
        firefox_options.add_argument("--headless")
        
    firefox_options.set_preference("general.useragent.override", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0")
    
    driver = None
    try:
        if GECKO_MANAGER_AVAILABLE:
            driver_path = GeckoDriverManager().install()
            service = Service(driver_path)
            driver = webdriver.Firefox(service=service, options=firefox_options)
        else:
            driver = webdriver.Firefox(options=firefox_options)
        
        driver.set_page_load_timeout(30)
        driver.implicitly_wait(5)
        
        driver.get('https://www.metal-archives.com/')
        return driver
        
    except Exception as e:
        print(f"Error initializing Firefox driver: {e}")
        if driver:
            try:
                driver.quit()
            except:
                pass
        raise e

def format_band_url(band_name: str) -> str:
    formatted_name = band_name.strip().replace(' ', '_')
    encoded = urllib.parse.quote(formatted_name, safe='_')
    return f"{BASE_URL}{encoded}/"

def scrape_band_genre(driver, band_name: str) -> str:
    """
    Scrape the genre for a band from Metal Archives using Firefox Selenium.
    """
    url = format_band_url(band_name)
    
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 5)
        wait.until(EC.presence_of_element_located((By.ID, "band_stats")))
        
        try:
            genre_dt = driver.find_element(By.XPATH, "//div[@id='band_stats']//dl[@class='float_right']//dt[text()='Genre:']")
            genre_dd = genre_dt.find_element(By.XPATH, "./following-sibling::dd[1]")
            genre = genre_dd.text.strip()
            if genre:
                return genre
        except NoSuchElementException:
            try:
                dts = driver.find_elements(By.CSS_SELECTOR, "#band_stats dl.float_right dt")
                dds = driver.find_elements(By.CSS_SELECTOR, "#band_stats dl.float_right dd")
                
                for i, dt in enumerate(dts):
                    if dt.text.strip() == "Genre:":
                        if i < len(dds):
                            genre = dds[i].text.strip()
                            if genre:
                                return genre
                        break
            except Exception:
                pass
        except Exception:
            try:
                genre_dd = driver.find_element(By.CSS_SELECTOR, "#band_stats dl.float_right dd:first-of-type")
                genre = genre_dd.text.strip()
                if genre:
                    return genre
            except Exception:
                pass
        
        return None
        
    except (TimeoutException, NoSuchElementException):
        return None
    except Exception:
        return None

JSON_PATH = "./data/dataset.json"

def scrape_all_genres(json_path = "./data/dataset.json" , output_path: str = DEFAULT_OUTPUT, delay: float = DELAY, headless: bool = True):
    """
    Extracts all band names from dataset.json and builds a flat JSON file mapping band -> genre.
    """
    if not os.path.exists("./data/dataset.json"):
        print(f"Error: JSON de fils de pute file not found at {json_path}")
        return

    print(f"Loading dataset from {json_path}...")
    with open("./data/dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = data.get("dataset", data)

    # Charger un fichier de résultats existant s'il y en a un (pour reprendre là où on s'est arrêté)
    results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded existing progress from {output_path} ({len(results)} bands already processed).")
        except Exception:
            pass

    # Extraire la liste unique des noms de groupes depuis ton dataset
    band_names = []
    for artist_key, artist_data in dataset.items():
        name = artist_data.get("name", artist_key)
        if name not in band_names:
            band_names.append(name)

    total_bands = len(band_names)
    print(f"Found {total_bands} unique bands to process.")
    print("Initializing Firefox browser...")
    
    driver = init_selenium_driver(headless=headless)

    try:
        print("=" * 60)
        for i, band_name in enumerate(band_names, 1):
            # Si le groupe est déjà dans le fichier de résultats (même avec la valeur null), on saute
            if band_name in results:
                print(f"[{i}/{total_bands}] ⏭️ Already processed: {band_name} -> {results[band_name]}")
                continue

            print(f"[{i}/{total_bands}] 🔍 Scraping genre for: {band_name}...", end=' ', flush=True)

            genre = scrape_band_genre(driver, band_name)
            results[band_name] = genre

            if genre:
                print(f"✓ {genre}")
            else:
                print("✗ Not found (saved as null)")

            # Sauvegarde intermédiaire toutes les 10 requêtes
            if i % 10 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

            time.sleep(delay)

    finally:
        print("\nClosing browser...")
        driver.quit()

        # Sauvegarde finale propre au format plat
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✨ Flat JSON successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape flat band-to-genre JSON using Firefox")
    parser.add_argument("-f", "--filepath", default="dataset.json", help="Chemin vers ton dataset.json d'origine")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Fichier JSON de sortie plat")
    parser.add_argument("-d", "--delay", type=float, default=DELAY, help="Délai entre chaque requête")
    parser.add_argument("--no-headless", action="store_true", help="Afficher Firefox graphiquement")
    args = parser.parse_args()

    scrape_all_genres(json_path=args.filepath, output_path=args.output, delay=args.delay, headless=not args.no_headless)