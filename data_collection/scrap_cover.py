import os
import urllib.parse
import requests
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
    print("Erreur : Paquets Selenium Firefox manquants. Installez-les avec : pip install selenium webdriver-manager")
    raise

BASE_ALBUM_URL = "https://www.metal-archives.com/albums/"

def init_selenium_driver(headless=True):
    """
    Initialise Firefox pour contourner Cloudflare de la même manière.
    """
    firefox_options = Options()
    if headless:
        firefox_options.add_argument("--headless")
        
    # User-Agent réaliste pour Firefox
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
        
        # Visite initiale de la page d'accueil pour établir la session Cloudflare
        driver.get('https://www.metal-archives.com/')
        return driver
        
    except Exception as e:
        print(f"Erreur d'initialisation du driver Firefox : {e}")
        if driver:
            try:
                driver.quit()
            except:
                pass
        raise e

def format_album_url(artist_name: str, album_name: str) -> str:
    f_artist = artist_name.strip().replace(' ', '_')
    f_album = album_name.strip().replace(' ', '_')
    encoded_artist = urllib.parse.quote(f_artist, safe='_')
    encoded_album = urllib.parse.quote(f_album, safe='_')
    return f"{BASE_ALBUM_URL}{encoded_artist}/{encoded_album}/"

def get_metal_archives_cover(driver, artist_name, album_name):
    """
    Visite la page de l'album via Firefox, récupère l'URL de la cover via id="cover",
    et télécharge l'image en binaire.
    """
    url = format_album_url(artist_name, album_name)
    
    try:
        driver.get(url)
        
        wait = WebDriverWait(driver, 5)
        cover_element = wait.until(EC.presence_of_element_located((By.ID, "cover")))
        
        cover_url = cover_element.get_attribute("href")
        if not cover_url:
            return None
            
        selenium_cookies = {cookie['name']: cookie['value'] for cookie in driver.get_cookies()}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Referer": url
        }
        
        img_res = requests.get(cover_url, cookies=selenium_cookies, headers=headers, timeout=10)
        if img_res.status_code == 200:
            filename = cover_url.split("/")[-1].split("?")[0]
            mime_type = img_res.headers.get("Content-Type", "image/jpeg")
            return filename, mime_type, img_res.content
            
    except (TimeoutException, NoSuchElementException):
        print(f"      ⚠️ Cover introuvable pour '{album_name}'")
    except Exception as e:
        print(f"      ⚠️ Erreur lors du scraping de la cover : {e}")
        
    return None