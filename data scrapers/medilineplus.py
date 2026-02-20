import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

SAVE_DIR = Path("../data/tests")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TESTS = {
    "hba1c": "https://medlineplus.gov/lab-tests/hemoglobin-a1c-hba1c-test/",
    "fasting_glucose": "https://medlineplus.gov/lab-tests/blood-glucose-test/",
    "creatinine": "https://medlineplus.gov/lab-tests/creatinine-test/",
    "lipid_panel": "https://medlineplus.gov/lab-tests/lipid-panel/",
    "tsh": "https://medlineplus.gov/lab-tests/thyroid-stimulating-hormone-tsh-test/"
}

headers = {"User-Agent": "Mozilla/5.0"}

def clean_page(url):
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    article = soup.find("div", class_="main")
    if article:
        return article.get_text(separator="\n")
    return ""

for test, url in TESTS.items():
    print("Fetching:", test)
    text = clean_page(url)

    with open(SAVE_DIR / f"{test}.txt", "w", encoding="utf-8") as f:
        f.write(text)

    time.sleep(1)

print("Lab test data ready")