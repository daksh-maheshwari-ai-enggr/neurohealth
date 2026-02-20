import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import time

BASE_API = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
SAVE_DIR = Path("../data/drugs_structured")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def get_setid(drug_name):
    url = f"{BASE_API}/spls.json?drug_name={drug_name}"
    r = requests.get(url, headers=HEADERS, timeout=15)

    if r.status_code != 200:
        print("Search failed:", r.status_code)
        return None

    data = r.json()

    if data.get("data"):
        return data["data"][0]["setid"]

    return None


def get_spl_xml(setid):
    url = f"{BASE_API}/spls/{setid}.xml"
    r = requests.get(url, headers=HEADERS, timeout=15)

    print("SPL STATUS:", r.status_code)

    if r.status_code != 200:
        print(r.text[:200])
        return None

    return r.text


def build_drug_file(drug_name):
    print(f"\n🔎 Processing: {drug_name}")

    setid = get_setid(drug_name)

    if not setid:
        print("No SETID found")
        return

    print("✔ SETID:", setid)

    xml_data = get_spl_xml(setid)

    if not xml_data:
        return

    output_file = SAVE_DIR / f"{drug_name}.xml"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_data)

    print("✅ Saved:", output_file)


PROTOTYPE_DRUGS = [
    "metformin",
    "glimepiride",
    "sitagliptin",
    "insulin",
    "amlodipine"
]

for drug in PROTOTYPE_DRUGS:
    build_drug_file(drug)
    time.sleep(1)

print("\n🚀 DailyMed XML ingestion complete.")