import os
import re
import json
import xml.etree.ElementTree as ET


def clean_text(text):
    if text:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return ""


def extract_full_document_from_spl(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ Failed parsing {xml_path}: {e}")
        return None

    namespace = {'ns': root.tag.split('}')[0].strip('{')}

    full_text_parts = []

    for section in root.findall('.//ns:section', namespace):

        title = section.find('.//ns:title', namespace)
        if title is not None and title.text:
            heading = clean_text(title.text)
            full_text_parts.append(f"\n\n=== {heading.upper()} ===\n")

        paragraphs = section.findall('.//ns:paragraph', namespace)
        for para in paragraphs:
            if para.text:
                full_text_parts.append(clean_text(para.text))

    return "\n".join(full_text_parts)


def process_spl_directory(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(input_dir) if f.endswith(".xml")]

    print(f"🔍 Found {len(xml_files)} XML files\n")

    for file_name in xml_files:
        xml_path = os.path.join(input_dir, file_name)

        print(f"⚙ Processing: {file_name}")

        full_text = extract_full_document_from_spl(xml_path)

        if not full_text:
            continue

        base_name = file_name.replace(".xml", "")

        # Save FULL text
        txt_path = os.path.join(output_dir, f"{base_name}_FULL.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        # Save META
        metadata = {
            "source_file": file_name,
            "document_length": len(full_text),
            "version": "raw_full_document_v1"
        }

        json_path = os.path.join(output_dir, f"{base_name}_META.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"   ✅ Saved: {base_name}")

    print("\n🎯 All files processed.")


# ====== USAGE ======

input_directory = r"C:\orchestration\data\drugs_structured"
output_directory = r"C:\orchestration\data\drugs_clean_final"

process_spl_directory(input_directory, output_directory)