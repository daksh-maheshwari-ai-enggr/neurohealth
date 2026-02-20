import os
import json
import time
import langextract as lx
from dotenv import load_dotenv
from langextract.providers.openai import OpenAILanguageModel




INPUT_DIR = r"C:\orchestration\data\drugs_clean_final"
OUTPUT_DIR = r"C:\orchestration\data\dosage_json"

MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
BASE_URL = "https://api.groq.com/openai/v1"

EXTRACTION_PASSES = 1      
COOLDOWN_SECONDS = 10   
MAX_RETRIES = 5
 



load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

model = OpenAILanguageModel(
    model_id=MODEL_ID,
    api_key=api_key,
    base_url=BASE_URL,
    temperature=0.1,
)




import textwrap

PROMPT = textwrap.dedent("""
You are a clinical-grade medical information extraction system.

Your task is STRICT span extraction from the provided drug document.

  ZERO HALLUCINATION POLICY:
- Extract ONLY phrases that appear EXACTLY in the text.
- NEVER generate medical information.
- NEVER infer.
- NEVER paraphrase.
- NEVER complete missing information.
- If a class is not present in the text → DO NOT extract anything for it.
- Do NOT guess typical doses.
- Do NOT use medical knowledge outside the document.

If information is not explicitly written, IGNORE it.

This is span extraction, NOT summarization.

━━━━━━━━━━━━━━━━━━━━━━
EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━

• Extract smallest medically meaningful phrase.
• Avoid full sentences when shorter phrase works.
• Avoid leading/trailing spaces.
• Do not overlap extractions.
• Do not merge separate items.
• Split list items into separate extractions.
• Keep original casing.
• Do not normalize units.
• Do not expand abbreviations.
• Do not modify punctuation.
• Do not explain anything.

If uncertain → DO NOT extract.

━━━━━━━━━━━━━━━━━━━━━━
CLASSES (USE ONLY THESE)
━━━━━━━━━━━━━━━━━━━━━━

1. indication  
   Approved medical use of the drug.

2. adult_dosage  
   Adult dosing instructions.

3. pediatric_dosage  
   Pediatric dosing instructions.

4. max_dose  
   Maximum dose limits.

5. contraindication  
   Situations where drug must not be used.

6. warning  
   Safety warnings or precautions.

7. boxed_warning  
   Black box warning text.

8. adverse_reaction  
   Side effects.

9. drug_interaction  
   Interactions with other drugs.

10. renal_adjustment  
    Dose adjustments in renal impairment.

11. hepatic_adjustment  
    Dose adjustments in hepatic impairment.

12. pregnancy_info  
    Pregnancy or lactation related information.

━━━━━━━━━━━━━━━━━━━━━━
CRITICAL SAFETY CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━

• If a dosage number is not explicitly present → DO NOT invent one.
• If pediatric dosing not mentioned → extract nothing.
• If pregnancy not mentioned → extract nothing.
• If boxed warning not clearly present → extract nothing.
• Never assume a drug has common side effects.
• Never assume class effects.

Only extract text that is directly supported by visible text in the document.

Output ONLY structured extractions.
No explanations.
No commentary.
No summary.
""")



import langextract as lx

examples = [

    lx.data.ExampleData(
        text="""
INDICATIONS AND USAGE:
Metformin is indicated for the treatment of type 2 diabetes mellitus.

DOSAGE AND ADMINISTRATION:
The recommended starting dose is 500 mg twice daily with meals.
Maximum recommended dose is 2000 mg per day.

CONTRAINDICATIONS:
Severe renal impairment.

WARNINGS:
Lactic acidosis has been reported.
""",
        extractions=[

            lx.data.Extraction(
                extraction_class="indication",
                extraction_text="type 2 diabetes mellitus",
                attributes={}
            ),

            lx.data.Extraction(
                extraction_class="adult_dosage",
                extraction_text="500 mg twice daily",
                attributes={}
            ),

            lx.data.Extraction(
                extraction_class="max_dose",
                extraction_text="2000 mg per day",
                attributes={}
            ),

            lx.data.Extraction(
                extraction_class="contraindication",
                extraction_text="Severe renal impairment",
                attributes={}
            ),

            lx.data.Extraction(
                extraction_class="warning",
                extraction_text="Lactic acidosis",
                attributes={}
            ),
        ]
    ),


    # NEGATIVE EXAMPLE (IMPORTANT FOR GUARDRAIL)

    lx.data.ExampleData(
        text="""
INDICATIONS:
This medication is used to treat hypertension.

DOSAGE:
Take as directed by your physician.
""",
        extractions=[

            lx.data.Extraction(
                extraction_class="indication",
                extraction_text="hypertension",
                attributes={}
            )

            
        ]
    )
]








def deduplicate_extractions(extractions):
    seen = set()
    clean = []

    for ext in extractions:
        key = (ext.extraction_class, ext.extraction_text.strip())
        if key not in seen:
            seen.add(key)
            clean.append({
                "class": ext.extraction_class,
                "text": ext.extraction_text.strip()
            })

    return clean


def safe_extract(text):
    delay = 5  

    for attempt in range(MAX_RETRIES):
        try:
            result = lx.extract(
                text_or_documents=text,
                prompt_description=PROMPT,
                examples=examples,
                model=model,
                extraction_passes=1  
            )
            return result

        except Exception as e:
            error_str = str(e).lower()

            if "rate_limit" in error_str or "429" in error_str:
                print(f"   Rate limit hit. Sleeping {delay}s...")
                time.sleep(delay)
                delay *= 2   
            else:
                raise e

    raise Exception("Max retries exceeded.")




def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(input_dir) if f.endswith("_FULL.txt")]
    print(f" Found {len(txt_files)} documents\n")

    for index, file_name in enumerate(txt_files):
        file_path = os.path.join(input_dir, file_name)
        print(f" Processing ({index+1}/{len(txt_files)}): {file_name}")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        try:
            result = safe_extract(text)

            print("    Raw extraction count:", len(result.extractions))

            if not result.extractions:
                print("    No extractions found.\n")
                time.sleep(COOLDOWN_SECONDS)
                continue

            if not result.extractions:
                print("    No extractions found.\n")
                time.sleep(COOLDOWN_SECONDS)
                continue

            clean_data = deduplicate_extractions(result.extractions)

            output_path = os.path.join(
                output_dir,
                file_name.replace("_FULL.txt", "_EXTRACTIONS.json")
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(clean_data, f, indent=2)

            print("    Saved\n")

        
            print(f"   Cooling down {COOLDOWN_SECONDS}s...\n")
            time.sleep(COOLDOWN_SECONDS)

        except Exception as e:
            print(f"    Error: {e}\n")
            print("    Cooling before next file...\n")
            time.sleep(COOLDOWN_SECONDS)

    print(" All files processed.")