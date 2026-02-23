import os
import json
import textwrap
import langextract as lx
from dotenv import load_dotenv
from langextract.providers.openai import OpenAILanguageModel


# ========= CONFIG =========

INPUT_FILE = r"C:\orchestration\data\drugs_clean_final\metformin_FULL.txt"
OUTPUT_FILE = r"C:\orchestration\data\dosage_json\metformin_EXTRACTIONS.json"

MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
BASE_URL = "https://api.groq.com/openai/v1"


# ========= LOAD ENV =========

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

model = OpenAILanguageModel(
    model_id=MODEL_ID,
    api_key=api_key,
    base_url=BASE_URL,
    temperature=0.1,
    max_concurrent_requests=1
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


# ========= MAIN =========

def run_extraction():

    print(" Reading file...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    print(" Running extraction...")

    result = lx.extract(
        text_or_documents=text,
        prompt_description=PROMPT,
        model=model,
        extraction_passes=1,
        examples=examples,
        max_char_buffer=4000,
        resolve=False
    )

    print(" Raw extraction count:", len(result.extractions))

    if not result.extractions:
        print(" No extractions found.")
        return

    clean_data = deduplicate_extractions(result.extractions)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2)

    print(" Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    run_extraction()