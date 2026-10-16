"""
Known Medicines Database & Saved Expected Medicines for Nepal.
Helps match noisy OCR text from blister packs/boxes and provides accurate
English-to-Nepali phonetic names and safety explanations.
"""

from typing import Dict, List, Optional
from rapidfuzz import fuzz, process

# Common medicines found in Nepal (boxes and blister packs)
KNOWN_MEDICINES: List[Dict[str, str]] = [
    {
        "id": "cetamol-500",
        "brand_en": "Cetamol",
        "generic_en": "Paracetamol",
        "name_ne": "सिटामोल (प्यारासिटामोल)",
        "default_strength": "500 mg",
        "strength_ne": "५०० मिलिग्राम",
        "purpose_ne": "ज्वरो आउँदा र टाउको वा शरीर दुख्दा खाने औषधि",
        "purpose_en": "Fever and pain relief",
        "keywords": ["cetamol", "paracetamol", "सिटामोल", "प्यारासिटामोल", "napa", "calpol", "flexon"],
    },
    {
        "id": "amlod-5",
        "brand_en": "Amlod",
        "generic_en": "Amlodipine",
        "name_ne": "एम्लोड (एम्लोडिपिन)",
        "default_strength": "5 mg",
        "strength_ne": "५ मिलिग्राम",
        "purpose_ne": "उच्च रक्तचाप (प्रेसर) नियन्त्रण गर्ने औषधि",
        "purpose_en": "High blood pressure (Hypertension)",
        "keywords": ["amlod", "amlodipine", "एम्लोड", "एम्लोडिपिन", "stamlo", "amlong"],
    },
    {
        "id": "metfor-500",
        "brand_en": "Metfor",
        "generic_en": "Metformin",
        "name_ne": "मेटफोर (मेटफर्मिन)",
        "default_strength": "500 mg",
        "strength_ne": "५०० मिलिग्राम",
        "purpose_ne": "सुगर (मधुमेह) नियन्त्रण गर्ने औषधि",
        "purpose_en": "Diabetes (Blood sugar control)",
        "keywords": ["metfor", "metformin", "glycomet", "मेटफोर", "मेटफर्मिन"],
    },
    {
        "id": "pantop-40",
        "brand_en": "Pantop",
        "generic_en": "Pantoprazole",
        "name_ne": "प्यान्टोप (प्यान्टोप्राजोल)",
        "default_strength": "40 mg",
        "strength_ne": "४० मिलिग्राम",
        "purpose_ne": "ग्यास्ट्रिक र पेट पोल्ने समस्याको औषधि",
        "purpose_en": "Gastritis and acid reflux",
        "keywords": ["pantop", "pantoprazole", "pan-40", "प्यान्टोप", "aciloc", "omeprazole"],
    },
    {
        "id": "azithro-500",
        "brand_en": "Azithro",
        "generic_en": "Azithromycin",
        "name_ne": "एजिथ्रो (एजिथ्रोमाइसिन)",
        "default_strength": "500 mg",
        "strength_ne": "५०० मिलिग्राम",
        "purpose_ne": "घाँटी दुख्ने र संक्रमणको एन्टिबायोटिक औषधि",
        "purpose_en": "Bacterial infection antibiotic",
        "keywords": ["azithro", "azithromycin", "azee", "एजिथ्रो", "zithromax"],
    },
    {
        "id": "losar-50",
        "brand_en": "Losar",
        "generic_en": "Losartan Potassium",
        "name_ne": "लोसार (लोसार्टन)",
        "default_strength": "50 mg",
        "strength_ne": "५० मिलिग्राम",
        "purpose_ne": "उच्च रक्तचाप (प्रेसर) को औषधि",
        "purpose_en": "Blood pressure medication",
        "keywords": ["losar", "losartan", "लोसार", "लोसार्टन"],
    },
]

# English digits to Nepali (Devanagari) digits mapping
EN_TO_NE_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")
NE_TO_EN_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def to_nepali_digits(text: str) -> str:
    """Convert English numerals (0-9) into Nepali Devanagari numerals (०-९)."""
    return text.translate(EN_TO_NE_DIGITS)


def to_english_digits(text: str) -> str:
    """Convert Nepali Devanagari numerals (०-९) into English numerals (0-9)."""
    return text.translate(NE_TO_EN_DIGITS)


def strength_to_nepali(strength_en: str) -> str:
    """Convert English strength like '500 mg' or '10 ml' into natural Nepali."""
    if not strength_en or strength_en == "Unknown":
        return "मात्रा स्पष्ट नभएको"
    s = strength_en.lower().strip()
    s = s.replace("mcg", " माइक्रोग्राम")
    s = s.replace("mg", " मिलिग्राम")
    s = s.replace("ml", " मिलिलिटर")
    s = s.replace("gm", " ग्राम")
    s = s.replace("g", " ग्राम")
    return to_nepali_digits(s).strip()


def match_known_medicine(raw_text: str) -> Optional[Dict]:
    """
    Match OCR text against our Nepal medicine dictionary using exact substring
    or fuzzy matching (RapidFuzz) so minor OCR typos (e.g. 'Cetamo1' or 'Paracetam0l')
    are still recognized along with a similarity score.
    """
    if not raw_text:
        return None

    cleaned = raw_text.lower()
    best_match = None
    best_score = 0.0

    # 1. Direct keyword search first
    for med in KNOWN_MEDICINES:
        for kw in med["keywords"]:
            if kw in cleaned:
                return {**med, "match_score": 98.0, "matched_keyword": kw}

    # 2. Fuzzy token matching across lines/words
    tokens = [
        w.strip(".,:;()[]{}|-_/")
        for w in cleaned.split()
        if len(w.strip(".,:;()[]{}|-_/")) >= 4
    ]
    for med in KNOWN_MEDICINES:
        for kw in med["keywords"]:
            for token in tokens:
                score = fuzz.ratio(token, kw)
                if score > best_score:
                    best_score = score
                    best_match = med

    if best_match and best_score >= 78.0:
        return {**best_match, "match_score": round(best_score, 1)}

    return None
