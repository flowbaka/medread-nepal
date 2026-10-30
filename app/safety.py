"""
Safety & Verification Guardrails for MedRead Nepal.
Ensures uncertain OCR readings, blurry images, expired medicines, or
wrong medicine packages are strictly flagged for human verification.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from rapidfuzz import fuzz
from app.medicine_db import to_english_digits, to_nepali_digits

NEPALI_MONTHS_EN = {
    1: "जनवरी (Jan)",
    2: "फेब्रुअरी (Feb)",
    3: "मार्च (Mar)",
    4: "अप्रिल (Apr)",
    5: "मे (May)",
    6: "जुन (Jun)",
    7: "जुलाई (Jul)",
    8: "अगस्ट (Aug)",
    9: "सेप्टेम्बर (Sep)",
    10: "अक्टोबर (Oct)",
    11: "नोभेम्बर (Nov)",
    12: "डिसेम्बर (Dec)",
}

MONTH_NAME_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def parse_expiry_date(raw_text: str) -> Dict[str, Any]:
    """
    Extract Expiry Date from OCR text and check whether the medicine is expired.
    Supports formats like:
      - EXP: 08/2027, EXP. 08-27, EXPIRY DATE: FEB 2028, EXP 2027/05
      - म्याद समाप्त: २०८४/०५ or २०२७/०८
    """
    text_en = to_english_digits(raw_text)

    # Look specifically near EXP / EXPIRY / USE BEFORE / म्याद keywords first
    exp_patterns = [
        # EXP: FEB 2027 or EXP. Aug-2026
        r"(?:exp(?:iry)?(?:\s*date)?|use\s*before|म्याद)[\s:.\-]*([a-zA-Z]{3,9})[\s/\-\.,]*(\d{2,4})",
        # EXP: 08/2027 or EXP 08-27
        r"(?:exp(?:iry)?(?:\s*date)?|use\s*before|म्याद)[\s:.\-]*(\d{1,2})[\s/\-\.](\d{2,4})",
        # EXP: 2027/08 or 2027-08
        r"(?:exp(?:iry)?(?:\s*date)?|use\s*before|म्याद)[\s:.\-]*(\d{4})[\s/\-\.](\d{1,2})",
    ]

    month: Optional[int] = None
    year: Optional[int] = None
    matched_str: Optional[str] = None

    for idx, pat in enumerate(exp_patterns):
        m = re.search(pat, text_en, re.IGNORECASE)
        if m:
            matched_str = m.group(0).strip()
            g1, g2 = m.group(1), m.group(2)
            if idx == 0:
                month = MONTH_NAME_MAP.get(g1.lower()[:3])
                year = int(g2)
            elif idx == 1:
                month = int(g1)
                year = int(g2)
            elif idx == 2:
                year = int(g1)
                month = int(g2)
            break

    # Fallback: if multiple MM/YYYY dates appear (e.g., MFG 03/2025 and EXP 02/2028), pick the latest date
    if not year:
        all_dates = re.findall(r"\b(0?[1-9]|1[0-2])[\s/\-](20[2-3]\d)\b", text_en)
        if all_dates:
            parsed_pairs = [(int(m_str), int(y_str)) for m_str, y_str in all_dates]
            parsed_pairs.sort(key=lambda x: (x[1], x[0]))
            month, year = parsed_pairs[-1]
            matched_str = f"{month:02d}/{year}"

    if year and year < 100:
        year += 2000

    # Convert Nepali Bikram Sambat (BS) years (e.g., 2080-2095 BS) to Gregorian AD (~ -57 years)
    is_bs_calendar = False
    if year and 2075 <= year <= 2099:
        is_bs_calendar = True
        year_ad = year - 57
    else:
        year_ad = year

    if not year_ad or not month or not (1 <= month <= 12):
        return {
            "found": False,
            "raw": "Not clearly visible",
            "formatted_en": "Unclear / Not Found",
            "formatted_ne": "म्याद सकिने मिति स्पष्ट देखिएन",
            "is_expired": None,
            "expiring_soon": False,
        }

    now = datetime.now()
    # Medicine is valid through the end of its expiry month
    is_expired = (year_ad < now.year) or (year_ad == now.year and month < now.month)
    months_remaining = (year_ad - now.year) * 12 + (month - now.month)
    expiring_soon = (not is_expired) and (0 <= months_remaining <= 2)

    month_ne = NEPALI_MONTHS_EN.get(month, str(month))
    year_ne = to_nepali_digits(str(year_ad))
    formatted_en = f"{month:02d}/{year_ad}" + (f" (BS {year})" if is_bs_calendar else "")
    formatted_ne = f"सन् {year_ne} {month_ne}"

    return {
        "found": True,
        "raw": matched_str or formatted_en,
        "month": month,
        "year": year_ad,
        "formatted_en": formatted_en,
        "formatted_ne": formatted_ne,
        "is_expired": is_expired,
        "expiring_soon": expiring_soon,
    }


def evaluate_safety_and_match(
    med_name_en: str,
    strength_en: str,
    expiry_info: Dict[str, Any],
    ocr_confidence: float,
    blur_score: float,
    expected_medicine: Optional[Dict[str, str]] = None,
    brightness_score: float = 128.0,
) -> Dict[str, Any]:
    """
    Compute overall safety status:
      - 'SAFE' (Green): High confidence, clear expiry date (not expired), matches expected medicine (if set).
      - 'WARNING_VERIFY' (Yellow/Orange): Uncertain OCR, missing expiry date, or blurry photo -> Must verify with human.
      - 'DANGER' (Red): Expired medicine OR does NOT match the user's saved expected medicine!
    """
    flags_en: List[str] = []
    flags_ne: List[str] = []
    status = "SAFE"  # SAFE | WARNING_VERIFY | DANGER

    # 1. Check blurriness & lighting
    if blur_score < 45.0:
        flags_en.append("Photo appears blurry. Please hold the camera steady or ask someone to check.")
        flags_ne.append("तस्बिर धमिलो देखिएको छ। कृपया सफा तस्बिर खिच्नुहोस् वा परिवारको सदस्यलाई देखाउनुहोस्।")
        status = "WARNING_VERIFY"

    if brightness_score < 45.0:
        flags_en.append("Photo is too dark. Please turn on a light or move near a window.")
        flags_ne.append("तस्बिर अँध्यारो छ। कृपया उज्यालो ठाउँमा राखेर फोटो खिच्नुहोस्।")
        status = "WARNING_VERIFY"
    elif brightness_score > 242.0:
        flags_en.append("Strong glare or overexposure detected on the package.")
        flags_ne.append("प्याकेटमा धेरै टल्किएको (Glare) देखियो। कृपया थोरै ढल्काएर फोटो खिच्नुहोस्।")
        if status == "SAFE":
            status = "WARNING_VERIFY"

    # 2. Check medicine name & confidence
    if not med_name_en or med_name_en == "Unknown" or ocr_confidence < 0.68:
        flags_en.append("Uncertain reading: Medicine name or label is not 100% clear. Human verification required!")
        flags_ne.append("औषधिको नाम स्पष्ट बुझिएन। गलत औषधि पर्न सक्ने हुँदा कृपया अरू कसैलाई जाँच गर्न लगाउनुहोस्।")
        status = "WARNING_VERIFY"

    # 3. Check strength
    if not strength_en or strength_en == "Unknown":
        flags_en.append("Medicine dosage/strength (mg/ml) could not be confirmed clearly.")
        flags_ne.append("औषधिको मात्रा (मिलिग्राम) स्पष्ट देखिएन।")
        if status == "SAFE":
            status = "WARNING_VERIFY"

    # 4. Check Expiry Date
    if not expiry_info.get("found"):
        flags_en.append("Expiry date was not clearly visible on this side of the package.")
        flags_ne.append("म्याद सकिने मिति (Expiry Date) स्पष्ट देखिएन। कृपया प्याकेटको अर्को भाग हेर्नुहोस्।")
        if status == "SAFE":
            status = "WARNING_VERIFY"
    elif expiry_info.get("is_expired") is True:
        flags_en.append(f"DANGER: This medicine EXPIRED on {expiry_info['formatted_en']}! Do NOT take this medicine.")
        flags_ne.append(
            f"सावधान! यो औषधिको म्याद {expiry_info['formatted_ne']} मा सकिएको छ! यो औषधि नखानुहोस्।"
        )
        status = "DANGER"
    elif expiry_info.get("expiring_soon") is True:
        flags_en.append(f"Note: This medicine is expiring very soon ({expiry_info['formatted_en']}).")
        flags_ne.append(
            f"ध्यान दिनुहोस्: यो औषधिको म्याद छिट्टै ({expiry_info['formatted_ne']} मा) सकिँदैछ।"
        )
        if status == "SAFE":
            status = "WARNING_VERIFY"

    # 5. Compare with Saved Expected Medicine (if user selected one)
    comparison_result = None
    if expected_medicine:
        exp_brand = expected_medicine.get("brand_en", "")
        exp_generic = expected_medicine.get("generic_en", "")
        exp_strength = expected_medicine.get("default_strength", "")
        exp_name_ne = expected_medicine.get("name_ne", exp_brand)

        name_sim = max(
            fuzz.partial_ratio(med_name_en.lower(), exp_brand.lower()),
            fuzz.partial_ratio(med_name_en.lower(), exp_generic.lower()),
        )
        strength_match = (
            exp_strength.lower().replace(" ", "") in strength_en.lower().replace(" ", "")
            if strength_en and strength_en != "Unknown"
            else False
        )

        if name_sim >= 80 and strength_match:
            comparison_result = {
                "matched": True,
                "status": "MATCH",
                "message_en": f"Matches your saved medicine: {exp_brand} ({exp_strength}).",
                "message_ne": f"यो तपाईंले खोज्नुभएको {exp_name_ne} ({to_nepali_digits(exp_strength)}) औषधिसँग मेल खान्छ।",
            }
        elif name_sim >= 80 and not strength_match:
            comparison_result = {
                "matched": False,
                "status": "STRENGTH_MISMATCH",
                "message_en": f"Medicine name matches ({exp_brand}), but strength '{strength_en}' could not be matched to expected '{exp_strength}'!",
                "message_ne": f"औषधिको नाम मिले पनि यसको मात्रा ({to_nepali_digits(strength_en)}) तपाईंको नियमित मात्रा ({to_nepali_digits(exp_strength)}) सँग यकिन भएन!",
            }
            flags_en.append(comparison_result["message_en"])
            flags_ne.append(comparison_result["message_ne"])
            if status == "SAFE":
                status = "WARNING_VERIFY"
        else:
            comparison_result = {
                "matched": False,
                "status": "MISMATCH",
                "message_en": f"WRONG MEDICINE! Expected '{exp_brand} {exp_strength}', but scanned '{med_name_en} {strength_en}'.",
                "message_ne": f"सावधान! यो तपाईंले खोज्नुभएको '{exp_name_ne}' होइन! यो '{med_name_en}' हो। कृपया नखानुहोस् र परिवारलाई देखाउनुहोस्।",
            }
            flags_en.insert(0, comparison_result["message_en"])
            flags_ne.insert(0, comparison_result["message_ne"])
            status = "DANGER"

    return {
        "status": status,
        "requires_human_verification": status != "SAFE",
        "flags_en": flags_en,
        "flags_ne": flags_ne,
        "comparison": comparison_result,
    }
