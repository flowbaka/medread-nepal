"""
Computer Vision & OCR Engine for MedRead Nepal.
Performs:
  1. Image Preprocessing & Glare/Contrast Enhancement (OpenCV CLAHE)
  2. Blur Detection (Laplacian Variance)
  3. Text Detection & Bilingual OCR (EasyOCR ['en', 'ne'] + optional Gemini Vision API)
  4. Entity Extraction (Medicine Name, Strength, Expiry Date) & Package Matching
"""

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.medicine_db import (
    match_known_medicine,
    strength_to_nepali,
    to_english_digits,
    to_nepali_digits,
)
from app.safety import evaluate_safety_and_match, parse_expiry_date
from app.tts_engine import build_spoken_messages, generate_nepali_audio

# Lazy-loaded EasyOCR reader singleton so server starts immediately
_EASYOCR_READER = None


def get_easyocr_reader():
    """Lazy-initialize EasyOCR with English ('en') and Nepali/Devanagari ('ne')."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        print("[MedRead OCR] Loading EasyOCR models for English ('en') & Nepali ('ne')...")
        _EASYOCR_READER = easyocr.Reader(["en", "ne"], gpu=False)
    return _EASYOCR_READER


def assess_and_preprocess_image(image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Decode image, compute Laplacian blur score and mean brightness score,
    and apply bilateral filtering + CLAHE contrast enhancement to reduce
    foil glare on metallic blister packs.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Invalid image file.")

    # Resize if overly large for faster OCR
    h, w = img_bgr.shape[:2]
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_score = float(np.mean(gray))

    # Mild edge-preserving denoising for shiny foil grain before CLAHE
    denoised_gray = cv2.bilateralFilter(gray, d=5, sigmaColor=45, sigmaSpace=45)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to reduce foil glare
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(denoised_gray)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    return img_bgr, enhanced_bgr, round(blur_score, 1), round(brightness_score, 1)


def extract_batch_number(raw_text: str) -> Optional[str]:
    """Extract Batch / Lot number (e.g., B.No: NP-402, BATCH: AM99) from label text."""
    text_en = to_english_digits(raw_text)
    pattern = r"(?:b\.?\s*no\.?|batch(?:\s*no\.?)?|lot)[\s:.\-]*([A-Z0-9\-]{3,14})"
    m = re.search(pattern, text_en, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def extract_strength(raw_text: str) -> Optional[str]:
    """Extract dosage strength such as '500 mg', '5 mg', '40mg', '10 ml', '५०० मि.ग्रा.'."""
    text_en = to_english_digits(raw_text)
    # Match patterns like 500mg, 500 mg, 5 mg/5 ml, 0.5 mg, 40 MG
    pattern = r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|ml|gm|g|मि\.?ग्रा\.?|मिलिग्राम)\b"
    matches = re.findall(pattern, text_en, flags=re.IGNORECASE)
    if matches:
        num, unit = matches[0]
        unit_clean = "mg" if "मि" in unit else unit.lower()
        return f"{num} {unit_clean}"
    return None


def annotate_image_base64(img_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> str:
    """Draw high-contrast bounding boxes on the image and return as a base64 data URI."""
    canvas = img_bgr.copy()
    for det in detections:
        bbox = det.get("bbox")
        conf = det.get("confidence", 0.0)
        if bbox and len(bbox) == 4:
            pts = np.array(bbox, dtype=np.int32)
            # Green for high confidence, Yellow/Orange for low confidence
            color = (0, 210, 80) if conf >= 0.68 else (0, 165, 255)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3)

    _, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def run_easyocr_extraction(img_bgr: np.ndarray, enhanced_bgr: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
    """Run EasyOCR on the image and return full text, average confidence, and bounding boxes."""
    reader = get_easyocr_reader()
    results = reader.readtext(enhanced_bgr)

    lines: List[str] = []
    confidences: List[float] = []
    detections: List[Dict[str, Any]] = []

    for bbox, text, conf in results:
        text_clean = text.strip()
        if not text_clean:
            continue
        lines.append(text_clean)
        confidences.append(float(conf))
        detections.append(
            {
                "bbox": [[int(pt[0]), int(pt[1])] for pt in bbox],
                "text": text_clean,
                "confidence": round(float(conf), 2),
            }
        )

    full_text = "\n".join(lines)
    avg_conf = round(float(np.mean(confidences)), 2) if confidences else 0.0
    return full_text, avg_conf, detections


def process_medicine_image(
    image_bytes: bytes,
    expected_medicine: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Full Computer Vision pipeline:
      1. Preprocess & blur check
      2. OCR text detection & recognition
      3. Package matching & entity extraction (Name, Strength, Expiry)
      4. Safety & human verification evaluation
      5. Nepali voice generation
    """
    img_bgr, enhanced_bgr, blur_score, brightness_score = assess_and_preprocess_image(image_bytes)

    raw_text, avg_conf, detections = run_easyocr_extraction(img_bgr, enhanced_bgr)
    annotated_img_uri = annotate_image_base64(img_bgr, detections)

    # Match against known Nepal medicine database
    matched_med = match_known_medicine(raw_text)
    extracted_strength = extract_strength(raw_text)
    batch_number = extract_batch_number(raw_text)
    expiry_info = parse_expiry_date(raw_text)

    if matched_med:
        name_en = f"{matched_med['brand_en']} ({matched_med['generic_en']})"
        name_ne = matched_med["name_ne"]
        purpose_ne = matched_med["purpose_ne"]
        purpose_en = matched_med["purpose_en"]
        strength_en = extracted_strength or matched_med["default_strength"]
        strength_ne = strength_to_nepali(strength_en)
        # Boost confidence slightly if dictionary matched strongly, but still respect low OCR conf
        effective_conf = max(avg_conf, round(matched_med["match_score"] / 100.0 * 0.85, 2))
    else:
        # Fallback: pick largest/first capitalized word if unknown medicine
        candidates = [
            line.strip()
            for line in raw_text.splitlines()
            if len(line.strip()) >= 4 and not re.search(r"(mfg|exp|batch|b\.no|mrp|ltd)", line, re.I)
        ]
        name_en = candidates[0] if candidates else "Unknown"
        name_ne = name_en if name_en != "Unknown" else "अज्ञात औषधि (नाम स्पष्ट नभएको)"
        purpose_ne = ""
        purpose_en = "Not in verified medicine database"
        strength_en = extracted_strength or "Unknown"
        strength_ne = strength_to_nepali(strength_en)
        effective_conf = min(avg_conf, 0.58)  # Force human verification if unknown medicine

    safety_info = evaluate_safety_and_match(
        med_name_en=name_en,
        strength_en=strength_en,
        expiry_info=expiry_info,
        ocr_confidence=effective_conf,
        blur_score=blur_score,
        expected_medicine=expected_medicine,
    )

    spoken = build_spoken_messages(
        name_en=name_en,
        name_ne=name_ne,
        strength_en=strength_en,
        strength_ne=strength_ne,
        purpose_ne=purpose_ne,
        expiry_info=expiry_info,
        safety_info=safety_info,
    )

    audio_url = generate_nepali_audio(spoken["speech_ne"])

    return {
        "medicine_name_en": name_en,
        "medicine_name_ne": name_ne,
        "strength_en": strength_en,
        "strength_ne": strength_ne,
        "purpose_en": purpose_en,
        "purpose_ne": purpose_ne,
        "expiry": expiry_info,
        "batch_number": batch_number or "N/A",
        "ocr_confidence": round(effective_conf * 100, 1),
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "raw_ocr_text": raw_text,
        "detections_count": len(detections),
        "annotated_image": annotated_img_uri,
        "safety": safety_info,
        "speech_ne": spoken["speech_ne"],
        "speech_en": spoken["speech_en"],
        "audio_url": audio_url,
    }
