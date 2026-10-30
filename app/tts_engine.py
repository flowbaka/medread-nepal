"""
English-to-Nepali Voice Synthesis Engine (TTS) for MedRead Nepal.
Builds natural Nepali spoken instructions and generates MP3 audio files
so older adults and low-vision users hear the reading aloud.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict
from gtts import gTTS

AUDIO_DIR = Path(__file__).resolve().parent.parent / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def build_spoken_messages(
    name_en: str,
    name_ne: str,
    strength_en: str,
    strength_ne: str,
    purpose_ne: str,
    expiry_info: Dict[str, Any],
    safety_info: Dict[str, Any],
) -> Dict[str, str]:
    """
    Construct clear, slow-paced Nepali and English spoken scripts.
    Prioritizes critical safety warnings first so the user hears them immediately.
    """
    parts_ne = ["नमस्कार। "]
    parts_en = []

    # 1. If there is a critical DANGER (e.g. wrong medicine or expired), speak warning FIRST
    if safety_info["status"] == "DANGER":
        parts_ne.append("विशेष सावधानी! ")
        parts_en.append("Important Warning! ")

    # 2. Medicine Name & Strength
    if name_en and name_en != "Unknown":
        parts_ne.append(f"यो औषधिको नाम {name_ne} हो। ")
        parts_en.append(f"This medicine is {name_en}. ")
        if strength_en and strength_en != "Unknown":
            parts_ne.append(f"यसको मात्रा {strength_ne} छ। ")
            parts_en.append(f"Strength is {strength_en}. ")
        if purpose_ne:
            parts_ne.append(f"यो {purpose_ne} हो। ")
    else:
        parts_ne.append("यो औषधिको नाम स्पष्ट पढ्न सकिएन। ")
        parts_en.append("The medicine name could not be read clearly. ")

    # 3. Expiry Date
    if expiry_info.get("found"):
        if expiry_info.get("is_expired"):
            parts_ne.append(
                f"सावधान! यो औषधिको म्याद {expiry_info['formatted_ne']} मा सकिइसकेको छ। कृपया यो औषधि नखानुहोस्! "
            )
            parts_en.append(
                f"Danger! This medicine expired on {expiry_info['formatted_en']}. Do not take it! "
            )
        else:
            parts_ne.append(f"यसको म्याद {expiry_info['formatted_ne']} सम्म छ। ")
            parts_en.append(f"It is valid until {expiry_info['formatted_en']}. ")
            if expiry_info.get("expiring_soon"):
                parts_ne.append("ध्यान दिनुहोस्, यसको म्याद छिट्टै सकिँदैछ। ")
                parts_en.append("Note that this medicine is expiring soon. ")
    else:
        parts_ne.append("प्याकेटमा म्याद सकिने मिति स्पष्ट देखिएन। ")
        parts_en.append("The expiry date was not clearly visible. ")

    # 4. Expected Medicine Match / Safety warnings
    comp = safety_info.get("comparison")
    if comp:
        parts_ne.append(comp["message_ne"] + " ")
        parts_en.append(comp["message_en"] + " ")

    if safety_info["requires_human_verification"] and safety_info["status"] != "DANGER":
        parts_ne.append("कृपया औषधि खानुअघि परिवारको सदस्य वा स्वास्थ्यकर्मीलाई एकपटक देखाउनुहोस्।")
        parts_en.append("Please ask a family member or pharmacist to verify before taking.")

    return {
        "speech_ne": "".join(parts_ne).strip(),
        "speech_en": "".join(parts_en).strip(),
    }


def generate_nepali_audio(text_ne: str) -> str:
    """
    Generate Nepali MP3 audio using Google Text-to-Speech (gTTS).
    Caches by hash so repeated readings are instant.
    Returns the relative URL `/static/audio/<hash>.mp3` or empty string if offline.
    """
    if not text_ne:
        return ""
    file_hash = hashlib.md5(text_ne.encode("utf-8")).hexdigest()[:12]
    filename = f"med_{file_hash}.mp3"
    filepath = AUDIO_DIR / filename

    if filepath.exists():
        return f"/static/audio/{filename}"

    try:
        tts = gTTS(text=text_ne, lang="ne", slow=False)
        tts.save(str(filepath))
        return f"/static/audio/{filename}"
    except Exception as exc:
        print(f"[TTS Warning] Could not reach gTTS online service, falling back to browser TTS: {exc}")
        return ""
