# MedRead Nepal: Medicine Package and Label Reader 💊🇳🇵

An assistive Computer Vision & Nepali Voice web application built with **FastAPI**, **OpenCV**, **EasyOCR (English + Devanagari/Nepali)**, **RapidFuzz**, and **Google Text-to-Speech (`gTTS`)** to help older adults and people with low vision safely identify medicine boxes and blister packs.

---

## ✨ Key Features
1. **Text Detection & Bilingual OCR (`app/ocr_engine.py`)**:
   - Uses OpenCV **Bilateral Filtering + CLAHE** contrast enhancement to reduce foil glare on metallic blister packs.
   - Computes a **Laplacian Blur Score** and **Brightness Score** to warn users if the photo is out of focus or too dark.
   - Extracts **Medicine Name**, **Strength (`mg`/`ml`)**, **Batch Number (`B.No`)**, and **Expiry Date** (both Gregorian AD and Nepali Bikram Sambat BS calendars).
2. **English-to-Nepali Spoken Audio (`app/tts_engine.py`)**:
   - Translates English medicine names, strengths (`500 mg` → `५०० मिलिग्राम`), and expiry dates (`08/2028` → `सन् २०२८ अगस्ट`) into natural Nepali and reads them aloud, with a **Slow Playback (`0.75x`)** button for elderly listeners.
3. **Expected Medicine Package Comparison (`app/safety.py`)**:
   - Users or caregivers can select or add their expected daily medicine (e.g., *Amlod 5 mg* for blood pressure). If they accidentally scan a different medicine (e.g., *Cetamol*), a loud **DANGER MISMATCH** warning is spoken in Nepali.
4. **Safety Guardrails for Uncertain Readings**:
   - Automatically flags **Expired medicines**, **Near-expiry medicines (≤ 2 months)**, **Blurry photos**, or **Low-confidence OCR readings** for human verification.
5. **Senior-Friendly Accessibility (`templates/index.html`)**:
   - Includes **Font Zoom (`A+` / `A-`)**, **High-Contrast Yellow-on-Black mode**, **Recent Scan History**, and 5 instant demo medicine labels.

---

## 🚀 How to Run the Project

### 1. Open a Terminal in the Project Folder
```powershell
cd C:\Users\kshit\Documents\Dashain_34_Projects\medread-nepal
```

### 2. Create a Virtual Environment & Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start the FastAPI Server
```powershell
uvicorn app.main:app --reload --port 8000
```

### 4. Run Unit Tests
```powershell
python -m unittest discover -s tests
```

### 5. Open in Browser
Visit **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.
