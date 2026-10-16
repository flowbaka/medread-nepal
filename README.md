# MedRead Nepal: Medicine Package and Label Reader 💊🇳🇵

An assistive Computer Vision & Nepali Voice web application built with **FastAPI**, **OpenCV**, **EasyOCR (English + Devanagari/Nepali)**, **RapidFuzz**, and **Google Text-to-Speech (`gTTS`)** to help older adults and people with low vision safely identify medicine boxes and blister packs.

---

## ✨ Key Features
1. **Text Detection & Bilingual OCR (`app/ocr_engine.py`)**:
   - Uses OpenCV **CLAHE** contrast enhancement to reduce foil glare on blister packs.
   - Computes a **Laplacian Blur Score** to warn users if the photo is out of focus.
   - Extracts **Medicine Name**, **Strength (`mg`/`ml`)**, and **Expiry Date** in English and Nepali.
2. **English-to-Nepali Spoken Audio (`app/tts_engine.py`)**:
   - Translates English medicine names, strengths (`500 mg` → `५०० मिलिग्राम`), and expiry dates (`08/2028` → `सन् २०२८ अगस्ट`) into natural Nepali and reads them aloud.
3. **Expected Medicine Package Comparison (`app/safety.py`)**:
   - Users or caregivers can select their expected daily medicine (e.g., *Amlod 5 mg* for blood pressure). If they accidentally scan a different medicine (e.g., *Cetamol*), a loud **DANGER MISMATCH** warning is spoken in Nepali.
4. **Safety Guardrails for Uncertain Readings**:
   - Automatically flags **Expired medicines**, **Blurry photos**, or **Low-confidence OCR readings** for human verification.

---

## 🚀 How to Run the Project

### 1. Open a Terminal in the Project Folder
```powershell
cd C:\Users\kshit\.gemini\antigravity\scratch\medread-nepal
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

### 4. Open in Browser
Visit **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.
