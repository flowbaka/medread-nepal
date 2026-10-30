"""
FastAPI Backend for MedRead Nepal: Medicine Package and Label Reader.
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.medicine_db import KNOWN_MEDICINES
from app.ocr_engine import process_medicine_image
from app.sample_generator import SAMPLES_DIR, generate_demo_samples

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="MedRead Nepal API",
    description="Assistive Medicine Package & Label Reader for Older Adults and Low-Vision Users in Nepal",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.on_event("startup")
async def startup_event():
    generate_demo_samples()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the high-contrast, low-vision accessible frontend."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "medicines": KNOWN_MEDICINES,
        },
    )


@app.get("/api/medicines")
async def get_known_medicines():
    """Return known/saved medicines for package comparison."""
    return {"medicines": KNOWN_MEDICINES}


@app.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "medicines_loaded": len(KNOWN_MEDICINES)}


@app.post("/api/medicines")
async def add_custom_expected_medicine(
    brand_en: str = Form(...),
    name_ne: str = Form(...),
    default_strength: str = Form("500 mg"),
    purpose_ne: str = Form("नियमित खाने औषधि"),
):
    """Allow a caregiver to save a custom prescription medicine for comparison."""
    med_id = brand_en.strip().lower().replace(" ", "-")
    new_med = {
        "id": med_id,
        "brand_en": brand_en.strip(),
        "generic_en": brand_en.strip(),
        "name_ne": name_ne.strip(),
        "default_strength": default_strength.strip(),
        "strength_ne": default_strength.strip(),
        "purpose_ne": purpose_ne.strip(),
        "purpose_en": "Custom saved prescription",
        "keywords": [brand_en.strip().lower(), name_ne.strip()],
    }
    KNOWN_MEDICINES.append(new_med)
    return {"status": "added", "medicine": new_med}


@app.post("/api/scan")
async def scan_medicine_package(
    file: UploadFile = File(...),
    expected_medicine_id: Optional[str] = Form(None),
):
    """
    Process a photographed medicine box or blister pack:
      - Runs OpenCV blur/glare preprocessing + EasyOCR
      - Extracts Name, Strength, and Expiry Date
      - Compares against expected_medicine_id if provided
      - Generates Nepali spoken audio (MP3)
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image uploaded.")

    expected_med = None
    if expected_medicine_id:
        expected_med = next((m for m in KNOWN_MEDICINES if m["id"] == expected_medicine_id), None)

    try:
        result = process_medicine_image(image_bytes, expected_medicine=expected_med)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(exc)}")


@app.post("/api/scan-sample/{sample_name}")
async def scan_sample_package(
    sample_name: str,
    expected_medicine_id: Optional[str] = Form(None),
):
    """Run the full CV/OCR pipeline on one of the pre-generated sample medicine labels."""
    sample_map = {
        "cetamol": SAMPLES_DIR / "cetamol_500_valid.jpg",
        "amlod": SAMPLES_DIR / "amlod_5_valid.jpg",
        "expired": SAMPLES_DIR / "pantop_40_expired.jpg",
        "blurry": SAMPLES_DIR / "blurry_package.jpg",
        "atorva": SAMPLES_DIR / "atorva_10_bs.jpg",
    }
    sample_path = sample_map.get(sample_name)
    if not sample_path or not sample_path.exists():
        generate_demo_samples()
    if not sample_path or not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found.")

    expected_med = None
    if expected_medicine_id:
        expected_med = next((m for m in KNOWN_MEDICINES if m["id"] == expected_medicine_id), None)

    image_bytes = sample_path.read_bytes()
    result = process_medicine_image(image_bytes, expected_medicine=expected_med)
    return JSONResponse(content=result)
