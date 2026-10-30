"""
Generates realistic synthetic medicine box/blister pack images for immediate testing
and classroom/project demonstrations (Valid, Expired, Mismatched, and Blurry samples).
"""

from pathlib import Path
import cv2
import numpy as np

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "static" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def generate_demo_samples() -> None:
    """Create 4 realistic test medicine package images if they don't exist yet."""
    # 1. Cetamol 500 mg (Clear, Valid)
    img1 = np.full((420, 700, 3), (245, 248, 252), dtype=np.uint8)
    cv2.rectangle(img1, (20, 20), (680, 400), (180, 70, 30), 4)
    cv2.rectangle(img1, (20, 20), (680, 95), (180, 70, 30), -1)
    cv2.putText(img1, " Rx  PARACETAMOL TABLETS IP", (35, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    cv2.putText(img1, "CETAMOL 500 mg", (45, 175), cv2.FONT_HERSHEY_DUPLEX, 1.85, (20, 30, 140), 4)
    cv2.putText(img1, "Each uncoated tablet contains: Paracetamol 500 mg", (45, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2)
    cv2.putText(img1, "B.No: NP-402   MFG: 08/2025   EXP: 08/2028", (45, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (10, 10, 10), 2)
    cv2.putText(img1, "Manufactured in Nepal", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 2)
    cv2.imwrite(str(SAMPLES_DIR / "cetamol_500_valid.jpg"), img1)

    # 2. Amlod 5 mg (Blood pressure medicine, Valid)
    img2 = np.full((420, 700, 3), (242, 250, 242), dtype=np.uint8)
    cv2.rectangle(img2, (20, 20), (680, 400), (30, 130, 60), 4)
    cv2.rectangle(img2, (20, 20), (680, 95), (30, 130, 60), -1)
    cv2.putText(img2, " AMLODIPINE BESILATE TABLETS", (35, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    cv2.putText(img2, "AMLOD 5 mg", (45, 175), cv2.FONT_HERSHEY_DUPLEX, 1.9, (20, 90, 30), 4)
    cv2.putText(img2, "Strength: 5 mg   (For Blood Pressure)", (45, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (40, 40, 40), 2)
    cv2.putText(img2, "BATCH: AM99   MFG: 11/2024   EXP: 11/2027", (45, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (10, 10, 10), 2)
    cv2.imwrite(str(SAMPLES_DIR / "amlod_5_valid.jpg"), img2)

    # 3. Pantop 40 mg (EXPIRED medicine warning test)
    img3 = np.full((420, 700, 3), (240, 244, 252), dtype=np.uint8)
    cv2.rectangle(img3, (20, 20), (680, 400), (40, 80, 160), 4)
    cv2.putText(img3, "PANTOP 40 mg", (45, 140), cv2.FONT_HERSHEY_DUPLEX, 1.9, (25, 45, 120), 4)
    cv2.putText(img3, "Pantoprazole Gastro-Resistant Tablets", (45, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.putText(img3, "MFG: 02/2022   EXP: 02/2024", (45, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 180), 3)
    cv2.imwrite(str(SAMPLES_DIR / "pantop_40_expired.jpg"), img3)

    # 4. Blurry / Faded Blister Pack (Triggers Human Verification Flag)
    img4 = np.full((420, 700, 3), (215, 215, 215), dtype=np.uint8)
    cv2.putText(img4, "METFOR 500 mg", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (140, 140, 140), 2)
    cv2.putText(img4, "EXP: ../../....", (60, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
    img4_blur = cv2.GaussianBlur(img4, (25, 25), 9)
    cv2.imwrite(str(SAMPLES_DIR / "blurry_package.jpg"), img4_blur)

    # 5. Atorva 10 mg with Bikram Sambat (BS) Expiry Year (e.g. EXP: 05/2085 BS)
    img5 = np.full((420, 700, 3), (250, 245, 240), dtype=np.uint8)
    cv2.rectangle(img5, (20, 20), (680, 400), (120, 40, 90), 4)
    cv2.rectangle(img5, (20, 20), (680, 95), (120, 40, 90), -1)
    cv2.putText(img5, " ATORVASTATIN TABLETS IP", (35, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    cv2.putText(img5, "ATORVA 10 mg", (45, 175), cv2.FONT_HERSHEY_DUPLEX, 1.85, (100, 20, 80), 4)
    cv2.putText(img5, "For Cholesterol & Heart Protection", (45, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 2)
    cv2.putText(img5, "B.No: AT-88   MFG: 05/2082   EXP: 05/2085", (45, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (10, 10, 10), 2)
    cv2.imwrite(str(SAMPLES_DIR / "atorva_10_bs.jpg"), img5)
