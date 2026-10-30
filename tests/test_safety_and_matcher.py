"""
Unit tests for MedRead Nepal medicine matching, strength translation,
expiry date parsing (AD & BS calendars), and safety guardrails.
"""

import unittest
from app.medicine_db import match_known_medicine, strength_to_nepali, to_nepali_digits
from app.safety import evaluate_safety_and_match, parse_expiry_date


class TestMedReadSafetyAndMatcher(unittest.TestCase):
    def test_nepali_digits_and_strength(self):
        self.assertEqual(to_nepali_digits("500"), "५००")
        self.assertEqual(strength_to_nepali("500 mg"), "५०० मिलिग्राम")
        self.assertEqual(strength_to_nepali("10 ml"), "१० मिलिलिटर")

    def test_fuzzy_medicine_matching(self):
        # Exact keyword match
        m1 = match_known_medicine("Rx CETAMOL 500 mg Tablets")
        self.assertIsNotNone(m1)
        self.assertEqual(m1["id"], "cetamol-500")

        # Minor OCR typo ("Paracetam0l")
        m2 = match_known_medicine("Each tablet contains Paracetam0l 500mg")
        self.assertIsNotNone(m2)
        self.assertEqual(m2["id"], "cetamol-500")

    def test_expiry_date_parsing(self):
        exp_valid = parse_expiry_date("MFG: 08/2025 EXP: 08/2029")
        self.assertTrue(exp_valid["found"])
        self.assertFalse(exp_valid["is_expired"])
        self.assertEqual(exp_valid["year"], 2029)

        exp_old = parse_expiry_date("EXP: 02/2023")
        self.assertTrue(exp_old["found"])
        self.assertTrue(exp_old["is_expired"])

    def test_safety_guardrails_mismatch_and_expired(self):
        exp_info = parse_expiry_date("EXP: 08/2029")
        expected_amlod = {
            "id": "amlod-5",
            "brand_en": "Amlod",
            "generic_en": "Amlodipine",
            "name_ne": "एम्लोड",
            "default_strength": "5 mg",
        }
        # Scanning Cetamol when expecting Amlod must trigger DANGER
        res = evaluate_safety_and_match(
            med_name_en="Cetamol (Paracetamol)",
            strength_en="500 mg",
            expiry_info=exp_info,
            ocr_confidence=0.92,
            blur_score=120.0,
            expected_medicine=expected_amlod,
        )
        self.assertEqual(res["status"], "DANGER")
        self.assertTrue(res["requires_human_verification"])


if __name__ == "__main__":
    unittest.main()
