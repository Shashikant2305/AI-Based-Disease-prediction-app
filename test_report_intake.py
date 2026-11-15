from contextlib import closing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from docx import Document
from PIL import Image
import pytesseract

import database
import report_analyzer
from pdf_report import create_pdf_bytes
from report_analyzer import (
    extract_lab_values,
    extract_patient_details,
    extract_report_text,
    flag_lab_values,
    validate_reviewed_lab_values,
    validate_reviewed_patient_details,
)


SYNTHETIC_TEXT = """SYNTHETIC TEST DATA - NOT A REAL PERSON
Patient Name: Example Patient 001
Patient ID: DEMO-001
Age: 46 years
Sex: Female
Phone: 555-0100
Fasting glucose: 106 mg/dL
HbA1c: 5.9 %
"""

EXPECTED_DETAILS = {
    "name": "Example Patient 001",
    "age": 46,
    "sex": "Female",
    "contact": "555-0100",
    "external_id": "DEMO-001",
}


class ReportIntakeTests(unittest.TestCase):
    def test_init_db_migrates_legacy_schema_without_changing_patient_ids(self):
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "legacy.sqlite"
            with closing(sqlite3.connect(db_path)) as con:
                with con:
                    con.execute(
                        """CREATE TABLE patients (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL, age INTEGER, sex TEXT, contact TEXT,
                            notes TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )"""
                    )
                    con.execute(
                        "INSERT INTO patients (name, age, sex, contact) "
                        "VALUES ('Existing Person', 38, 'Female', '555-0300')"
                    )
                    con.execute(
                        """CREATE TABLE analyses (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            patient_id INTEGER NOT NULL, analysis_type TEXT NOT NULL,
                            summary TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )"""
                    )

            with patch.object(database, "DB_PATH", db_path):
                database.init_db()
                database.init_db()

            with closing(sqlite3.connect(db_path)) as con:
                patient = con.execute(
                    "SELECT id, name FROM patients"
                ).fetchone()
                sequence = con.execute(
                    "SELECT seq FROM sqlite_sequence WHERE name = 'patients'"
                ).fetchone()[0]
                index_sql = con.execute(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type = 'index' AND name = 'idx_patients_name_contact'"
                ).fetchone()[0]
                user_version = con.execute("PRAGMA user_version").fetchone()[0]
                foreign_keys = con.execute(
                    "PRAGMA foreign_key_list(analyses)"
                ).fetchall()

            self.assertEqual(patient, (1, "Existing Person"))
            self.assertEqual(sequence, 1)
            self.assertIn("(external_id IS NULL OR trim(external_id) = '')", index_sql)
            self.assertEqual(user_version, 1)
            self.assertEqual(foreign_keys[0][2], "patients")
            with patch.object(database, "DB_PATH", db_path):
                with self.assertRaises(sqlite3.IntegrityError):
                    database.save_analysis(999, "Invalid", "Summary")

    def test_migration_reports_existing_duplicate_ids_without_deleting_data(self):
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "legacy.sqlite"
            with closing(sqlite3.connect(db_path)) as con:
                with con:
                    con.execute(
                        """CREATE TABLE patients (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL, age INTEGER, sex TEXT, contact TEXT,
                            notes TEXT, external_id TEXT,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )"""
                    )
                    con.executemany(
                        "INSERT INTO patients(name, external_id) VALUES(?, ?)",
                        [("First", "REC-1"), ("Second", "rec-1")],
                    )
                    con.execute(
                        """CREATE TABLE analyses (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            patient_id INTEGER NOT NULL, analysis_type TEXT NOT NULL,
                            summary TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )"""
                    )

            with patch.object(database, "DB_PATH", db_path):
                with self.assertRaisesRegex(RuntimeError, "No records were merged or deleted"):
                    database.init_db()

            with closing(sqlite3.connect(db_path)) as con:
                self.assertEqual(
                    con.execute("SELECT COUNT(*) FROM patients").fetchone()[0],
                    2,
                )

    def test_extracts_demographics_and_units_from_digital_pdf(self):
        pdf_bytes = create_pdf_bytes("Synthetic report", SYNTHETIC_TEXT.splitlines())
        text = extract_report_text("report.pdf", pdf_bytes)

        self.assertEqual(extract_patient_details(text), EXPECTED_DETAILS)
        values = extract_lab_values(text)
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0]["unit"], "mg/dL")
        self.assertEqual(values[1]["unit"], "%")

    def test_extracts_labeled_fields_from_docx_and_long_csv(self):
        document = Document()
        for line in SYNTHETIC_TEXT.splitlines():
            document.add_paragraph(line)
        docx_bytes = BytesIO()
        document.save(docx_bytes)
        docx_text = extract_report_text("report.docx", docx_bytes.getvalue())
        self.assertEqual(extract_patient_details(docx_text), EXPECTED_DETAILS)

        csv_bytes = (
            b"Patient Name,Patient ID,Age,Sex,Phone,Test,Result,Unit,Reference Range\n"
            b"Example Patient 001,DEMO-001,46,Female,555-0100,HbA1c,5.9,%,4.0-5.6 %\n"
            b"Example Patient 001,DEMO-001,46,Female,555-0100,Glucose,106,mg/dL,70-99 mg/dL\n"
        )
        csv_text = extract_report_text("report.csv", csv_bytes)
        self.assertEqual(extract_patient_details(csv_text), EXPECTED_DETAILS)
        csv_values = extract_lab_values(csv_text)
        self.assertEqual(len(csv_values), 2)
        hba1c = next(item for item in csv_values if item["parameter"] == "HbA1c")
        self.assertEqual(hba1c["reported_reference_range"], "4.0-5.6 %")

    def test_rejects_multiple_patient_csv(self):
        csv_bytes = (
            b"Patient Name,Patient ID,Test,Result\n"
            b"Person One,ID-1,HbA1c,5.9\n"
            b"Person Two,ID-2,HbA1c,6.0\n"
        )
        with self.assertRaisesRegex(ValueError, "one patient's report"):
            extract_report_text("report.csv", csv_bytes)

    def test_does_not_compare_lab_value_when_units_differ(self):
        values = extract_lab_values("Glucose: 5.5 mmol/L")
        flagged = flag_lab_values(values)
        self.assertEqual(
            flagged[0]["status"],
            "Not compared (unit differs from illustrative interval)",
        )

    def test_supports_pipe_layout_parenthesized_units_and_report_range(self):
        values = extract_lab_values(
            "Glucose (mg/dL): 1,234 (reference range: 70-99 mg/dL)\n"
            "HbA1c | 5.9 | %"
        )
        self.assertEqual(values[0]["value"], 1234)
        self.assertEqual(values[0]["unit"], "mg/dL")
        self.assertEqual(values[0]["reported_reference_range"], "70-99 mg/dL")
        self.assertEqual(values[1]["unit"], "%")
        self.assertEqual(extract_lab_values("Glucose: 1e3 mg/dL"), [])
        self.assertEqual(extract_lab_values("Glucose: 70-99 mg/dL"), [])
        self.assertEqual(extract_lab_values("Glucose: -1 mg/dL"), [])

    def test_missing_units_are_not_compared_and_intervals_are_illustrative(self):
        results = flag_lab_values(extract_lab_values("Glucose: 106"))
        self.assertEqual(results[0]["status"], "Not compared (report unit missing)")
        self.assertIn("illustrative", results[0]["interval_note"].casefold())

    def test_glucose_is_compared_only_when_fasting_context_is_explicit(self):
        fasting = flag_lab_values(extract_lab_values("Fasting glucose: 106 mg/dL"))
        post_load = flag_lab_values(
            extract_lab_values("2-hour post-load plasma glucose: 106 mg/dL")
        )
        unspecified = flag_lab_values(extract_lab_values("Glucose: 106 mg/dL"))

        self.assertEqual(fasting[0]["context"], "Fasting")
        self.assertEqual(
            fasting[0]["status"],
            "Outside illustrative interval (not diagnostic)",
        )
        self.assertEqual(post_load[0]["context"], "Post-load")
        self.assertEqual(
            post_load[0]["status"],
            "Not compared (fasting context not confirmed)",
        )
        self.assertEqual(
            unspecified[0]["status"],
            "Not compared (fasting context not confirmed)",
        )

    def test_rejects_oversized_upload_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "10 MB"):
            extract_report_text("report.csv", b"x" * (10 * 1024 * 1024 + 1))

    def test_missing_tesseract_has_actionable_local_setup_message(self):
        image = Image.new("RGB", (1, 1))
        with patch(
            "pytesseract.image_to_string",
            side_effect=pytesseract.TesseractNotFoundError(),
        ):
            with self.assertRaisesRegex(RuntimeError, "Tesseract executable"):
                report_analyzer._ocr_image(image)

    def test_requires_labeled_name_before_saving_upload(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB_PATH", Path(temp_dir) / "test.sqlite"):
                database.init_db()
                with self.assertRaisesRegex(ValueError, "labeled patient name"):
                    database.save_uploaded_report(
                        {"name": None},
                        "Report summary",
                        "fingerprint-1",
                    )
                self.assertEqual(database.list_patients(), [])

    def test_uploaded_report_creates_matches_and_deduplicates(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB_PATH", Path(temp_dir) / "test.sqlite"):
                database.init_db()
                patient_id, status = database.save_uploaded_report(
                    EXPECTED_DETAILS,
                    "Report summary",
                    "fingerprint-1",
                    create_patient=True,
                    identity_confirmed=True,
                )
                self.assertEqual(status, "created")

                repeated_id, repeated_status = database.save_uploaded_report(
                    EXPECTED_DETAILS,
                    "Duplicate summary",
                    "fingerprint-1",
                    create_patient=True,
                    identity_confirmed=True,
                )
                self.assertEqual((repeated_id, repeated_status), (patient_id, "already_processed"))
                self.assertEqual(len(database.list_analyses()), 1)

                linked_id, linked_status = database.save_uploaded_report(
                    {**EXPECTED_DETAILS, "age": 47},
                    "Second report",
                    "fingerprint-2",
                    patient_id=patient_id,
                    identity_confirmed=True,
                )
                self.assertEqual((linked_id, linked_status), (patient_id, "linked"))
                patients = database.list_patients()
                self.assertEqual(len(patients), 1)
                self.assertEqual(patients[0]["age"], 46)
                self.assertEqual(len(database.list_analyses()), 2)

    def test_strong_identity_checks_preserve_ambiguous_profiles(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB_PATH", Path(temp_dir) / "test.sqlite"):
                database.init_db()
                first_id, status = database.add_patient(
                    "Example Person",
                    40,
                    "Female",
                    "555-0100",
                    "",
                    "REC-1",
                )
                self.assertEqual(status, "created")

                repeated_id, repeated_status = database.add_patient(
                    " example-person ",
                    40,
                    "Female",
                    "(555) 0100",
                    "",
                    "rec-1",
                )
                self.assertEqual((repeated_id, repeated_status), (first_id, "existing"))
                self.assertEqual(len(database.list_patients()), 1)

                second_id, second_status = database.save_uploaded_report(
                    {
                        "name": "Example Person",
                        "age": 40,
                        "sex": "Female",
                        "contact": "555-0200",
                        "external_id": "REC-2",
                    },
                    "Second person's report",
                    "fingerprint-2",
                    create_patient=True,
                    identity_confirmed=True,
                )
                self.assertNotEqual(second_id, first_id)
                self.assertEqual(second_status, "created_similar")
                self.assertEqual(len(database.list_patients()), 2)

                with self.assertRaisesRegex(ValueError, "different name"):
                    database.add_patient(
                        "Different Person",
                        40,
                        "Female",
                        "555-0300",
                        "",
                        "REC-1",
                    )

    def test_patient_id_is_not_reused_after_deletion(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB_PATH", Path(temp_dir) / "test.sqlite"):
                database.init_db()
                first_id, _ = database.add_patient("First", 30, "Other", "", "")
                removed_id, _ = database.add_patient("Removed", 31, "Other", "", "")
                database.delete_patient(removed_id)
                next_id, _ = database.add_patient("Next", 32, "Other", "", "")
                self.assertGreater(next_id, removed_id)
                self.assertEqual(next_id, removed_id + 1)
                self.assertEqual(len(database.list_patients()), 2)
                self.assertNotEqual(first_id, next_id)

    def test_existing_profile_is_not_changed_by_repeated_manual_entry(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB_PATH", Path(temp_dir) / "test.sqlite"):
                database.init_db()
                patient_id, _ = database.add_patient(
                    "Example Person",
                    None,
                    "Not specified",
                    "555-0100",
                    "",
                )
                matched_id, status = database.add_patient(
                    "Example Person",
                    40,
                    "Female",
                    "(555) 0100",
                    "",
                    "REC-1",
                )
                self.assertEqual((matched_id, status), (patient_id, "existing"))
                patient = database.list_patients()[0]
                self.assertEqual(len(database.list_patients()), 1)
                self.assertIsNone(patient["age"])
                self.assertEqual(patient["sex"], "Not specified")
                self.assertIsNone(patient["external_id"])

    def test_upload_requires_explicit_profile_choice_and_confirmation(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB_PATH", Path(temp_dir) / "test.sqlite"):
                database.init_db()
                with self.assertRaisesRegex(ValueError, "Select an existing profile"):
                    database.save_uploaded_report(
                        EXPECTED_DETAILS,
                        "Report summary",
                        "fingerprint-1",
                        identity_confirmed=True,
                    )
                with self.assertRaisesRegex(ValueError, "Confirm"):
                    database.save_uploaded_report(
                        EXPECTED_DETAILS,
                        "Report summary",
                        "fingerprint-1",
                        create_patient=True,
                    )
                self.assertEqual(database.list_patients(), [])
                self.assertEqual(database.list_analyses(), [])

    def test_report_match_preview_is_read_only_and_conflicts_are_visible(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB_PATH", Path(temp_dir) / "test.sqlite"):
                database.init_db()
                patient_id, _ = database.add_patient(
                    "Example Person", 46, "Female", "555-0100", "", "REC-1"
                )
                matches = database.find_patient_matches(
                    {
                        **EXPECTED_DETAILS,
                        "age": 47,
                        "contact": "555-0200",
                        "external_id": "REC-1",
                    }
                )
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0]["id"], patient_id)
                self.assertIn("same patient/record ID", matches[0]["match_reasons"])
                self.assertIn("contact differs", matches[0]["conflicts"])
                self.assertIn("age differs", matches[0]["conflicts"])
                self.assertTrue(matches[0]["requires_confirmation"])
                self.assertEqual(len(database.list_patients()), 1)
                self.assertEqual(database.list_analyses(), [])

    def test_report_save_does_not_overwrite_selected_profile(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB_PATH", Path(temp_dir) / "test.sqlite"):
                database.init_db()
                patient_id, _ = database.add_patient(
                    "Example Patient 001",
                    46,
                    "Female",
                    "555-0100",
                    "",
                    "DEMO-001",
                )
                saved_id, status = database.save_uploaded_report(
                    {**EXPECTED_DETAILS, "age": 99, "contact": "555-9999"},
                    "Reviewed summary",
                    "fingerprint-1",
                    patient_id=patient_id,
                    identity_confirmed=True,
                )
                self.assertEqual((saved_id, status), (patient_id, "linked"))
                patient = database.list_patients()[0]
                self.assertEqual(patient["age"], 46)
                self.assertEqual(patient["contact"], "555-0100")
                self.assertEqual(patient["external_id"], "DEMO-001")

    def test_concurrent_duplicate_report_saves_are_serialized(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB_PATH", Path(temp_dir) / "test.sqlite"):
                database.init_db()

                def save_report():
                    return database.save_uploaded_report(
                        EXPECTED_DETAILS,
                        "Report summary",
                        "fingerprint-shared",
                        create_patient=True,
                        identity_confirmed=True,
                    )

                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _: save_report(), range(2)))
                self.assertEqual(len(database.list_patients()), 1)
                self.assertEqual(len(database.list_analyses()), 1)
                self.assertEqual(results[0][0], results[1][0])
                self.assertCountEqual(
                    [result[1] for result in results],
                    ["created", "already_processed"],
                )

    def test_concurrent_patient_registration_does_not_duplicate_identity(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB_PATH", Path(temp_dir) / "test.sqlite"):
                database.init_db()

                def add_patient():
                    return database.add_patient(
                        "Example Person",
                        46,
                        "Female",
                        "555-0100",
                        "",
                        "REC-1",
                    )

                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _: add_patient(), range(2)))
                self.assertEqual(len(database.list_patients()), 1)
                self.assertEqual(results[0][0], results[1][0])
                self.assertCountEqual(
                    [result[1] for result in results],
                    ["created", "existing"],
                )

    def test_patient_and_linked_analyses_deletion_roll_back_together(self):
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            with patch.object(database, "DB_PATH", db_path):
                database.init_db()
                patient_id, _ = database.add_patient("Example", 30, "Other", "", "")
                database.save_analysis(patient_id, "Test", "Summary")
                with database.connect() as con:
                    con.execute(
                        """CREATE TRIGGER prevent_patient_delete
                        BEFORE DELETE ON patients
                        BEGIN SELECT RAISE(ABORT, 'delete blocked for test'); END"""
                    )
                with self.assertRaises(sqlite3.IntegrityError):
                    database.delete_patient(patient_id)
                self.assertEqual(len(database.list_patients()), 1)
                self.assertEqual(len(database.list_analyses()), 1)

    def test_reviewed_fields_are_validated_without_guessing(self):
        self.assertEqual(
            validate_reviewed_patient_details(
                " Example Person ", "46", "Female", "555-0100", "REC-1"
            ),
            {
                "name": "Example Person",
                "age": 46,
                "sex": "Female",
                "contact": "555-0100",
                "external_id": "REC-1",
            },
        )
        with self.assertRaisesRegex(ValueError, "whole number"):
            validate_reviewed_patient_details("Example", "46.5", None, None, None)
        with self.assertRaisesRegex(ValueError, "patient name"):
            validate_reviewed_patient_details("", "", None, None, None)
        with self.assertRaisesRegex(ValueError, "finite, non-negative"):
            validate_reviewed_lab_values(
                [{"parameter": "Glucose", "value": float("inf"), "unit": "mg/dL"}]
            )

    def test_streamlit_report_flow_waits_for_review_and_profile_confirmation(self):
        from streamlit.testing.v1 import AppTest

        original_db_path = database.DB_PATH
        try:
            with TemporaryDirectory() as temp_dir:
                database.DB_PATH = Path(temp_dir) / "mediai.db"
                app_test = AppTest.from_file(
                    str(Path(__file__).parent / "app.py"),
                    default_timeout=30,
                ).run()
                self.assertFalse(app_test.exception)
                app_test.sidebar.radio[0].set_value("Medical Report Analyzer").run()
                self.assertFalse(app_test.exception)

                csv_bytes = (
                    b"Patient Name,Patient ID,Age,Sex,Phone,Test,Result,Unit\n"
                    b"Synthetic User,DEMO-TEST,44,Other,555-0101,Glucose,100,mg/dL\n"
                )
                app_test.file_uploader[0].set_value(
                    ("synthetic.csv", csv_bytes, "text/csv")
                ).run()
                self.assertFalse(app_test.exception)
                self.assertEqual(database.list_patients(), [])
                self.assertEqual(database.list_analyses(), [])

                next(
                    button
                    for button in app_test.button
                    if button.label == "Review profile matches"
                ).click().run()
                self.assertFalse(app_test.exception)
                self.assertEqual(database.list_patients(), [])
                self.assertEqual(database.list_analyses(), [])

                next(
                    box
                    for box in app_test.selectbox
                    if box.label == "Profile to use"
                ).set_value("Create a new patient profile").run()
                next(
                    box
                    for box in app_test.checkbox
                    if "confirm this is a new patient profile" in box.label
                ).check().run()
                next(
                    button
                    for button in app_test.button
                    if button.label == "Save reviewed report once"
                ).click().run()
                self.assertFalse(app_test.exception)
                self.assertEqual(len(database.list_patients()), 1)
                self.assertEqual(len(database.list_analyses()), 1)

                app_test.file_uploader[0].set_value(
                    ("synthetic.csv", csv_bytes, "text/csv")
                ).run()
                self.assertFalse(app_test.exception)
                self.assertEqual(len(database.list_patients()), 1)
                self.assertEqual(len(database.list_analyses()), 1)
        finally:
            database.DB_PATH = original_db_path


if __name__ == "__main__":
    unittest.main()
