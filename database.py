from contextlib import contextmanager
import re
import sqlite3
import unicodedata
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "mediai.db"


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _migrate_analysis_foreign_key(con):
    columns = [
        row["name"] for row in con.execute("PRAGMA table_info(analyses)")
    ]
    expected_columns = {
        "id",
        "patient_id",
        "analysis_type",
        "summary",
        "source_fingerprint",
        "created_at",
    }
    if set(columns) != expected_columns:
        raise RuntimeError(
            "The legacy analyses table has an unsupported schema; no records were changed."
        )

    schema_objects = [
        row["sql"]
        for row in con.execute(
            """SELECT sql FROM sqlite_master
            WHERE tbl_name = 'analyses' AND sql IS NOT NULL
            AND type IN ('index', 'trigger')"""
        )
    ]
    con.execute(
        """CREATE TABLE analyses_migrating (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            analysis_type TEXT NOT NULL,
            summary TEXT,
            source_fingerprint TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )"""
    )
    con.execute(
        """INSERT INTO analyses_migrating(
            id, patient_id, analysis_type, summary, source_fingerprint, created_at
        )
        SELECT id, patient_id, analysis_type, summary, source_fingerprint, created_at
        FROM analyses"""
    )
    con.execute("DROP TABLE analyses")
    con.execute("ALTER TABLE analyses_migrating RENAME TO analyses")
    for sql in schema_objects:
        con.execute(sql)


def init_db():
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute("""CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, age INTEGER, sex TEXT, contact TEXT,
            notes TEXT, external_id TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL, analysis_type TEXT NOT NULL,
            summary TEXT, source_fingerprint TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )""")
        patient_columns = {
            row["name"] for row in con.execute("PRAGMA table_info(patients)")
        }
        if "external_id" not in patient_columns:
            con.execute("ALTER TABLE patients ADD COLUMN external_id TEXT")
        analysis_columns = {
            row["name"] for row in con.execute("PRAGMA table_info(analyses)")
        }
        if "source_fingerprint" not in analysis_columns:
            con.execute("ALTER TABLE analyses ADD COLUMN source_fingerprint TEXT")
        orphan_analyses = con.execute(
            """SELECT COUNT(*) FROM analyses a
            LEFT JOIN patients p ON p.id = a.patient_id
            WHERE p.id IS NULL"""
        ).fetchone()[0]
        if orphan_analyses:
            raise RuntimeError(
                "The database contains analyses without a patient profile. "
                "No records were changed; resolve these links before restarting."
            )
        analysis_foreign_keys = con.execute(
            "PRAGMA foreign_key_list(analyses)"
        ).fetchall()
        if not any(
            row["table"] == "patients" and row["from"] == "patient_id"
            for row in analysis_foreign_keys
        ):
            _migrate_analysis_foreign_key(con)
        duplicate_fingerprints = con.execute(
            """SELECT COUNT(*) FROM (
                SELECT source_fingerprint FROM analyses
                WHERE source_fingerprint IS NOT NULL
                GROUP BY source_fingerprint HAVING COUNT(*) > 1
            )"""
        ).fetchone()[0]
        duplicate_external_ids = con.execute(
            """SELECT COUNT(*) FROM (
                SELECT lower(trim(external_id)) FROM patients
                WHERE external_id IS NOT NULL AND trim(external_id) <> ''
                GROUP BY lower(trim(external_id)) HAVING COUNT(*) > 1
            )"""
        ).fetchone()[0]
        duplicate_name_contacts = con.execute(
            """SELECT COUNT(*) FROM (
                SELECT lower(trim(name)), lower(trim(contact)) FROM patients
                WHERE contact IS NOT NULL AND trim(contact) <> ''
                AND (external_id IS NULL OR trim(external_id) = '')
                GROUP BY lower(trim(name)), lower(trim(contact))
                HAVING COUNT(*) > 1
            )"""
        ).fetchone()[0]
        identity_rows = con.execute(
            "SELECT name, contact, external_id FROM patients"
        ).fetchall()
        normalized_external_ids = {}
        normalized_name_contacts = {}
        for patient in identity_rows:
            external_id_key = _normalize_external_id(patient["external_id"])
            if external_id_key:
                normalized_external_ids[external_id_key] = (
                    normalized_external_ids.get(external_id_key, 0) + 1
                )
            elif _normalize_contact(patient["contact"]):
                name_contact_key = (
                    _normalize_text(patient["name"]),
                    _normalize_contact(patient["contact"]),
                )
                normalized_name_contacts[name_contact_key] = (
                    normalized_name_contacts.get(name_contact_key, 0) + 1
                )
        duplicate_external_ids = max(
            duplicate_external_ids,
            sum(count > 1 for count in normalized_external_ids.values()),
        )
        duplicate_name_contacts = max(
            duplicate_name_contacts,
            sum(count > 1 for count in normalized_name_contacts.values()),
        )
        if duplicate_fingerprints or duplicate_external_ids or duplicate_name_contacts:
            raise RuntimeError(
                "The database contains conflicting duplicate report fingerprints "
                "or patient identifiers. No records were merged or deleted; resolve "
                "the duplicates and restart the app."
            )
        con.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "idx_analyses_source_fingerprint ON analyses(source_fingerprint) "
            "WHERE source_fingerprint IS NOT NULL"
        )
        con.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "idx_patients_external_id ON patients(lower(trim(external_id))) "
            "WHERE external_id IS NOT NULL AND trim(external_id) <> ''"
        )
        name_contact_index_sql = (
            "CREATE UNIQUE INDEX idx_patients_name_contact "
            "ON patients(lower(trim(name)), lower(trim(contact))) "
            "WHERE contact IS NOT NULL AND trim(contact) <> '' "
            "AND (external_id IS NULL OR trim(external_id) = '')"
        )
        existing_index = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'index' AND name = 'idx_patients_name_contact'"
        ).fetchone()
        if existing_index is None or existing_index["sql"] != name_contact_index_sql:
            con.execute("DROP INDEX IF EXISTS idx_patients_name_contact")
            con.execute(name_contact_index_sql)
        con.execute("PRAGMA user_version = 1")


def _normalize_text(value):
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(
        "".join(
            character if character.isalnum() else " " for character in normalized
        ).split()
    )


def _normalize_contact(value):
    return re.sub(r"\D", "", str(value or ""))


def _normalize_external_id(value):
    return "".join(str(value or "").split()).casefold()


def _patient_match_details(patient, name, contact, external_id, age, sex):
    name_key = _normalize_text(name)
    external_id_key = _normalize_external_id(external_id)
    contact_key = _normalize_contact(contact)
    saved_name = _normalize_text(patient["name"])
    saved_contact = _normalize_contact(patient["contact"])
    saved_external_id = _normalize_external_id(patient["external_id"])
    reasons = []
    conflicts = []

    if external_id_key and saved_external_id == external_id_key:
        reasons.append("same patient/record ID")
        if name_key and saved_name != name_key:
            conflicts.append("name differs")
    if name_key and saved_name == name_key:
        if contact_key and saved_contact == contact_key:
            reasons.append("same normalized name and contact")
        elif (
            age is not None
            and sex
            and patient["age"] == age
            and _normalize_text(patient["sex"]) == _normalize_text(sex)
        ):
            reasons.append("same name, age, and sex (not a unique identifier)")
        else:
            reasons.append("same normalized name only")
    if contact_key and saved_contact == contact_key and saved_name != name_key:
        reasons.append("same contact but name differs")

    if external_id_key and saved_external_id and saved_external_id != external_id_key:
        conflicts.append("patient/record ID differs")
    if contact_key and saved_contact and saved_contact != contact_key:
        conflicts.append("contact differs")
    if contact_key and saved_contact == contact_key and saved_name != name_key:
        conflicts.append("name differs")
    if age is not None and patient["age"] is not None and age != patient["age"]:
        conflicts.append("age differs")
    if sex and patient["sex"] and _normalize_text(patient["sex"]) not in {
        "",
        "not specified",
    } and _normalize_text(patient["sex"]) != _normalize_text(sex):
        conflicts.append("sex differs")

    if not reasons:
        return None
    exact_identifier_match = (
        "same patient/record ID" in reasons
        or "same normalized name and contact" in reasons
    )
    return {
        **dict(patient),
        "match_reasons": reasons,
        "conflicts": conflicts,
        "requires_confirmation": bool(conflicts) or not exact_identifier_match,
    }


def _find_patient_matches(con, details):
    matches = []
    for patient in con.execute(
        "SELECT id, name, age, sex, contact, external_id FROM patients"
    ):
        match = _patient_match_details(
            patient,
            details.get("name"),
            details.get("contact"),
            details.get("external_id"),
            details.get("age"),
            details.get("sex"),
        )
        if match:
            matches.append(match)
    return sorted(matches, key=lambda match: match["id"])


def find_patient_matches(details):
    """Return possible identity matches without changing any stored data."""
    with connect() as con:
        return _find_patient_matches(con, details)


def find_uploaded_report(source_fingerprint):
    """Return the patient linked to a previously saved report, if any."""
    if not source_fingerprint:
        return None
    with connect() as con:
        row = con.execute(
            """SELECT p.id AS patient_id, p.name
            FROM analyses a JOIN patients p ON p.id = a.patient_id
            WHERE a.source_fingerprint = ?""",
            (source_fingerprint,),
        ).fetchone()
    return dict(row) if row else None


def add_patient(name, age, sex, contact, notes, external_id=None):
    name = " ".join((name or "").split())
    contact = " ".join((contact or "").split())
    external_id = " ".join((external_id or "").split()) or None
    if not name:
        raise ValueError("Patient name cannot be empty.")

    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        details = {
            "name": name,
            "age": age,
            "sex": sex,
            "contact": contact,
            "external_id": external_id,
        }
        matches = _find_patient_matches(con, details)
        if external_id:
            id_matches = [
                match
                for match in matches
                if "same patient/record ID" in match["match_reasons"]
            ]
            if id_matches:
                if len(id_matches) > 1:
                    raise RuntimeError(
                        "This patient/record ID matches multiple profiles; resolve the "
                        "ambiguous records before adding another patient."
                    )
                patient = id_matches[0]
                if patient["conflicts"]:
                    if "name differs" in patient["conflicts"]:
                        raise ValueError(
                            "This patient/record ID is already associated with a different name."
                        )
                    raise ValueError(
                        "This patient/record ID matches a profile with conflicting "
                        "details; review it instead of creating or changing a profile."
                    )
                return patient["id"], "existing"
        exact_matches = [
            match
            for match in matches
            if "same normalized name and contact" in match["match_reasons"]
        ]
        if exact_matches:
            if len(exact_matches) > 1:
                raise RuntimeError(
                    "This normalized name and contact match multiple profiles; "
                    "resolve the ambiguous records before adding another patient."
                )
            if external_id and exact_matches[0]["conflicts"]:
                raise ValueError(
                    "The matching name and contact are associated with a different "
                    "patient/record ID."
                )
            return exact_matches[0]["id"], "existing"
        cursor = con.execute(
            "INSERT INTO patients(name,age,sex,contact,notes,external_id) "
            "VALUES(?,?,?,?,?,?)",
            (name, age, sex, contact, notes, external_id),
        )
        patient_id = cursor.lastrowid
        status = "created_similar" if matches else "created"
    return patient_id, status


def delete_patient(patient_id):
    if type(patient_id) is not int or patient_id <= 0:
        raise ValueError("patient_id must be a positive integer.")

    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        patient = con.execute(
            "SELECT id FROM patients WHERE id = ?",
            (patient_id,),
        ).fetchone()
        if patient is None:
            raise ValueError(f"Patient {patient_id} does not exist.")

        deleted_analyses = con.execute(
            "DELETE FROM analyses WHERE patient_id = ?",
            (patient_id,),
        ).rowcount
        deleted_patient = con.execute(
            "DELETE FROM patients WHERE id = ?",
            (patient_id,),
        ).rowcount
        if deleted_patient != 1:
            raise RuntimeError(f"Could not remove patient {patient_id}.")
    return deleted_analyses


def list_patients():
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM patients ORDER BY id DESC")]


def save_analysis(patient_id, analysis_type, summary):
    with connect() as con:
        con.execute(
            "INSERT INTO analyses(patient_id,analysis_type,summary) VALUES(?,?,?)",
            (patient_id, analysis_type, summary),
        )


def save_uploaded_report(
    details,
    summary,
    source_fingerprint,
    *,
    patient_id=None,
    create_patient=False,
    identity_confirmed=False,
):
    name = " ".join((details.get("name") or "").split())
    if not name:
        raise ValueError("A labeled patient name is required to save this report.")
    if not source_fingerprint:
        raise ValueError("A report fingerprint is required to save this upload.")
    if (patient_id is None) == (not create_patient):
        raise ValueError("Select an existing profile or explicitly create a new one.")
    if not identity_confirmed:
        raise ValueError("Confirm that the reviewed patient details and profile are correct.")

    reviewed_details = {
        **details,
        "name": name,
        "external_id": " ".join((details.get("external_id") or "").split()) or None,
        "contact": " ".join((details.get("contact") or "").split()),
    }

    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        previous = con.execute(
            """SELECT patient_id FROM analyses
            WHERE source_fingerprint = ?""",
            (source_fingerprint,),
        ).fetchone()
        if previous:
            return previous["patient_id"], "already_processed"

        matches = _find_patient_matches(con, reviewed_details)
        if patient_id is not None:
            patient = con.execute(
                "SELECT id FROM patients WHERE id = ?",
                (patient_id,),
            ).fetchone()
            if patient is None:
                raise ValueError(f"Patient {patient_id} does not exist.")
            selected_match = next(
                (match for match in matches if match["id"] == patient_id),
                None,
            )
            if selected_match and selected_match["requires_confirmation"]:
                if not identity_confirmed:
                    raise ValueError(
                        "Confirm the uncertain or conflicting identity match before saving."
                    )
            status = "linked"
        else:
            exact_contact_match = next(
                (
                    match
                    for match in matches
                    if "same normalized name and contact" in match["match_reasons"]
                ),
                None,
            )
            if exact_contact_match:
                raise ValueError(
                    "A profile with this normalized name and contact already exists. "
                    "Select that profile or review the contact details."
                )
            id_match = next(
                (
                    match
                    for match in matches
                    if "same patient/record ID" in match["match_reasons"]
                ),
                None,
            )
            if id_match:
                raise ValueError(
                    "This patient/record ID already belongs to a profile. "
                    "Select that profile or correct the reviewed ID."
                )
            if matches and not identity_confirmed:
                raise ValueError(
                    "Review the possible profile matches before creating a new profile."
                )
            cursor = con.execute(
                """INSERT INTO patients(name, age, sex, contact, notes, external_id)
                    VALUES(?, ?, ?, ?, '', ?)""",
                (
                    name,
                    reviewed_details.get("age"),
                    reviewed_details.get("sex") or "Not specified",
                    reviewed_details.get("contact"),
                    reviewed_details.get("external_id"),
                ),
            )
            patient_id = cursor.lastrowid
            status = "created_similar" if matches else "created"

        con.execute(
            """INSERT INTO analyses(
                patient_id, analysis_type, summary, source_fingerprint
            ) VALUES(?, 'Medical report', ?, ?)""",
            (patient_id, summary, source_fingerprint),
        )
    return patient_id, status


def list_analyses():
    with connect() as con:
        return [dict(r) for r in con.execute("""
            SELECT a.id, p.name AS patient, a.analysis_type, a.summary, a.created_at
            FROM analyses a JOIN patients p ON p.id=a.patient_id ORDER BY a.id DESC
        """)]
