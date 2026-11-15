from io import BytesIO
import math
import re
import zipfile

import pandas as pd
import pymupdf


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 30
MAX_IMAGE_PIXELS = 20_000_000
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_DOCX_MEMBERS = 500
MAX_CSV_ROWS = 1000
MAX_EXTRACTED_TEXT_CHARS = 200_000


# Illustrative intervals only; compare only when the report unit matches.
REFERENCE_RANGES = {
    "glucose": (70, 99, "mg/dL"),
    "hba1c": (0, 5.6, "%"),
    "total cholesterol": (0, 199, "mg/dL"),
    "hemoglobin": (12, 17.5, "g/dL"),
    "creatinine": (0.6, 1.3, "mg/dL"),
}
DISPLAY_NAMES = {
    "glucose": "Glucose",
    "hba1c": "HbA1c",
    "total cholesterol": "Total cholesterol",
    "hemoglobin": "Hemoglobin",
    "creatinine": "Creatinine",
}
SUPPORTED_PARAMETERS = tuple(DISPLAY_NAMES.values())

_UNIT_TOKEN = r"mg\s*/?\s*dL|mmol\s*/?\s*L|g\s*/?\s*dL|g\s*/?\s*L|(?:u|µ|μ)mol\s*/?\s*L|%"
_UNIT_PATTERN = rf"(?P<unit>{_UNIT_TOKEN})?"


def _ocr_image(image):
    try:
        import pytesseract
    except ImportError as error:
        raise RuntimeError("Install the pytesseract package to use local OCR.") from error

    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "Local OCR needs the Tesseract executable installed on this computer."
        ) from error
    except pytesseract.TesseractError as error:
        raise RuntimeError(
            "Local Tesseract OCR could not read this image. Try a clearer scan or a text-based file."
        ) from error


def extract_pdf_text(pdf_bytes):
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as error:
        raise ValueError("The uploaded PDF could not be opened.") from error

    page_text = []
    try:
        if document.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")
        if len(document) > MAX_PDF_PAGES:
            raise ValueError(
                f"PDFs are limited to {MAX_PDF_PAGES} pages for safe local processing."
            )
        for page in document:
            text = page.get_text("text")
            if not isinstance(text, str):
                raise ValueError("PDF text extraction returned a non-text value.")
            if text.strip():
                page_text.append(text)
            else:
                rendered_pixels = (
                    page.rect.width
                    * page.rect.height
                    * (200 / 72) ** 2
                )
                if rendered_pixels > MAX_IMAGE_PIXELS:
                    raise ValueError(
                        "A scanned PDF page exceeds the 20 megapixel processing limit."
                    )
                pixmap = page.get_pixmap(dpi=200)
                page_text.append(_ocr_image(pixmap.pil_image()))
            if sum(map(len, page_text)) > MAX_EXTRACTED_TEXT_CHARS:
                raise ValueError(
                    "The extracted PDF text is too large to review safely."
                )
    finally:
        document.close()
    return "\n".join(page_text)


def _extract_docx_text(file_bytes):
    try:
        from docx import Document
        from docx.opc.exceptions import PackageNotFoundError
    except ImportError as error:
        raise RuntimeError("Install python-docx to read Word reports.") from error

    try:
        with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
            members = archive.infolist()
            expanded_size = sum(member.file_size for member in members)
            if len(members) > MAX_DOCX_MEMBERS or expanded_size > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "The DOCX contains too much compressed content to process safely."
                )
            if any(
                member.compress_size
                and member.file_size / member.compress_size > 1000
                for member in members
            ):
                raise ValueError("The DOCX contains an unsafe compression ratio.")
        document = Document(BytesIO(file_bytes))
    except (
        ValueError,
        KeyError,
        zipfile.BadZipFile,
        PackageNotFoundError,
    ) as error:
        if isinstance(error, ValueError) and str(error).startswith("The DOCX"):
            raise
        raise ValueError("The uploaded DOCX file could not be opened.") from error

    sections = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                sections.append(" | ".join(cells))
    for section in document.sections:
        for part in (section.header, section.footer):
            sections.extend(
                paragraph.text for paragraph in part.paragraphs if paragraph.text.strip()
            )
    return "\n".join(sections)


def _normalized_header(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _extract_csv_text(file_bytes):
    try:
        frame = pd.read_csv(
            BytesIO(file_bytes),
            dtype=str,
            keep_default_na=False,
            nrows=MAX_CSV_ROWS + 1,
        )
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
    ) as error:
        raise ValueError("The uploaded CSV could not be parsed.") from error
    if len(frame) > MAX_CSV_ROWS:
        raise ValueError(f"CSV reports are limited to {MAX_CSV_ROWS} rows.")
    if frame.empty:
        raise ValueError("The uploaded CSV contains no report rows.")

    aliases = {
        "patientname": "Patient Name",
        "fullname": "Patient Name",
        "name": "Patient Name",
        "patientid": "Patient ID",
        "medicalrecordnumber": "Patient ID",
        "mrn": "Patient ID",
        "recordid": "Patient ID",
        "age": "Age",
        "patientage": "Age",
        "sex": "Sex",
        "gender": "Sex",
        "phone": "Phone",
        "telephone": "Phone",
        "mobile": "Phone",
        "contact": "Phone",
        "test": "test",
        "analyte": "test",
        "parameter": "test",
        "testname": "test",
        "result": "result",
        "value": "result",
        "units": "unit",
        "unit": "unit",
        "referencerange": "reference range",
        "refrange": "reference range",
        "referenceinterval": "reference range",
        "normalrange": "reference range",
        "glucose": "Glucose",
        "fastingglucose": "Fasting glucose",
        "hba1c": "HbA1c",
        "a1c": "HbA1c",
        "glycatedhemoglobin": "HbA1c",
        "cholesterol": "Total cholesterol",
        "totalcholesterol": "Total cholesterol",
        "hemoglobin": "Hemoglobin",
        "creatinine": "Creatinine",
    }
    columns = {
        column: aliases.get(_normalized_header(column), str(column).strip())
        for column in frame.columns
    }
    normalized_columns = {label: column for column, label in columns.items()}

    patient_columns = [
        normalized_columns[label]
        for label in ("Patient Name", "Patient ID", "Age", "Sex", "Phone")
        if label in normalized_columns
    ]
    for column in patient_columns:
        distinct_values = {
            value.strip().casefold()
            for value in frame[column]
            if value.strip()
        }
        if len(distinct_values) > 1:
            raise ValueError("Upload one patient's report per CSV file.")

    lines = []
    test_column = normalized_columns.get("test")
    result_column = normalized_columns.get("result")
    unit_column = normalized_columns.get("unit")
    reference_column = normalized_columns.get("reference range")
    if test_column and result_column:
        for column, label in columns.items():
            if label in {"Patient Name", "Patient ID", "Age", "Sex", "Phone"}:
                value = next((item.strip() for item in frame[column] if item.strip()), "")
                if value:
                    lines.append(f"{label}: {value}")
        for _, row in frame.iterrows():
            test = row[test_column].strip()
            result = row[result_column].strip()
            unit = row[unit_column].strip() if unit_column else ""
            if test and result:
                reference = row[reference_column].strip() if reference_column else ""
                line = f"{test}: {result} {unit}".strip()
                if reference:
                    line += f" (reference range: {reference})"
                lines.append(line)
    elif len(frame) == 1:
        for column, label in columns.items():
            value = frame.iloc[0][column].strip()
            if value:
                lines.append(f"{label}: {value}")
    else:
        raise ValueError(
            "A multi-row CSV must use test/analyte and result/value columns, "
            "and contain only one patient's report."
        )
    return "\n".join(lines)


def extract_report_text(filename, file_bytes):
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploads are limited to 10 MB.")
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension == "pdf":
        text = extract_pdf_text(file_bytes)
        if len(text) > MAX_EXTRACTED_TEXT_CHARS:
            raise ValueError("The extracted report text is too large to review safely.")
        return text
    if extension in {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}:
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as error:
            raise RuntimeError("Install Pillow to read image reports.") from error

        try:
            image = Image.open(BytesIO(file_bytes))
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    "Images are limited to 20 megapixels for safe local processing."
                )
            image.load()
        except (
            OSError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
        ) as error:
            raise ValueError("The uploaded image could not be opened.") from error
        try:
            text = _ocr_image(image)
        finally:
            image.close()
        if len(text) > MAX_EXTRACTED_TEXT_CHARS:
            raise ValueError("The extracted report text is too large to review safely.")
        return text
    if extension == "docx":
        text = _extract_docx_text(file_bytes)
        if len(text) > MAX_EXTRACTED_TEXT_CHARS:
            raise ValueError("The extracted report text is too large to review safely.")
        return text
    if extension == "csv":
        text = _extract_csv_text(file_bytes)
        if len(text) > MAX_EXTRACTED_TEXT_CHARS:
            raise ValueError("The extracted report text is too large to review safely.")
        return text
    raise ValueError("Supported formats are PDF, PNG/JPG/TIFF/BMP/WEBP, DOCX, and CSV.")


def _extract_labeled_value(text, labels, value_pattern):
    label_pattern = "|".join(labels)
    match = re.search(
        rf"(?im)^\s*(?:{label_pattern})\s*(?:[:#\-]|\|)\s*({value_pattern})\s*$",
        text,
    )
    return match.group(1).strip() if match else None


def extract_patient_details(text):
    name = _extract_labeled_value(
        text,
        (r"patient\s+name", r"full\s+name"),
        r"[^\r\n|]+",
    )
    if name and name.casefold() in {"unknown", "n/a", "na", "not available"}:
        name = None

    age_value = _extract_labeled_value(
        text,
        (r"patient\s+age", r"age"),
        r"\d{1,3}(?:\s*(?:years?|yrs?))?",
    )
    age_match = re.search(r"\d{1,3}", age_value or "")
    age = int(age_match.group()) if age_match else None
    if age is not None and not 0 <= age <= 120:
        age = None

    raw_sex = _extract_labeled_value(
        text,
        (r"sex", r"gender"),
        r"[^\r\n|,;]+",
    )
    sex = None
    if raw_sex:
        normalized_sex = re.split(
            r"[\(/]", raw_sex.strip(), maxsplit=1
        )[0].strip().casefold()
        sex = {
            "m": "Male",
            "male": "Male",
            "man": "Male",
            "f": "Female",
            "female": "Female",
            "woman": "Female",
            "other": "Other",
        }.get(normalized_sex)

    contact = _extract_labeled_value(
        text,
        (r"phone", r"mobile", r"telephone"),
        r"[+()\d.\-\s]{7,}",
    )
    patient_identifier = _extract_labeled_value(
        text,
        (
            r"patient\s+(?:id|no\.?|number)",
            r"medical\s+record\s+(?:id|no\.?|number)",
            r"record\s+(?:id|no\.?|number)",
            r"mrn",
        ),
        r"[A-Za-z0-9\-_/]+",
    )
    return {
        "name": name.strip() if name else None,
        "age": age,
        "sex": sex,
        "contact": contact.strip() if contact else None,
        "external_id": patient_identifier.strip() if patient_identifier else None,
    }


def validate_reviewed_patient_details(name, age, sex, contact, external_id):
    normalized_name = " ".join(str(name or "").split())
    if not normalized_name:
        raise ValueError("Enter a patient name before continuing the review.")

    age_text = str(age or "").strip()
    if age_text:
        if not age_text.isdecimal() or not 0 <= int(age_text) <= 120:
            raise ValueError("Age must be a whole number from 0 to 120, or left blank.")
        reviewed_age = int(age_text)
    else:
        reviewed_age = None

    return {
        "name": normalized_name,
        "age": reviewed_age,
        "sex": sex if sex and sex != "Not specified" else None,
        "contact": " ".join(str(contact or "").split()) or None,
        "external_id": " ".join(str(external_id or "").split()) or None,
    }


def validate_reviewed_lab_values(rows):
    reviewed = []
    for row in rows:
        parameter = str(row.get("parameter") or "").strip()
        if parameter not in SUPPORTED_PARAMETERS:
            raise ValueError(f"Unsupported laboratory parameter: {parameter or '(blank)'}")
        raw_value = row.get("value")
        if pd.isna(raw_value) or str(raw_value).strip() == "":
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{parameter} must be a numeric value or blank.") from error
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{parameter} must be a finite, non-negative value.")
        context = str(row.get("context") or "Unspecified").strip()
        if parameter == "Glucose":
            if context in {"", "Not applicable"}:
                context = "Unspecified"
            if context not in {"Unspecified", "Fasting", "Post-load"}:
                raise ValueError(
                    "Glucose context must be Unspecified, Fasting, or Post-load."
                )
        else:
            context = ""
        reviewed.append(
            {
                "parameter": parameter,
                "value": value,
                "unit": " ".join(str(row.get("unit") or "").split()),
                "context": context,
                "reported_reference_range": str(
                    row.get("reported_reference_range") or ""
                ).strip(),
            }
        )
    return reviewed


def extract_lab_values(text):
    patterns = [
        (
            "glucose",
            r"(?im)^\s*(?:(?:2\s*[- ]?\s*(?:hour|hr)|2h)\s*"
            r"(?:OGTT\s+)?(?:post[- ]load\s+)?(?:plasma\s+)?|"
            r"(?:fasting\s+)?(?:plasma\s+)?)glucose"
            r"\s*(?:\((?P<annotation>[^)]*)\))?\s*(?:[:|]|-\s+)?\s*"
            rf"(?P<value>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)(?![\w.,])"
            rf"(?!\s*(?:-|–|to)\s*\d)"
            rf"\s*(?:\|\s*)?{_UNIT_PATTERN}(?![A-Za-z%])",
        ),
        (
            "hba1c",
            r"(?im)^\s*(?:hba1c|hb\s+a1c|a1c|glycated\s+hemoglobin)"
            r"\s*(?:\((?P<annotation>[^)]*)\))?\s*(?:[:|]|-\s+)?\s*"
            rf"(?P<value>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)(?![\w.,])"
            rf"(?!\s*(?:-|–|to)\s*\d)"
            rf"\s*(?:\|\s*)?{_UNIT_PATTERN}(?![A-Za-z%])",
        ),
        (
            "total cholesterol",
            r"(?im)^\s*(?:total\s+cholesterol|cholesterol)"
            r"\s*(?:\((?P<annotation>[^)]*)\))?\s*(?:[:|]|-\s+)?\s*"
            rf"(?P<value>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)(?![\w.,])"
            rf"(?!\s*(?:-|–|to)\s*\d)"
            rf"\s*(?:\|\s*)?{_UNIT_PATTERN}(?![A-Za-z%])",
        ),
        (
            "hemoglobin",
            r"(?im)^\s*hemoglobin\s*(?:\((?P<annotation>[^)]*)\))?\s*(?:[:|]|-\s+)?\s*"
            rf"(?P<value>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)(?![\w.,])"
            rf"(?!\s*(?:-|–|to)\s*\d)"
            rf"\s*(?:\|\s*)?{_UNIT_PATTERN}(?![A-Za-z%])",
        ),
        (
            "creatinine",
            r"(?im)^\s*creatinine\s*(?:\((?P<annotation>[^)]*)\))?\s*(?:[:|]|-\s+)?\s*"
            rf"(?P<value>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)(?![\w.,])"
            rf"(?!\s*(?:-|–|to)\s*\d)"
            rf"\s*(?:\|\s*)?{_UNIT_PATTERN}(?![A-Za-z%])",
        ),
    ]
    found = []
    for name, pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            source_unit = match.group("unit") or ""
            if not source_unit:
                annotation = match.group("annotation") or ""
                unit_match = re.search(_UNIT_TOKEN, annotation, re.IGNORECASE)
                source_unit = unit_match.group() if unit_match else ""
            normalized_unit = re.sub(r"\s+", "", source_unit).lower()
            unit_aliases = {
                "mg/dl": "mg/dL",
                "mgdl": "mg/dL",
                "mmol/l": "mmol/L",
                "mmoll": "mmol/L",
                "g/dl": "g/dL",
                "gdl": "g/dL",
                "g/l": "g/L",
                "gl": "g/L",
                "umol/l": "umol/L",
                "µmol/l": "umol/L",
                "μmol/l": "umol/L",
                "%": "%",
            }
            line_end = text.find("\n", match.start())
            if line_end < 0:
                line_end = len(text)
            source_line = text[match.start() : line_end]
            context = ""
            if name == "glucose":
                value_start = match.start("value") - match.start()
                label_text = source_line[:value_start]
                if re.search(
                    r"\b(?:2\s*[- ]?\s*(?:hour|hr)|2h|ogtt|post[- ]load)\b",
                    label_text,
                    re.IGNORECASE,
                ):
                    context = "Post-load"
                elif re.search(r"\bfasting\b", label_text, re.IGNORECASE):
                    context = "Fasting"
                else:
                    context = "Unspecified"
            found.append(
                {
                    "parameter": DISPLAY_NAMES[name],
                    "value": float(match.group("value").replace(",", "")),
                    "unit": unit_aliases.get(normalized_unit, ""),
                    "context": context,
                    "reported_reference_range": _reported_reference_range(
                        source_line
                    ),
                }
            )
    return found


def _reported_reference_range(line):
    match = re.search(
        r"\b(?:reference|ref|normal)\s*(?:range|interval)?\s*[:=]\s*"
        r"([^;\r\n]+)",
        line,
        re.IGNORECASE,
    )
    return match.group(1).strip(" ()") if match else ""


def flag_lab_values(values):
    output = []
    for item in values:
        key = item["parameter"].lower()
        low, high, expected_unit = REFERENCE_RANGES[key]
        reported_unit = item.get("unit", "")
        if not reported_unit:
            status = "Not compared (report unit missing)"
        elif reported_unit.casefold() != expected_unit.casefold():
            status = "Not compared (unit differs from illustrative interval)"
        elif key == "glucose" and item.get("context") != "Fasting":
            status = "Not compared (fasting context not confirmed)"
        elif low <= item["value"] <= high:
            status = "Within illustrative interval (not diagnostic)"
        else:
            status = "Outside illustrative interval (not diagnostic)"
        output.append(
            {
                **item,
                "reference_interval": f"{low}–{high} {expected_unit}",
                "interval_note": (
                    "Illustrative only; compare with the report's own range and "
                    "clinical context. The glucose interval is only compared for "
                    "explicitly fasting results."
                ),
                "status": status,
            }
        )
    return output
