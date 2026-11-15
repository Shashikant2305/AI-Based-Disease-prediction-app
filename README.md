# MediAI — Educational Symptom Pattern Explorer

MediAI is a local Streamlit educational prototype for exploring symptom patterns from a supplied dataset and reviewing a limited set of fields in laboratory reports. It is not a diagnostic, screening, treatment, or clinical product. Symptom scores are uncalibrated pattern-matching outputs, not diagnoses or disease probabilities. Nothing in this app is intended to guide patient care.

## Important privacy and deployment boundary

The app has **no authentication, authorization, access controls, or encryption**. It is intended only for local demonstrations with synthetic data. Do not enter real patient information, expose the app to a public or shared network, or treat its local SQLite file as protected storage. Adding authentication alone would not make it appropriate for clinical use.

The database at `data/mediai.db` stores patient profile fields (including optional notes), confirmed report summaries and supported lab values, SHA-256 upload fingerprints for duplicate prevention, and any symptom exploration explicitly linked to a patient. The original report file and full extracted text are not written to SQLite. Streamlit temporarily holds an uploaded file in session memory while it is being reviewed; reviewed fields are held in that session until the user saves or leaves the report page. Files are not sent to an external extraction service. Clear the upload or leave the page when finished.

## Report review and persistence

Uploading a report only extracts a draft; it does not create a patient or analysis. The user can correct extracted patient details and supported lab values, inspect possible profile matches and conflicting fields, then explicitly select an existing profile or confirm creation of a new one. Existing patient fields are never overwritten by report extraction. Each confirmed file fingerprint can be saved only once, including when two saves race.

Supported input formats: text or scanned PDF, PNG/JPG/TIFF/BMP/WEBP, DOCX, and single-patient CSV. Uploads are limited to 10 MB. PDF page count, image pixel count, DOCX expansion, CSV row count, and extracted text are also bounded. Scanned PDFs and images require the optional local Tesseract executable; `pytesseract` alone does not install it. Check with `tesseract --version`. If OCR is unavailable or extraction is unreadable, the app explains that nothing was saved and suggests a searchable PDF/DOCX/CSV or local OCR setup.

Supported extracted profile fields are name, age, sex, phone/contact, and patient/record ID. Supported lab fields are glucose, HbA1c, total cholesterol, hemoglobin, and creatinine. Extraction is label/pattern based and can miss or misread fields; review and correct the draft against the source before saving. Unknown units are not normalized. A value is compared only when its stated unit exactly matches the configured interval unit; missing or incompatible units are not compared. Glucose is compared only when the report explicitly confirms fasting context; post-load or unspecified glucose is not compared with the fasting interval. The app does not infer a diagnosis from a value. Configured intervals are illustrative only and may differ from the reporting laboratory's range and patient context.

## Supplied symptom dataset and evaluation

The symptom explorer uses root-level `Training.csv` and `Testing.csv`. `Training.csv` has 4,920 rows and 132 binary symptom features across 41 labels, but only 304 unique exact symptom/label patterns after removing 4,616 duplicate rows. The unique-pattern class sizes range from 5 to 10 examples. `Testing.csv` has 42 rows; 41 share an exact symptom pattern with the training data, so it is not an independent test set and is not used to report the model metrics.

Training fits one model row per unique pattern and reports a reproducible stratified 20% internal holdout of those unique patterns (seed 42). The supplied model artifact currently reports a perfect score on this small internal holdout. That result is not evidence that the model is reliable: the dataset is small, class sizes are limited, the held-out rows come from the same supplied dataset, and there is no external, prospective, or clinical validation. Random-forest scores are uncalibrated and must not be interpreted as disease probabilities. The app treats every unselected symptom as absent, so incomplete input can change the matches.

Dataset provenance, licensing, population, symptom definitions, and label quality have not been independently established. Confirm these before reuse or research. No independent dataset or clinical expertise has been used to validate the results.

## Features

- Educational symptom-pattern exploration with limitations displayed next to results.
- Local patient directory and analysis history in SQLite.
- Review-first report extraction with editable fields, visible identity suggestions/conflicts, explicit profile confirmation, and duplicate protection.
- Limited report parsing and illustrative unit-aware lab interval flags.
- PDF summaries of report extraction, symptom exploration, and history.

`predictor.py` and `train_demo_model.py` implement a separate Pima diabetes experiment and are not used by the current Streamlit workflow.

## Setup

Use the same `.venv` name on Windows, macOS, and Linux. Do not add a virtual environment to source control.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_symptom_model.py
streamlit run app.py
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_symptom_model.py
streamlit run app.py
```

The trainer creates `models/symptom_disease_model.joblib` and its metrics JSON. Install the system Tesseract OCR executable separately only if scanned-document OCR is needed. The Python requirements do not install that executable.

## Tests

Run the automated unit and Streamlit workflow tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

On macOS/Linux, use `python -m unittest discover -v` after activating `.venv`.

## Remaining limitations

- No authentication, access control, encryption, consent flow, audit trail, backup/restore, or multi-user deployment support.
- No clinical validation, calibration, external evaluation, prospective evaluation, fairness analysis, or intended-use approval.
- Report extraction is not guaranteed to understand all layouts; supported analytes and units are deliberately limited, no unit conversion is performed, and OCR may fail.
- Reference intervals are illustrative and are not personalized or validated by clinical experts.
- Dataset provenance and redistribution permissions require confirmation.
- This prototype must not be used for diagnosis, screening, treatment, or patient-care decisions.
