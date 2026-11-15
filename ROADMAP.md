# MediAI Development Roadmap

## Phase 0 — Current educational MVP
- Local patient records and analysis history in SQLite (no authentication, authorization, or encryption)
- Review-first report extraction with editable details, explicit profile selection/creation, and duplicate-save protection
- Pattern-based PDF, image, DOCX, and single-patient CSV extraction with upload/processing limits
- Limited supported lab parsing with unit-matched illustrative intervals; no diagnosis or unit conversion
- Educational multi-class symptom-pattern explorer trained on the supplied dataset
- Visible duplicate-row, class-size, train/test-overlap, and non-validation disclosures alongside model scores
- Streamlit UI, history analytics, and PDF exports
- Synthetic/local demo-data boundary documented; not safe for sensitive real patient data or public deployment

## Phase 1 — Reliable report intelligence
1. Improve OCR robustness for scanned PDFs and images; local Tesseract is currently optional and may fail.
2. Expand extraction of analyte, value, unit, date, and report-specific reference ranges without guessing.
3. Design and clinically review any unit normalization before implementation; current app does not convert units.
4. Evaluate extraction accuracy on permitted, de-identified reports; current parser has no confidence estimate.
5. Add additional lab panels only with reviewed parsing and reference-range requirements.
6. Continue parser tests with synthetic and permitted de-identified report layouts.

## Phase 2 — Validated ML pipeline
1. Select one disease/use case first (e.g. diabetes) and document intended population.
2. Verify dataset provenance, label definitions, permissions, and intended population; the current symptom dataset does not yet meet this bar.
3. Data quality checks, missing-value handling, outlier review and leakage prevention.
4. Stratified train/validation/test split and cross-validation on data independent at the subject/source level; the current holdout is a small split of unique supplied patterns only.
5. Compare Logistic Regression, Random Forest and Gradient Boosting.
6. Report metrics appropriate to the task, calibration where probability estimates are intended, and uncertainty.
7. Evaluate class imbalance, subgroup performance and confidence intervals.
8. Model versioning, reproducible training and model cards.
9. Keep external validation as a separate milestone.

## Phase 3 — Multimodal patient profile
1. Structured EHR-like history and medication/allergy list.
2. CSV upload for clinical/lab history.
3. Sensor data ingestion (manual CSV first; device API later).
4. Time-series charts for heart rate, SpO2, temperature and blood pressure.
5. Image upload pipeline; DICOM support and validated imaging models only after dataset and expertise are available.
6. Patient timeline combining reports, observations and predictions.

## Phase 4 — Explainability and clinician workflow
1. Validate explanations and model outputs before any clinical use; the current symptom explorer does not provide causal explanations.
2. Show missing inputs, model version, prediction timestamp and limitations.
3. Clinician review queue and approve/reject/comment workflow.
4. Referral/consultation notes, not automated treatment prescriptions.
5. Feedback capture for prospective evaluation; never silently retrain on feedback.

## Phase 5 — Security, interoperability and deployment
1. Authentication with secure password hashing and role-based authorization.
2. Consent, data retention, deletion and access history.
3. Encryption in transit and at rest; secrets management.
4. FHIR-compatible import/export after defining interoperability requirements.
5. PostgreSQL, migrations, backups and restore tests.
6. FastAPI service layer, Docker, CI tests, monitoring and cloud deployment.
7. Threat modelling, penetration testing and privacy/legal review.
8. Keep the current app local and synthetic-data-only until authentication, authorization, data protection, retention, and deployment controls are designed, implemented, and independently reviewed.

## Phase 6 — Research validation and final-year demonstration
1. Freeze a protocol and intended-use statement before evaluation.
2. Obtain required approvals and permissions for any human data or clinical study.
3. Validate on representative, independent data.
4. Document bias, limitations, failure cases and data drift.
5. Conduct usability testing with intended users.
6. Prepare architecture diagrams, sequence diagrams, test report, model card and dissertation evidence.

## Recommended priorities
The current app is a local educational demo, not a validated or secure clinical product. Prioritize permitted dataset provenance and independent validation, extraction-quality assessment, and security/privacy controls before adding modalities or considering real data. Treat clinical expertise, privacy/legal review, and deployment security review as required gates before any real-world or networked use.
