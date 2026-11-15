import hashlib
import sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime
from database import (
    init_db,
    add_patient,
    delete_patient,
    find_patient_matches,
    find_uploaded_report,
    list_patients,
    save_analysis,
    save_uploaded_report,
    list_analyses,
)
from report_analyzer import (
    extract_patient_details,
    extract_report_text,
    extract_lab_values,
    flag_lab_values,
    validate_reviewed_lab_values,
    validate_reviewed_patient_details,
    MAX_UPLOAD_BYTES,
)
from symptom_predictor import (
    DATASET_NAME,
    MODEL_VERSION,
    format_symptom_name,
    load_model as load_symptom_model,
    load_model_metrics as load_symptom_model_metrics,
    load_symptom_features,
    predict_disease_candidates,
)
from pdf_report import create_pdf_bytes

def _navigate_to_page(page):
    st.session_state["workspace_page_v2"] = page


@st.cache_resource
def _get_symptom_model():
    return load_symptom_model()


@st.cache_data
def _get_symptom_model_metrics():
    return load_symptom_model_metrics()


@st.cache_data
def _get_symptom_features():
    return load_symptom_features()


def _apply_app_styles():
    st.markdown(
        """
        <style>
        :root {
            --ink: #0f241f;
            --muted: #5a706a;
            --teal: #135a4b;
            --teal-dark: #09382e;
            --mint: #e9f2ef;
            --line: #e4e9e7;
            --canvas: #fafbf9;
            --champagne: #c59b5f;
            --amber-wash: #fdf9ee;
            --amber-line: #e6d3a3;
            --amber-text: #7c5812;
        }

        .stApp, .stApp button, .stApp input, .stApp textarea {
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .stApp {
            background: var(--canvas);
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: rgba(250, 251, 249, 0.9);
        }
        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 1440px;
            padding: 1.25rem 2.6rem 3.5rem;
        }
        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: -0.035em;
        }
        h1, h2, h3 { font-family: Georgia, "Times New Roman", serif; }
        h1 { font-weight: 600; }
        h2, h3 { font-weight: 600; }
        p, label, [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f4f7f5 0%, #fafbf9 78%);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.3rem;
        }
        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.65rem 0.45rem 1.25rem;
        }
        .brand-mark {
            display: grid;
            width: 2.7rem;
            height: 2.7rem;
            place-items: center;
            border-radius: 0.95rem;
            background: linear-gradient(145deg, #135a4b, #09382e);
            color: white;
            font-size: 1.5rem;
            font-weight: 700;
            box-shadow: 0 8px 18px rgba(9, 56, 46, 0.18);
        }
        .brand-name {
            color: var(--ink);
            font-size: 1.18rem;
            font-weight: 760;
            letter-spacing: -0.04em;
            line-height: 1.1;
        }
        .brand-caption {
            color: #81938f;
            font-size: 0.64rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            margin-top: 0.3rem;
        }
        .sidebar-section-label {
            color: #82948f;
            font-size: 0.67rem;
            font-weight: 750;
            letter-spacing: 0.14em;
            margin: 0.4rem 0 0.5rem 0.5rem;
            text-transform: uppercase;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] > div {
            gap: 0.28rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 0.8rem;
            padding: 0.42rem 0.55rem;
            transition: background 150ms ease, color 150ms ease;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: #e9f2ef;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: #fff;
            box-shadow: 0 2px 8px rgba(9, 56, 46, 0.05);
            color: var(--teal-dark);
            font-weight: 700;
        }
        .sidebar-note {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line);
            border-radius: 0.9rem;
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.5;
            margin: 1.1rem 0.2rem;
            padding: 0.85rem;
        }
        .sidebar-note strong {
            color: var(--ink);
            display: block;
            margin-bottom: 0.2rem;
        }

        .page-overline {
            color: var(--teal);
            font-size: 0.7rem;
            font-weight: 760;
            letter-spacing: 0.16em;
            margin: 0.2rem 0 0.3rem;
            text-transform: uppercase;
        }
        .workspace-bar {
            align-items: center;
            border-bottom: 1px solid var(--line);
            color: var(--muted);
            display: flex;
            font-size: 0.78rem;
            gap: 0.65rem;
            justify-content: space-between;
            margin: 0.1rem 0 1.5rem;
            padding: 0.35rem 0 0.9rem;
        }
        .workspace-crumb { align-items: center; display: flex; gap: 0.6rem; }
        .workspace-crumb strong { color: var(--ink); font-weight: 700; }
        .workspace-divider { color: #b6c2be; }
        .page-heading {
            align-items: flex-end;
            display: flex;
            gap: 1.5rem;
            justify-content: space-between;
            margin: 0.4rem 0 1.35rem;
        }
        .page-heading h1 {
            font-size: clamp(1.9rem, 3vw, 2.65rem);
            letter-spacing: -0.045em;
            line-height: 1.08;
            margin: 0;
        }
        .page-heading p {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.6;
            margin: 0.45rem 0 0;
            max-width: 44rem;
        }
        .page-heading-badge {
            background: #fff;
            border: 1px solid var(--line);
            border-radius: 0.85rem;
            color: var(--muted);
            flex: 0 0 auto;
            font-size: 0.63rem;
            font-weight: 760;
            letter-spacing: 0.1em;
            padding: 0.65rem 0.8rem;
            text-align: center;
            text-transform: uppercase;
        }
        .page-heading-badge strong {
            color: var(--champagne);
            display: block;
            font-family: "Courier New", monospace;
            font-size: 0.65rem;
            letter-spacing: 0.08em;
            margin-top: 0.2rem;
        }
        .research-disclaimer {
            background: var(--amber-wash);
            border: 1px solid var(--amber-line);
            border-radius: 0.95rem;
            color: var(--amber-text);
            font-size: 0.82rem;
            line-height: 1.55;
            margin: 0 0 1.2rem;
            padding: 0.8rem 1rem;
        }
        .hero-panel {
            align-items: center;
            background:
                radial-gradient(circle at 86% 18%, rgba(197, 155, 95, 0.22), transparent 34%),
                radial-gradient(circle at 14% 88%, rgba(25, 114, 95, 0.42), transparent 42%),
                linear-gradient(125deg, #072c24 0%, #0d473b 58%, #135a4b 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 1.5rem;
            box-shadow: 0 16px 36px rgba(9, 56, 46, 0.14);
            color: white;
            display: flex;
            justify-content: space-between;
            margin: 0.9rem 0 1.5rem;
            min-height: 15.5rem;
            overflow: hidden;
            padding: 2.25rem 2.6rem;
            position: relative;
        }
        .hero-copy { max-width: 650px; position: relative; z-index: 1; }
        .hero-label {
            color: #f0d7a8;
            font-size: 0.68rem;
            font-weight: 760;
            letter-spacing: 0.16em;
            margin-bottom: 0.85rem;
            text-transform: uppercase;
        }
        .hero-title {
            color: #fff;
            font-size: clamp(2rem, 3.5vw, 3.1rem);
            font-weight: 760;
            letter-spacing: -0.055em;
            line-height: 1.04;
        }
        .hero-copy p {
            color: rgba(239, 250, 246, 0.82);
            font-size: 1rem;
            line-height: 1.65;
            margin: 0.9rem 0 0;
            max-width: 580px;
        }
        .hero-aside {
            align-items: center;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 1.2rem;
            display: flex;
            gap: 0.9rem;
            margin-left: 1.5rem;
            max-width: 250px;
            padding: 1.1rem;
            position: relative;
            z-index: 1;
        }
        .hero-orb {
            align-items: center;
            background: rgba(197, 155, 95, 0.2);
            border: 1px solid rgba(197, 155, 95, 0.45);
            border-radius: 50%;
            color: #f0d7a8;
            display: flex;
            flex: 0 0 2.8rem;
            font-size: 1.4rem;
            height: 2.8rem;
            justify-content: center;
            width: 2.8rem;
        }
        .hero-aside span {
            color: #c7e2d8;
            display: block;
            font-size: 0.58rem;
            font-weight: 750;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
        }
        .hero-aside strong {
            color: white;
            font-size: 0.95rem;
            line-height: 1.35;
        }
        .section-heading {
            color: var(--ink);
            font-size: 1.15rem;
            font-weight: 720;
            letter-spacing: -0.025em;
            margin: 1.55rem 0 0.8rem;
        }
        .workflow-card {
            background: #fff;
            border: 1px solid var(--line);
            border-radius: 1.05rem;
            box-shadow: 0 5px 16px rgba(31, 74, 66, 0.035);
            height: 100%;
            padding: 1.1rem 1.15rem;
        }
        .workflow-number {
            align-items: center;
            background: var(--mint);
            border-radius: 0.65rem;
            color: var(--teal-dark);
            display: flex;
            font-size: 0.76rem;
            font-weight: 760;
            height: 2rem;
            justify-content: center;
            margin-bottom: 0.85rem;
            width: 2rem;
        }
        .workflow-title {
            color: var(--ink);
            font-size: 0.96rem;
            font-weight: 720;
            margin-bottom: 0.35rem;
        }
        .workflow-copy {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.55;
        }

        [data-testid="stMetric"] {
            background: #fff;
            border: 1px solid var(--line);
            border-radius: 1rem;
            box-shadow: 0 5px 16px rgba(31, 74, 66, 0.035);
            padding: 1rem 1.1rem;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 620;
        }
        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 760;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #fff;
            border-color: var(--line);
            border-radius: 1rem;
            box-shadow: 0 5px 16px rgba(31, 74, 66, 0.035);
        }
        .stButton > button, .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] button {
            border: 1px solid #d5e5df;
            border-radius: 0.75rem;
            color: var(--teal-dark);
            font-weight: 680;
            min-height: 2.7rem;
            transition: background 150ms ease, box-shadow 150ms ease, transform 150ms ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            border-color: #b5c9c1;
            box-shadow: 0 5px 14px rgba(9, 56, 46, 0.1);
            color: var(--teal-dark);
            transform: translateY(-1px);
        }
        .stButton > button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            background: var(--teal);
            border-color: var(--teal);
            color: white;
        }
        .stButton > button[kind="primary"]:hover,
        [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
            background: var(--teal-dark);
            border-color: var(--teal-dark);
            color: white;
        }
        [data-testid="stAlert"] {
            border-radius: 0.9rem;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid var(--line);
            border-radius: 0.85rem;
            overflow: hidden;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: #fafbf9;
            border: 1px dashed #b5c9c1;
            border-radius: 1rem;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea {
            border-radius: 0.7rem;
        }
        :focus-visible {
            outline: 3px solid rgba(19, 90, 75, 0.42);
            outline-offset: 2px;
        }
        @media (max-width: 760px) {
            [data-testid="stAppViewContainer"] .main .block-container {
                padding: 1.2rem 1rem 2rem;
            }
            .page-heading { align-items: flex-start; }
            .page-heading-badge { display: none; }
            .hero-panel {
                align-items: flex-start;
                border-radius: 1.15rem;
                flex-direction: column;
                min-height: auto;
                padding: 1.6rem;
            }
            .hero-aside {
                margin: 1.25rem 0 0;
                max-width: none;
            }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important;
                transition-duration: 0.01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_page_heading(overline, title, description):
    st.markdown(
        f"""
        <div class="page-heading">
            <div>
                <div class="page-overline">{overline}</div>
                <h1>{title}</h1>
                <p>{description}</p>
            </div>
            <div class="page-heading-badge">
                Research workspace
                <strong>LOCAL · EDUCATIONAL</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_symptom_candidates(selected_symptoms, patient_id):
    try:
        metrics = _get_symptom_model_metrics()
        candidates = predict_disease_candidates(
            _get_symptom_model(),
            selected_symptoms,
        )
        class_counts = metrics["class_counts_unique"]
        st.warning(
            "Educational pattern matches only — not diagnosis, screening, treatment "
            "guidance, or disease probabilities. The supplied dataset has "
            f"{metrics['training_rows']:,} rows but only "
            f"{metrics['unique_training_patterns']:,} unique exact patterns after "
            f"removing {metrics['duplicate_training_rows_removed']:,} duplicate rows; "
            f"{metrics['disease_classes']} labels have {metrics['min_unique_patterns_per_class']}–"
            f"{metrics['max_unique_patterns_per_class']} unique patterns each. "
            f"{metrics['provided_test_rows_overlapping_training']}/"
            f"{metrics['provided_test_rows']} supplied test rows overlap training. "
            f"These uncalibrated scores come from a {metrics['holdout_rows']}-pattern "
            "internal holdout only; there is no external validation. Even a perfect "
            "holdout score does not establish reliability."
        )
        st.subheader("Educational dataset pattern matches")
        for candidate in candidates:
            st.metric(
                f"Dataset label: {candidate['label']}",
                f"{candidate['score']:.1%} educational pattern-match score",
            )
        st.dataframe(
            pd.DataFrame(
                {
                    "Selected symptom": [
                        format_symptom_name(symptom)
                        for symptom in selected_symptoms
                    ]
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Dataset and internal evaluation details"):
            st.write(f"Model: {MODEL_VERSION}")
            st.write(f"Dataset: {DATASET_NAME}")
            st.write(
                f"{metrics['training_rows']:,} source rows, "
                f"{metrics['unique_training_patterns']:,} unique patterns, "
                f"{metrics['symptom_features']} binary symptom fields, and "
                f"{metrics['disease_classes']} labels."
            )
            st.write(f"Unique-pattern class sizes: {class_counts}")
            st.write(
                f"{metrics['holdout_method']} (seed {metrics['holdout_seed']}): "
                f"accuracy {metrics['holdout_accuracy']:.3f}, balanced accuracy "
                f"{metrics['holdout_balanced_accuracy']:.3f}, macro-F1 "
                f"{metrics['holdout_macro_f1']:.3f}, top-3 accuracy "
                f"{metrics['holdout_top_3_accuracy']:.3f}."
            )
            st.caption(
                f"Testing.csv was not used for the metrics: "
                f"{metrics['provided_test_rows_overlapping_training']} of "
                f"{metrics['provided_test_rows']} rows overlap. "
                f"{metrics['evaluation_note']}"
            )
        input_summary = ", ".join(
            format_symptom_name(symptom) for symptom in selected_symptoms
        )
        candidate_summary = "; ".join(
            f"{item['label']}: {item['score']:.3f} uncalibrated pattern score"
            for item in candidates
        )
        if patient_id:
            save_analysis(
                patient_id,
                "Symptom pattern exploration",
                f"Educational dataset pattern matches only; not a diagnosis, screening, "
                f"treatment recommendation, or probability. {candidate_summary} | "
                f"Selected symptoms: {input_summary}. Dataset has "
                f"{metrics['unique_training_patterns']} unique patterns "
                f"({metrics['duplicate_training_rows_removed']} duplicate rows removed), "
                f"{metrics['disease_classes']} labels with "
                f"{metrics['min_unique_patterns_per_class']}–"
                f"{metrics['max_unique_patterns_per_class']} patterns each; "
                f"{metrics['provided_test_rows_overlapping_training']}/"
                f"{metrics['provided_test_rows']} supplied test rows overlap. "
                "Internal holdout is not external validation."
            )
        prediction_lines = [
            f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
            f"Model: {MODEL_VERSION}",
            f"Dataset: {DATASET_NAME}",
            "Top pattern matches:",
            *[
                f"{item['label']}: {item['score']:.4f} uncalibrated model score"
                for item in candidates
            ],
            "",
            "Selected symptoms:",
            *[format_symptom_name(symptom) for symptom in selected_symptoms],
            "",
            "EDUCATIONAL PATTERN MATCHES ONLY — not diagnoses, screening, treatment "
            "guidance, or disease probabilities.",
            (
                f"The dataset has {metrics['training_rows']} rows, "
                f"{metrics['unique_training_patterns']} unique patterns, and "
                f"{metrics['duplicate_training_rows_removed']} duplicate rows."
            ),
            (
                f"{metrics['disease_classes']} labels have "
                f"{metrics['min_unique_patterns_per_class']} to "
                f"{metrics['max_unique_patterns_per_class']} unique patterns each."
            ),
            (
                f"{metrics['provided_test_rows_overlapping_training']} of "
                f"{metrics['provided_test_rows']} supplied test rows overlap "
                "training patterns; that file was not used for these metrics."
            ),
            (
                "The small internal holdout is not external, prospective, or clinical "
                "validation. Perfect holdout metrics do not establish reliability. "
                "Do not use for patient care."
            ),
            "",
            f"Unique-pattern internal holdout metrics: accuracy "
            f"{metrics['holdout_accuracy']:.3f}; balanced accuracy "
            f"{metrics['holdout_balanced_accuracy']:.3f}; macro-F1 "
            f"{metrics['holdout_macro_f1']:.3f}; top-3 accuracy "
            f"{metrics['holdout_top_3_accuracy']:.3f}.",
        ]
        st.download_button(
            "Download symptom exploration summary (PDF)",
            data=create_pdf_bytes("Educational Symptom Pattern Summary", prediction_lines),
            file_name="symptom-pattern-exploration.pdf",
            mime="application/pdf",
        )
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Could not run symptom exploration: {error}")


st.set_page_config(page_title="MediAI | Educational Symptom Explorer", page_icon="🩺", layout="wide")
_apply_app_styles()
try:
    init_db()
except (sqlite3.Error, RuntimeError) as error:
    st.error(f"Could not initialize the local database: {error}")
    st.stop()

st.sidebar.markdown(
    """
    <div class="brand-lockup">
        <div class="brand-mark">+</div>
        <div>
            <div class="brand-name">MediAI</div>
            <div class="brand-caption">HEALTH WORKSPACE</div>
        </div>
    </div>
    <div class="sidebar-section-label">Workspace modules</div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Workspace",
    ["Overview", "Patient Management", "Medical Report Analyzer", "Symptom Explorer", "Patient History"],
    key="workspace_page_v2",
    label_visibility="collapsed",
)
if page != "Medical Report Analyzer":
    for state_key in list(st.session_state):
        if state_key.startswith("report_review_"):
            st.session_state.pop(state_key, None)
st.sidebar.markdown(
    """
    <div class="sidebar-note">
        <strong>Local educational demo only</strong>
        No authentication or access controls. Use synthetic data only; records are stored in a local SQLite file.
    </div>
    """,
    unsafe_allow_html=True,
)

bar, action = st.columns([5, 1.35])
with bar:
    st.markdown(
        f"""
        <div class="workspace-bar">
            <div class="workspace-crumb">
                <span>MediAI Workspace</span>
                <span class="workspace-divider">/</span>
                <strong>{page}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with action:
    st.button(
        "＋  New analysis",
        key="header_new_analysis",
        on_click=_navigate_to_page,
        args=("Medical Report Analyzer",),
        use_container_width=True,
    )

st.markdown(
    """
    <div class="research-disclaimer">
        <strong>Education-only prototype:</strong> Symptom matches are dataset pattern comparisons,
        not diagnoses, screening, treatment guidance, or disease probabilities. No authentication,
        access controls, or clinical validation are implemented. Use synthetic demo data only; do not
        deploy this app on a public network or use it for patient care.
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Overview":
    _render_page_heading(
        "Education & exploratory pattern matching",
        "MediAI Symptom Workspace",
        "Explore symptom patterns from the supplied dataset alongside saved patient records and laboratory report summaries.",
    )
    st.markdown(
        """
        <section class="hero-panel">
            <div class="hero-copy">
                <div class="hero-label">EDUCATION &amp; EXPLORATION</div>
                <div class="hero-title">Explore symptom<br>patterns.</div>
                <p>Review educational model matches from the supplied symptom dataset. Results are not diagnoses or care guidance.</p>
            </div>
            <div class="hero-aside">
                <div class="hero-orb">✦</div>
                <div>
                    <span>DESIGNED FOR CLARITY</span>
                    <strong>One workspace.<br>More context.</strong>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    patients = list_patients()
    analyses = list_analyses()
    workspace_patients, workspace_analyses = st.columns(2)
    workspace_patients.metric(
        "Registered patients",
        len(patients),
        help="Patient records saved in this workspace",
    )
    workspace_analyses.metric(
        "Saved analyses",
        len(analyses),
        help="Reports and experimental predictions in patient history",
    )

    st.markdown('<div class="section-heading">Symptom dataset &amp; model snapshot</div>', unsafe_allow_html=True)
    try:
        model_metrics = _get_symptom_model_metrics()
    except (FileNotFoundError, ValueError):
        model_metrics = None

    if model_metrics:
        dataset_rows, unique_patterns, disease_count, balanced_accuracy = st.columns(4)
        dataset_rows.metric("Training rows", model_metrics["training_rows"])
        unique_patterns.metric(
            "Unique symptom patterns",
            model_metrics["unique_training_patterns"],
        )
        disease_count.metric("Prognosis labels", model_metrics["disease_classes"])
        balanced_accuracy.metric(
            "Internal balanced accuracy",
            f"{model_metrics['holdout_balanced_accuracy']:.3f}",
            help="Stratified holdout of unique training patterns; not external validation",
        )
        st.caption(
            f"{DATASET_NAME}: {model_metrics['symptom_features']} binary symptom fields. "
            f"{model_metrics['unique_training_patterns']} unique patterns; each of "
            f"{model_metrics['disease_classes']} labels has "
            f"{model_metrics['min_unique_patterns_per_class']}–"
            f"{model_metrics['max_unique_patterns_per_class']} unique patterns. "
            "The internal holdout is not evidence of reliability or clinical validation."
        )
        st.warning(
            f"Training.csv has {model_metrics['training_rows']:,} rows and "
            f"{model_metrics['duplicate_training_rows_removed']:,} duplicate rows. "
            f"Testing.csv overlaps {model_metrics['provided_test_rows_overlapping_training']} of "
            f"{model_metrics['provided_test_rows']} rows in training; it is not an independent test set. "
            "A perfect internal holdout score does not establish real-world reliability."
        )
    else:
        st.info("Run `python train_symptom_model.py` to train the supplied dataset and display metrics.")

    st.markdown('<div class="section-heading">Workspace shortcuts</div>', unsafe_allow_html=True)
    action_columns = st.columns(3)
    with action_columns[0]:
        st.button(
            "＋  Add a patient",
            key="overview_add_patient",
            on_click=_navigate_to_page,
            args=("Patient Management",),
            use_container_width=True,
            type="primary",
        )
    with action_columns[1]:
        st.button(
            "↥  Analyze a report",
            key="overview_analyze_report",
            on_click=_navigate_to_page,
            args=("Medical Report Analyzer",),
            use_container_width=True,
        )
    with action_columns[2]:
        st.button(
            "◉  Explore symptoms",
            key="overview_open_prediction",
            on_click=_navigate_to_page,
            args=("Symptom Explorer",),
            use_container_width=True,
        )

    st.markdown('<div class="section-heading">A clear workflow</div>', unsafe_allow_html=True)
    workflow_columns = st.columns(3)
    workflow_cards = [
        ("01", "Set up a patient profile", "Create a patient record to keep reports and analyses organized."),
        ("02", "Review a lab report", "Upload a text-based PDF and inspect the values MediAI can identify."),
        ("03", "Explore symptom patterns", "Select symptoms and review non-diagnostic matches from the supplied dataset."),
    ]
    for column, (number, title, copy) in zip(workflow_columns, workflow_cards):
        with column:
            st.markdown(
                f"""
                <div class="workflow-card">
                    <div class="workflow-number">{number}</div>
                    <div class="workflow-title">{title}</div>
                    <div class="workflow-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption(
        "Current support: educational symptom-pattern matching and text extraction from text-based PDFs. "
        "Imaging, live sensors, EHR integrations and clinical validation are not implemented."
    )
    if analyses:
        st.markdown('<div class="section-heading">Recent workspace activity</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(analyses).head(8),
            use_container_width=True,
            hide_index=True,
        )

elif page == "Patient Management":
    _render_page_heading(
        "Cohort directory",
        "Patient Management",
        "Create and organize patient records for report reviews and saved analysis history.",
    )
    with st.container(border=True):
        st.markdown('<div class="section-heading">Register a patient</div>', unsafe_allow_html=True)
        with st.form("patient_form"):
            name = st.text_input("Patient name")
            details_left, details_right = st.columns(2)
            with details_left:
                age = st.number_input("Age", 0, 120, 25)
            with details_right:
                sex = st.selectbox("Sex", ["Not specified", "Female", "Male", "Other"])
            contact = st.text_input("Contact (optional)")
            external_id = st.text_input("Patient/record ID (optional)")
            notes = st.text_area("Relevant medical history (optional)")
            submitted = st.form_submit_button("Save patient", type="primary")
    if submitted:
        if not name.strip():
            st.error("Please enter a patient name.")
        else:
            try:
                patient_id, add_status = add_patient(
                    name.strip(),
                    int(age),
                    sex,
                    contact.strip(),
                    notes.strip(),
                    external_id.strip() or None,
                )
            except (ValueError, sqlite3.Error, RuntimeError) as error:
                st.error(f"Could not save patient: {error}")
            else:
                if add_status == "existing":
                    st.info(
                        f"Matching profile reused (ID {patient_id}); its existing details "
                        "were not changed."
                    )
                elif add_status == "created_similar":
                    st.warning(
                        f"Patient profile created separately (ID {patient_id}). One or more "
                        "profiles have similar identifiers or demographics; they were not merged."
                    )
                else:
                    st.success(f"Patient record saved (ID {patient_id}).")
    delete_notice = st.session_state.pop("patient_delete_notice", None)
    if delete_notice:
        st.success(delete_notice)
    st.markdown('<div class="section-heading">Patient directory</div>', unsafe_allow_html=True)
    rows = list_patients()
    if rows:
        with st.container(border=True):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with st.expander("Remove a patient", expanded=False):
            patient_options = {
                f"{patient['name']} (ID {patient['id']})": patient["id"]
                for patient in rows
            }
            with st.form("delete_patient_form"):
                selected_patient = st.selectbox(
                    "Select patient to remove",
                    list(patient_options),
                )
                confirmed = st.checkbox(
                    "I understand this permanently deletes the patient and all linked analyses."
                )
                remove_submitted = st.form_submit_button(
                    "Remove patient and linked history",
                    type="secondary",
                )
            if remove_submitted:
                if not confirmed:
                    st.error("Confirm the permanent removal before continuing.")
                else:
                    try:
                        deleted_analyses = delete_patient(
                            patient_options[selected_patient]
                        )
                    except (ValueError, sqlite3.Error, RuntimeError) as error:
                        st.error(f"Could not remove patient: {error}")
                    else:
                        st.session_state["patient_delete_notice"] = (
                            f"Patient and {deleted_analyses} linked analysis record(s) removed."
                        )
                        st.rerun()
    else:
        st.info("No patients yet. Add a demo patient above.")

elif page == "Medical Report Analyzer":
    _render_page_heading(
        "Review before saving",
        "Medical Report Analyzer",
        "Inspect and edit extracted details and laboratory values, explicitly choose a patient profile, then save only after confirmation.",
    )
    st.info(
        "Nothing is written to patient records or history during upload or extraction. Existing "
        "profiles are never changed by extracted values. You must review the extraction and explicitly "
        "confirm an existing profile or creation of a new one before a single save is made."
    )
    st.caption(
        "Supported formats: digital or scanned PDF, PNG/JPG/TIFF/BMP/WEBP, DOCX, and single-patient CSV. "
        "Scanned documents require the optional local Tesseract OCR executable. Files are limited to "
        "10 MB; PDF pages, image pixels, document expansion, and extracted text are also bounded. "
        "Only labeled demographics and supported lab fields are mapped; other extracted text is not interpreted."
    )
    saved_notice = st.session_state.pop("report_saved_notice", None)
    if saved_notice:
        st.success(saved_notice)
    upload_nonce = st.session_state.setdefault("report_upload_nonce", 0)
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Upload a laboratory report",
            type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "docx", "csv"],
            key=f"report_upload_{upload_nonce}",
            help=(
                "Processed locally by this app. The original is not written to the database, "
                "but Streamlit temporarily holds the upload in this session until it is cleared."
            ),
        )
    if not uploaded:
        for state_key in list(st.session_state):
            if state_key.startswith("report_review_"):
                st.session_state.pop(state_key, None)
    if uploaded:
        if uploaded.size > MAX_UPLOAD_BYTES:
            for state_key in list(st.session_state):
                if state_key.startswith("report_review_"):
                    st.session_state.pop(state_key, None)
            st.error("This file exceeds the 10 MB limit. Choose a smaller report.")
        else:
            try:
                file_bytes = uploaded.getvalue()
                fingerprint = hashlib.sha256(file_bytes).hexdigest()
                report_state = st.session_state.get("report_review_state")
                if report_state and report_state["fingerprint"] != fingerprint:
                    for state_key in list(st.session_state):
                        if state_key.startswith("report_review_"):
                            st.session_state.pop(state_key, None)
                duplicate = find_uploaded_report(fingerprint)
                if duplicate:
                    for state_key in list(st.session_state):
                        if state_key.startswith("report_review_"):
                            st.session_state.pop(state_key, None)
                    st.session_state["report_saved_notice"] = (
                        f"This exact file was already saved once for "
                        f"{duplicate['name']} (patient ID {duplicate['patient_id']}); "
                        "no new profile or history entry was created."
                    )
                    st.session_state["report_upload_nonce"] += 1
                    st.rerun()

                raw_text = extract_report_text(uploaded.name, file_bytes)
            except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
                st.error(f"Could not process this report: {error}")
            else:
                del file_bytes
                if not raw_text.strip():
                    st.error(
                        "No readable text was extracted, so nothing was saved. "
                        "Try a text-based PDF or DOCX/CSV. For a scan or image, install "
                        "the Tesseract executable locally and confirm `tesseract --version` works."
                    )
                else:
                    with st.expander("Extracted report text (read-only)", expanded=False):
                        st.code(raw_text[:12000])
                        if len(raw_text) > 12000:
                            st.caption("Display truncated to 12,000 characters; extraction remains bounded.")

                    report_state = st.session_state.get("report_review_state")
                    if report_state is None:
                        extracted_details = extract_patient_details(raw_text)
                        extracted_values = extract_lab_values(raw_text)
                        st.subheader("Review and edit extracted patient details")
                        st.caption(
                            "These values are a draft only. Correct them or enter missing details; "
                            "no patient profile or analysis is saved at this stage."
                        )
                        with st.form(f"report_review_{fingerprint[:12]}"):
                            patient_name = st.text_input(
                                "Patient name",
                                extracted_details["name"] or "",
                                key=f"report_review_name_{fingerprint[:12]}",
                            )
                            age = st.text_input(
                                "Age in years (optional)",
                                str(extracted_details["age"])
                                if extracted_details["age"] is not None
                                else "",
                                key=f"report_review_age_{fingerprint[:12]}",
                            )
                            sex_options = ["Not specified", "Female", "Male", "Other"]
                            extracted_sex = extracted_details["sex"] or "Not specified"
                            sex = st.selectbox(
                                "Sex (optional)",
                                sex_options,
                                index=sex_options.index(extracted_sex),
                                key=f"report_review_sex_{fingerprint[:12]}",
                            )
                            contact = st.text_input(
                                "Contact (optional)",
                                extracted_details["contact"] or "",
                                key=f"report_review_contact_{fingerprint[:12]}",
                            )
                            external_id = st.text_input(
                                "Patient/record ID (optional)",
                                extracted_details["external_id"] or "",
                                key=f"report_review_id_{fingerprint[:12]}",
                            )
                            st.subheader("Review and edit recognized laboratory values")
                            if extracted_values:
                                lab_frame = pd.DataFrame(
                                    [
                                        {
                                            "parameter": item["parameter"],
                                            "value": item["value"],
                                            "unit": item["unit"],
                                            "context": item.get("context", ""),
                                            "reported_reference_range": item.get(
                                                "reported_reference_range", ""
                                            ),
                                        }
                                        for item in extracted_values
                                    ]
                                )
                                reviewed_lab_frame = st.data_editor(
                                    lab_frame,
                                    disabled=["parameter"],
                                    column_config={
                                        "context": st.column_config.SelectboxColumn(
                                            "Glucose context",
                                            options=[
                                                "",
                                                "Not applicable",
                                                "Unspecified",
                                                "Fasting",
                                                "Post-load",
                                            ],
                                            help=(
                                                "For glucose only. Confirm fasting status "
                                                "from the source report; post-load or "
                                                "unspecified values are not compared."
                                            ),
                                        )
                                    },
                                    num_rows="fixed",
                                    use_container_width=True,
                                    key=f"report_review_labs_{fingerprint[:12]}",
                                )
                            else:
                                reviewed_lab_frame = pd.DataFrame(
                                    columns=[
                                        "parameter",
                                        "value",
                                        "unit",
                                        "context",
                                        "reported_reference_range",
                                    ]
                                )
                                st.info(
                                    "No supported results were recognized. You may still review "
                                    "the extracted text, correct the patient details, and save a "
                                    "report summary with no interpreted lab values."
                                )
                            review_submitted = st.form_submit_button(
                                "Review profile matches",
                                type="primary",
                            )

                        if review_submitted:
                            try:
                                reviewed_details = validate_reviewed_patient_details(
                                    patient_name, age, sex, contact, external_id
                                )
                                reviewed_values = validate_reviewed_lab_values(
                                    reviewed_lab_frame.to_dict("records")
                                )
                            except ValueError as error:
                                st.error(str(error))
                            else:
                                st.session_state["report_review_state"] = {
                                    "fingerprint": fingerprint,
                                    "details": reviewed_details,
                                    "values": reviewed_values,
                                }
                                st.rerun()
                    else:
                        details = report_state["details"]
                        results = flag_lab_values(report_state["values"])
                        st.subheader("Reviewed report details")
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "Patient name": details["name"],
                                        "Age": details["age"],
                                        "Sex": details["sex"],
                                        "Contact": details["contact"],
                                        "Patient/record ID": details["external_id"],
                                    }
                                ]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.subheader("Reviewed laboratory values")
                        if results:
                            st.dataframe(
                                pd.DataFrame(results),
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.warning(
                                "Configured ranges are illustrative, not medical reference "
                                "ranges. No abnormal value is a diagnosis. Never compare "
                                "incompatible or missing units; glucose is compared only when "
                                "the source explicitly confirms fasting. Verify the report's "
                                "own range and clinical context."
                            )
                        else:
                            st.info("No supported lab values will be attached to this report summary.")

                        try:
                            matches = find_patient_matches(details)
                            patient_profiles = list_patients()
                        except sqlite3.Error as error:
                            st.error(f"Could not load patient identity matches: {error}")
                            st.stop()
                        st.subheader("Choose the patient profile explicitly")
                        if matches:
                            st.warning(
                                "Possible identity matches need your review. Similar name, age, "
                                "or sex alone is not proof of identity. Differences are shown; "
                                "MediAI will not silently merge or update profiles."
                            )
                            st.dataframe(
                                pd.DataFrame(
                                    [
                                        {
                                            "Possible profile": (
                                                f"{match['name']} (ID {match['id']})"
                                            ),
                                            "Match clues": "; ".join(match["match_reasons"]),
                                            "Differences": "; ".join(match["conflicts"]) or "None detected",
                                        }
                                        for match in matches
                                    ]
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info(
                                "No likely profile match was found. Check the directory before "
                                "choosing to create a new profile."
                            )

                        profile_options = {
                            "Create a new patient profile": None,
                            **{
                                f"{patient['name']} (ID {patient['id']})": patient["id"]
                                for patient in patient_profiles
                            },
                        }
                        selected_profile = st.selectbox(
                            "Profile to use",
                            ["Choose a profile..."] + list(profile_options),
                            key=f"report_review_profile_{fingerprint[:12]}",
                        )
                        selected_patient_id = profile_options.get(selected_profile)
                        if selected_profile in profile_options and selected_patient_id is not None:
                            selected_match = next(
                                (match for match in matches if match["id"] == selected_patient_id),
                                None,
                            )
                            selected_patient = next(
                                patient
                                for patient in patient_profiles
                                if patient["id"] == selected_patient_id
                            )
                            st.write(
                                "Selected existing profile: "
                                f"{selected_patient['name']} (ID {selected_patient_id}), "
                                f"age {selected_patient['age']}, sex {selected_patient['sex']}, "
                                f"contact {selected_patient['contact'] or 'not recorded'}, "
                                f"record ID {selected_patient['external_id'] or 'not recorded'}."
                            )
                            if selected_match and selected_match["conflicts"]:
                                st.warning(
                                    "The selected profile conflicts with the reviewed report: "
                                    + "; ".join(selected_match["conflicts"])
                                    + ". Confirm only if you have verified the identity."
                                )
                            st.caption(
                                "The selected profile's existing details will not be overwritten "
                                "by report values."
                            )
                        elif selected_profile == "Create a new patient profile":
                            st.caption(
                                "A new profile will use the reviewed name, age, sex, contact, "
                                "and record ID. Possible matches remain visible above."
                            )

                        if selected_profile != "Choose a profile...":
                            confirmed = st.checkbox(
                                (
                                    "I reviewed the extracted values and verified the "
                                    "selected patient profile."
                                    if selected_patient_id is not None
                                    else "I reviewed the extracted values and confirm this "
                                    "is a new patient profile."
                                ),
                                key=f"report_review_confirm_{fingerprint[:12]}",
                            )
                            if st.button(
                                "Save reviewed report once",
                                type="primary",
                                key=f"report_review_save_{fingerprint[:12]}",
                            ):
                                summary_lines = [
                                    "Educational report extraction summary; not a diagnosis.",
                                    *[
                                        f"{item['parameter']}: {item['value']} "
                                        f"{item['unit'] or '(unit not stated)'}; "
                                        f"context {item.get('context') or 'not applicable'}; "
                                        f"{item['status']}; illustrative interval "
                                        f"{item['reference_interval']}; report range "
                                        f"{item.get('reported_reference_range') or 'not extracted'}."
                                        for item in results
                                    ],
                                    "Configured intervals are illustrative; confirm the "
                                    "report's own interval and context.",
                                ]
                                try:
                                    patient_id, save_status = save_uploaded_report(
                                        details,
                                        "\n".join(summary_lines),
                                        fingerprint,
                                        patient_id=selected_patient_id,
                                        create_patient=selected_patient_id is None,
                                        identity_confirmed=confirmed,
                                    )
                                except (ValueError, sqlite3.Error) as error:
                                    st.error(f"Could not save this reviewed report: {error}")
                                else:
                                    if save_status == "already_processed":
                                        message = (
                                            f"This exact file was already saved for patient ID "
                                            f"{patient_id}; no duplicate was created."
                                        )
                                    elif save_status == "created_similar":
                                        message = (
                                            f"Reviewed report saved once with separate patient "
                                            f"profile ID {patient_id}. A similar profile was shown "
                                            "during review and was not merged."
                                        )
                                    elif save_status == "created":
                                        message = (
                                            f"Reviewed report saved once with new patient profile "
                                            f"ID {patient_id}."
                                        )
                                    else:
                                        message = (
                                            f"Reviewed report saved once to patient profile "
                                            f"ID {patient_id}. Existing profile fields were not changed."
                                        )
                                    st.session_state["report_saved_notice"] = message
                                    for state_key in list(st.session_state):
                                        if state_key.startswith("report_review_"):
                                            st.session_state.pop(state_key, None)
                                    st.session_state["report_upload_nonce"] += 1
                                    st.rerun()

                        report_lines = [
                            "EDUCATIONAL EXTRACTION SUMMARY — NOT A DIAGNOSIS",
                            f"Patient profile selected: {selected_profile}",
                            f"Source file: {uploaded.name}",
                            f"Processed: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
                            "",
                            *[
                                f"{item['parameter']}: {item['value']} "
                                f"{item['unit'] or '(unit not stated)'} - {item['status']} "
                                f"(context {item.get('context') or 'not applicable'}; "
                                f"illustrative interval {item['reference_interval']}; "
                                f"report range {item.get('reported_reference_range') or 'not extracted'})"
                                for item in results
                            ],
                            "",
                            "Review extracted details against the source. Illustrative "
                            "intervals do not replace the report's own range or clinical context.",
                        ]
                        st.download_button(
                            "Download reviewed extraction summary (PDF)",
                            data=create_pdf_bytes("Educational Report Extraction", report_lines),
                            file_name="report-extraction-summary.pdf",
                            mime="application/pdf",
                        )
                        if st.button(
                            "Edit reviewed details",
                            key=f"report_review_edit_{fingerprint[:12]}",
                        ):
                            for state_key in list(st.session_state):
                                if state_key.startswith("report_review_"):
                                    st.session_state.pop(state_key, None)
                            st.rerun()

elif page == "Symptom Explorer":
    _render_page_heading(
        "Dataset-based education",
        "Symptom Pattern Explorer",
        "Compare selected symptoms with patterns in the supplied multi-class dataset. The output is not a diagnosis.",
    )
    st.caption(
        "This exploratory model has no independent external or clinical validation. It must not be used for screening, diagnosis, or treatment."
    )
    try:
        symptom_features = _get_symptom_features()
    except (FileNotFoundError, ValueError) as error:
        symptom_features = []
        st.error(f"Could not load the supplied symptom dataset: {error}")

    if symptom_features:
        patients = list_patients()
        patient_id = None
        if patients:
            patient_map = {f"{p['name']} (ID {p['id']})": p["id"] for p in patients}
            selected_patient = st.selectbox(
                "Link this exploration to a patient (optional)",
                ["No patient selected"] + list(patient_map),
            )
            if selected_patient != "No patient selected":
                patient_id = patient_map[selected_patient]

        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">Select observed symptoms</div>',
                unsafe_allow_html=True,
            )
            st.info(
                "Select every symptom that is present. The dataset uses binary values, so "
                "unselected symptoms are sent to the model as absent; do not submit an incomplete history."
            )
            with st.form("symptom_explorer_form"):
                selected_symptoms = st.multiselect(
                    "Symptoms",
                    options=symptom_features,
                    format_func=format_symptom_name,
                    help=(
                        f"The source dataset contains {len(symptom_features)} "
                        "binary symptom indicators."
                    ),
                )
                run = st.form_submit_button(
                    "Explore pattern matches",
                    type="primary",
                    use_container_width=True,
                )
        if run:
            if not selected_symptoms:
                st.error("Select at least one symptom before exploring patterns.")
            else:
                _render_symptom_candidates(selected_symptoms, patient_id)

elif page == "Patient History":
    _render_page_heading(
        "Longitudinal trajectory",
        "Patient History",
        "Filter saved report and symptom exploration records, inspect recent activity and export a history summary.",
    )
    rows = list_analyses()
    if rows:
        history = pd.DataFrame(rows)
        history["created_dt"] = pd.to_datetime(history["created_at"], errors="coerce")
        patient_options = ["All patients"] + sorted(history["patient"].dropna().unique().tolist())
        patient_filter = st.selectbox("Filter by patient", patient_options)
        type_options = ["All analysis types"] + sorted(history["analysis_type"].dropna().unique().tolist())
        type_filter = st.selectbox("Filter by analysis type", type_options)
        filtered = history.copy()
        if patient_filter != "All patients":
            filtered = filtered[filtered["patient"] == patient_filter]
        if type_filter != "All analysis types":
            filtered = filtered[filtered["analysis_type"] == type_filter]
        filtered = filtered.sort_values(["created_dt", "id"], ascending=[False, False])

        recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
        recent_count = int((filtered["created_dt"] >= recent_cutoff).sum())
        total, patient_count, recent = st.columns(3)
        total.metric("Visible analyses", len(filtered))
        patient_count.metric("Patients represented", filtered["patient"].nunique())
        recent.metric("Analyses in last 30 days", recent_count)

        analytics, timeline = st.columns(2)
        with analytics:
            st.subheader("Analysis type breakdown")
            type_counts = filtered["analysis_type"].value_counts()
            if not type_counts.empty:
                st.bar_chart(type_counts)
            st.subheader("Activity over time")
            dated = filtered.dropna(subset=["created_dt"]).copy()
            if not dated.empty:
                dated["date"] = dated["created_dt"].dt.strftime("%Y-%m-%d")
                daily_counts = dated.groupby("date").size().sort_index()
                st.line_chart(daily_counts)
        with timeline:
            st.subheader("Timeline")
            if filtered.empty:
                st.info("No analyses match the selected filters.")
            for record in filtered.to_dict("records"):
                created = record["created_dt"]
                display_time = (
                    created.strftime("%Y-%m-%d %H:%M")
                    if not pd.isna(created)
                    else str(record["created_at"])
                )
                with st.container(border=True):
                    st.markdown(f"**{record['analysis_type']}**")
                    st.caption(f"{display_time} | Patient: {record['patient']}")
                    st.write(record["summary"] or "No summary was saved.")
                    entry_pdf = create_pdf_bytes(
                        "Patient Analysis History",
                        [
                            f"Patient: {record['patient']}",
                            f"Analysis type: {record['analysis_type']}",
                            f"Created: {display_time}",
                            "",
                            record["summary"] or "No summary was saved.",
                            "",
                            "MediAI academic prototype. This is not a diagnosis.",
                        ],
                    )
                    st.download_button(
                        "Download this entry (PDF)",
                        data=entry_pdf,
                        file_name=f"patient-analysis-{record['id']}.pdf",
                        mime="application/pdf",
                        key=f"history_entry_pdf_{record['id']}",
                    )

        history_lines = ["Filtered patient history", ""]
        for record in filtered.to_dict("records"):
            created = record["created_dt"]
            display_time = (
                created.strftime("%Y-%m-%d %H:%M")
                if not pd.isna(created)
                else str(record["created_at"])
            )
            history_lines.extend(
                [
                    f"{display_time} | {record['patient']} | {record['analysis_type']}",
                    record["summary"] or "No summary was saved.",
                    "",
                ]
            )
        history_lines.append("MediAI academic prototype. This is not a diagnosis.")
        st.download_button(
            "Download filtered history (PDF)",
            data=create_pdf_bytes("Patient History Export", history_lines),
            file_name="patient-history.pdf",
            mime="application/pdf",
            key="filtered_history_pdf",
        )
    else:
        st.info("No saved analyses yet. Report reviews and symptom explorations will appear here.")
