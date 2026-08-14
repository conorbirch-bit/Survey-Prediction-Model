from pathlib import Path
import io

import pandas as pd
import streamlit as st

from duration_predictor import (
    DurationPredictor,
    FEATURE_COLUMNS,
)

st.set_page_config(page_title="Site Survey Duration Predictor", layout="wide")
st.title("Site Survey Duration Predictor")
st.caption(
    "Predicts survey time from building height, ground-floor area and Sovereign flats. "
    "Missing inputs automatically trigger the appropriate fallback model."
)

DEFAULT_FILE = Path(__file__).with_name("Predictive Model.xlsx")

with st.sidebar:
    st.header("Training data")
    uploaded = st.file_uploader(
        "Upload completed-surveys Excel export",
        type=["xlsx", "xls"],
        key="training_file",
        help="If no file is uploaded, the bundled Predictive Model.xlsx is used.",
    )
    min_duration = st.number_input(
        "Minimum completed duration (minutes)",
        min_value=1,
        max_value=30,
        value=6,
        help="Rows below this duration are excluded as likely aborted/failed visits.",
    )
    buffer_pct = st.slider(
        "Scheduling buffer",
        min_value=0,
        max_value=50,
        value=15,
        step=5,
        format="%d%%",
    )

@st.cache_resource(show_spinner=False)
def train_from_path(path: str, min_duration: int, buffer_pct: int):
    return DurationPredictor(
        min_completed_duration=min_duration,
        planning_buffer_pct=buffer_pct / 100,
    ).load_excel(path)

@st.cache_resource(show_spinner=False)
def train_from_bytes(file_bytes: bytes, min_duration: int, buffer_pct: int):
    df = pd.read_excel(io.BytesIO(file_bytes))
    return DurationPredictor(
        min_completed_duration=min_duration,
        planning_buffer_pct=buffer_pct / 100,
    ).fit(df)

try:
    if uploaded is not None:
        predictor = train_from_bytes(uploaded.getvalue(), min_duration, buffer_pct)
    else:
        predictor = train_from_path(str(DEFAULT_FILE), min_duration, buffer_pct)
except Exception as exc:
    st.error(f"Could not train model: {exc}")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "Predict a building",
    "Predict upcoming surveys",
    "Model diagnostics",
])

with tab1:
    st.subheader("Building inputs")
    st.write("Leave any unavailable field blank.")

    c1, c2, c3 = st.columns(3)
    with c1:
        height_text = st.text_input("Building Height", placeholder="e.g. 24")
    with c2:
        area_text = st.text_input(
            "Internal Ground Floor Area (m²)", placeholder="e.g. 520"
        )
    with c3:
        flats_text = st.text_input("Sovereign Flats", placeholder="e.g. 42")

    def optional_number(text):
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    if st.button("Predict survey duration", type="primary"):
        height = optional_number(height_text)
        area = optional_number(area_text)
        flats = optional_number(flats_text)

        try:
            prediction = predictor.predict(
                building_height=height,
                ground_floor_area=area,
                flats=flats,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted duration", f"{prediction.predicted_minutes:.0f} min")
            m2.metric("Planning duration", f"{prediction.planning_minutes} min")
            m3.metric("Confidence", prediction.confidence)

            st.success(f"Model used: {prediction.model_label}")
            st.caption(
                f"Trained on {prediction.training_rows} usable surveys"
                + (
                    f" · leave-one-out MAE ≈ {prediction.validation_mae_minutes:.1f} min"
                    if prediction.validation_mae_minutes is not None else ""
                )
            )

            if prediction.missing_inputs:
                pretty_missing = ", ".join(
                    DurationPredictor.pretty_feature(k)
                    for k in prediction.missing_inputs
                )
                st.info(
                    "Missing data handled automatically. "
                    f"Not used: {pretty_missing}."
                )

        except ValueError as exc:
            st.warning(str(exc))

with tab2:
    st.subheader("Upcoming surveys")
    st.write(
        "Upload an Excel spreadsheet containing upcoming buildings. "
        "The app will add predicted and planning durations to every row."
    )

    upcoming_file = st.file_uploader(
        "Upload upcoming surveys spreadsheet",
        type=["xlsx", "xls"],
        key="upcoming_file",
    )

    if upcoming_file is not None:
        try:
            upcoming_df = pd.read_excel(upcoming_file)
            upcoming_df = predictor._normalise_columns(upcoming_df)

            required_any = [
                FEATURE_COLUMNS["building_height"],
                FEATURE_COLUMNS["ground_floor_area"],
                FEATURE_COLUMNS["flats"],
            ]

            present_inputs = [c for c in required_any if c in upcoming_df.columns]

            if not present_inputs:
                st.error(
                    "The upcoming-surveys spreadsheet needs at least one of these columns: "
                    "Building Height, Internal Ground Floor Area (m2), or Sovereign Flat."
                )
            else:
                results = []
                success_count = 0
                failed_count = 0

                for _, row in upcoming_df.iterrows():
                    height = (
                        row.get(FEATURE_COLUMNS["building_height"])
                        if FEATURE_COLUMNS["building_height"] in upcoming_df.columns
                        else None
                    )
                    area = (
                        row.get(FEATURE_COLUMNS["ground_floor_area"])
                        if FEATURE_COLUMNS["ground_floor_area"] in upcoming_df.columns
                        else None
                    )
                    flats = (
                        row.get(FEATURE_COLUMNS["flats"])
                        if FEATURE_COLUMNS["flats"] in upcoming_df.columns
                        else None
                    )

                    try:
                        p = predictor.predict(
                            building_height=height,
                            ground_floor_area=area,
                            flats=flats,
                        )
                        results.append({
                            "Predicted Survey Duration (Minutes)": p.predicted_minutes,
                            "Planning Duration (Minutes)": p.planning_minutes,
                            "Prediction Confidence": p.confidence,
                            "Prediction Model Used": p.model_label,
                            "Validation MAE (Minutes)": p.validation_mae_minutes,
                            "Prediction Status": "Predicted",
                        })
                        success_count += 1
                    except ValueError:
                        results.append({
                            "Predicted Survey Duration (Minutes)": None,
                            "Planning Duration (Minutes)": None,
                            "Prediction Confidence": "None",
                            "Prediction Model Used": "No usable input data",
                            "Validation MAE (Minutes)": None,
                            "Prediction Status": "Could not predict",
                        })
                        failed_count += 1

                result_df = pd.concat(
                    [upcoming_df.reset_index(drop=True), pd.DataFrame(results)],
                    axis=1
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Buildings", len(result_df))
                c2.metric("Predicted", success_count)
                c3.metric("Could not predict", failed_count)

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True,
                )

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    result_df.to_excel(
                        writer,
                        sheet_name="Upcoming Surveys Predictions",
                        index=False,
                    )

                st.download_button(
                    "Download predictions spreadsheet",
                    data=output.getvalue(),
                    file_name="Upcoming_Surveys_With_Predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                )

                st.caption(
                    "Rows with all three predictor fields blank are retained in the output "
                    "and marked 'Could not predict'."
                )

        except Exception as exc:
            st.error(f"Could not process upcoming surveys spreadsheet: {exc}")

with tab3:
    st.subheader("Fallback models")
    summary = predictor.model_summary()
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.caption(
        "MAE is leave-one-out cross-validation error on the current historical data. "
        "It should be monitored as more completed surveys are added."
    )
