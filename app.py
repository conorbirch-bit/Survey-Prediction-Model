from pathlib import Path
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import io
import os

import pandas as pd
import streamlit as st

from duration_predictor_height import DurationPredictor, FEATURE_COLUMNS
from google_routes import GoogleTransitRouter, GoogleRoutesError
from scheduler import DailyTransitScheduler, postcode_district


LONDON_TZ = ZoneInfo("Europe/London")
DEFAULT_FILE = Path(__file__).with_name("Predictive Model.xlsx")

st.set_page_config(page_title="Site Survey Scheduling Agent", layout="wide")
st.title("Site Survey Scheduling Agent")
st.caption("Version 8 — Google API key via Streamlit Secrets")
st.caption(
    "Predict survey durations, then create a public-transport day route "
    "starting and finishing at Harpenden Station."
)

with st.sidebar:
    st.header("Duration model")
    training_file = st.file_uploader(
        "Completed-surveys training spreadsheet",
        type=["xlsx", "xls"],
        key="training_file",
        help="If omitted, the bundled Predictive Model.xlsx is used.",
    )
    min_duration = st.number_input(
        "Minimum completed duration (minutes)",
        min_value=1,
        max_value=30,
        value=6,
        help="Historical rows below this are excluded from model training.",
    )
    buffer_pct = st.slider(
        "Survey scheduling buffer",
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
    if training_file is not None:
        predictor = train_from_bytes(
            training_file.getvalue(), min_duration, buffer_pct
        )
    else:
        predictor = train_from_path(
            str(DEFAULT_FILE), min_duration, buffer_pct
        )
except Exception as exc:
    st.error(f"Could not train duration model: {exc}")
    st.stop()


def optional_number(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def normalise_upcoming_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = predictor._normalise_columns(df.copy())

    # Exact Salesforce export aliases used in Future Surveys.xlsx.
    aliases = {
        "Customer Reference": [
            "Customer Reference Code  ↑",
            "Customer Reference Code",
            "Customer Reference",
        ],
        "Building Name": ["Building Name"],
        "Postcode": ["Postcode", "Postal Code"],
        "Resource Name": ["Resource Name: Name", "Resource Name"],
        "Planned Start": [
            "Planned Start",
            "Primary Service Appointment: Scheduled Start",
        ],
    }

    rename = {}
    stripped = {str(c).strip(): c for c in df.columns}
    for canonical, possibilities in aliases.items():
        for option in possibilities:
            if option in stripped:
                rename[stripped[option]] = canonical
                break

    return df.rename(columns=rename)


def predict_upcoming(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        try:
            p = predictor.predict(
                building_height=row.get(
                    FEATURE_COLUMNS["building_height"]
                ),
                ground_floor_area=row.get(
                    FEATURE_COLUMNS["ground_floor_area"]
                ),
                flats=row.get(FEATURE_COLUMNS["flats"]),
            )
            rows.append({
                "Predicted Survey Duration (Minutes)": p.predicted_minutes,
                "Planning Duration (Minutes)": p.planning_minutes,
                "Prediction Confidence": p.confidence,
                "Prediction Model Used": p.model_label,
                "Validation MAE (Minutes)": p.validation_mae_minutes,
                "Prediction Status": "Predicted",
            })
        except ValueError:
            rows.append({
                "Predicted Survey Duration (Minutes)": None,
                "Planning Duration (Minutes)": None,
                "Prediction Confidence": "None",
                "Prediction Model Used": "No usable input data",
                "Validation MAE (Minutes)": None,
                "Prediction Status": "Could not predict",
            })

    return pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(rows)],
        axis=1,
    )


tab1, tab2, tab3 = st.tabs([
    "Predict one building",
    "Upcoming surveys + routing",
    "Model diagnostics",
])

with tab1:
    st.subheader("Single-building prediction")
    st.write("Leave any unavailable input blank.")

    c1, c2, c3 = st.columns(3)
    with c1:
        height_text = st.text_input(
            "Building Height", placeholder="e.g. 24"
        )
    with c2:
        area_text = st.text_input(
            "Internal Ground Floor Area (m²)", placeholder="e.g. 520"
        )
    with c3:
        flats_text = st.text_input(
            "Sovereign Flats", placeholder="e.g. 42"
        )

    if st.button("Predict survey duration", type="primary"):
        try:
            p = predictor.predict(
                building_height=optional_number(height_text),
                ground_floor_area=optional_number(area_text),
                flats=optional_number(flats_text),
            )
            a, b, c = st.columns(3)
            a.metric("Predicted duration", f"{p.predicted_minutes:.0f} min")
            b.metric("Planning duration", f"{p.planning_minutes} min")
            c.metric("Confidence", p.confidence)
            st.success(f"Model used: {p.model_label}")
        except ValueError as exc:
            st.warning(str(exc))


with tab2:
    st.subheader("Upload future surveys")
    upcoming_file = st.file_uploader(
        "Future Surveys spreadsheet",
        type=["xlsx", "xls"],
        key="upcoming_file",
        help=(
            "Designed for the Salesforce Future Surveys.xlsx structure, "
            "including Building Name, Postcode, Building Height, "
            "Sovereign Flat and Internal Ground Floor Area."
        ),
    )

    if upcoming_file is None:
        st.info("Upload Future Surveys.xlsx to predict and route the sites.")
    else:
        try:
            raw_upcoming = pd.read_excel(upcoming_file)
            upcoming = normalise_upcoming_columns(raw_upcoming)

            if "Postcode" not in upcoming.columns:
                st.error(
                    "The future-surveys file needs a Postcode column for routing."
                )
                st.stop()

            predictions = predict_upcoming(upcoming)

            st.markdown("### Duration predictions")
            p1, p2, p3 = st.columns(3)
            p1.metric("Buildings", len(predictions))
            p2.metric(
                "Predicted",
                int((predictions["Prediction Status"] == "Predicted").sum()),
            )
            p3.metric(
                "Missing prediction",
                int((predictions["Prediction Status"] != "Predicted").sum()),
            )

            display_cols = [
                c for c in [
                    "Customer Reference",
                    "Building Name",
                    "Postcode",
                    "Resource Name",
                    "Building Height",
                    "Sovereign Flat",
                    "Internal Ground Floor Area (m2)",
                    "Predicted Survey Duration (Minutes)",
                    "Planning Duration (Minutes)",
                    "Prediction Confidence",
                    "Prediction Model Used",
                ] if c in predictions.columns
            ]
            st.dataframe(
                predictions[display_cols],
                use_container_width=True,
                hide_index=True,
            )

            pred_output = io.BytesIO()
            with pd.ExcelWriter(pred_output, engine="openpyxl") as writer:
                predictions.to_excel(
                    writer,
                    sheet_name="Upcoming Surveys Predictions",
                    index=False,
                )
            st.download_button(
                "Download duration predictions",
                data=pred_output.getvalue(),
                file_name="Upcoming_Surveys_With_Predictions.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )

            st.divider()
            st.markdown("### Build public-transport schedules")

            routable = predictions[
                (predictions["Prediction Status"] == "Predicted")
                & predictions["Postcode"].notna()
                & (predictions["Postcode"].astype(str).str.strip() != "")
            ].copy()

            if routable.empty:
                st.warning("There are no sites with both a prediction and postcode.")
            else:
                planning_mode = st.radio(
                    "Planning mode",
                    ["Single day", "Full working week"],
                    horizontal=True,
                )

                controls1 = st.columns(3)

                # Resource selector defaults to Conor Birch if present.
                resource_options = ["All resources"]
                if "Resource Name" in routable.columns:
                    names = sorted(
                        {
                            str(x).strip()
                            for x in routable["Resource Name"].dropna()
                            if str(x).strip()
                        }
                    )
                    resource_options += names

                default_resource_index = (
                    resource_options.index("Conor Birch")
                    if "Conor Birch" in resource_options else 0
                )

                with controls1[0]:
                    resource = st.selectbox(
                        "Sites to consider",
                        resource_options,
                        index=default_resource_index,
                    )

                filtered = routable.copy()
                if resource != "All resources":
                    filtered = filtered[
                        filtered["Resource Name"].astype(str) == resource
                    ].copy()

                # Default date = earliest planned date in selected data, otherwise today.
                default_date = datetime.now(LONDON_TZ).date()
                if "Planned Start" in filtered.columns:
                    parsed = pd.to_datetime(
                        filtered["Planned Start"],
                        dayfirst=True,
                        errors="coerce",
                    )
                    valid_dates = parsed.dropna()
                    if not valid_dates.empty:
                        default_date = valid_dates.min().date()

                with controls1[1]:
                    if planning_mode == "Single day":
                        route_date = st.date_input(
                            "Survey date",
                            value=default_date,
                        )
                        week_start = None
                    else:
                        # Default to the Monday containing the earliest selected date.
                        default_monday = default_date - timedelta(
                            days=default_date.weekday()
                        )
                        week_start = st.date_input(
                            "Week commencing (Monday)",
                            value=default_monday,
                        )
                        route_date = None
                with controls1[2]:
                    home_location = st.text_input(
                        "Start / finish location",
                        value="Harpenden Station",
                    )

                controls2 = st.columns(4)
                with controls2[0]:
                    start_clock = st.time_input(
                        "Leave home",
                        value=time(7, 50),
                    )
                with controls2[1]:
                    finish_clock = st.time_input(
                        "Latest return",
                        value=time(16, 0),
                    )
                with controls2[2]:
                    same_postcode_minutes = st.number_input(
                        "Same-postcode transfer",
                        min_value=0,
                        max_value=30,
                        value=5,
                        step=1,
                        help=(
                            "Used instead of a transit API call when two surveys "
                            "share exactly the same postcode."
                        ),
                    )
                with controls2[3]:
                    transit_choice = st.selectbox(
                        "Transit preference",
                        ["Fastest / default", "Less walking", "Fewer transfers"],
                    )

                buffer_controls = st.columns(3)
                with buffer_controls[0]:
                    travel_leeway = st.number_input(
                        "Travel leeway per journey (min)",
                        min_value=0,
                        max_value=30,
                        value=5,
                        step=1,
                        help=(
                            "Extra time added to every public-transport journey "
                            "for delays, platform finding, crossings and general slack."
                        ),
                    )
                with buffer_controls[1]:
                    pre_survey_buffer = st.number_input(
                        "Before each survey (min)",
                        min_value=0,
                        max_value=30,
                        value=5,
                        step=1,
                        help=(
                            "Time after arriving for getting bearings, finding "
                            "the entrance, preparing equipment and getting started."
                        ),
                    )
                with buffer_controls[2]:
                    post_survey_buffer = st.number_input(
                        "After each survey (min)",
                        min_value=0,
                        max_value=30,
                        value=5,
                        step=1,
                        help=(
                            "Time after the survey for packing bags, notes, "
                            "orientating yourself and leaving the site."
                        ),
                    )

                if planning_mode == "Full working week":
                    working_days = st.multiselect(
                        "Working days",
                        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                        default=[
                            "Monday",
                            "Tuesday",
                            "Wednesday",
                            "Thursday",
                            "Friday",
                        ],
                    )
                else:
                    working_days = []

                st.caption(
                    f"{len(filtered)} predicted sites are currently eligible "
                    "for this routing run."
                )

                # Google API key: prefer Streamlit Secrets.
                # On Streamlit Community Cloud add:
                # GOOGLE_MAPS_API_KEY = "your-key"
                # under App settings -> Secrets.
                try:
                    api_key = st.secrets["GOOGLE_MAPS_API_KEY"]
                except (KeyError, FileNotFoundError):
                    # Optional local fallback for developers who prefer
                    # an environment variable.
                    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")

                if api_key:
                    st.success("Google Maps API key loaded from secrets.")
                else:
                    st.error(
                        "Google Maps API key not found. Add "
                        "GOOGLE_MAPS_API_KEY to Streamlit Secrets."
                    )

                preference_map = {
                    "Fastest / default": None,
                    "Less walking": "LESS_WALKING",
                    "Fewer transfers": "FEWER_TRANSFERS",
                }

                if st.button(
                    "Create public-transport schedule",
                    type="primary",
                    disabled=not bool(api_key.strip()),
                ):
                    if planning_mode == "Single day":
                        start_dt = datetime.combine(
                            route_date, start_clock, tzinfo=LONDON_TZ
                        )
                        finish_dt = datetime.combine(
                            route_date, finish_clock, tzinfo=LONDON_TZ
                        )
                    else:
                        start_dt = None
                        finish_dt = None

                    if finish_clock <= start_clock:
                        st.error("Latest return must be after the start time.")
                    elif planning_mode == "Full working week" and not working_days:
                        st.error("Choose at least one working day.")
                    else:
                        sites = []
                        for _, row in filtered.iterrows():
                            building_name = str(
                                row.get("Building Name", "")
                            ).strip()
                            postcode = str(row.get("Postcode", "")).strip()

                            # Building Name already contains a street address in the
                            # supplied Salesforce export. Including postcode gives
                            # Google more precise door-to-door routing context.
                            route_location = (
                                f"{building_name}, {postcode}"
                                if building_name else postcode
                            )

                            sites.append({
                                "customer_reference": row.get(
                                    "Customer Reference", ""
                                ),
                                "building_name": building_name or postcode,
                                "postcode": postcode,
                                "route_location": route_location,
                                "planning_minutes": int(
                                    row["Planning Duration (Minutes)"]
                                ),
                                "predicted_minutes": float(
                                    row["Predicted Survey Duration (Minutes)"]
                                ),
                                "confidence": row["Prediction Confidence"],
                                "model_used": row["Prediction Model Used"],
                            })

                        try:
                            with st.spinner(
                                "Calculating live timetable-based transit journeys "
                                "and testing which surveys fit..."
                            ):
                                router = GoogleTransitRouter(
                                    api_key=api_key,
                                    transit_preference=preference_map[
                                        transit_choice
                                    ],
                                )
                                scheduler = DailyTransitScheduler(
                                    router=router,
                                    home_location=home_location,
                                    same_postcode_transfer_minutes=int(
                                        same_postcode_minutes
                                    ),
                                    travel_leeway_minutes=int(travel_leeway),
                                    pre_survey_buffer_minutes=int(
                                        pre_survey_buffer
                                    ),
                                    post_survey_buffer_minutes=int(
                                        post_survey_buffer
                                    ),
                                )
                                if planning_mode == "Single day":
                                    schedule = scheduler.build_day(
                                        sites=sites,
                                        start_time=start_dt,
                                        latest_return=finish_dt,
                                    )
                                    weekly_schedule = None
                                else:
                                    day_names = [
                                        "Monday", "Tuesday", "Wednesday",
                                        "Thursday", "Friday"
                                    ]
                                    chosen_dates = []
                                    for offset, name in enumerate(day_names):
                                        if name in working_days:
                                            chosen_dates.append(
                                                week_start + timedelta(days=offset)
                                            )

                                    weekly_schedule = scheduler.build_week(
                                        sites=sites,
                                        dates=chosen_dates,
                                        start_clock=start_clock,
                                        latest_return_clock=finish_clock,
                                        timezone=LONDON_TZ,
                                    )
                                    schedule = None

                            if planning_mode == "Single day":
                                if not schedule.items:
                                    st.warning(
                                        "No survey could be fitted into this day while "
                                        "still returning by the deadline."
                                    )
                                else:
                                    st.success(
                                        f"Scheduled {len(schedule.items)} surveys. "
                                        f"Expected return to {home_location}: "
                                        f"{schedule.return_time.strftime('%H:%M')}."
                                    )

                                    k1, k2, k3, k4 = st.columns(4)
                                    k1.metric("Surveys", len(schedule.items))
                                    k2.metric(
                                        "Survey time",
                                        f"{schedule.survey_minutes} min",
                                    )
                                    k3.metric(
                                        "Transit time",
                                        f"{schedule.travel_minutes} min",
                                    )
                                    k4.metric(
                                        "Return home",
                                        schedule.return_time.strftime("%H:%M"),
                                    )

                                    schedule_df = schedule.to_dataframe()
                                    st.dataframe(
                                        schedule_df,
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                                    clusters = [
                                        item.cluster for item in schedule.items
                                        if item.cluster
                                    ]
                                    if clusters:
                                        st.caption(
                                            "Route cluster: "
                                            + " → ".join(dict.fromkeys(clusters))
                                        )

                                    schedule_output = io.BytesIO()
                                    with pd.ExcelWriter(
                                        schedule_output,
                                        engine="openpyxl",
                                    ) as writer:
                                        schedule_df.to_excel(
                                            writer,
                                            sheet_name="Daily Schedule",
                                            index=False,
                                        )
                                        filtered.to_excel(
                                            writer,
                                            sheet_name="Candidate Sites",
                                            index=False,
                                        )

                                    st.download_button(
                                        "Download daily schedule",
                                        data=schedule_output.getvalue(),
                                        file_name=(
                                            f"Survey_Schedule_"
                                            f"{route_date.isoformat()}.xlsx"
                                        ),
                                        mime=(
                                            "application/vnd.openxmlformats-officedocument."
                                            "spreadsheetml.sheet"
                                        ),
                                        type="primary",
                                    )

                                    if schedule.unscheduled_count:
                                        st.info(
                                            f"{schedule.unscheduled_count} eligible sites "
                                            "were left for another day."
                                        )

                            else:
                                if weekly_schedule.total_surveys == 0:
                                    st.warning(
                                        "No surveys could be fitted into the selected "
                                        "working week while respecting the return deadline."
                                    )
                                else:
                                    st.success(
                                        f"Scheduled {weekly_schedule.total_surveys} surveys "
                                        f"across {len(weekly_schedule.days)} days."
                                    )

                                    w1, w2, w3, w4 = st.columns(4)
                                    w1.metric(
                                        "Surveys scheduled",
                                        weekly_schedule.total_surveys,
                                    )
                                    w2.metric(
                                        "Survey time",
                                        f"{weekly_schedule.total_survey_minutes} min",
                                    )
                                    w3.metric(
                                        "Transit time",
                                        f"{weekly_schedule.total_travel_minutes} min",
                                    )
                                    w4.metric(
                                        "Sites remaining",
                                        len(weekly_schedule.unscheduled_sites),
                                    )

                                    st.markdown("#### Week summary")
                                    week_summary_df = (
                                        weekly_schedule.summary_dataframe()
                                    )
                                    st.dataframe(
                                        week_summary_df,
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                                    st.markdown("#### Full schedule")
                                    full_week_df = (
                                        weekly_schedule.full_schedule_dataframe()
                                    )
                                    st.dataframe(
                                        full_week_df,
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                                    if weekly_schedule.unscheduled_sites:
                                        remaining_df = pd.DataFrame(
                                            weekly_schedule.unscheduled_sites
                                        )
                                        remaining_display = [
                                            c for c in [
                                                "customer_reference",
                                                "building_name",
                                                "postcode",
                                                "planning_minutes",
                                                "confidence",
                                            ] if c in remaining_df.columns
                                        ]
                                        with st.expander(
                                            "Sites not fitted into this week"
                                        ):
                                            st.dataframe(
                                                remaining_df[remaining_display],
                                                use_container_width=True,
                                                hide_index=True,
                                            )

                                    week_output = io.BytesIO()
                                    with pd.ExcelWriter(
                                        week_output,
                                        engine="openpyxl",
                                    ) as writer:
                                        week_summary_df.to_excel(
                                            writer,
                                            sheet_name="Week Summary",
                                            index=False,
                                        )
                                        full_week_df.to_excel(
                                            writer,
                                            sheet_name="Full Week Schedule",
                                            index=False,
                                        )

                                        for day in weekly_schedule.days:
                                            if not day.items:
                                                continue
                                            sheet_name = day.start_time.strftime(
                                                "%a %d %b"
                                            )[:31]
                                            day.to_dataframe().to_excel(
                                                writer,
                                                sheet_name=sheet_name,
                                                index=False,
                                            )

                                        if weekly_schedule.unscheduled_sites:
                                            pd.DataFrame(
                                                weekly_schedule.unscheduled_sites
                                            ).to_excel(
                                                writer,
                                                sheet_name="Unscheduled Sites",
                                                index=False,
                                            )

                                        filtered.to_excel(
                                            writer,
                                            sheet_name="Candidate Sites",
                                            index=False,
                                        )

                                    st.download_button(
                                        "Download full-week schedule",
                                        data=week_output.getvalue(),
                                        file_name=(
                                            f"Survey_Week_"
                                            f"{week_start.isoformat()}.xlsx"
                                        ),
                                        mime=(
                                            "application/vnd.openxmlformats-officedocument."
                                            "spreadsheetml.sheet"
                                        ),
                                        type="primary",
                                    )

                        except GoogleRoutesError as exc:
                            st.error(str(exc))
                        except Exception as exc:
                            st.error(f"Could not build schedule: {exc}")

        except Exception as exc:
            st.error(f"Could not process Future Surveys spreadsheet: {exc}")


with tab3:
    st.subheader("Duration fallback models")
    st.dataframe(
        predictor.model_summary(),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "MAE is leave-one-out cross-validation error on the historical "
        "completed-survey data."
    )
