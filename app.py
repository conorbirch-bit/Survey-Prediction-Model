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
from ai_planner import OpenAISchedulePlanner
from tfl_client import TfLClient
from metoffice_client import MetOfficeClient


LONDON_TZ = ZoneInfo("Europe/London")
DEFAULT_FILE = Path(__file__).with_name("Predictive Model.xlsx")

st.set_page_config(page_title="Site Survey Scheduling Agent", layout="wide")
st.title("Site Survey Scheduling Agent")
st.caption("Version 10 — AI week summary and schedule reasoning")
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



def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets[name])
    except (KeyError, FileNotFoundError):
        return os.getenv(name, default)


def planned_date_from_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def build_future_cluster_context(df: pd.DataFrame, week_start):
    """
    Count how many candidate sites share a postcode district in this week,
    next week, and the following three weeks. Existing Planned Start is used
    as a planning signal, not as a hard appointment constraint.
    """
    records = []
    for _, row in df.iterrows():
        pc = str(row.get("Postcode", "")).strip()
        d = postcode_district(pc)
        pd_date = planned_date_from_value(row.get("Planned Start"))
        records.append((d, pd_date))

    def count_for(district, start, end):
        return sum(
            1 for d, dt in records
            if d == district and dt is not None and start <= dt <= end
        )

    this_end = week_start + timedelta(days=6)
    next_start = week_start + timedelta(days=7)
    next_end = week_start + timedelta(days=13)
    three_end = week_start + timedelta(days=27)

    rows = []
    summary = {}
    districts = sorted({d for d, _ in records if d})
    for d in districts:
        this_count = count_for(d, week_start, this_end)
        next_count = count_for(d, next_start, next_end)
        next3_count = count_for(d, next_start, three_end)
        summary[d] = {
            "this_week": this_count,
            "next_week": next_count,
            "next_3_weeks": next3_count,
        }
        rows.append(
            f"{d}: this week={this_count}, next week={next_count}, "
            f"next 3 weeks={next3_count}"
        )
    return summary, "\n".join(rows)


def apply_ai_decisions(sites, decisions):
    by_ref = {d.customer_reference: d for d in decisions}
    for site in sites:
        ref = str(site.get("customer_reference", ""))
        decision = by_ref.get(ref)
        if decision:
            site["ai_priority"] = decision.priority
            site["ai_decision"] = decision.decision
            site["ai_reason"] = decision.reason
        else:
            site["ai_priority"] = 50
            site["ai_decision"] = "neutral"
            site["ai_reason"] = "No specific AI decision returned."
    return sites

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

                google_api_key = get_secret("GOOGLE_MAPS_API_KEY")
                openai_api_key = get_secret("OPENAI_API_KEY")
                openai_model = get_secret("OPENAI_MODEL", "gpt-5.6")
                tfl_api_key = get_secret("TFL_API_KEY")
                metoffice_api_key = get_secret("MET_OFFICE_API_KEY")
                metoffice_endpoint = get_secret("MET_OFFICE_GLOBAL_SPOT_URL")

                integration_cols = st.columns(4)
                integration_cols[0].metric(
                    "Google Routes", "Ready" if google_api_key else "Missing"
                )
                integration_cols[1].metric(
                    "OpenAI", "Ready" if openai_api_key else "Missing"
                )
                integration_cols[2].metric(
                    "TfL", "Ready" if tfl_api_key else "Optional"
                )
                integration_cols[3].metric(
                    "Met Office",
                    "Ready" if (metoffice_api_key and metoffice_endpoint)
                    else "Optional",
                )

                api_key = google_api_key

                preference_map = {
                    "Fastest / default": None,
                    "Less walking": "LESS_WALKING",
                    "Fewer transfers": "FEWER_TRANSFERS",
                }

                use_ai_planner = st.checkbox(
                    "Use OpenAI planning/reasoning layer",
                    value=True,
                    help=(
                        "AI ranks sites using cluster timing, future-week cluster "
                        "opportunities, TfL disruption and weather. Google/Python "
                        "still enforce actual routes and hard time constraints."
                    ),
                )

                ai_priority_weight = st.slider(
                    "AI influence on site choice",
                    min_value=0,
                    max_value=40,
                    value=20,
                    step=5,
                    help=(
                        "Maximum equivalent minutes the AI priority can improve "
                        "a site's routing score. Travel time remains dominant."
                    ),
                )

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
                                "postcode_district": postcode_district(postcode),
                                "route_location": route_location,
                                "planning_minutes": int(
                                    row["Planning Duration (Minutes)"]
                                ),
                                "predicted_minutes": float(
                                    row["Predicted Survey Duration (Minutes)"]
                                ),
                                "confidence": row["Prediction Confidence"],
                                "model_used": row["Prediction Model Used"],
                                "planned_start": str(
                                    row.get("Planned Start", "") or ""
                                ),
                            })

                        try:
                            with st.spinner(
                                "Reviewing cluster strategy, disruptions/weather, "
                                "then calculating transit journeys..."
                            ):
                                ai_strategy = ""
                                tfl_summary = "TfL integration not configured."
                                weather_summary = "Met Office integration not configured."

                                # Determine the relevant planning period.
                                if planning_mode == "Single day":
                                    context_week_start = (
                                        route_date - timedelta(
                                            days=route_date.weekday()
                                        )
                                    )
                                    period_start = route_date
                                    period_end = route_date
                                else:
                                    context_week_start = week_start
                                    period_start = week_start
                                    period_end = week_start + timedelta(days=6)

                                cluster_counts, cluster_summary = (
                                    build_future_cluster_context(
                                        filtered,
                                        context_week_start,
                                    )
                                )

                                for site in sites:
                                    counts = cluster_counts.get(
                                        site["postcode_district"],
                                        {},
                                    )
                                    site["same_district_this_week"] = (
                                        counts.get("this_week", 0)
                                    )
                                    site["same_district_next_week"] = (
                                        counts.get("next_week", 0)
                                    )
                                    site["same_district_next_3_weeks"] = (
                                        counts.get("next_3_weeks", 0)
                                    )

                                if tfl_api_key:
                                    tfl_summary = TfLClient(
                                        tfl_api_key
                                    ).disruption_summary(
                                        period_start,
                                        period_end,
                                    )

                                # Harpenden/London operating area midpoint is used
                                # for broad weekly weather risk. Route-level weather
                                # can be added later if desired.
                                if metoffice_api_key and metoffice_endpoint:
                                    weather_summary = MetOfficeClient(
                                        metoffice_api_key,
                                        metoffice_endpoint,
                                    ).forecast_summary(
                                        latitude=51.65,
                                        longitude=-0.20,
                                        start_date=period_start,
                                        end_date=period_end,
                                    )

                                if use_ai_planner:
                                    if not openai_api_key:
                                        raise ValueError(
                                            "OPENAI_API_KEY is missing from secrets."
                                        )
                                    planner = OpenAISchedulePlanner(
                                        api_key=openai_api_key,
                                        model=openai_model,
                                    )
                                    decisions, ai_strategy = planner.rank_sites(
                                        sites=sites,
                                        week_label=(
                                            f"{period_start} to {period_end}"
                                        ),
                                        tfl_summary=tfl_summary,
                                        weather_summary=weather_summary,
                                        future_cluster_summary=cluster_summary,
                                    )
                                    sites = apply_ai_decisions(
                                        sites,
                                        decisions,
                                    )

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
                                    ai_priority_weight_minutes=float(
                                        ai_priority_weight
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

                            if use_ai_planner and ai_strategy:
                                st.markdown("#### AI planning strategy")
                                st.info(ai_strategy)

                                ai_rows = []
                                for site in sites:
                                    ai_rows.append({
                                        "Customer Reference": site.get(
                                            "customer_reference", ""
                                        ),
                                        "Building": site.get("building_name", ""),
                                        "Postcode": site.get("postcode", ""),
                                        "AI Priority": site.get("ai_priority", 50),
                                        "AI Decision": site.get(
                                            "ai_decision", "neutral"
                                        ),
                                        "AI Reason": site.get("ai_reason", ""),
                                        "Same District This Week": site.get(
                                            "same_district_this_week", 0
                                        ),
                                        "Same District Next Week": site.get(
                                            "same_district_next_week", 0
                                        ),
                                        "Same District Next 3 Weeks": site.get(
                                            "same_district_next_3_weeks", 0
                                        ),
                                    })
                                with st.expander("AI site decisions"):
                                    st.dataframe(
                                        pd.DataFrame(ai_rows),
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                            with st.expander("External planning context"):
                                st.markdown("**TfL disruption**")
                                st.text(tfl_summary)
                                st.markdown("**Met Office weather**")
                                st.text(weather_summary)

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

                                    if use_ai_planner and openai_api_key:
                                        st.markdown("#### AI explanation of the day")
                                        try:
                                            day_summary_df = pd.DataFrame([{
                                                "Date": route_date.isoformat(),
                                                "Surveys": len(schedule.items),
                                                "Leave Harpenden": (
                                                    schedule.start_time
                                                    .strftime("%H:%M")
                                                ),
                                                "Return Harpenden": (
                                                    schedule.return_time
                                                    .strftime("%H:%M")
                                                ),
                                                "Survey Time (Minutes)": (
                                                    schedule.survey_minutes
                                                ),
                                                "Travel Time (Minutes)": (
                                                    schedule.travel_minutes
                                                ),
                                            }])

                                            day_ai_decisions = []
                                            for site in sites:
                                                day_ai_decisions.append({
                                                    "customer_reference": site.get(
                                                        "customer_reference", ""
                                                    ),
                                                    "building_name": site.get(
                                                        "building_name", ""
                                                    ),
                                                    "postcode": site.get(
                                                        "postcode", ""
                                                    ),
                                                    "ai_priority": site.get(
                                                        "ai_priority", 50
                                                    ),
                                                    "ai_decision": site.get(
                                                        "ai_decision", "neutral"
                                                    ),
                                                    "ai_reason": site.get(
                                                        "ai_reason", ""
                                                    ),
                                                })

                                            day_narrative = (
                                                OpenAISchedulePlanner(
                                                    api_key=openai_api_key,
                                                    model=openai_model,
                                                ).summarise_week(
                                                    week_summary=(
                                                        day_summary_df.to_dict(
                                                            orient="records"
                                                        )
                                                    ),
                                                    full_schedule=(
                                                        schedule_df.to_dict(
                                                            orient="records"
                                                        )
                                                    ),
                                                    ai_site_decisions=(
                                                        day_ai_decisions
                                                    ),
                                                    unscheduled_sites=[],
                                                    tfl_summary=tfl_summary,
                                                    weather_summary=(
                                                        weather_summary
                                                    ),
                                                    future_cluster_summary=(
                                                        cluster_summary
                                                    ),
                                                )
                                            )
                                            st.write(day_narrative)
                                        except Exception as exc:
                                            st.warning(
                                                "The day was created, but the AI "
                                                "explanation could not be generated: "
                                                f"{exc}"
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

                                    if use_ai_planner and openai_api_key:
                                        st.markdown("#### AI summary of the week")

                                        ai_decision_rows = []
                                        for site in sites:
                                            ai_decision_rows.append({
                                                "customer_reference": site.get(
                                                    "customer_reference", ""
                                                ),
                                                "building_name": site.get(
                                                    "building_name", ""
                                                ),
                                                "postcode": site.get(
                                                    "postcode", ""
                                                ),
                                                "ai_priority": site.get(
                                                    "ai_priority", 50
                                                ),
                                                "ai_decision": site.get(
                                                    "ai_decision", "neutral"
                                                ),
                                                "ai_reason": site.get(
                                                    "ai_reason", ""
                                                ),
                                                "planning_minutes": site.get(
                                                    "planning_minutes"
                                                ),
                                                "confidence": site.get(
                                                    "confidence", ""
                                                ),
                                            })

                                        unscheduled_for_summary = []
                                        for remaining_site in (
                                            weekly_schedule.unscheduled_sites
                                        ):
                                            unscheduled_for_summary.append({
                                                "customer_reference": (
                                                    remaining_site.get(
                                                        "customer_reference", ""
                                                    )
                                                ),
                                                "building_name": (
                                                    remaining_site.get(
                                                        "building_name", ""
                                                    )
                                                ),
                                                "postcode": remaining_site.get(
                                                    "postcode", ""
                                                ),
                                                "ai_priority": remaining_site.get(
                                                    "ai_priority", 50
                                                ),
                                                "ai_decision": remaining_site.get(
                                                    "ai_decision", "neutral"
                                                ),
                                                "ai_reason": remaining_site.get(
                                                    "ai_reason", ""
                                                ),
                                            })

                                        try:
                                            planner_for_summary = (
                                                OpenAISchedulePlanner(
                                                    api_key=openai_api_key,
                                                    model=openai_model,
                                                )
                                            )
                                            week_narrative = (
                                                planner_for_summary.summarise_week(
                                                    week_summary=(
                                                        week_summary_df
                                                        .to_dict(
                                                            orient="records"
                                                        )
                                                    ),
                                                    full_schedule=(
                                                        full_week_df
                                                        .to_dict(
                                                            orient="records"
                                                        )
                                                    ),
                                                    ai_site_decisions=(
                                                        ai_decision_rows
                                                    ),
                                                    unscheduled_sites=(
                                                        unscheduled_for_summary
                                                    ),
                                                    tfl_summary=tfl_summary,
                                                    weather_summary=(
                                                        weather_summary
                                                    ),
                                                    future_cluster_summary=(
                                                        cluster_summary
                                                    ),
                                                )
                                            )
                                            st.write(week_narrative)
                                        except Exception as exc:
                                            st.warning(
                                                "The schedule was created, but "
                                                "the AI week summary could not "
                                                f"be generated: {exc}"
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
