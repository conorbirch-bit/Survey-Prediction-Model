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
from salesforce_master import read_salesforce_or_standard_excel
from team_scheduler import (
    SurveyorConfig,
    representative_sites,
    home_to_cluster_matrix,
    allocate_cluster_targets,
    build_team_shortlists,
    allocations_dataframe,
)
from portfolio_clusterer import (
    add_portfolio_fields,
    build_cluster_summary,
    deterministic_cluster_choices,
    shortlist_sites,
    build_drawing_priority_queue,
)


LONDON_TZ = ZoneInfo("Europe/London")
DEFAULT_FILE = Path(__file__).with_name("Predictive Model.xlsx")

st.set_page_config(page_title="Site Survey Scheduling Agent", layout="wide")
st.title("Site Survey Scheduling Agent")
st.caption("Version 13 — Salesforce master import + team scheduling")
st.caption(
    "Predict survey durations, shortlist strategic postcode clusters, then use "
    "Google transit routing only on realistic candidates."
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
        "Drawing Status": [
            "Drawing Status",
            "Draw Status",
            "Drawing State",
        ],
        "Earliest Survey Date": [
            "Earliest Survey Date",
            "Earliest Survey",
            "Available From",
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

tab1, tab2, tab3, tab4 = st.tabs([
    "Predict one building",
    "Upcoming surveys + routing",
    "Model diagnostics",
    "Team weekly schedule",
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
            raw_upcoming, detected_header_row = (
                read_salesforce_or_standard_excel(
                    upcoming_file.getvalue()
                )
            )
            upcoming = normalise_upcoming_columns(raw_upcoming)

            if detected_header_row > 0:
                st.caption(
                    f"Salesforce report detected: table header found on Excel row "
                    f"{detected_header_row + 1}. Report/filter rows above it were ignored."
                )

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
                    "Drawing Status",
                    "Earliest Survey Date",
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
                    f"{len(filtered)} predicted sites are in the current portfolio "
                    "before week eligibility and cluster filtering."
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

                shortlist_controls = st.columns(2)
                with shortlist_controls[0]:
                    default_google_candidates = (
                        25 if planning_mode == "Single day" else 60
                    )
                    max_sites_for_google = st.number_input(
                        "Maximum sites sent to Google Routes",
                        min_value=10,
                        max_value=200,
                        value=default_google_candidates,
                        step=5,
                        help=(
                            "The full portfolio is clustered first. Google Route "
                            "Matrix only sees this shortlisted number of sites."
                        ),
                    )
                with shortlist_controls[1]:
                    use_ai_cluster_filter = st.checkbox(
                        "Use AI to choose portfolio clusters",
                        value=True,
                        help=(
                            "OpenAI reviews cheap postcode-cluster summaries first. "
                            "If disabled, clusters are shortlisted deterministically "
                            "by eligible site volume."
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
                        try:
                            with st.spinner(
                                "Filtering the portfolio into useful clusters before "
                                "calling Google Routes..."
                            ):
                                ai_strategy = ""
                                cluster_strategy = ""
                                deferred_clusters = []
                                tfl_summary = "TfL integration not configured."
                                weather_summary = "Met Office integration not configured."

                                # Determine the requested planning period.
                                if planning_mode == "Single day":
                                    context_week_start = (
                                        route_date - timedelta(
                                            days=route_date.weekday()
                                        )
                                    )
                                    period_start = route_date
                                    period_end = route_date
                                    working_day_count = 1
                                else:
                                    context_week_start = week_start
                                    period_start = week_start
                                    period_end = week_start + timedelta(days=6)
                                    working_day_count = len(working_days)

                                # STEP 1: cheap portfolio eligibility / postcode clustering.
                                portfolio_for_week = add_portfolio_fields(
                                    filtered,
                                    target_week_start=context_week_start,
                                    today=datetime.now(LONDON_TZ).date(),
                                )
                                eligible_portfolio = portfolio_for_week[
                                    portfolio_for_week[
                                        "Eligible for Selected Week"
                                    ] == True
                                ].copy()

                                if eligible_portfolio.empty:
                                    raise ValueError(
                                        "No sites are eligible for the selected week "
                                        "after applying drawing/earliest-date rules."
                                    )

                                portfolio_cluster_summary_df = (
                                    build_cluster_summary(
                                        portfolio_for_week,
                                        context_week_start,
                                    )
                                )

                                # External context is gathered once for the week.
                                if tfl_api_key:
                                    tfl_summary = TfLClient(
                                        tfl_api_key
                                    ).disruption_summary(
                                        period_start,
                                        period_end,
                                    )

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

                                # STEP 2: AI chooses strategic clusters BEFORE Google.
                                planner = None
                                if (
                                    use_ai_cluster_filter
                                    or use_ai_planner
                                ):
                                    if not openai_api_key:
                                        raise ValueError(
                                            "OPENAI_API_KEY is missing from secrets."
                                        )
                                    planner = OpenAISchedulePlanner(
                                        api_key=openai_api_key,
                                        model=openai_model,
                                    )

                                if use_ai_cluster_filter:
                                    (
                                        cluster_decisions,
                                        deferred_clusters,
                                        cluster_strategy,
                                    ) = planner.select_clusters(
                                        cluster_summary=(
                                            portfolio_cluster_summary_df
                                            .to_dict(orient="records")
                                        ),
                                        week_label=(
                                            f"{period_start} to {period_end}"
                                        ),
                                        working_days=working_day_count,
                                        max_sites_for_google=int(
                                            max_sites_for_google
                                        ),
                                        tfl_summary=tfl_summary,
                                        weather_summary=weather_summary,
                                    )
                                    cluster_choices = [
                                        {
                                            "cluster": d.cluster,
                                            "priority": d.priority,
                                            "target_sites": d.target_sites,
                                            "decision": d.decision,
                                            "reason": d.reason,
                                        }
                                        for d in cluster_decisions
                                    ]
                                else:
                                    cluster_choices = (
                                        deterministic_cluster_choices(
                                            portfolio_cluster_summary_df,
                                            int(max_sites_for_google),
                                        )
                                    )
                                    cluster_strategy = (
                                        "AI cluster filter disabled. The portfolio "
                                        "was shortlisted deterministically from the "
                                        "largest eligible postcode clusters."
                                    )

                                if not cluster_choices:
                                    raise ValueError(
                                        "No postcode clusters were selected for routing."
                                    )

                                cluster_choices_df = pd.DataFrame(cluster_choices)

                                # STEP 3: select only a small number of individual sites.
                                shortlist_df = shortlist_sites(
                                    portfolio=portfolio_for_week,
                                    cluster_choices=cluster_choices,
                                    max_sites_for_google=int(
                                        max_sites_for_google
                                    ),
                                    target_week_start=context_week_start,
                                )

                                if shortlist_df.empty:
                                    raise ValueError(
                                        "The cluster filter produced no route candidates."
                                    )

                                # Existing future-date context is still useful for
                                # site-level AI ranking within the cheap shortlist.
                                (
                                    cluster_counts,
                                    future_cluster_summary_text,
                                ) = build_future_cluster_context(
                                    portfolio_for_week,
                                    context_week_start,
                                )

                                sites = []
                                for _, row in shortlist_df.iterrows():
                                    building_name = str(
                                        row.get("Building Name", "")
                                    ).strip()
                                    postcode = str(
                                        row.get("Postcode", "")
                                    ).strip()
                                    route_location = (
                                        f"{building_name}, {postcode}"
                                        if building_name else postcode
                                    )

                                    site = {
                                        "customer_reference": row.get(
                                            "Customer Reference", ""
                                        ),
                                        "building_name": (
                                            building_name or postcode
                                        ),
                                        "postcode": postcode,
                                        "postcode_district": (
                                            postcode_district(postcode)
                                        ),
                                        "route_location": route_location,
                                        "planning_minutes": int(
                                            row[
                                                "Planning Duration (Minutes)"
                                            ]
                                        ),
                                        "predicted_minutes": float(
                                            row[
                                                "Predicted Survey Duration (Minutes)"
                                            ]
                                        ),
                                        "confidence": row[
                                            "Prediction Confidence"
                                        ],
                                        "model_used": row[
                                            "Prediction Model Used"
                                        ],
                                        "planned_start": str(
                                            row.get("Planned Start", "") or ""
                                        ),
                                        "drawing_status": str(
                                            row.get(
                                                "Normalised Drawing Status",
                                                "Unknown",
                                            )
                                        ),
                                        "cluster_priority": row.get(
                                            "AI Cluster Priority", 50
                                        ),
                                        "cluster_reason": row.get(
                                            "AI Cluster Reason", ""
                                        ),
                                    }

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
                                    sites.append(site)

                                # STEP 4: optional site-level reasoning only on shortlist.
                                if use_ai_planner:
                                    decisions, ai_strategy = planner.rank_sites(
                                        sites=sites,
                                        week_label=(
                                            f"{period_start} to {period_end}"
                                        ),
                                        tfl_summary=tfl_summary,
                                        weather_summary=weather_summary,
                                        future_cluster_summary=(
                                            future_cluster_summary_text
                                        ),
                                    )
                                    sites = apply_ai_decisions(
                                        sites,
                                        decisions,
                                    )
                                else:
                                    sites = apply_ai_decisions(sites, [])

                                # STEP 5: paid / precise routing sees shortlist only.
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
                                                week_start
                                                + timedelta(days=offset)
                                            )

                                    weekly_schedule = scheduler.build_week(
                                        sites=sites,
                                        dates=chosen_dates,
                                        start_clock=start_clock,
                                        latest_return_clock=finish_clock,
                                        timezone=LONDON_TZ,
                                    )
                                    schedule = None

                                eligible_portfolio_count = len(
                                    eligible_portfolio
                                )
                                google_candidate_count = len(shortlist_df)
                                portfolio_count = len(portfolio_for_week)
                                excluded_before_google_count = max(
                                    0,
                                    eligible_portfolio_count
                                    - google_candidate_count,
                                )
                                reduction_pct = (
                                    100
                                    * excluded_before_google_count
                                    / eligible_portfolio_count
                                    if eligible_portfolio_count
                                    else 0
                                )

                                planning_context_summary = (
                                    "Portfolio cluster strategy:\n"
                                    f"{cluster_strategy}\n\n"
                                    "Future cluster counts:\n"
                                    f"{future_cluster_summary_text}"
                                )

                            st.markdown("#### Portfolio pre-filter")
                            f1, f2, f3, f4 = st.columns(4)
                            f1.metric("Portfolio sites", portfolio_count)
                            f2.metric(
                                "Eligible this week",
                                eligible_portfolio_count,
                            )
                            f3.metric(
                                "Sent to Google",
                                google_candidate_count,
                            )
                            f4.metric(
                                "Filtered before Google",
                                f"{reduction_pct:.0f}%",
                            )

                            st.caption(
                                f"{excluded_before_google_count} eligible sites "
                                "were excluded before any detailed Google Route "
                                "Matrix comparison."
                            )

                            if cluster_strategy:
                                st.markdown("#### AI cluster strategy")
                                st.info(cluster_strategy)

                            with st.expander(
                                "Portfolio cluster summary and Google shortlist"
                            ):
                                st.markdown("**Portfolio clusters**")
                                st.dataframe(
                                    portfolio_cluster_summary_df,
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                st.markdown("**Selected clusters**")
                                st.dataframe(
                                    cluster_choices_df,
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                if deferred_clusters:
                                    st.markdown("**AI-deferred clusters**")
                                    st.dataframe(
                                        pd.DataFrame(deferred_clusters),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                st.markdown(
                                    f"**Sites sent to Google ({len(shortlist_df)})**"
                                )
                                shortlist_display = [
                                    c for c in [
                                        "Customer Reference",
                                        "Building Name",
                                        "Postcode",
                                        "Postcode Cluster",
                                        "Drawing Status",
                                        "Normalised Drawing Status",
                                        "Planning Duration (Minutes)",
                                        "Prediction Confidence",
                                        "AI Cluster Priority",
                                        "AI Cluster Reason",
                                    ]
                                    if c in shortlist_df.columns
                                ]
                                st.dataframe(
                                    shortlist_df[shortlist_display],
                                    use_container_width=True,
                                    hide_index=True,
                                )

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
                                        portfolio_for_week.to_excel(
                                            writer,
                                            sheet_name="Portfolio",
                                            index=False,
                                        )
                                        portfolio_cluster_summary_df.to_excel(
                                            writer,
                                            sheet_name="Cluster Summary",
                                            index=False,
                                        )
                                        cluster_choices_df.to_excel(
                                            writer,
                                            sheet_name="Selected Clusters",
                                            index=False,
                                        )
                                        shortlist_df.to_excel(
                                            writer,
                                            sheet_name="Google Shortlist",
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
                                            f"{schedule.unscheduled_count} shortlisted sites "
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
                                                        planning_context_summary
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
                                        "Shortlist remaining",
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
                                                        planning_context_summary
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

                                        portfolio_for_week.to_excel(
                                            writer,
                                            sheet_name="Portfolio",
                                            index=False,
                                        )
                                        portfolio_cluster_summary_df.to_excel(
                                            writer,
                                            sheet_name="Cluster Summary",
                                            index=False,
                                        )
                                        cluster_choices_df.to_excel(
                                            writer,
                                            sheet_name="Selected Clusters",
                                            index=False,
                                        )
                                        shortlist_df.to_excel(
                                            writer,
                                            sheet_name="Google Shortlist",
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


with tab4:
    st.subheader("Team weekly schedule")
    st.write(
        "Plan one upcoming week for multiple surveyors. The full portfolio is "
        "clustered once; only each surveyor's small allocated shortlist is sent "
        "to Google Routes."
    )

    team_file = st.file_uploader(
        "Master portfolio spreadsheet",
        type=["xlsx", "xls"],
        key="team_master_file",
        help=(
            "Can contain both survey-ready and Needs Drawing sites. "
            "Recommended columns include Customer Reference, Building Name, "
            "Postcode, Building Height, Sovereign Flat, Internal Ground Floor "
            "Area, Drawing Status and Earliest Survey Date."
        ),
    )

    if team_file is None:
        st.info(
            "Upload the master portfolio to build a multi-surveyor weekly schedule."
        )
    else:
        try:
            team_raw, team_detected_header_row = (
                read_salesforce_or_standard_excel(
                    team_file.getvalue()
                )
            )
            team_upcoming = normalise_upcoming_columns(team_raw)

            if team_detected_header_row > 0:
                st.caption(
                    f"Salesforce report detected: table header found on Excel row "
                    f"{team_detected_header_row + 1}. Report/filter rows above it were ignored."
                )

            if "Postcode" not in team_upcoming.columns:
                st.error(
                    "The master portfolio needs a Postcode column for clustering "
                    "and routing."
                )
            else:
                team_predictions = predict_upcoming(team_upcoming)
                team_routable = team_predictions[
                    (team_predictions["Prediction Status"] == "Predicted")
                    & team_predictions["Postcode"].notna()
                    & (
                        team_predictions["Postcode"]
                        .astype(str)
                        .str.strip()
                        != ""
                    )
                ].copy()

                if team_routable.empty:
                    st.warning(
                        "There are no portfolio sites with both a duration prediction "
                        "and a postcode."
                    )
                else:
                    today = datetime.now(LONDON_TZ).date()
                    current_monday = today - timedelta(days=today.weekday())
                    default_team_week = current_monday + timedelta(days=7)

                    t1, t2, t3 = st.columns(3)
                    with t1:
                        team_week_start = st.date_input(
                            "Team week commencing",
                            value=default_team_week,
                            key="team_week_start",
                        )
                    with t2:
                        team_start_clock = st.time_input(
                            "Team leave time",
                            value=time(7, 50),
                            key="team_start_clock",
                        )
                    with t3:
                        team_finish_clock = st.time_input(
                            "Team latest return",
                            value=time(16, 0),
                            key="team_finish_clock",
                        )

                    team_working_days = st.multiselect(
                        "Team working days",
                        [
                            "Monday",
                            "Tuesday",
                            "Wednesday",
                            "Thursday",
                            "Friday",
                        ],
                        default=[
                            "Monday",
                            "Tuesday",
                            "Wednesday",
                            "Thursday",
                            "Friday",
                        ],
                        key="team_working_days",
                    )

                    st.markdown("#### Surveyors")
                    st.caption(
                        "Add the four people now. Only rows marked Active, with an "
                        "Active From date on/before the selected week and a start "
                        "location, will generate paid Google routing."
                    )

                    default_surveyors = pd.DataFrame([
                        {
                            "Name": "Conor Birch",
                            "Active": True,
                            "Start / Finish Location": "Harpenden Station",
                            "Active From": today,
                        },
                        {
                            "Name": "Surveyor 2",
                            "Active": False,
                            "Start / Finish Location": "",
                            "Active From": default_team_week,
                        },
                        {
                            "Name": "Surveyor 3",
                            "Active": False,
                            "Start / Finish Location": "",
                            "Active From": default_team_week,
                        },
                        {
                            "Name": "Surveyor 4",
                            "Active": False,
                            "Start / Finish Location": "",
                            "Active From": default_team_week,
                        },
                    ])

                    edited_surveyors = st.data_editor(
                        default_surveyors,
                        key="team_surveyor_editor",
                        num_rows="fixed",
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Active": st.column_config.CheckboxColumn(
                                "Active",
                                help=(
                                    "Only active surveyors are scheduled."
                                ),
                            ),
                            "Active From": st.column_config.DateColumn(
                                "Active From",
                                format="DD/MM/YYYY",
                            ),
                        },
                    )

                    team_settings = st.columns(4)
                    with team_settings[0]:
                        team_max_candidates = st.number_input(
                            "Max Google candidates per surveyor",
                            min_value=15,
                            max_value=100,
                            value=40,
                            step=5,
                            help=(
                                "Each active surveyor gets their own capped shortlist. "
                                "The whole portfolio is never sent to Google."
                            ),
                        )
                    with team_settings[1]:
                        team_transit_choice = st.selectbox(
                            "Team transit preference",
                            [
                                "Fastest / default",
                                "Less walking",
                                "Fewer transfers",
                            ],
                            key="team_transit_choice",
                        )
                    with team_settings[2]:
                        team_same_postcode = st.number_input(
                            "Same-postcode transfer (min)",
                            min_value=0,
                            max_value=30,
                            value=5,
                            step=1,
                            key="team_same_postcode",
                        )
                    with team_settings[3]:
                        use_team_ai_clusters = st.checkbox(
                            "AI cluster selection",
                            value=True,
                            key="team_ai_clusters",
                            help=(
                                "OpenAI sees cheap cluster summaries, not all detailed "
                                "Google route combinations."
                            ),
                        )

                    team_buffers = st.columns(3)
                    with team_buffers[0]:
                        team_travel_leeway = st.number_input(
                            "Team travel leeway (min)",
                            min_value=0,
                            max_value=30,
                            value=5,
                            step=1,
                            key="team_travel_leeway",
                        )
                    with team_buffers[1]:
                        team_pre_buffer = st.number_input(
                            "Before survey (min)",
                            min_value=0,
                            max_value=30,
                            value=5,
                            step=1,
                            key="team_pre_buffer",
                        )
                    with team_buffers[2]:
                        team_post_buffer = st.number_input(
                            "After survey (min)",
                            min_value=0,
                            max_value=30,
                            value=5,
                            step=1,
                            key="team_post_buffer",
                        )

                    generate_team_ai_summary = st.checkbox(
                        "Generate one AI summary of the final team week",
                        value=True,
                        key="team_ai_summary",
                    )

                    team_google_key = get_secret("GOOGLE_MAPS_API_KEY")
                    team_openai_key = get_secret("OPENAI_API_KEY")
                    team_openai_model = get_secret("OPENAI_MODEL", "gpt-5.6")
                    team_tfl_key = get_secret("TFL_API_KEY")
                    team_met_key = get_secret("MET_OFFICE_API_KEY")
                    team_met_endpoint = get_secret(
                        "MET_OFFICE_GLOBAL_SPOT_URL"
                    )

                    status_cols = st.columns(4)
                    status_cols[0].metric(
                        "Portfolio rows",
                        len(team_predictions),
                    )
                    status_cols[1].metric(
                        "Predicted/routable",
                        len(team_routable),
                    )
                    status_cols[2].metric(
                        "Google Routes",
                        "Ready" if team_google_key else "Missing",
                    )
                    status_cols[3].metric(
                        "OpenAI",
                        "Ready" if team_openai_key else "Missing",
                    )

                    if st.button(
                        "Create team weekly schedules",
                        type="primary",
                        key="create_team_week",
                        disabled=not bool(team_google_key.strip()),
                    ):
                        if team_finish_clock <= team_start_clock:
                            st.error(
                                "Latest return must be after the leave time."
                            )
                        elif not team_working_days:
                            st.error("Choose at least one team working day.")
                        else:
                            active_surveyors = []
                            invalid_active_rows = []

                            for _, row in edited_surveyors.iterrows():
                                is_active = bool(row.get("Active", False))
                                name = str(row.get("Name", "")).strip()
                                location = str(
                                    row.get(
                                        "Start / Finish Location",
                                        "",
                                    )
                                ).strip()
                                active_from_raw = row.get("Active From")
                                active_from = pd.to_datetime(
                                    active_from_raw,
                                    errors="coerce",
                                )

                                if not is_active:
                                    continue
                                if not name:
                                    invalid_active_rows.append(
                                        "An active surveyor has no name."
                                    )
                                    continue
                                if not location:
                                    invalid_active_rows.append(
                                        f"{name} has no start / finish location."
                                    )
                                    continue
                                if pd.isna(active_from):
                                    invalid_active_rows.append(
                                        f"{name} has no valid Active From date."
                                    )
                                    continue

                                active_from_date = active_from.date()
                                if active_from_date > (
                                    team_week_start + timedelta(days=6)
                                ):
                                    # They exist in the team, but are not active yet.
                                    continue

                                active_surveyors.append(
                                    SurveyorConfig(
                                        name=name,
                                        start_location=location,
                                        active_from=active_from_date,
                                    )
                                )

                            if invalid_active_rows:
                                for message in invalid_active_rows:
                                    st.error(message)
                            elif not active_surveyors:
                                st.error(
                                    "No surveyors are active for the selected week."
                                )
                            else:
                                try:
                                    with st.spinner(
                                        "Clustering the portfolio, allocating work "
                                        "across the team, then routing only each "
                                        "person's shortlist..."
                                    ):
                                        team_portfolio = add_portfolio_fields(
                                            team_routable,
                                            target_week_start=team_week_start,
                                            today=today,
                                        )
                                        team_eligible = team_portfolio[
                                            team_portfolio[
                                                "Eligible for Selected Week"
                                            ] == True
                                        ].copy()

                                        if team_eligible.empty:
                                            raise ValueError(
                                                "No sites are eligible for the "
                                                "selected team week."
                                            )

                                        team_cluster_summary = (
                                            build_cluster_summary(
                                                team_portfolio,
                                                team_week_start,
                                            )
                                        )

                                        team_period_end = (
                                            team_week_start
                                            + timedelta(days=6)
                                        )

                                        team_tfl_summary = (
                                            "TfL integration not configured."
                                        )
                                        if team_tfl_key:
                                            team_tfl_summary = TfLClient(
                                                team_tfl_key
                                            ).disruption_summary(
                                                team_week_start,
                                                team_period_end,
                                            )

                                        team_weather_summary = (
                                            "Met Office integration not configured."
                                        )
                                        if (
                                            team_met_key
                                            and team_met_endpoint
                                        ):
                                            team_weather_summary = (
                                                MetOfficeClient(
                                                    team_met_key,
                                                    team_met_endpoint,
                                                ).forecast_summary(
                                                    latitude=51.65,
                                                    longitude=-0.20,
                                                    start_date=team_week_start,
                                                    end_date=team_period_end,
                                                )
                                            )

                                        total_team_candidate_cap = (
                                            int(team_max_candidates)
                                            * len(active_surveyors)
                                        )

                                        team_planner = None
                                        if (
                                            use_team_ai_clusters
                                            or generate_team_ai_summary
                                        ):
                                            if not team_openai_key:
                                                raise ValueError(
                                                    "OPENAI_API_KEY is missing "
                                                    "from secrets."
                                                )
                                            team_planner = (
                                                OpenAISchedulePlanner(
                                                    api_key=team_openai_key,
                                                    model=team_openai_model,
                                                )
                                            )

                                        if use_team_ai_clusters:
                                            (
                                                team_cluster_decisions,
                                                team_deferred_clusters,
                                                team_cluster_strategy,
                                            ) = team_planner.select_clusters(
                                                cluster_summary=(
                                                    team_cluster_summary
                                                    .to_dict(
                                                        orient="records"
                                                    )
                                                ),
                                                week_label=(
                                                    f"{team_week_start} to "
                                                    f"{team_period_end}"
                                                ),
                                                working_days=len(
                                                    team_working_days
                                                ),
                                                max_sites_for_google=(
                                                    total_team_candidate_cap
                                                ),
                                                tfl_summary=(
                                                    team_tfl_summary
                                                ),
                                                weather_summary=(
                                                    team_weather_summary
                                                ),
                                                team_size=len(
                                                    active_surveyors
                                                ),
                                            )
                                            team_cluster_choices = [
                                                {
                                                    "cluster": d.cluster,
                                                    "priority": d.priority,
                                                    "target_sites": (
                                                        d.target_sites
                                                    ),
                                                    "decision": d.decision,
                                                    "reason": d.reason,
                                                }
                                                for d in (
                                                    team_cluster_decisions
                                                )
                                            ]
                                        else:
                                            team_cluster_choices = (
                                                deterministic_cluster_choices(
                                                    team_cluster_summary,
                                                    total_team_candidate_cap,
                                                )
                                            )
                                            team_deferred_clusters = []
                                            team_cluster_strategy = (
                                                "AI cluster selection disabled. "
                                                "The largest eligible postcode "
                                                "clusters were used."
                                            )

                                        if not team_cluster_choices:
                                            raise ValueError(
                                                "No clusters were selected for "
                                                "the team week."
                                            )

                                        drawing_priority_queue = (
                                            build_drawing_priority_queue(
                                                portfolio=team_portfolio,
                                                cluster_summary=team_cluster_summary,
                                                selected_cluster_choices=(
                                                    team_cluster_choices
                                                ),
                                                target_week_start=(
                                                    team_week_start
                                                ),
                                            )
                                        )

                                        team_router_pref = {
                                            "Fastest / default": None,
                                            "Less walking": "LESS_WALKING",
                                            "Fewer transfers": (
                                                "FEWER_TRANSFERS"
                                            ),
                                        }

                                        # Google enters here for the first time:
                                        # tiny surveyor-home -> selected-cluster matrix.
                                        team_router = GoogleTransitRouter(
                                            api_key=team_google_key,
                                            transit_preference=(
                                                team_router_pref[
                                                    team_transit_choice
                                                ]
                                            ),
                                        )

                                        team_representatives = (
                                            representative_sites(
                                                team_portfolio,
                                                team_cluster_choices,
                                                team_week_start,
                                            )
                                        )

                                        first_working_offset = min(
                                            [
                                                [
                                                    "Monday",
                                                    "Tuesday",
                                                    "Wednesday",
                                                    "Thursday",
                                                    "Friday",
                                                ].index(day)
                                                for day in team_working_days
                                            ]
                                        )
                                        allocation_departure = datetime.combine(
                                            (
                                                team_week_start
                                                + timedelta(
                                                    days=first_working_offset
                                                )
                                            ),
                                            team_start_clock,
                                            tzinfo=LONDON_TZ,
                                        )

                                        team_home_cluster_matrix = (
                                            home_to_cluster_matrix(
                                                team_router,
                                                active_surveyors,
                                                team_representatives,
                                                allocation_departure,
                                            )
                                        )

                                        team_allocations = (
                                            allocate_cluster_targets(
                                                cluster_choices=(
                                                    team_cluster_choices
                                                ),
                                                cluster_summary=(
                                                    team_cluster_summary
                                                ),
                                                surveyors=active_surveyors,
                                                travel_matrix=(
                                                    team_home_cluster_matrix
                                                ),
                                                max_sites_per_surveyor=int(
                                                    team_max_candidates
                                                ),
                                            )
                                        )

                                        if not team_allocations:
                                            raise ValueError(
                                                "Google could not create any "
                                                "usable surveyor-to-cluster "
                                                "allocations."
                                            )

                                        team_shortlists = (
                                            build_team_shortlists(
                                                portfolio=team_portfolio,
                                                allocations=team_allocations,
                                                surveyors=active_surveyors,
                                                target_week_start=(
                                                    team_week_start
                                                ),
                                                max_sites_per_surveyor=int(
                                                    team_max_candidates
                                                ),
                                            )
                                        )

                                        day_names = [
                                            "Monday",
                                            "Tuesday",
                                            "Wednesday",
                                            "Thursday",
                                            "Friday",
                                        ]
                                        chosen_team_dates = [
                                            team_week_start
                                            + timedelta(days=offset)
                                            for offset, day_name in enumerate(
                                                day_names
                                            )
                                            if day_name in team_working_days
                                        ]

                                        team_results = {}
                                        combined_candidate_frames = []

                                        for surveyor in active_surveyors:
                                            surveyor_df = team_shortlists.get(
                                                surveyor.name,
                                                pd.DataFrame(),
                                            )
                                            if surveyor_df.empty:
                                                team_results[
                                                    surveyor.name
                                                ] = None
                                                continue

                                            combined_candidate_frames.append(
                                                surveyor_df
                                            )

                                            surveyor_sites = []
                                            for _, site_row in (
                                                surveyor_df.iterrows()
                                            ):
                                                building_name = str(
                                                    site_row.get(
                                                        "Building Name",
                                                        "",
                                                    )
                                                ).strip()
                                                postcode = str(
                                                    site_row.get(
                                                        "Postcode",
                                                        "",
                                                    )
                                                ).strip()
                                                route_location = (
                                                    f"{building_name}, "
                                                    f"{postcode}"
                                                    if building_name
                                                    else postcode
                                                )

                                                surveyor_sites.append({
                                                    "customer_reference": (
                                                        site_row.get(
                                                            "Customer Reference",
                                                            "",
                                                        )
                                                    ),
                                                    "building_name": (
                                                        building_name
                                                        or postcode
                                                    ),
                                                    "postcode": postcode,
                                                    "postcode_district": (
                                                        postcode_district(
                                                            postcode
                                                        )
                                                    ),
                                                    "route_location": (
                                                        route_location
                                                    ),
                                                    "planning_minutes": int(
                                                        site_row[
                                                            "Planning Duration "
                                                            "(Minutes)"
                                                        ]
                                                    ),
                                                    "predicted_minutes": float(
                                                        site_row[
                                                            "Predicted Survey "
                                                            "Duration (Minutes)"
                                                        ]
                                                    ),
                                                    "confidence": site_row[
                                                        "Prediction Confidence"
                                                    ],
                                                    "model_used": site_row[
                                                        "Prediction Model Used"
                                                    ],
                                                    "planned_start": str(
                                                        site_row.get(
                                                            "Planned Start",
                                                            "",
                                                        )
                                                        or ""
                                                    ),
                                                    "ai_priority": float(
                                                        site_row.get(
                                                            "AI Cluster Priority",
                                                            50,
                                                        )
                                                    ),
                                                    "ai_decision": "neutral",
                                                    "ai_reason": str(
                                                        site_row.get(
                                                            "AI Cluster Reason",
                                                            "",
                                                        )
                                                    ),
                                                })

                                            surveyor_scheduler = (
                                                DailyTransitScheduler(
                                                    router=team_router,
                                                    home_location=(
                                                        surveyor.start_location
                                                    ),
                                                    same_postcode_transfer_minutes=int(
                                                        team_same_postcode
                                                    ),
                                                    travel_leeway_minutes=int(
                                                        team_travel_leeway
                                                    ),
                                                    pre_survey_buffer_minutes=int(
                                                        team_pre_buffer
                                                    ),
                                                    post_survey_buffer_minutes=int(
                                                        team_post_buffer
                                                    ),
                                                    ai_priority_weight_minutes=(
                                                        15.0
                                                    ),
                                                )
                                            )

                                            surveyor_dates = [
                                                d
                                                for d in chosen_team_dates
                                                if (
                                                    surveyor.active_from is None
                                                    or d >= surveyor.active_from
                                                )
                                            ]

                                            if not surveyor_dates:
                                                team_results[
                                                    surveyor.name
                                                ] = None
                                                continue

                                            team_results[
                                                surveyor.name
                                            ] = (
                                                surveyor_scheduler.build_week(
                                                    sites=surveyor_sites,
                                                    dates=surveyor_dates,
                                                    start_clock=(
                                                        team_start_clock
                                                    ),
                                                    latest_return_clock=(
                                                        team_finish_clock
                                                    ),
                                                    timezone=LONDON_TZ,
                                                )
                                            )

                                        team_allocations_df = (
                                            allocations_dataframe(
                                                team_allocations
                                            )
                                        )

                                        if combined_candidate_frames:
                                            team_google_shortlist = pd.concat(
                                                combined_candidate_frames,
                                                ignore_index=True,
                                            ).drop_duplicates()
                                        else:
                                            team_google_shortlist = (
                                                team_eligible.head(0)
                                            )

                                        # Build combined team outputs.
                                        team_summary_rows = []
                                        combined_schedule_frames = []

                                        for surveyor in active_surveyors:
                                            result = team_results.get(
                                                surveyor.name
                                            )
                                            if result is None:
                                                team_summary_rows.append({
                                                    "Surveyor": (
                                                        surveyor.name
                                                    ),
                                                    "Start / Finish": (
                                                        surveyor.start_location
                                                    ),
                                                    "Surveys": 0,
                                                    "Survey Minutes": 0,
                                                    "Travel Minutes": 0,
                                                    "Shortlist Sites": 0,
                                                })
                                                continue

                                            team_summary_rows.append({
                                                "Surveyor": surveyor.name,
                                                "Start / Finish": (
                                                    surveyor.start_location
                                                ),
                                                "Surveys": (
                                                    result.total_surveys
                                                ),
                                                "Survey Minutes": (
                                                    result.total_survey_minutes
                                                ),
                                                "Travel Minutes": (
                                                    result.total_travel_minutes
                                                ),
                                                "Shortlist Sites": len(
                                                    team_shortlists.get(
                                                        surveyor.name,
                                                        pd.DataFrame(),
                                                    )
                                                ),
                                            })

                                            full_df = (
                                                result
                                                .full_schedule_dataframe()
                                                .copy()
                                            )
                                            if not full_df.empty:
                                                full_df.insert(
                                                    0,
                                                    "Surveyor",
                                                    surveyor.name,
                                                )
                                                combined_schedule_frames.append(
                                                    full_df
                                                )

                                        team_summary_df = pd.DataFrame(
                                            team_summary_rows
                                        )
                                        if combined_schedule_frames:
                                            combined_team_schedule = (
                                                pd.concat(
                                                    combined_schedule_frames,
                                                    ignore_index=True,
                                                )
                                            )
                                        else:
                                            combined_team_schedule = (
                                                pd.DataFrame()
                                            )

                                        cluster_matrix_rows = []
                                        for surveyor in active_surveyors:
                                            for rep in (
                                                team_representatives
                                            ):
                                                cluster_matrix_rows.append({
                                                    "Surveyor": (
                                                        surveyor.name
                                                    ),
                                                    "Cluster": (
                                                        rep["cluster"]
                                                    ),
                                                    "Home to Cluster "
                                                    "(Minutes)": (
                                                        team_home_cluster_matrix
                                                        .get(
                                                            (
                                                                surveyor.name,
                                                                rep[
                                                                    "cluster"
                                                                ],
                                                            )
                                                        )
                                                    ),
                                                })
                                        team_home_cluster_df = pd.DataFrame(
                                            cluster_matrix_rows
                                        )

                                        team_strategy_context = (
                                            "Strategic cluster selection:\n"
                                            f"{team_cluster_strategy}\n\n"
                                            "Team cluster allocations:\n"
                                            f"{team_allocations_df.to_dict(orient='records')}"
                                        )

                                    st.markdown("#### Team portfolio pre-filter")
                                    tm1, tm2, tm3, tm4 = st.columns(4)
                                    tm1.metric(
                                        "Active surveyors",
                                        len(active_surveyors),
                                    )
                                    tm2.metric(
                                        "Eligible portfolio",
                                        len(team_eligible),
                                    )
                                    tm3.metric(
                                        "Google shortlist total",
                                        len(team_google_shortlist),
                                    )
                                    reduction = (
                                        100
                                        * (
                                            len(team_eligible)
                                            - len(team_google_shortlist)
                                        )
                                        / len(team_eligible)
                                        if len(team_eligible)
                                        else 0
                                    )
                                    tm4.metric(
                                        "Filtered before routing",
                                        f"{reduction:.0f}%",
                                    )

                                    st.caption(
                                        "The only cross-team Google comparison is "
                                        f"{len(active_surveyors)} surveyor homes × "
                                        f"{len(team_representatives)} selected cluster "
                                        "representatives, followed by each person's "
                                        "own capped shortlist."
                                    )

                                    st.markdown("#### Team cluster strategy")
                                    st.info(team_cluster_strategy)

                                    st.markdown("#### Drawing priority")
                                    if drawing_priority_queue.empty:
                                        st.caption(
                                            "No sites in this portfolio are currently "
                                            "marked Needs Drawing."
                                        )
                                    else:
                                        st.caption(
                                            f"{len(drawing_priority_queue)} sites need "
                                            "drawing. Work down this numbered list; "
                                            "clusters needed for the target survey week "
                                            "are placed first."
                                        )
                                        drawing_display_cols = [
                                            c for c in [
                                                "Drawing Order",
                                                "Customer Reference",
                                                "Building Name",
                                                "Postcode",
                                                "Postcode Cluster",
                                                "Building Height",
                                                "Sovereign Flat",
                                                "Drawing Priority Tier",
                                                "Drawing Cluster Priority",
                                                "Drawing Priority Reason",
                                            ]
                                            if c in drawing_priority_queue.columns
                                        ]
                                        st.dataframe(
                                            drawing_priority_queue[
                                                drawing_display_cols
                                            ].head(100),
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                                        if len(drawing_priority_queue) > 100:
                                            st.caption(
                                                "Showing the first 100 drawing "
                                                "priorities here; the downloaded "
                                                "workbook contains the full queue."
                                            )

                                    with st.expander(
                                        "Cluster allocation details"
                                    ):
                                        st.markdown(
                                            "**Portfolio cluster summary**"
                                        )
                                        st.dataframe(
                                            team_cluster_summary,
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                                        st.markdown(
                                            "**Home → selected cluster travel**"
                                        )
                                        st.dataframe(
                                            team_home_cluster_df,
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                                        st.markdown(
                                            "**Cluster / workload allocation**"
                                        )
                                        st.dataframe(
                                            team_allocations_df,
                                            use_container_width=True,
                                            hide_index=True,
                                        )

                                    st.markdown("#### Team week")
                                    st.dataframe(
                                        team_summary_df,
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                                    total_team_surveys = int(
                                        team_summary_df["Surveys"].sum()
                                    )
                                    total_team_travel = int(
                                        team_summary_df[
                                            "Travel Minutes"
                                        ].sum()
                                    )
                                    total_team_survey_minutes = int(
                                        team_summary_df[
                                            "Survey Minutes"
                                        ].sum()
                                    )

                                    summary_metrics = st.columns(3)
                                    summary_metrics[0].metric(
                                        "Team surveys scheduled",
                                        total_team_surveys,
                                    )
                                    summary_metrics[1].metric(
                                        "Team survey time",
                                        f"{total_team_survey_minutes} min",
                                    )
                                    summary_metrics[2].metric(
                                        "Team transit time",
                                        f"{total_team_travel} min",
                                    )

                                    for surveyor in active_surveyors:
                                        result = team_results.get(
                                            surveyor.name
                                        )
                                        with st.expander(
                                            f"{surveyor.name} — "
                                            f"{surveyor.start_location}",
                                            expanded=True,
                                        ):
                                            shortlist = (
                                                team_shortlists.get(
                                                    surveyor.name,
                                                    pd.DataFrame(),
                                                )
                                            )
                                            st.caption(
                                                f"{len(shortlist)} sites were "
                                                "allocated to this person's Google "
                                                "candidate pool."
                                            )

                                            if result is None:
                                                st.warning(
                                                    "No candidate sites were "
                                                    "allocated."
                                                )
                                                continue

                                            st.dataframe(
                                                result.summary_dataframe(),
                                                use_container_width=True,
                                                hide_index=True,
                                            )
                                            full_person = (
                                                result
                                                .full_schedule_dataframe()
                                            )
                                            if not full_person.empty:
                                                st.dataframe(
                                                    full_person,
                                                    use_container_width=True,
                                                    hide_index=True,
                                                )

                                    if (
                                        generate_team_ai_summary
                                        and team_openai_key
                                        and not combined_team_schedule.empty
                                    ):
                                        st.markdown(
                                            "#### AI summary of the team week"
                                        )
                                        try:
                                            team_narrative = (
                                                team_planner.summarise_week(
                                                    week_summary=(
                                                        team_summary_df
                                                        .to_dict(
                                                            orient="records"
                                                        )
                                                    ),
                                                    full_schedule=(
                                                        combined_team_schedule
                                                        .to_dict(
                                                            orient="records"
                                                        )
                                                    ),
                                                    ai_site_decisions=[],
                                                    unscheduled_sites=[],
                                                    tfl_summary=(
                                                        team_tfl_summary
                                                    ),
                                                    weather_summary=(
                                                        team_weather_summary
                                                    ),
                                                    future_cluster_summary=(
                                                        team_strategy_context
                                                    ),
                                                )
                                            )
                                            st.write(team_narrative)
                                        except Exception as exc:
                                            st.warning(
                                                "Schedules were created, but "
                                                "the AI team summary failed: "
                                                f"{exc}"
                                            )

                                    team_output = io.BytesIO()
                                    with pd.ExcelWriter(
                                        team_output,
                                        engine="openpyxl",
                                    ) as writer:
                                        team_summary_df.to_excel(
                                            writer,
                                            sheet_name="Team Summary",
                                            index=False,
                                        )
                                        combined_team_schedule.to_excel(
                                            writer,
                                            sheet_name="Full Team Schedule",
                                            index=False,
                                        )
                                        team_cluster_summary.to_excel(
                                            writer,
                                            sheet_name="Cluster Summary",
                                            index=False,
                                        )
                                        pd.DataFrame(
                                            team_cluster_choices
                                        ).to_excel(
                                            writer,
                                            sheet_name="Selected Clusters",
                                            index=False,
                                        )
                                        team_allocations_df.to_excel(
                                            writer,
                                            sheet_name="Team Allocations",
                                            index=False,
                                        )
                                        team_home_cluster_df.to_excel(
                                            writer,
                                            sheet_name="Home Cluster Matrix",
                                            index=False,
                                        )
                                        team_google_shortlist.to_excel(
                                            writer,
                                            sheet_name="Google Shortlists",
                                            index=False,
                                        )
                                        drawing_priority_queue.to_excel(
                                            writer,
                                            sheet_name="Drawing Priority",
                                            index=False,
                                        )
                                        team_portfolio.to_excel(
                                            writer,
                                            sheet_name="Portfolio",
                                            index=False,
                                        )

                                        for surveyor in active_surveyors:
                                            result = team_results.get(
                                                surveyor.name
                                            )
                                            if result is None:
                                                continue
                                            safe_name = "".join(
                                                ch
                                                for ch in surveyor.name
                                                if ch
                                                not in r'[]:*?/\\'
                                            )[:23]
                                            sheet_name = (
                                                f"{safe_name} Week"
                                            )[:31]
                                            result.full_schedule_dataframe().to_excel(
                                                writer,
                                                sheet_name=sheet_name,
                                                index=False,
                                            )

                                    st.download_button(
                                        "Download team weekly schedule",
                                        data=team_output.getvalue(),
                                        file_name=(
                                            "Team_Survey_Week_"
                                            f"{team_week_start.isoformat()}.xlsx"
                                        ),
                                        mime=(
                                            "application/vnd.openxmlformats-"
                                            "officedocument.spreadsheetml.sheet"
                                        ),
                                        type="primary",
                                    )

                                except GoogleRoutesError as exc:
                                    st.error(str(exc))
                                except Exception as exc:
                                    st.error(
                                        f"Could not build team schedule: {exc}"
                                    )

        except Exception as exc:
            st.error(
                f"Could not process the master portfolio spreadsheet: {exc}"
            )
