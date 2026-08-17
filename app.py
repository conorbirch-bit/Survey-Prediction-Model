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
st.caption("Version 15 — unified weekly scheduling + same-campus routing bypass")
st.caption(
    "Upload the master portfolio, set surveyor availability for one week, then "
    "use Google transit routing only for that selected week."
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

tab1, tab2, tab3 = st.tabs([
    "Predict one building",
    "Weekly scheduling",
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
    st.subheader("Weekly scheduling")
    st.write(
        "Create the selected week's schedules for one or more surveyors. "
        "The wider portfolio is used for cheap clustering and drawing priority; "
        "Google Routes is only used for this one selected week."
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

                    st.markdown("#### Surveyor availability")
                    st.caption(
                        "Tick the exact dates each person is available in the selected "
                        "week. A surveyor with no dates ticked generates no Google "
                        "routing calls."
                    )

                    selected_week_dates = [
                        team_week_start + timedelta(days=offset)
                        for offset in range(5)
                    ]
                    availability_columns = {
                        d: d.strftime("%a %d %b")
                        for d in selected_week_dates
                    }

                    default_surveyors = pd.DataFrame([
                        {
                            "Name": "Conor Birch",
                            "Start / Finish Location": "Harpenden Station",
                            **{
                                label: True
                                for label in availability_columns.values()
                            },
                        },
                        {
                            "Name": "Surveyor 2",
                            "Start / Finish Location": "",
                            **{
                                label: False
                                for label in availability_columns.values()
                            },
                        },
                        {
                            "Name": "Surveyor 3",
                            "Start / Finish Location": "",
                            **{
                                label: False
                                for label in availability_columns.values()
                            },
                        },
                        {
                            "Name": "Surveyor 4",
                            "Start / Finish Location": "",
                            **{
                                label: False
                                for label in availability_columns.values()
                            },
                        },
                    ])

                    availability_config = {
                        label: st.column_config.CheckboxColumn(
                            label,
                            help=f"Available on {d.strftime('%A %d %B %Y')}",
                        )
                        for d, label in availability_columns.items()
                    }

                    edited_surveyors = st.data_editor(
                        default_surveyors,
                        key=(
                            "team_surveyor_editor_"
                            f"{team_week_start.isoformat()}"
                        ),
                        num_rows="fixed",
                        hide_index=True,
                        use_container_width=True,
                        column_config=availability_config,
                    )

                    st.info(
                        "Cost rule: Google route calculations are restricted to "
                        f"{team_week_start.strftime('%d %b %Y')}–"
                        f"{(team_week_start + timedelta(days=6)).strftime('%d %b %Y')}. "
                        "Future weeks influence clustering/drawing priority only."
                    )

                    team_settings = st.columns(4)
                    with team_settings[0]:
                        team_max_candidates = st.number_input(
                            "Max Google candidates per full 5-day surveyor",
                            min_value=15,
                            max_value=100,
                            value=40,
                            step=5,
                            help=(
                                "This is the five-day cap. Someone available for fewer "
                                "days gets a proportionally smaller shortlist. The whole "
                                "portfolio is never sent to Google."
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
                            "Same-campus transfer (min)",
                            min_value=0,
                            max_value=30,
                            value=5,
                            step=1,
                            key="team_same_postcode",
                            help=(
                                "Used instead of Google when two consecutive "
                                "buildings have the same full postcode and similar "
                                "building/address names."
                            ),
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
                        "Create weekly schedules",
                        type="primary",
                        key="create_team_week",
                        disabled=not bool(team_google_key.strip()),
                    ):
                        if team_week_start.weekday() != 0:
                            st.error(
                                "Week commencing must be a Monday."
                            )
                        elif team_finish_clock <= team_start_clock:
                            st.error(
                                "Latest return must be after the leave time."
                            )
                        else:
                            active_surveyors = []
                            invalid_active_rows = []

                            for _, row in edited_surveyors.iterrows():
                                name = str(row.get("Name", "")).strip()
                                location = str(
                                    row.get(
                                        "Start / Finish Location",
                                        "",
                                    )
                                ).strip()

                                available_dates = [
                                    d
                                    for d, label in availability_columns.items()
                                    if bool(row.get(label, False))
                                ]

                                # No dates ticked = not working this week.
                                if not available_dates:
                                    continue

                                if not name:
                                    invalid_active_rows.append(
                                        "A surveyor with availability selected "
                                        "has no name."
                                    )
                                    continue
                                if not location:
                                    invalid_active_rows.append(
                                        f"{name} has availability selected but "
                                        "no start / finish location."
                                    )
                                    continue

                                active_surveyors.append(
                                    SurveyorConfig(
                                        name=name,
                                        start_location=location,
                                        available_dates=available_dates,
                                    )
                                )

                            all_available_dates = sorted({
                                d
                                for surveyor in active_surveyors
                                for d in (surveyor.available_dates or [])
                            })

                            if invalid_active_rows:
                                for message in invalid_active_rows:
                                    st.error(message)
                            elif not active_surveyors:
                                st.error(
                                    "No surveyors have any availability selected for this week."
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

                                        effective_candidate_caps = {
                                            surveyor.name: max(
                                                1,
                                                round(
                                                    int(team_max_candidates)
                                                    * min(
                                                        5,
                                                        len(
                                                            surveyor.available_dates
                                                            or []
                                                        ),
                                                    )
                                                    / 5
                                                ),
                                            )
                                            for surveyor in active_surveyors
                                        }
                                        total_team_candidate_cap = sum(
                                            effective_candidate_caps.values()
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
                                                    all_available_dates
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

                                        allocation_departure = datetime.combine(
                                            min(all_available_dates),
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

                                            surveyor_dates = list(
                                                surveyor.available_dates or []
                                            )

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
                                                    "Available Dates": ", ".join(
                                                        d.strftime("%a %d %b")
                                                        for d in (
                                                            surveyor.available_dates
                                                            or []
                                                        )
                                                    ),
                                                    "Google Candidate Cap": (
                                                        effective_candidate_caps.get(
                                                            surveyor.name, 0
                                                        )
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
                                                "Available Dates": ", ".join(
                                                    d.strftime("%a %d %b")
                                                    for d in (
                                                        surveyor.available_dates
                                                        or []
                                                    )
                                                ),
                                                "Google Candidate Cap": (
                                                    effective_candidate_caps.get(
                                                        surveyor.name, 0
                                                    )
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


