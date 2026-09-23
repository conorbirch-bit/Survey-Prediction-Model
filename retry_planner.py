from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# RETRY BUSINESS RULES
# ---------------------------------------------------------------------------
# Kept as explicit constants so the agreed behaviour is easy to change later.
BLANK_CUSTOMER_FAILURE_ACTION = "IGNORE"
CUSTOMER_CLIENT_HELP_ATTEMPT = 3
TIME_PERIOD_SPLIT_HOUR = 12

RETRY_DECISIONS = {
    "RETRY",
    "RETRY_WITH_CONSTRAINT",
    "CLIENT_ACCESS_REQUIRED",
    "IGNORE",
}


@dataclass
class RetryPlan:
    decisions: pd.DataFrame
    client_access_required: pd.DataFrame
    warnings: List[str]
    stats: Dict[str, int]
    booking_exclusions: pd.DataFrame = field(default_factory=pd.DataFrame)


def _clean_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def _normalise_header(value) -> str:
    text = _clean_text(value)
    text = re.sub(r"\s+[↑↓]\s*$", "", text).strip()
    return text


def _normalise_work_order(value) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    if text.lower() in {"subtotal", "total", "sum", "count"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if text.isdigit() and len(text) < 8:
        text = text.zfill(8)
    return text


def _normalise_sa_id(value) -> str:
    text = _clean_text(value)
    return text if text.lower().startswith("08p") else ""


def _as_int(value, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        if value is None:
            return default
    try:
        return int(round(float(value)))
    except Exception:
        return default


def _as_optional_float(value):
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _parse_datetime(value):
    if value is None:
        return pd.NaT
    return pd.to_datetime(value, dayfirst=True, errors="coerce")


def _parse_booking_datetime(value):
    """Read UK report dates and ISO timestamps as UK-local, Excel-safe times."""
    text = _clean_text(value)
    if not text:
        return pd.NaT
    # Explicit ISO dates must not be interpreted as UK day/month strings.
    parsed = pd.to_datetime(value, dayfirst=not bool(re.match(r"^\d{4}-\d{2}-\d{2}", text)), errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("Europe/London").tz_localize(None)
    return parsed


def booking_week_start(value=None):
    """Monday of the UK calendar week, independent of the planning week."""
    parsed = _parse_booking_datetime(value) if value is not None else pd.Timestamp.now(tz="Europe/London")
    if pd.isna(parsed):
        raise ValueError("An exclusion week needs a valid date.")
    day = parsed.date()
    return day - timedelta(days=day.weekday())


def _period_from_datetime(value) -> str:
    parsed = _parse_datetime(value)
    if pd.isna(parsed):
        return ""
    return "Morning" if int(parsed.hour) < TIME_PERIOD_SPLIT_HOUR else "Afternoon"


def _opposite_period(period: str) -> str:
    p = _clean_text(period).lower()
    if p == "morning":
        return "Afternoon"
    if p == "afternoon":
        return "Morning"
    return ""


def _weekday_name(value) -> str:
    parsed = _parse_datetime(value)
    if pd.isna(parsed):
        return ""
    return parsed.day_name()


def _weekday_number(value):
    parsed = _parse_datetime(value)
    if pd.isna(parsed):
        return None
    return int(parsed.weekday())


def _read_report_table(excel_file: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
    if raw.empty:
        return pd.DataFrame()

    header_row = None
    best_score = -1
    header_markers = {
        "Work Order Number",
        "Service Appointment ID",
        "Primary Service Appointment: Service Appointment ID",
        "Reason Not Complete",
        "Failed Visits (Customer)",
        "Failed Visits (Metro)",
        "Actual Start",
    }

    for idx, row in raw.iterrows():
        values = {_normalise_header(v) for v in row.tolist() if _clean_text(v)}
        score = len(values.intersection(header_markers))
        if "Work Order Number" in values and score > best_score:
            best_score = score
            header_row = int(idx)

    if header_row is None:
        return pd.DataFrame()

    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row)
    # A present but entirely blank booking column means no bookings. Preserve
    # it so it is not confused with a report missing the required column.
    empty_columns = [c for c in df.columns
                     if df[c].isna().all() and _normalise_header(c) != "Scheduled Start"]
    df = df.drop(columns=empty_columns).dropna(axis=0, how="all")
    df.columns = [_normalise_header(c) for c in df.columns]

    # Keep names unique without changing the meaningful Salesforce headers.
    seen: Dict[str, int] = {}
    unique_columns = []
    for col in df.columns:
        base = col or "Unnamed"
        count = seen.get(base, 0)
        seen[base] = count + 1
        unique_columns.append(base if count == 0 else f"{base}__{count + 1}")
    df.columns = unique_columns
    return df.reset_index(drop=True)


def _detect_retry_tabs(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))

    failure_df = None
    sa_df = None

    for sheet_name in excel_file.sheet_names:
        table = _read_report_table(excel_file, sheet_name)
        columns = set(table.columns)

        if {
            "Work Order Number",
            "Failed Visits (Customer)",
            "Failed Visits (Metro)",
        }.issubset(columns) and (
            "Primary Service Appointment: Service Appointment ID" in columns
        ):
            failure_df = table
            continue

        if {
            "Work Order Number",
            "Service Appointment ID",
            "Actual Start",
        }.issubset(columns):
            sa_df = table

    if failure_df is None:
        raise ValueError(
            "Could not find the Cannot Complete history tab. It must contain "
            "Work Order Number, the failed Service Appointment ID, Failed Visits "
            "(Customer) and Failed Visits (Metro)."
        )
    if sa_df is None:
        raise ValueError(
            "Could not find the old/new Service Appointment tab. It must contain "
            "Work Order Number, Service Appointment ID and Actual Start."
        )

    return sa_df.copy(), failure_df.copy()


def _prepare_sa_mapping(sa_df: pd.DataFrame) -> pd.DataFrame:
    working = sa_df.copy()
    working["_wo_raw"] = working["Work Order Number"].apply(_normalise_work_order)
    working["_wo_group"] = working["_wo_raw"].replace("", pd.NA).ffill()
    working["_sa_id"] = working["Service Appointment ID"].apply(_normalise_sa_id)

    working = working[
        working["_wo_group"].notna() & working["_sa_id"].ne("")
    ].copy()

    working["Work Order Number"] = working["_wo_group"].astype(str)
    working["Service Appointment ID"] = working["_sa_id"]
    working["_actual_start_dt"] = working.get(
        "Actual Start", pd.Series(index=working.index, dtype=object)
    ).apply(_parse_datetime)
    working["_created_dt"] = working.get(
        "Created Date", pd.Series(index=working.index, dtype=object)
    ).apply(_parse_datetime)

    return working.drop(columns=["_wo_raw", "_wo_group", "_sa_id"], errors="ignore")


BOOKING_EXCLUSION_COLUMNS = [
    "Work Order Number", "Customer Reference", "Building Name",
    "Replacement SA ID", "Scheduled Start", "Excluded Week Commencing", "Reason",
]


def _scheduled_booking_dates(sa_mapping):
    if "Scheduled Start" not in sa_mapping.columns:
        raise ValueError(
            "Booking exclusions require a Scheduled Start column on the old/new "
            "Service Appointment tab. Add that column and upload the refreshed report, "
            "or clear the exclusion-week selection to run without booking exclusions."
        )
    dates = sa_mapping["Scheduled Start"].map(_parse_booking_datetime)
    invalid = sa_mapping["Scheduled Start"].map(_clean_text).ne("") & dates.isna()
    if invalid.any():
        raise ValueError(
            f"Scheduled Start contains {int(invalid.sum())} unreadable date(s). "
            "Correct the dates before using booking exclusions."
        )
    return dates


def available_booking_weeks(file_bytes):
    """Weeks present in the appointment report, for the UI selector only."""
    sa_raw, _ = _detect_retry_tabs(file_bytes)
    dates = _scheduled_booking_dates(_prepare_sa_mapping(sa_raw))
    return sorted({booking_week_start(value) for value in dates.dropna()})


def _existing_booking_exclusions(sa_mapping, latest, all_history, excluded_week_starts):
    exclusions = pd.DataFrame(columns=BOOKING_EXCLUSION_COLUMNS)
    selected_weeks = {booking_week_start(value) for value in (excluded_week_starts or [])}
    if not selected_weeks:
        return exclusions

    dates = _scheduled_booking_dates(sa_mapping)
    if latest.empty:
        return exclusions
    failed_ids = set(all_history["Old Service Appointment ID"].map(_normalise_sa_id))
    # Never mistake a known failed visit for an outstanding replacement, even
    # when its Actual Start is missing. Check all unstarted replacements so an
    # existing booking cannot be bypassed by choosing a different/newer SA.
    candidates = sa_mapping[
        sa_mapping["_actual_start_dt"].isna()
        & ~sa_mapping["Service Appointment ID"].isin(failed_ids)
        & sa_mapping["Work Order Number"].isin(latest["Work Order Number"])
    ].copy()
    candidates["_booking_dt"] = dates.loc[candidates.index]
    rows = []
    by_wo = latest.set_index("Work Order Number")
    for _, appointment in candidates.iterrows():
        scheduled = appointment["_booking_dt"]
        if pd.isna(scheduled):
            continue
        week = booking_week_start(scheduled)
        if week not in selected_weeks:
            continue
        wo = appointment["Work Order Number"]
        event = by_wo.loc[wo]
        rows.append({
            "Work Order Number": wo,
            "Customer Reference": _clean_text(event.get("Customer Reference Code")),
            "Building Name": _clean_text(event.get("Building Name")),
            "Replacement SA ID": appointment["Service Appointment ID"],
            "Scheduled Start": scheduled,
            "Excluded Week Commencing": week.isoformat(),
            "Reason": "Unstarted replacement appointment booked in a selected exclusion week.",
        })
    return pd.DataFrame(rows, columns=BOOKING_EXCLUSION_COLUMNS).drop_duplicates(
        subset=["Work Order Number", "Replacement SA ID", "Scheduled Start"]
    ).reset_index(drop=True)


def _prepare_failure_history(failure_df: pd.DataFrame) -> pd.DataFrame:
    working = failure_df.copy()
    working["Work Order Number"] = working["Work Order Number"].apply(
        _normalise_work_order
    )
    old_sa_col = "Primary Service Appointment: Service Appointment ID"
    working["Old Service Appointment ID"] = working[old_sa_col].apply(
        _normalise_sa_id
    )

    working = working[
        working["Work Order Number"].ne("")
        & working["Old Service Appointment ID"].ne("")
    ].copy()

    customer_flag = pd.to_numeric(
        working.get("Failed Visits (Customer)"), errors="coerce"
    ).fillna(0) > 0
    metro_flag = pd.to_numeric(
        working.get("Failed Visits (Metro)"), errors="coerce"
    ).fillna(0) > 0

    working["Failure Type"] = "Unknown"
    working.loc[customer_flag & ~metro_flag, "Failure Type"] = "Customer"
    working.loc[metro_flag & ~customer_flag, "Failure Type"] = "Metro"
    working.loc[metro_flag & customer_flag, "Failure Type"] = "Ambiguous"

    working["_actual_start_dt"] = working.get(
        "Actual Start", pd.Series(index=working.index, dtype=object)
    ).apply(_parse_datetime)
    working["_scheduled_start_dt"] = working.get(
        "Primary Service Appointment: Scheduled Start",
        pd.Series(index=working.index, dtype=object),
    ).apply(_parse_datetime)
    working["_actual_finish_dt"] = working.get(
        "Actual Finish", pd.Series(index=working.index, dtype=object)
    ).apply(_parse_datetime)
    working["_event_dt"] = working["_actual_start_dt"].combine_first(
        working["_scheduled_start_dt"]
    ).combine_first(working["_actual_finish_dt"])
    working["_row_order"] = range(len(working))

    def failure_reason(row) -> str:
        for col in [
            "Primary Service Appointment: Reason Description",
            "Cancelation Reason Description",
            "Reason Not Complete",
        ]:
            value = _clean_text(row.get(col))
            if value:
                return value
        return ""

    working["Failure Reason"] = working.apply(failure_reason, axis=1)

    # Support both Salesforce shapes we may receive:
    #   1) one failed-visit row per attempt with a value of 1; or
    #   2) a roll-up count on the current row (e.g. value 2 after two failures).
    # Taking the larger of row-count evidence and the largest supplied count
    # preserves the third-attempt rule under either report design.
    customer_numeric = pd.to_numeric(
        working.get("Failed Visits (Customer)"), errors="coerce"
    ).fillna(0)
    metro_numeric = pd.to_numeric(
        working.get("Failed Visits (Metro)"), errors="coerce"
    ).fillna(0)
    working["_customer_numeric"] = customer_numeric
    working["_metro_numeric"] = metro_numeric

    customer_row_counts = (
        working.assign(_customer=(working["Failure Type"] == "Customer").astype(int))
        .groupby("Work Order Number")["_customer"]
        .sum()
        .to_dict()
    )
    metro_row_counts = (
        working.assign(_metro=(working["Failure Type"] == "Metro").astype(int))
        .groupby("Work Order Number")["_metro"]
        .sum()
        .to_dict()
    )
    customer_max_values = working.groupby("Work Order Number")[
        "_customer_numeric"
    ].max().to_dict()
    metro_max_values = working.groupby("Work Order Number")[
        "_metro_numeric"
    ].max().to_dict()

    customer_counts = {
        wo: max(
            int(customer_row_counts.get(wo, 0)),
            int(round(float(customer_max_values.get(wo, 0) or 0))),
        )
        for wo in set(customer_row_counts) | set(customer_max_values)
    }
    metro_counts = {
        wo: max(
            int(metro_row_counts.get(wo, 0)),
            int(round(float(metro_max_values.get(wo, 0) or 0))),
        )
        for wo in set(metro_row_counts) | set(metro_max_values)
    }
    working["Customer Failure Count"] = working["Work Order Number"].map(
        customer_counts
    ).fillna(0).astype(int)
    working["Metro Failure Count"] = working["Work Order Number"].map(
        metro_counts
    ).fillna(0).astype(int)

    # Report is normally sorted newest first. Explicit datetimes take priority;
    # the original row order is the deterministic fallback for missing times.
    working = working.sort_values(
        ["Work Order Number", "_event_dt", "_row_order"],
        ascending=[True, False, True],
        na_position="last",
    )
    latest = working.drop_duplicates("Work Order Number", keep="first").copy()

    latest.attrs["all_history"] = working.copy()
    return latest.reset_index(drop=True)


def _resolve_replacement_sa(
    latest_event: pd.Series,
    all_history: pd.DataFrame,
    sa_mapping: pd.DataFrame,
) -> Dict[str, object]:
    wo = _normalise_work_order(latest_event.get("Work Order Number"))
    old_sa = _normalise_sa_id(latest_event.get("Old Service Appointment ID"))

    wo_history = all_history[
        all_history["Work Order Number"].astype(str) == wo
    ]
    failed_sa_ids = {
        _normalise_sa_id(v)
        for v in wo_history["Old Service Appointment ID"].tolist()
        if _normalise_sa_id(v)
    }

    group = sa_mapping[
        sa_mapping["Work Order Number"].astype(str) == wo
    ].copy()

    if group.empty:
        return {
            "Replacement Service Appointment ID": "",
            "Mapping Status": "Work Order not found on old/new SA tab",
            "Previous Visit": latest_event.get("_event_dt"),
            "Postcode from SA Report": "",
        }

    old_rows = group[group["Service Appointment ID"].astype(str) == old_sa]
    previous_visit = latest_event.get("_event_dt")
    if not old_rows.empty:
        old_actual = old_rows.iloc[0].get("_actual_start_dt")
        if not pd.isna(old_actual):
            previous_visit = old_actual

    candidates = group[
        ~group["Service Appointment ID"].astype(str).isin(failed_sa_ids)
    ].copy()

    # A replacement that already has an Actual Start is not a safe future
    # appointment to schedule automatically.
    candidates_without_actual = candidates[candidates["_actual_start_dt"].isna()].copy()

    replacement = None
    status = ""

    if old_rows.empty:
        status = "Failed SA not found on old/new SA tab"
    elif len(candidates_without_actual) == 1:
        replacement = candidates_without_actual.iloc[0]
        status = "OK"
    elif len(candidates_without_actual) > 1:
        ordered = candidates_without_actual.sort_values(
            "_created_dt", ascending=False, na_position="last"
        )
        if len(ordered) >= 2:
            first_created = ordered.iloc[0].get("_created_dt")
            second_created = ordered.iloc[1].get("_created_dt")
            if (
                not pd.isna(first_created)
                and (pd.isna(second_created) or first_created > second_created)
            ):
                replacement = ordered.iloc[0]
                status = "OK - newest unstarted replacement selected"
            else:
                status = "Multiple possible replacement SAs"
        else:
            status = "Multiple possible replacement SAs"
    elif len(candidates) > 0:
        status = "Replacement SA already has Actual Start"
    else:
        status = "No replacement SA found"

    postcode = ""
    for col in ["Zip/Postal Code", "Postcode", "Postal Code"]:
        if col in group.columns:
            postcode = _clean_text(group.iloc[0].get(col))
            if postcode:
                break

    # Prefer coordinates from the replacement SA row. If that row is blank,
    # fall back to any valid coordinates attached to the same Work Order.
    latitude = None
    longitude = None
    coordinate_rows = []
    if replacement is not None:
        coordinate_rows.append(replacement)
    coordinate_rows.extend(
        group.iloc[idx]
        for idx in range(len(group))
    )

    for coordinate_row in coordinate_rows:
        if latitude is None:
            latitude = _as_optional_float(
                coordinate_row.get("Latitude")
            )
        if longitude is None:
            longitude = _as_optional_float(
                coordinate_row.get("Longitude")
            )
        if latitude is not None and longitude is not None:
            break

    return {
        "Replacement Service Appointment ID": (
            _normalise_sa_id(replacement.get("Service Appointment ID"))
            if replacement is not None
            else ""
        ),
        "Mapping Status": status,
        "Previous Visit": previous_visit,
        "Postcode from SA Report": postcode,
        "Latitude from SA Report": latitude,
        "Longitude from SA Report": longitude,
    }


def _parse_json_output(text: str) -> dict:
    cleaned = _clean_text(text)
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


class CannotCompleteAIPlanner:
    """Narrow LLM layer for interpreting customer-fault access notes only."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing for Cannot Complete triage.")
        if not model:
            raise ValueError("OPENAI_MODEL is missing for Cannot Complete triage.")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def triage(self, cases: Sequence[dict], batch_size: int = 40) -> Dict[str, dict]:
        self.warnings = []
        if not cases:
            return {}

        instructions = """
You triage CUSTOMER-FAULT failed access visits for a UK residential building
survey programme. You are NOT a scheduler and you must not invent access facts.
Python has already applied deterministic business rules such as Metro-fault
retries, blank-description handling, and escalation before the client's third
customer-fault attempt.

For each supplied case choose exactly one decision:
- RETRY: another normal visit is reasonable.
- RETRY_WITH_CONSTRAINT: another visit is reasonable but the note gives a useful
  scheduling preference such as a particular weekday/time.
- CLIENT_ACCESS_REQUIRED: sending another surveyor without client/access action is
  not reasonable.
- IGNORE: the description is too unclear/meaningless to make a safe decision.

Access judgement guidance:
- "no answer", "no residents in", or similar occupancy-only failures are normally
  RETRY. Prefer the opposite time-of-day from the previous attempt where known.
- "key doesn't work and no residents in" can still be RETRY because resident entry
  may work at another occupancy time.
- a key/fob that does not work "inside and out", inability to pass internal secure
  doors, or a failed key combined with no working intercom/no way to call residents
  normally requires CLIENT_ACCESS_REQUIRED.
- explicit resident refusal, construction/building site conditions, a boarded or
  inaccessible site, or a secure facility requiring staff/escort normally requires
  CLIENT_ACCESS_REQUIRED.
- if the note says an appointment/prior notice/client arrangement is required,
  choose CLIENT_ACCESS_REQUIRED and state the action needed before release back to
  scheduling.
- if the note gives a useful preference (e.g. "Mondays are best"), use
  RETRY_WITH_CONSTRAINT and extract it. A preference is not a hard constraint unless
  the note explicitly says access is only possible then; downstream Python still
  protects route efficiency.
- numeric codes or unintelligible text with no interpretable access meaning should
  be IGNORE rather than guessed.

preferred_period must be one of MORNING, AFTERNOON, OPPOSITE_PREVIOUS, NONE.
preferred_weekdays must contain only Monday..Sunday and should be empty when none
is stated.
confidence must be HIGH, MEDIUM or LOW.
Keep reasoning and recommended_client_action brief.

Return JSON only:
{
  "decisions": [
    {
      "work_order": "01000000",
      "decision": "RETRY|RETRY_WITH_CONSTRAINT|CLIENT_ACCESS_REQUIRED|IGNORE",
      "reason_category": "short machine-readable category",
      "reasoning": "brief explanation",
      "preferred_period": "MORNING|AFTERNOON|OPPOSITE_PREVIOUS|NONE",
      "preferred_weekdays": ["Monday"],
      "recommended_client_action": "brief action or empty",
      "confidence": "HIGH|MEDIUM|LOW"
    }
  ]
}
"""

        results: Dict[str, dict] = {}
        cases = list(cases)

        for start in range(0, len(cases), max(1, int(batch_size))):
            batch = cases[start:start + max(1, int(batch_size))]
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=json.dumps({"cases": batch}, ensure_ascii=False, default=str),
                )
                data = _parse_json_output(response.output_text)
                if not isinstance(data, dict) or not isinstance(data.get("decisions"), list):
                    raise ValueError("AI triage did not return a decisions list.")
            except Exception as exc:
                self.warnings.append(
                    "Customer-fault AI access triage stopped. "
                    f"Kept {len(results)} completed decision(s); cases without "
                    f"a valid decision remain held out. Error: {exc}"
                )
                break

            for item in data.get("decisions", []):
                if not isinstance(item, dict):
                    continue
                wo = _normalise_work_order(item.get("work_order"))
                if not wo or wo not in {
                    _normalise_work_order(case.get("work_order")) for case in batch
                }:
                    continue
                decision = _clean_text(item.get("decision")).upper()
                if decision not in RETRY_DECISIONS:
                    decision = "IGNORE"

                preferred_period = _clean_text(item.get("preferred_period")).upper()
                if preferred_period not in {
                    "MORNING", "AFTERNOON", "OPPOSITE_PREVIOUS", "NONE"
                }:
                    preferred_period = "NONE"

                weekdays = item.get("preferred_weekdays") or []
                if not isinstance(weekdays, list):
                    weekdays = []
                valid_days = {
                    "Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday",
                }
                weekdays = [
                    str(v).strip().title()
                    for v in weekdays
                    if str(v).strip().title() in valid_days
                ]

                confidence = _clean_text(item.get("confidence")).upper()
                if confidence not in {"HIGH", "MEDIUM", "LOW"}:
                    confidence = "LOW"

                results[wo] = {
                    "Decision": decision,
                    "Reason Category": _clean_text(item.get("reason_category")),
                    "Decision Reason": _clean_text(item.get("reasoning")),
                    "AI Preferred Period": preferred_period,
                    "Preferred Weekdays": weekdays,
                    "Recommended Client Action": _clean_text(
                        item.get("recommended_client_action")
                    ),
                    "AI Confidence": confidence,
                    "Decision Source": "AI customer access triage",
                }

        return results


def build_retry_plan(
    file_bytes: bytes,
    openai_api_key: str = "",
    openai_model: str = "gpt-5.6",
    excluded_week_starts: Optional[Sequence] = None,
) -> RetryPlan:
    sa_raw, failures_raw = _detect_retry_tabs(file_bytes)
    sa_mapping = _prepare_sa_mapping(sa_raw)
    latest = _prepare_failure_history(failures_raw)
    all_history = latest.attrs.get("all_history", pd.DataFrame()).copy()
    booking_exclusions = _existing_booking_exclusions(
        sa_mapping, latest, all_history, excluded_week_starts,
    )
    excluded_work_orders = set(booking_exclusions["Work Order Number"])

    rows: List[dict] = []
    ai_cases: List[dict] = []
    warnings: List[str] = []

    for _, event in latest.iterrows():
        # Filter before business/AI triage. These work orders are accounted for
        # separately in the booking audit, never sent back to the retry pool.
        if event["Work Order Number"] in excluded_work_orders:
            continue
        mapping = _resolve_replacement_sa(
            event,
            all_history=all_history,
            sa_mapping=sa_mapping,
        )
        previous_visit = mapping.get("Previous Visit")
        previous_period = _period_from_datetime(previous_visit)
        previous_weekday = _weekday_name(previous_visit)
        previous_weekday_number = _weekday_number(previous_visit)

        base = {
            "Work Order Number": _normalise_work_order(
                event.get("Work Order Number")
            ),
            "Customer Reference": _clean_text(
                event.get("Customer Reference Code")
            ),
            "Building Name": _clean_text(event.get("Building Name")),
            "Postcode": _clean_text(mapping.get("Postcode from SA Report")),
            "Source Latitude": mapping.get("Latitude from SA Report"),
            "Source Longitude": mapping.get("Longitude from SA Report"),
            "Old Service Appointment ID": _normalise_sa_id(
                event.get("Old Service Appointment ID")
            ),
            "Replacement Service Appointment ID": _clean_text(
                mapping.get("Replacement Service Appointment ID")
            ),
            "Mapping Status": _clean_text(mapping.get("Mapping Status")),
            "Failure Type": _clean_text(event.get("Failure Type")),
            "Failure Reason": _clean_text(event.get("Failure Reason")),
            "Customer Failure Count": _as_int(
                event.get("Customer Failure Count")
            ),
            "Metro Failure Count": _as_int(event.get("Metro Failure Count")),
            "Previous Visit": previous_visit,
            "Previous Weekday": previous_weekday,
            "Previous Weekday Number": previous_weekday_number,
            "Previous Period": previous_period,
            "Decision": "",
            "Decision Source": "",
            "Reason Category": "",
            "Decision Reason": "",
            "Preferred Retry Period": "",
            "Preferred Weekdays": [],
            "Forbidden Weekday": "",
            "Forbidden Weekday Number": None,
            "Recommended Client Action": "",
            "AI Confidence": "",
            "Client Action Required": False,
            "Retry Eligible": False,
            # Source fields used only if this retry is missing from the normal
            # future-surveys portfolio and must be reconstructed.
            "Source Status": _clean_text(event.get("Status")),
            "Source Work Type Name": _clean_text(event.get("Work Type Name")),
            "Source Building Height": event.get("Building Height"),
            "Source Sovereign Flat": event.get("Sovereign Flat"),
            "Source Planned Start": event.get("Planned Start"),
            "Source Estimated Site Time": event.get("Estimated Site Time (Mins)"),
        }

        failure_type = base["Failure Type"]
        reason = base["Failure Reason"]
        customer_count = int(base["Customer Failure Count"])

        if failure_type == "Metro":
            base.update({
                "Decision": "RETRY",
                "Decision Source": "Python Metro-fault rule",
                "Reason Category": "METRO_FAULT",
                "Decision Reason": (
                    "Metro-fault Cannot Complete: automatically returned to the "
                    "future scheduling pool."
                ),
                "Preferred Retry Period": _opposite_period(previous_period),
                "Forbidden Weekday": previous_weekday,
                "Forbidden Weekday Number": previous_weekday_number,
                "Client Action Required": False,
            })

        elif failure_type == "Customer":
            if customer_count >= CUSTOMER_CLIENT_HELP_ATTEMPT - 1:
                base.update({
                    "Decision": "CLIENT_ACCESS_REQUIRED",
                    "Decision Source": "Python third-attempt rule",
                    "Reason Category": "THIRD_ATTEMPT_CLIENT_SUPPORT",
                    "Decision Reason": (
                        f"{customer_count} customer-fault failures are already "
                        "recorded; client support is required before the third "
                        "attempt is released back to scheduling."
                    ),
                    "Recommended Client Action": (
                        "Client to arrange/support access before another visit."
                    ),
                    "Client Action Required": True,
                })
            elif not reason:
                base.update({
                    "Decision": BLANK_CUSTOMER_FAILURE_ACTION,
                    "Decision Source": "Python blank-description rule",
                    "Reason Category": "BLANK_DESCRIPTION",
                    "Decision Reason": (
                        "Blank customer-fault description ignored under the current "
                        "configurable rule."
                    ),
                })
            else:
                ai_cases.append({
                    "work_order": base["Work Order Number"],
                    "building": base["Building Name"],
                    "failure_reason": reason,
                    "customer_failure_count": customer_count,
                    "previous_visit": str(previous_visit or ""),
                    "previous_weekday": previous_weekday,
                    "previous_period": previous_period,
                })
        else:
            base.update({
                "Decision": "IGNORE",
                "Decision Source": "Python ambiguous-failure rule",
                "Reason Category": "AMBIGUOUS_FAILURE_TYPE",
                "Decision Reason": (
                    "Could not unambiguously classify the latest failure as Metro "
                    "or Customer, so it was held out of automatic retry scheduling."
                ),
            })

        rows.append(base)

    ai_results: Dict[str, dict] = {}
    if ai_cases:
        if not _clean_text(openai_api_key):
            warnings.append(
                f"{len(ai_cases)} customer-fault cases required AI access triage, "
                "but OPENAI_API_KEY was not available. They were ignored rather "
                "than scheduled automatically."
            )
        else:
            try:
                triage_planner = CannotCompleteAIPlanner(
                    openai_api_key,
                    openai_model,
                )
                ai_results = triage_planner.triage(ai_cases)
                warnings.extend(triage_planner.warnings)
            except Exception as exc:
                warnings.append(
                    "Customer-fault AI access triage failed. Affected cases were "
                    f"ignored rather than guessed. Error: {exc}"
                )

    for row in rows:
        if row["Failure Type"] != "Customer" or row["Decision"]:
            continue

        wo = row["Work Order Number"]
        ai = ai_results.get(wo)
        if ai is None:
            row.update({
                "Decision": "IGNORE",
                "Decision Source": "AI unavailable / no decision",
                "Reason Category": "AI_TRIAGE_UNAVAILABLE",
                "Decision Reason": (
                    "Customer-fault reason required AI interpretation, but no valid "
                    "AI decision was available; held out of automatic scheduling."
                ),
            })
            continue

        row.update(ai)
        ai_period = _clean_text(ai.get("AI Preferred Period")).upper()
        if ai_period == "OPPOSITE_PREVIOUS":
            row["Preferred Retry Period"] = _opposite_period(row["Previous Period"])
        elif ai_period == "MORNING":
            row["Preferred Retry Period"] = "Morning"
        elif ai_period == "AFTERNOON":
            row["Preferred Retry Period"] = "Afternoon"
        else:
            row["Preferred Retry Period"] = ""
        row["Client Action Required"] = (
            row["Decision"] == "CLIENT_ACCESS_REQUIRED"
        )

    decisions = pd.DataFrame(rows)

    if decisions.empty:
        return RetryPlan(
            decisions=decisions,
            client_access_required=decisions.copy(),
            warnings=warnings,
            stats={"work_orders": int(len(latest)), "bookings_excluded": len(excluded_work_orders)},
            booking_exclusions=booking_exclusions,
        )

    mapping_ok = decisions["Mapping Status"].astype(str).str.startswith("OK")
    retry_decision = decisions["Decision"].isin(["RETRY", "RETRY_WITH_CONSTRAINT"])
    decisions["Retry Eligible"] = mapping_ok & retry_decision

    # A business retry decision without a usable replacement SA is visible in
    # the audit output but is never injected into the scheduler.
    mapping_blocked = retry_decision & ~mapping_ok
    if mapping_blocked.any():
        warnings.append(
            f"{int(mapping_blocked.sum())} retryable Work Order(s) could not be "
            "scheduled because the replacement Service Appointment mapping was "
            "not unambiguous."
        )

    client_access = decisions[
        decisions["Decision"].eq("CLIENT_ACCESS_REQUIRED")
    ].copy()
    if not client_access.empty and "Preferred Weekdays" in client_access.columns:
        client_access["Preferred Weekdays"] = client_access[
            "Preferred Weekdays"
        ].apply(
            lambda value: ", ".join(value)
            if isinstance(value, list)
            else _clean_text(value)
        )

    stats = {
        "work_orders": int(len(latest)),
        "bookings_excluded": len(excluded_work_orders),
        "retry_eligible": int(decisions["Retry Eligible"].sum()),
        "metro_retries": int(
            ((decisions["Failure Type"] == "Metro") & decisions["Retry Eligible"]).sum()
        ),
        "customer_retries": int(
            ((decisions["Failure Type"] == "Customer") & decisions["Retry Eligible"]).sum()
        ),
        "client_access_required": int(len(client_access)),
        "ignored": int(decisions["Decision"].eq("IGNORE").sum()),
        "mapping_blocked": int(mapping_blocked.sum()),
    }

    return RetryPlan(
        decisions=decisions,
        client_access_required=client_access,
        warnings=warnings,
        stats=stats,
        booking_exclusions=booking_exclusions,
    )


def _portfolio_work_order_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["Work Order Number", "Work Order"]:
        if candidate in df.columns:
            return candidate
    return None


def _candidate_row_from_retry(decision: pd.Series) -> dict:
    return {
        "Customer Reference": decision.get("Customer Reference", ""),
        "Building Name": decision.get("Building Name", ""),
        "Postcode": decision.get("Postcode", ""),
        "Work Order Number": decision.get("Work Order Number", ""),
        "Work Type Name": decision.get("Source Work Type Name", "Geospatial Asset Mapping"),
        "Status": decision.get("Source Status", "Released"),
        # Do not carry the failed appointment's old planned date into a newly
        # reconstructed retry row. The selected-week scheduler decides the new
        # timing from scratch.
        "Planned Start": None,
        "Building Height": decision.get("Source Building Height"),
        "Sovereign Flat": decision.get("Source Sovereign Flat"),
        "Latitude": decision.get("Source Latitude"),
        "Longitude": decision.get("Source Longitude"),
        "Service Appointment ID": decision.get(
            "Replacement Service Appointment ID", ""
        ),
        "Primary Service Appointment: Service Appointment ID": decision.get(
            "Replacement Service Appointment ID", ""
        ),
    }


def apply_retry_plan_to_portfolio(
    portfolio: pd.DataFrame,
    decisions: pd.DataFrame,
    booking_exclusions: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply the retry gate BEFORE duration prediction / geographic clustering.

    - Customer failures held for client action or ignored are removed from the
      schedulable master portfolio when their Work Order is present.
    - Retryable rows are annotated and their replacement SA ID is enforced.
    - If a retryable Work Order is missing from the master portfolio, a minimal
      row is reconstructed from the Cannot Complete workbook so it can still be
      predicted/clusted (postcode clustering is used if coordinates are absent).
    """
    result = portfolio.copy()
    removed_existing_bookings = 0
    if booking_exclusions is not None and not booking_exclusions.empty:
        excluded_wos = set(booking_exclusions["Work Order Number"].map(_normalise_work_order)) - {""}
        wo_col = _portfolio_work_order_column(result)
        if not result.empty and wo_col is None:
            raise ValueError("The master portfolio needs Work Order Number to apply booking exclusions.")
        if wo_col is not None:
            remove = result[wo_col].map(_normalise_work_order).isin(excluded_wos)
            removed_existing_bookings = int(remove.sum())
            result = result.loc[~remove].copy()
        if decisions is not None and not decisions.empty:
            decisions = decisions.loc[
                ~decisions["Work Order Number"].map(_normalise_work_order).isin(excluded_wos)
            ].copy()
    if decisions is None or decisions.empty:
        return result, {
            "removed_non_retry": 0,
            "removed_existing_bookings": removed_existing_bookings,
            "annotated_existing": 0,
            "appended_missing": 0,
        }

    wo_col = _portfolio_work_order_column(result)
    if wo_col is None:
        result["Work Order Number"] = ""
        wo_col = "Work Order Number"

    # Salesforce identifiers are identifiers, not numeric measures. Excel/pandas
    # can infer an all-numeric/blank Work Order or Service Appointment column as
    # float64; retry injection then needs to write values such as "01023173" or
    # "08pR..." into that column. Keep only these identifier fields object-typed
    # so leading zeroes and Salesforce IDs are preserved safely.
    identifier_columns = {
        wo_col,
        "Work Order Number",
        "Work Order",
        "Service Appointment ID",
        "Primary Service Appointment: Service Appointment ID",
    }
    for identifier_col in identifier_columns:
        if identifier_col in result.columns:
            result[identifier_col] = result[identifier_col].astype(object)

    result["_retry_wo"] = result[wo_col].apply(_normalise_work_order)

    decision_by_wo = {
        str(row["Work Order Number"]): row
        for _, row in decisions.iterrows()
        if _normalise_work_order(row.get("Work Order Number"))
    }

    non_retry_wos = {
        wo
        for wo, row in decision_by_wo.items()
        if not bool(row.get("Retry Eligible", False))
    }
    before = len(result)
    result = result[~result["_retry_wo"].isin(non_retry_wos)].copy()
    removed_non_retry = before - len(result)

    annotation_columns = [
        "Is Retry",
        "Retry Failure Type",
        "Retry Failure Reason",
        "Retry Decision",
        "Retry Decision Reason",
        "Retry Reason Category",
        "Retry Previous Visit",
        "Retry Previous Weekday",
        "Retry Previous Period",
        "Retry Forbidden Weekday",
        "Retry Forbidden Weekday Number",
        "Retry Preferred Period",
        "Retry Preferred Weekdays",
        "Retry Old Service Appointment ID",
        "Retry Replacement Service Appointment ID",
        "Retry AI Confidence",
    ]

    # Retry metadata is intentionally mixed-type: booleans, datetimes, weekday
    # numbers and text all live in this annotation layer. Pandas 3 can infer a
    # newly-created "" column as strict StringDtype, which then rejects values
    # such as True or a Timestamp. Keep the metadata columns object-typed so the
    # agreed retry values can be written without coercing business data.
    for col in annotation_columns:
        if col not in result.columns:
            default_value = False if col == "Is Retry" else None
            result[col] = pd.Series(
                [default_value] * len(result),
                index=result.index,
                dtype=object,
            )
        else:
            result[col] = result[col].astype(object)

    annotated_existing = 0
    appended_missing = 0

    retry_rows = decisions[decisions["Retry Eligible"] == True].copy()

    for _, decision in retry_rows.iterrows():
        wo = _normalise_work_order(decision.get("Work Order Number"))
        matches = result.index[result["_retry_wo"] == wo].tolist()

        if matches:
            # Keep one active row for this retry Work Order. Prefer an existing row
            # already carrying the replacement SA ID when possible.
            replacement = _clean_text(
                decision.get("Replacement Service Appointment ID")
            )
            preferred_index = None
            for idx in matches:
                existing_ids = [
                    _clean_text(result.at[idx, c])
                    for c in [
                        "Service Appointment ID",
                        "Primary Service Appointment: Service Appointment ID",
                    ]
                    if c in result.columns
                ]
                if replacement and replacement in existing_ids:
                    preferred_index = idx
                    break
            if preferred_index is None:
                preferred_index = matches[0]

            duplicate_indices = [i for i in matches if i != preferred_index]
            if duplicate_indices:
                result = result.drop(index=duplicate_indices)
            row_index = preferred_index
            annotated_existing += 1
        else:
            candidate = _candidate_row_from_retry(decision)
            candidate["_retry_wo"] = wo
            new_index = (
                int(result.index.max()) + 1
                if len(result.index) and isinstance(result.index.max(), (int, float))
                else len(result)
            )

            # Concatenating a one-row frame lets pandas widen only the affected
            # identifier columns to object where necessary. Direct
            # result.loc[new_index] = candidate can fail when the Salesforce
            # export inferred Work Order Number as float64.
            candidate_frame = pd.DataFrame(
                [candidate],
                index=[new_index],
            )
            result = pd.concat(
                [result, candidate_frame],
                axis=0,
                sort=False,
            )
            row_index = new_index
            appended_missing += 1

        replacement = _clean_text(
            decision.get("Replacement Service Appointment ID")
        )
        result.at[row_index, "Work Order Number"] = wo
        result.at[row_index, "Service Appointment ID"] = replacement

        source_latitude = _as_optional_float(
            decision.get("Source Latitude")
        )
        source_longitude = _as_optional_float(
            decision.get("Source Longitude")
        )

        # Preserve any richer Future Surveys coordinates already present.
        # Only fill a coordinate field when the existing row is blank/missing.
        if source_latitude is not None:
            if "Latitude" not in result.columns:
                result["Latitude"] = pd.Series(
                    [None] * len(result),
                    index=result.index,
                    dtype=object,
                )
            existing_latitude = _as_optional_float(
                result.at[row_index, "Latitude"]
            )
            if existing_latitude is None:
                result.at[row_index, "Latitude"] = source_latitude

        if source_longitude is not None:
            if "Longitude" not in result.columns:
                result["Longitude"] = pd.Series(
                    [None] * len(result),
                    index=result.index,
                    dtype=object,
                )
            existing_longitude = _as_optional_float(
                result.at[row_index, "Longitude"]
            )
            if existing_longitude is None:
                result.at[row_index, "Longitude"] = source_longitude
        result.at[
            row_index,
            "Primary Service Appointment: Service Appointment ID",
        ] = replacement

        values = {
            "Is Retry": True,
            "Retry Failure Type": decision.get("Failure Type", ""),
            "Retry Failure Reason": decision.get("Failure Reason", ""),
            "Retry Decision": decision.get("Decision", ""),
            "Retry Decision Reason": decision.get("Decision Reason", ""),
            "Retry Reason Category": decision.get("Reason Category", ""),
            "Retry Previous Visit": decision.get("Previous Visit"),
            "Retry Previous Weekday": decision.get("Previous Weekday", ""),
            "Retry Previous Period": decision.get("Previous Period", ""),
            "Retry Forbidden Weekday": decision.get("Forbidden Weekday", ""),
            "Retry Forbidden Weekday Number": decision.get(
                "Forbidden Weekday Number"
            ),
            "Retry Preferred Period": decision.get("Preferred Retry Period", ""),
            "Retry Preferred Weekdays": ", ".join(
                decision.get("Preferred Weekdays", [])
                if isinstance(decision.get("Preferred Weekdays", []), list)
                else []
            ),
            "Retry Old Service Appointment ID": decision.get(
                "Old Service Appointment ID", ""
            ),
            "Retry Replacement Service Appointment ID": replacement,
            "Retry AI Confidence": decision.get("AI Confidence", ""),
        }
        for col, value in values.items():
            result.at[row_index, col] = value

    result = result.drop(columns=["_retry_wo"], errors="ignore")
    return result.reset_index(drop=True), {
        "removed_non_retry": int(removed_non_retry),
        "removed_existing_bookings": removed_existing_bookings,
        "annotated_existing": int(annotated_existing),
        "appended_missing": int(appended_missing),
    }


def _schedule_lookup(schedule_df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    by_ref: Dict[str, pd.Series] = {}
    by_building: Dict[str, pd.Series] = {}
    if schedule_df is None or schedule_df.empty:
        return by_ref, by_building

    working = schedule_df.copy()
    if "Sequence" in working.columns:
        numeric = pd.to_numeric(working["Sequence"], errors="coerce")
        working = working[numeric.notna()].copy()

    for _, row in working.iterrows():
        ref = _clean_text(row.get("Customer Reference"))
        building = _clean_text(row.get("Building Name"))
        if ref and ref not in by_ref:
            by_ref[ref] = row
        if building and building not in by_building:
            by_building[building] = row
    return by_ref, by_building


def build_retry_audit(
    decisions: pd.DataFrame,
    combined_schedule: pd.DataFrame,
) -> pd.DataFrame:
    if decisions is None or decisions.empty:
        return pd.DataFrame()

    by_ref, by_building = _schedule_lookup(combined_schedule)
    rows = []

    for _, decision in decisions.iterrows():
        ref = _clean_text(decision.get("Customer Reference"))
        building = _clean_text(decision.get("Building Name"))
        scheduled = by_ref.get(ref) if ref else None
        if scheduled is None and building:
            scheduled = by_building.get(building)

        scheduled_date = ""
        scheduled_time = ""
        scheduled_period = ""
        surveyor = ""
        if scheduled is not None:
            scheduled_date = _clean_text(scheduled.get("Date"))
            scheduled_time = _clean_text(scheduled.get("Survey Start"))
            surveyor = _clean_text(scheduled.get("Surveyor"))
            if scheduled_time:
                try:
                    hour = int(scheduled_time.split(":", 1)[0])
                    scheduled_period = (
                        "Morning" if hour < TIME_PERIOD_SPLIT_HOUR else "Afternoon"
                    )
                except Exception:
                    scheduled_period = ""

        preferred_period = _clean_text(decision.get("Preferred Retry Period"))
        if not preferred_period:
            preference_honoured = "N/A"
            override_reason = "No morning/afternoon preference applied."
        elif scheduled is None:
            preference_honoured = "Not scheduled"
            override_reason = "Retry was not placed in the final weekly schedule."
        elif scheduled_period.lower() == preferred_period.lower():
            preference_honoured = "Yes"
            override_reason = "Preferred retry period was honoured."
        else:
            preference_honoured = "No"
            override_reason = (
                "Soft retry-period preference was overridden by the normal route/"
                "time optimiser; hard feasibility rules remained in force."
            )

        rows.append({
            "Work Order Number": decision.get("Work Order Number", ""),
            "Customer Reference": ref,
            "Building Name": building,
            "Old / Failed SA ID": decision.get("Old Service Appointment ID", ""),
            "Replacement SA ID": decision.get(
                "Replacement Service Appointment ID", ""
            ),
            "SA Mapping Status": decision.get("Mapping Status", ""),
            "Failure Type": decision.get("Failure Type", ""),
            "Failure Reason": decision.get("Failure Reason", ""),
            "Customer Failure Count": decision.get("Customer Failure Count", 0),
            "Metro Failure Count": decision.get("Metro Failure Count", 0),
            "Previous Visit": decision.get("Previous Visit"),
            "Previous Weekday": decision.get("Previous Weekday", ""),
            "Previous Period": decision.get("Previous Period", ""),
            "Decision": decision.get("Decision", ""),
            "Decision Source": decision.get("Decision Source", ""),
            "Reason Category": decision.get("Reason Category", ""),
            "Decision Reason": decision.get("Decision Reason", ""),
            "Preferred Retry Period": preferred_period,
            "Preferred Weekdays": ", ".join(
                decision.get("Preferred Weekdays", [])
                if isinstance(decision.get("Preferred Weekdays", []), list)
                else []
            ),
            "Forbidden Weekday": decision.get("Forbidden Weekday", ""),
            "Retry Eligible": bool(decision.get("Retry Eligible", False)),
            "Client Action Required": bool(
                decision.get("Client Action Required", False)
            ),
            "Recommended Client Action": decision.get(
                "Recommended Client Action", ""
            ),
            "AI Confidence": decision.get("AI Confidence", ""),
            "Scheduled?": "Yes" if scheduled is not None else "No",
            "Scheduled Surveyor": surveyor,
            "Actual Scheduled Date": scheduled_date,
            "Actual Scheduled Time": scheduled_time,
            "Actual Scheduled Period": scheduled_period,
            "Period Preference Honoured?": preference_honoured,
            "Scheduling / Override Reason": override_reason,
        })

    return pd.DataFrame(rows)


def sense_check_retry_outputs(
    decisions: pd.DataFrame,
    audit: pd.DataFrame,
    salesforce_copy: pd.DataFrame,
) -> pd.DataFrame:
    """Return explicit PASS/WARN/FAIL checks for the human audit workbook."""
    checks = []

    def add(check, status, details):
        checks.append({"Check": check, "Status": status, "Details": details})

    if decisions is None or decisions.empty:
        add("Retry input", "PASS", "No Cannot Complete retry workbook rows to assess.")
        return pd.DataFrame(checks)

    duplicate_wos = decisions["Work Order Number"].astype(str).duplicated().sum()
    add(
        "One retry decision per Work Order",
        "PASS" if duplicate_wos == 0 else "FAIL",
        f"Duplicate Work Order decisions: {int(duplicate_wos)}",
    )

    bad_mapping = (
        decisions["Retry Eligible"].astype(bool)
        & ~decisions["Mapping Status"].astype(str).str.startswith("OK")
    ).sum()
    add(
        "Retryable rows have replacement SA",
        "PASS" if bad_mapping == 0 else "FAIL",
        f"Retryable rows without safe replacement mapping: {int(bad_mapping)}",
    )

    scheduled_non_retry = 0
    metro_same_weekday = 0
    if audit is not None and not audit.empty:
        scheduled_non_retry = int((
            audit["Scheduled?"].eq("Yes")
            & ~audit["Decision"].isin(["RETRY", "RETRY_WITH_CONSTRAINT"])
        ).sum())

        metro_rows = audit[
            audit["Failure Type"].eq("Metro")
            & audit["Scheduled?"].eq("Yes")
        ]
        for _, row in metro_rows.iterrows():
            scheduled_date = pd.to_datetime(
                row.get("Actual Scheduled Date"), errors="coerce"
            )
            forbidden = _clean_text(row.get("Forbidden Weekday"))
            if not pd.isna(scheduled_date) and forbidden:
                if scheduled_date.day_name() == forbidden:
                    metro_same_weekday += 1

    add(
        "Held/client-access rows not scheduled",
        "PASS" if scheduled_non_retry == 0 else "FAIL",
        f"Non-retry decisions found in final schedule: {scheduled_non_retry}",
    )
    add(
        "Metro retry different-weekday hard rule",
        "PASS" if metro_same_weekday == 0 else "FAIL",
        f"Metro retries scheduled on the failed weekday: {metro_same_weekday}",
    )

    # Check that the Salesforce copy uses the replacement 08p ID for every
    # retry that actually made the weekly schedule.
    sf_mismatch = 0
    if (
        salesforce_copy is not None
        and not salesforce_copy.empty
        and audit is not None
        and not audit.empty
    ):
        sf_by_wo = {
            _normalise_work_order(row.get("Work Order Number")): _clean_text(
                row.get("Service Appointment ID")
            )
            for _, row in salesforce_copy.iterrows()
        }
        scheduled_retry = audit[
            audit["Scheduled?"].eq("Yes")
            & audit["Decision"].isin(["RETRY", "RETRY_WITH_CONSTRAINT"])
        ]
        for _, row in scheduled_retry.iterrows():
            wo = _normalise_work_order(row.get("Work Order Number"))
            expected = _clean_text(row.get("Replacement SA ID"))
            actual = sf_by_wo.get(wo, "")
            if expected and actual != expected:
                sf_mismatch += 1

    add(
        "Salesforce Copy uses replacement SA IDs",
        "PASS" if sf_mismatch == 0 else "FAIL",
        f"Scheduled retries with mismatched replacement SA ID: {sf_mismatch}",
    )

    period_overrides = 0
    unscheduled_retries = 0
    if audit is not None and not audit.empty:
        period_overrides = int(
            audit["Period Preference Honoured?"].eq("No").sum()
        )
        unscheduled_retries = int((
            audit["Retry Eligible"].astype(bool)
            & audit["Scheduled?"].eq("No")
        ).sum())

    add(
        "Soft period preferences",
        "WARN" if period_overrides else "PASS",
        f"Retry period preferences overridden for route efficiency: {period_overrides}",
    )
    add(
        "Eligible retries scheduled this week",
        "WARN" if unscheduled_retries else "PASS",
        (
            f"Eligible retries not selected/placed this week: {unscheduled_retries}. "
            "They remain valid future work rather than being forced into an inefficient week."
        ),
    )

    return pd.DataFrame(checks)
