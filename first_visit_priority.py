"""Building-level visit evidence and a transparent first-visit coverage register."""
from __future__ import annotations

import re
import pandas as pd

from retry_planner import _clean_text, _normalise_work_order


def _truth(value):
    return _clean_text(value).casefold() in {"true", "1", "1.0"}


def _reference(row):
    for column in ("Customer Reference", "Customer Reference Code", "Customer Reference Code  ↑"):
        value = _clean_text(row.get(column))
        if value:
            return re.sub(r"\.0$", "", value) if re.fullmatch(r"\d+\.0", value) else value.upper()
    return ""


def _work_order(row):
    return _normalise_work_order(row.get("Work Order Number", row.get("Work Order")))


def annotate_first_visits(portfolio, retry_plan=None, completed_history=None):
    """No recorded visit is assessed against the supplied history, not a new SA.

    References join evidence across Work Orders. Work Orders bridge appointment
    rows that omit the reference. Missing history or identity stays unverified.
    A scheduled/cancelled appointment without an actual visit is not evidence.
    This annotation changes priority only; the existing eligibility gates remain.
    """
    result = portfolio.copy()
    failures = getattr(retry_plan, "visit_history", pd.DataFrame())
    appointments = getattr(retry_plan, "appointment_history", pd.DataFrame())
    completed = completed_history if completed_history is not None else pd.DataFrame()
    decisions = getattr(retry_plan, "report_decisions", pd.DataFrame())
    if decisions.empty and retry_plan is not None:
        decisions = retry_plan.decisions
    frames = [result, failures, appointments, decisions, completed]
    refs_by_wo = {}
    for frame in frames:
        for _, row in frame.iterrows():
            ref, wo = _reference(row), _work_order(row)
            if ref and wo:
                refs_by_wo.setdefault(wo, set()).add(ref)

    evidence_by_ref, evidence_by_wo = {}, {}

    def record(row, reason):
        wo, ref = _work_order(row), _reference(row)
        if wo:
            evidence_by_wo.setdefault(wo, set()).add(reason)
        for reference in ({ref} if ref else set()) | refs_by_wo.get(wo, set()):
            evidence_by_ref.setdefault(reference, set()).add(reason)

    for _, row in failures.iterrows():
        record(row, "Recorded cannot-complete visit")
    for _, row in appointments.iterrows():
        status = _clean_text(row.get("_sa_status")).casefold()
        if status in {"completed", "cannot complete"}:
            record(row, f"Service Appointment status: {status}")
        elif pd.notna(row.get("_actual_start_dt", pd.NaT)):
            record(row, "Service Appointment has an Actual Start")
    for _, row in completed.iterrows():
        record(row, "Completed-survey training history")
    for _, row in decisions.iterrows():
        if _clean_text(row.get("Decision")) == "RESOLVED":
            record(row, "Completed Service Appointment")
        elif _clean_text(row.get("Old Service Appointment ID")):
            record(row, "Recorded failed Service Appointment")
    # Keep explicit evidence in the master too, including when no retry file is
    # supplied. Never infer a visit just from Work Order Status or Scheduled Start.
    for _, row in result.iterrows():
        if _truth(row.get("Is Retry")):
            record(row, "Approved retry in the scheduling portfolio")
        for column in ("Failed Visits (Customer)", "Failed Visits (Metro)",
                       "Customer Failure Count", "Metro Failure Count"):
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(value) and value > 0:
                record(row, "Recorded failure count in the master portfolio")
        for column in ("Actual Start", "Actual Finish", "Retry Previous Visit"):
            if pd.notna(pd.to_datetime(row.get(column), errors="coerce")):
                record(row, f"Master portfolio: {column}")
        for column in ("Service Appointment Status", "Service Appointment: Status"):
            status = _clean_text(row.get(column)).casefold()
            if status in {"completed", "cannot complete"}:
                record(row, f"Service Appointment status: {status}")

    annotations = []
    for _, row in result.iterrows():
        ref, wo = _reference(row), _work_order(row)
        reasons = evidence_by_ref.get(ref, set()) | evidence_by_wo.get(wo, set())
        if reasons:
            classification, never, reason = "Previously visited", False, "; ".join(sorted(reasons))
        elif not ref:
            classification, never, reason = "Unverified", False, "Missing building reference; cannot verify building-level visit history."
        elif retry_plan is None:
            classification, never, reason = "Unverified", False, "No cannot-complete/appointment history supplied; first-visit status is unverified."
        else:
            classification, never, reason = "No recorded visit", True, "No completed, failed or started visit matched in the supplied history."
        annotations.append((never, classification, reason))
    for i, column in enumerate(("Never Visited", "Visit Classification", "Visit Evidence")):
        result[column] = [values[i] for values in annotations]
    return result


def retains_first_visits(baseline, trial, first_visit_references):
    """Weekly-note trials may move first visits, but cannot discard them."""
    def refs(result):
        return {str(item.customer_reference).strip().upper()
                for day in result.days for item in day.items} if result else set()
    return (refs(baseline) & set(first_visit_references)) <= refs(trial)


def build_first_visit_review(source, portfolio, team_results, surveyors,
                            retry_plan=None, routing_attempts=None):
    """Report exclusions and observed route failures without claiming optimality."""
    from scheduler_v20_10 import _site_allowed_today, _site_allowed_for_surveyor

    by_ref = {_reference(row): row for _, row in portfolio.iterrows() if _reference(row)}
    bookings = getattr(retry_plan, "booking_exclusions", pd.DataFrame())
    booked_wos = {_work_order(row) for _, row in bookings.iterrows()}
    booked_refs = {_reference(row) for _, row in bookings.iterrows()} - {""}
    placements = {}
    for name, result in team_results.items():
        if result:
            for day in result.days:
                for item in day.items:
                    placements.setdefault(str(item.customer_reference).strip().upper(), []).append(
                        (name, (day.first_survey_target or day.start_time).date().isoformat()))
    attempts = {}
    for attempt in routing_attempts or []:
        ref = _clean_text(attempt.get("Customer Reference")).upper()
        attempts.setdefault(ref, set()).add(attempt["Reason"])
    rows = []
    for _, row in source.iterrows():
        if row.get("Visit Classification") == "Previously visited":
            continue
        ref, wo = _reference(row), _work_order(row)
        current = by_ref.get(ref)
        placed = placements.get(ref, [])
        never = _truth(row.get("Never Visited"))
        if placed:
            outcome, reason = "Scheduled", "First visit scheduled." if never else "Scheduled without verified first-visit priority."
        elif not never:
            outcome, reason = "Unverified", row.get("Visit Evidence", "")
        elif wo in booked_wos or ref in booked_refs:
            outcome, reason = "Excluded booking", "An existing booking is in an excluded week; retained the booking exclusion."
        elif not _clean_text(row.get("Postcode")):
            outcome, reason = "Missing location", "No usable postcode in the supplied master portfolio."
        elif current is None:
            outcome, reason = "Excluded", "Removed by the existing booking/retry gate; inspect its audit before rescheduling."
        elif not _truth(current.get("Eligible for Selected Week")):
            outcome = "Not eligible"
            reason = _clean_text(current.get("Eligibility Reason"))
            error = _clean_text(current.get("Prediction Error"))
            if error:
                reason += " " + error
        else:
            site = {
                "is_retry": _truth(current.get("Is Retry")),
                "retry_forbidden_weekday": current.get("Retry Forbidden Weekday Number"),
                "retry_required_weekdays": current.get("Retry Required Weekdays"),
                "retry_review_original_surveyor": current.get("Retry Review Original Surveyor"),
                "special_request_date": current.get("Special Request Date"),
            }
            slots = [(person.name, date) for person in surveyors for date in person.available_dates or []
                     if date.weekday() < 5 and _site_allowed_today(site, date)
                     and _site_allowed_for_surveyor(site, person.name)]
            if not slots:
                outcome, reason = "No accessible weekday", "No selected Monday–Friday surveyor slot satisfies the access/reviewer restrictions. Saturday remains retry-only."
            else:
                outcome = "Unplaced eligible"
                reason = "Not placed after first-visit planning across all available weekday slots."
                if attempts.get(ref):
                    reason += " Observed checks: " + "; ".join(sorted(attempts[ref])) + "."
                else:
                    reason += " No remaining route capacity was found; review staffing or allocation."
        rows.append({
            "Customer Reference": row.get("Customer Reference", ref),
            "Work Order Number": row.get("Work Order Number", ""),
            "Building Name": row.get("Building Name", ""), "Postcode": row.get("Postcode", ""),
            "Never Visited": never, "Visit Classification": row.get("Visit Classification", ""),
            "Visit Evidence": row.get("Visit Evidence", ""),
            "Eligible for Selected Week": bool(current is not None and _truth(current.get("Eligible for Selected Week"))),
            "Outcome": outcome, "Reason": reason,
            "Surveyor": ", ".join(sorted({name for name, _ in placed})),
            "Scheduled Date": ", ".join(sorted({date for _, date in placed})),
        })
    return pd.DataFrame(rows, columns=[
        "Customer Reference", "Work Order Number", "Building Name", "Postcode", "Never Visited",
        "Visit Classification", "Visit Evidence", "Eligible for Selected Week", "Outcome", "Reason",
        "Surveyor", "Scheduled Date",
    ]).drop_duplicates(["Customer Reference", "Work Order Number"])
