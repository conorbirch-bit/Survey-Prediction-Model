from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from scheduler_v20_9_1 import postcode_district, same_campus


READY_WORDS = {
    "ready", "drawn", "complete", "completed", "drawing complete",
    "survey ready", "ready for survey", "yes",
}

NEEDS_DRAWING_WORDS = {
    "needs drawing", "need drawing", "needs to be drawn", "not drawn",
    "undrawn", "drawing required", "to draw", "no",
}


def _clean_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _parse_date(value) -> Optional[date]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def monday_of(day: date) -> date:
    return day - timedelta(days=day.weekday())


def next_monday(day: date) -> date:
    return monday_of(day) + timedelta(days=7)


def normalise_drawing_status(value) -> str:
    text = _clean_text(value).lower()
    if not text:
        return "Unknown"
    if text in READY_WORDS:
        return "Ready"
    if text in NEEDS_DRAWING_WORDS:
        return "Needs Drawing"

    if "draw" in text and any(x in text for x in ("need", "require", "not", "to ")):
        return "Needs Drawing"
    if "complete" in text or "ready" in text or "drawn" in text:
        return "Ready"
    return _clean_text(value)




def _is_future_pipeline_row(row: pd.Series) -> bool:
    """
    Cheap, non-routing signal that a site may become surveyable later.

    The master file is a to-do portfolio, so future planning counts:
      - all Plan Drafting rows, including Work Request and Work Done;
      - Geospatial Asset Mapping work that is not yet Released;
      - Status = Under Preparation;
      - Status = Work Done where it appears on the main work order;
      - Parent Work Order: Status = Work Done;
      - any row explicitly marked Needs Drawing.

    Released Geospatial rows are current-week candidates, not future pipeline,
    even if their parent work order is already Work Done.
    """
    work_type = _clean_text(row.get("Work Type Name")).lower()
    sf_status = _clean_text(row.get("Status")).lower()
    parent_status = _clean_text(
        row.get("Parent Work Order: Status")
    ).lower()
    drawing = _clean_text(row.get("Normalised Drawing Status")).lower()

    # Never double-count work that is already a current-week Released GAM site.
    if work_type == "geospatial asset mapping" and sf_status == "released":
        return False

    if work_type == "plan drafting":
        return True
    if sf_status in {"under preparation", "work done"}:
        return True
    if parent_status == "work done":
        return True
    if work_type == "geospatial asset mapping":
        return True
    if drawing == "needs drawing":
        return True
    return False


def _is_work_done_future_row(row: pd.Series) -> bool:
    """
    Stronger near-term pipeline signal.

    A future row is tagged Work Done when either its own Salesforce Status or
    its Parent Work Order Status is Work Done. Released GAM rows are excluded
    because they are current-week work, not future pipeline.
    """
    if not _is_future_pipeline_row(row):
        return False

    sf_status = _clean_text(row.get("Status")).lower()
    parent_status = _clean_text(
        row.get("Parent Work Order: Status")
    ).lower()

    return (
        sf_status == "work done"
        or parent_status == "work done"
    )


def _suggest_anchor_reserve(eligible_count: int, future_count: int) -> int:
    """
    Advisory reserve used by the endgame guardrail.

    It intentionally stays small: the aim is to leave enough released work to
    act as a future geographic anchor, not to starve the current week. The
    guardrail can release these anchors again when candidate capacity would
    otherwise be lost.
    """
    eligible_count = max(0, int(eligible_count or 0))
    future_count = max(0, int(future_count or 0))
    if eligible_count <= 0 or future_count <= 0:
        return 0
    if future_count >= 10 and eligible_count >= 8:
        return min(3, eligible_count)
    if future_count >= 5 and eligible_count >= 5:
        return min(2, eligible_count)
    if future_count >= 2 and eligible_count >= 2:
        return 1
    if future_count == 1 and eligible_count >= 4:
        return 1
    return 0


def _endgame_risk(eligible_count: int, future_count: int, reserve: int) -> str:
    if future_count <= 0:
        return "Low"
    if reserve >= 2 or (future_count >= 5 and eligible_count >= 4):
        return "High"
    if reserve >= 1 or future_count >= 2:
        return "Medium"
    return "Low"


def _annotate_endgame_fields(result: pd.DataFrame) -> pd.DataFrame:
    """
    Add portfolio-wide endgame/orphan-risk signals without paid routing calls.

    Within a postcode district, released sites that best overlap the future
    pipeline (same campus first, then same full postcode) are marked as anchor
    candidates. Those rows are sorted later in the candidate order so they are
    preserved when the selected cluster target is smaller than all released
    sites in that district.
    """
    result = result.copy()
    result["Future Pipeline Candidate"] = result.apply(_is_future_pipeline_row, axis=1)
    result["Future Work Done Candidate"] = result.apply(
        _is_work_done_future_row,
        axis=1,
    )
    result["Future Pipeline Sites in Cluster"] = 0
    result["Future Work Done Sites in Cluster"] = 0
    result["Suggested Anchor Reserve"] = 0
    result["Endgame Risk"] = "Low"
    result["Future Same Postcode Sites"] = 0
    result["Future Same Campus Sites"] = 0
    result["Endgame Anchor Score"] = 0.0
    result["Endgame Anchor Candidate"] = False
    result["Endgame Planning Note"] = "No future pipeline detected in this cluster."

    for cluster, group in result.groupby("Postcode Cluster", dropna=False):
        indices = list(group.index)
        if not str(cluster or "").strip():
            continue

        eligible = group[group["Eligible for Selected Week"] == True]
        future = group[group["Future Pipeline Candidate"] == True]
        future_work_done = future[
            future.get(
                "Future Work Done Candidate",
                pd.Series(False, index=future.index),
            ) == True
        ]
        eligible_count = len(eligible)
        future_count = len(future)
        future_work_done_count = len(future_work_done)
        reserve = _suggest_anchor_reserve(eligible_count, future_count)
        risk = _endgame_risk(eligible_count, future_count, reserve)

        result.loc[indices, "Future Pipeline Sites in Cluster"] = future_count
        result.loc[
            indices,
            "Future Work Done Sites in Cluster",
        ] = future_work_done_count
        result.loc[indices, "Suggested Anchor Reserve"] = reserve
        result.loc[indices, "Endgame Risk"] = risk

        if future_count > 0:
            work_done_note = (
                f" {future_work_done_count} are already Work Done / parent Work Done."
                if future_work_done_count > 0
                else ""
            )
            note = (
                f"{future_count} future pipeline site(s) remain in {cluster}."
                f"{work_done_note} "
                f"Suggested released-site anchor reserve: {reserve}."
            )
            result.loc[indices, "Endgame Planning Note"] = note

        if eligible.empty or future.empty or reserve <= 0:
            continue

        future_by_postcode = {}
        for _, future_row in future.iterrows():
            future_pc = _clean_text(future_row.get("Postcode")).upper()
            if not future_pc:
                continue
            future_by_postcode.setdefault(future_pc, []).append(
                _clean_text(future_row.get("Building Name"))
            )

        scored = []
        for idx, row in eligible.iterrows():
            pc = _clean_text(row.get("Postcode")).upper()
            name = _clean_text(row.get("Building Name"))
            future_names = future_by_postcode.get(pc, [])
            same_postcode_count = len(future_names)
            same_campus_count = 0
            for future_name in future_names:
                try:
                    if same_campus(name, pc, future_name, pc):
                        same_campus_count += 1
                except Exception:
                    pass

            # Same-campus support is strongest, then exact postcode support;
            # cluster-level pipeline provides a small baseline score.
            score = (
                same_campus_count * 100.0
                + same_postcode_count * 20.0
                + future_count
            )
            scored.append((score, same_campus_count, same_postcode_count, idx))

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        anchor_indices = [item[3] for item in scored[:reserve]]

        for score, same_campus_count, same_postcode_count, idx in scored:
            result.at[idx, "Future Same Postcode Sites"] = same_postcode_count
            result.at[idx, "Future Same Campus Sites"] = same_campus_count
            result.at[idx, "Endgame Anchor Score"] = round(score, 1)

        if anchor_indices:
            result.loc[anchor_indices, "Endgame Anchor Candidate"] = True
            for idx in anchor_indices:
                support = int(result.at[idx, "Future Same Campus Sites"] or 0)
                same_pc = int(result.at[idx, "Future Same Postcode Sites"] or 0)
                if support:
                    reason = f"Preserve if practical: supports {support} future same-campus site(s)."
                elif same_pc:
                    reason = f"Preserve if practical: supports {same_pc} future site(s) at the same postcode."
                else:
                    reason = "Preserve if practical as a geographic anchor for future work in this postcode district."
                result.at[idx, "Endgame Planning Note"] = reason

    return result


def endgame_adjust_cluster_choices(
    cluster_choices: Sequence[dict],
    cluster_summary: pd.DataFrame,
    max_sites_for_google: int,
) -> Tuple[List[dict], pd.DataFrame]:
    """
    Preserve future anchors without reducing the candidate pool unnecessarily.

    1) Start with the strategic choices already produced by AI/deterministic logic.
    2) Where a selected cluster has future pipeline, reduce its target by the
       small suggested anchor reserve.
    3) Replace those removed candidates from non-anchor capacity elsewhere.
    4) Only if replacement capacity is insufficient, release the lowest-risk
       anchors so the current week's candidate count is restored.

    This is deliberately a cheap portfolio operation; it never calls Google.
    """
    if cluster_summary is None or cluster_summary.empty:
        return list(cluster_choices), pd.DataFrame()

    summary_map = {
        str(r.get("Cluster", "")).strip(): r
        for r in cluster_summary.to_dict(orient="records")
        if str(r.get("Cluster", "")).strip()
    }

    original_total = min(
        int(max_sites_for_google),
        sum(max(0, int(c.get("target_sites", 0) or 0)) for c in cluster_choices),
    )
    if original_total <= 0:
        return list(cluster_choices), pd.DataFrame()

    adjusted = []
    audit = []
    selected_clusters = set()

    risk_rank = {"Low": 0, "Medium": 1, "High": 2}

    for choice in cluster_choices:
        cluster = str(choice.get("cluster", "")).strip()
        if not cluster or cluster not in summary_map:
            continue
        row = summary_map[cluster]
        eligible = int(row.get("Eligible This Week", 0) or 0)
        reserve = int(row.get("Suggested Anchor Reserve", 0) or 0)
        requested = min(eligible, max(0, int(choice.get("target_sites", 0) or 0)))
        non_anchor_capacity = max(0, eligible - reserve)
        target = min(requested, non_anchor_capacity) if reserve else requested

        if target > 0:
            new_choice = dict(choice)
            new_choice["target_sites"] = target
            if target < requested:
                new_choice["reason"] = (
                    str(choice.get("reason", ""))
                    + f" Endgame guardrail preserves {requested - target} released anchor site(s) for future {cluster} work where practical."
                ).strip()
            adjusted.append(new_choice)
            selected_clusters.add(cluster)

        audit.append({
            "Cluster": cluster,
            "Eligible This Week": eligible,
            "Future Pipeline Sites": int(row.get("Future Pipeline Sites", 0) or 0),
            "Endgame Risk": str(row.get("Endgame Risk", "Low")),
            "Suggested Anchor Reserve": reserve,
            "Original Target": requested,
            "Adjusted Target": target,
            "Released Anchors Preserved": max(0, eligible - target),
            "Guardrail Action": (
                "Preserved anchor capacity" if target < requested else "No target reduction"
            ),
        })

    current_total = sum(int(c.get("target_sites", 0) or 0) for c in adjusted)
    missing = max(0, original_total - current_total)

    # First replace removed anchor candidates with non-anchor capacity from the
    # already selected clusters, then from other low-risk/strong clusters.
    candidate_rows = sorted(
        cluster_summary.to_dict(orient="records"),
        key=lambda r: (
            risk_rank.get(str(r.get("Endgame Risk", "Low")), 1),
            -int(r.get("Eligible This Week", 0) or 0),
            -float(r.get("Planning Hours Available", 0) or 0),
        ),
    )

    def choice_for(cluster):
        for c in adjusted:
            if str(c.get("cluster", "")).strip() == cluster:
                return c
        return None

    for row in candidate_rows:
        if missing <= 0:
            break
        cluster = str(row.get("Cluster", "")).strip()
        eligible = int(row.get("Eligible This Week", 0) or 0)
        reserve = int(row.get("Suggested Anchor Reserve", 0) or 0)
        non_anchor_capacity = max(0, eligible - reserve)
        if non_anchor_capacity <= 0:
            continue

        existing = choice_for(cluster)
        used = int(existing.get("target_sites", 0) or 0) if existing else 0
        spare = max(0, non_anchor_capacity - used)
        if spare <= 0:
            continue
        add = min(spare, missing)

        if existing:
            existing["target_sites"] = used + add
            existing["reason"] = (
                str(existing.get("reason", ""))
                + f" Added {add} non-anchor candidate(s) to replace preserved anchors elsewhere."
            ).strip()
        else:
            adjusted.append({
                "cluster": cluster,
                "priority": 45,
                "target_sites": add,
                "decision": "endgame_replacement",
                "reason": (
                    "Added as non-anchor replacement capacity so stronger future anchors can be preserved elsewhere."
                ),
            })
            selected_clusters.add(cluster)
        missing -= add

    # If there still is not enough candidate capacity, release anchors from the
    # lowest-risk clusters first. This prevents the endgame logic from leaving
    # the current week under-supplied.
    if missing > 0:
        release_rows = sorted(
            candidate_rows,
            key=lambda r: (
                risk_rank.get(str(r.get("Endgame Risk", "Low")), 1),
                int(r.get("Future Pipeline Sites", 0) or 0),
            ),
        )
        for row in release_rows:
            if missing <= 0:
                break
            cluster = str(row.get("Cluster", "")).strip()
            eligible = int(row.get("Eligible This Week", 0) or 0)
            existing = choice_for(cluster)
            used = int(existing.get("target_sites", 0) or 0) if existing else 0
            spare = max(0, eligible - used)
            if spare <= 0:
                continue
            add = min(spare, missing)
            if existing:
                existing["target_sites"] = used + add
                existing["reason"] = (
                    str(existing.get("reason", ""))
                    + f" Released {add} anchor candidate(s) because replacement capacity was insufficient for this week's shortlist."
                ).strip()
            else:
                adjusted.append({
                    "cluster": cluster,
                    "priority": 35,
                    "target_sites": add,
                    "decision": "anchor_released_for_capacity",
                    "reason": "Anchor capacity released because the current week otherwise lacked enough candidate sites.",
                })
            missing -= add

    # Rebuild audit using final targets.
    final_target_map = {
        str(c.get("cluster", "")).strip(): int(c.get("target_sites", 0) or 0)
        for c in adjusted
    }
    audit_rows = []
    for _, row in cluster_summary.iterrows():
        cluster = str(row.get("Cluster", "")).strip()
        eligible = int(row.get("Eligible This Week", 0) or 0)
        future = int(row.get("Future Pipeline Sites", 0) or 0)
        reserve = int(row.get("Suggested Anchor Reserve", 0) or 0)
        final_target = final_target_map.get(cluster, 0)
        if eligible <= 0 and future <= 0:
            continue
        audit_rows.append({
            "Cluster": cluster,
            "Eligible Released Now": eligible,
            "Future Pipeline Sites": future,
            "Future Drawing Pipeline": int(row.get("Future Drawing Pipeline", 0) or 0),
            "Future Plan Drafting": int(row.get("Future Plan Drafting", 0) or 0),
            "Future Under Preparation": int(row.get("Future Under Preparation", 0) or 0),
            "Future Work Done": int(row.get("Future Work Done", 0) or 0),
            "Future Plan Drafting Work Done": int(
                row.get("Future Plan Drafting Work Done", 0) or 0
            ),
            "Future Parent Work Done": int(row.get("Future Parent Work Done", 0) or 0),
            "Future GAM Awaiting Release": int(row.get("Future GAM Awaiting Release", 0) or 0),
            "Endgame Risk": str(row.get("Endgame Risk", "Low")),
            "Suggested Anchor Reserve": reserve,
            "Candidate Target This Week": final_target,
            "Released Sites Left Outside Shortlist": max(0, eligible - final_target),
            "Endgame Reason": str(row.get("Endgame Reason", "")),
        })

    adjusted = sorted(
        adjusted,
        key=lambda c: int(c.get("priority", 50)),
        reverse=True,
    )
    return adjusted, pd.DataFrame(audit_rows)


def add_portfolio_fields(
    df: pd.DataFrame,
    target_week_start: date,
    today: date,
) -> pd.DataFrame:
    """
    Adds cheap planning metadata without calling Google.

    Drawing rule:
      - Ready/unknown status: eligible immediately.
      - Needs Drawing: eligible from the Monday after the current week, giving
        the drawing team the requested one-week lead time.
      - If an explicit Earliest Survey Date exists, it takes precedence and the
        site is only eligible once that date has been reached.
      - HARD GATE: the selected-week schedule only accepts rows where
        Work Type Name = Geospatial Asset Mapping AND Status = Released.
    """
    result = df.copy()

    if "Drawing Status" in result.columns:
        result["Normalised Drawing Status"] = result["Drawing Status"].apply(
            normalise_drawing_status
        )
    else:
        result["Normalised Drawing Status"] = "Unknown"

    current_next_monday = next_monday(today)

    earliest_dates = []
    eligibility = []
    reasons = []

    for _, row in result.iterrows():
        explicit_earliest = _parse_date(row.get("Earliest Survey Date"))
        status = str(row.get("Normalised Drawing Status", "Unknown"))

        if explicit_earliest is not None:
            earliest = explicit_earliest
            reason = f"Explicit earliest survey date: {earliest.isoformat()}"
        elif status == "Needs Drawing":
            earliest = current_next_monday
            reason = (
                "Needs drawing: held until the following Monday to provide "
                "a one-week drawing lead time."
            )
        else:
            earliest = today
            reason = "Survey-ready / no drawing hold."

        date_eligible = target_week_start >= monday_of(earliest)

        # Hard weekly scheduling gate: a site can only enter the selected
        # week's routing/schedule when Salesforce has explicitly released the
        # Geospatial Asset Mapping work. Other rows remain in the portfolio so
        # they can still contribute to drawing priority and future planning.
        work_type = _clean_text(row.get("Work Type Name")).lower()
        sf_status = _clean_text(row.get("Status")).lower()
        released_for_survey = (
            work_type == "geospatial asset mapping"
            and sf_status == "released"
        )

        prediction_status = _clean_text(row.get("Prediction Status")).lower()
        duration_ready = prediction_status == "predicted"
        eligible = bool(date_eligible and released_for_survey and duration_ready)

        if not released_for_survey:
            reason = (
                "Future-planning only: not currently released for weekly survey "
                "scheduling. This row still contributes to cluster/endgame planning."
            )
        elif not duration_ready:
            reason = (
                "Released but no usable duration prediction yet. Included in the "
                "portfolio strategy, but not routed into the current-week schedule."
            )

        earliest_dates.append(earliest)
        eligibility.append(eligible)
        reasons.append(reason)

    result["Calculated Earliest Survey Date"] = earliest_dates
    result["Eligible for Selected Week"] = eligibility
    result["Eligibility Reason"] = reasons
    if "Work Type Name" in result.columns and "Status" in result.columns:
        result["Released for Weekly Scheduling"] = (
            result["Work Type Name"].astype(str).str.strip().str.lower().eq("geospatial asset mapping")
            & result["Status"].astype(str).str.strip().str.lower().eq("released")
        )
    else:
        result["Released for Weekly Scheduling"] = False
    result["Postcode Cluster"] = result["Postcode"].apply(postcode_district)
    result = _annotate_endgame_fields(result)

    return result


def _confidence_counts(group: pd.DataFrame) -> Tuple[int, int, int]:
    if "Prediction Confidence" not in group.columns:
        return 0, 0, 0
    values = group["Prediction Confidence"].astype(str).str.lower()
    return (
        int((values == "high").sum()),
        int((values == "medium").sum()),
        int((values == "low").sum()),
    )


def build_cluster_summary(
    portfolio: pd.DataFrame,
    target_week_start: date,
) -> pd.DataFrame:
    """
    Summarise the whole portfolio by postcode district. No paid routing calls.
    """
    rows = []
    week_end = target_week_start + timedelta(days=6)
    next_start = target_week_start + timedelta(days=7)
    next_end = target_week_start + timedelta(days=13)
    three_week_end = target_week_start + timedelta(days=27)

    planned_dates = (
        portfolio["Planned Start"].apply(_parse_date)
        if "Planned Start" in portfolio.columns
        else pd.Series([None] * len(portfolio), index=portfolio.index)
    )

    working = portfolio.copy()
    working["_planned_date"] = planned_dates

    for cluster, group in working.groupby("Postcode Cluster", dropna=False):
        cluster = str(cluster or "").strip()
        if not cluster:
            continue

        eligible = group[group["Eligible for Selected Week"] == True]
        ready = group[
            group["Normalised Drawing Status"].astype(str).str.lower() == "ready"
        ]
        needs_drawing = group[
            group["Normalised Drawing Status"].astype(str).str.lower()
            == "needs drawing"
        ]

        high, medium, low = _confidence_counts(eligible)

        durations = pd.to_numeric(
            eligible.get(
                "Planning Duration (Minutes)",
                pd.Series(index=eligible.index, dtype=float),
            ),
            errors="coerce",
        ).dropna()

        def planned_count(start: date, end: date) -> int:
            return int(
                group["_planned_date"].apply(
                    lambda d: d is not None and start <= d <= end
                ).sum()
            )

        sample_buildings = []
        if "Building Name" in group.columns:
            for value in group["Building Name"].dropna().astype(str).head(3):
                sample_buildings.append(value[:80])

        future_pipeline = group[
            group.get(
                "Future Pipeline Candidate",
                pd.Series(False, index=group.index),
            ) == True
        ]
        future_count = len(future_pipeline)
        future_drawing = int(
            future_pipeline["Normalised Drawing Status"]
            .astype(str).str.lower().eq("needs drawing").sum()
        ) if "Normalised Drawing Status" in future_pipeline.columns else 0
        future_gam_awaiting = 0
        future_plan_drafting = 0
        future_under_preparation = 0
        future_work_done = 0
        future_plan_drafting_work_done = 0
        future_parent_work_done = 0
        if "Work Type Name" in future_pipeline.columns:
            future_plan_drafting = int(
                future_pipeline["Work Type Name"]
                .astype(str).str.strip().str.lower().eq("plan drafting").sum()
            )
        if "Status" in future_pipeline.columns:
            future_under_preparation = int(
                future_pipeline["Status"]
                .astype(str).str.strip().str.lower().eq("under preparation").sum()
            )
        if "Work Type Name" in future_pipeline.columns and "Status" in future_pipeline.columns:
            future_gam_awaiting = int((
                future_pipeline["Work Type Name"].astype(str).str.strip().str.lower().eq("geospatial asset mapping")
                & ~future_pipeline["Status"].astype(str).str.strip().str.lower().eq("released")
            ).sum())

        if not future_pipeline.empty:
            main_status = (
                future_pipeline["Status"]
                .astype(str).str.strip().str.lower()
                if "Status" in future_pipeline.columns
                else pd.Series("", index=future_pipeline.index)
            )
            parent_status = (
                future_pipeline["Parent Work Order: Status"]
                .astype(str).str.strip().str.lower()
                if "Parent Work Order: Status" in future_pipeline.columns
                else pd.Series("", index=future_pipeline.index)
            )
            work_type = (
                future_pipeline["Work Type Name"]
                .astype(str).str.strip().str.lower()
                if "Work Type Name" in future_pipeline.columns
                else pd.Series("", index=future_pipeline.index)
            )

            work_done_mask = (
                main_status.eq("work done")
                | parent_status.eq("work done")
            )
            future_work_done = int(work_done_mask.sum())
            future_parent_work_done = int(
                parent_status.eq("work done").sum()
            )
            future_plan_drafting_work_done = int((
                work_type.eq("plan drafting")
                & work_done_mask
            ).sum())

        reserve = _suggest_anchor_reserve(len(eligible), future_count)
        endgame_risk = _endgame_risk(len(eligible), future_count, reserve)
        if future_count <= 0:
            endgame_reason = "No known future pipeline in this postcode district; no anchor reserve needed."
        elif reserve <= 0:
            endgame_reason = (
                f"{future_count} future site(s) remain, but current released volume is too small to reserve an anchor without risking this week's capacity."
            )
        else:
            endgame_reason = (
                f"{future_count} future site(s) remain; preserve about {reserve} released anchor site(s) where practical so later work is less likely to become isolated."
            )

        rows.append({
            "Cluster": cluster,
            "Total Portfolio Sites": len(group),
            "Eligible This Week": len(eligible),
            "Ready Sites": len(ready),
            "Needs Drawing Sites": len(needs_drawing),
            "Planning Hours Available": round(float(durations.sum()) / 60, 1),
            "Average Planning Minutes": (
                round(float(durations.mean()), 1) if len(durations) else None
            ),
            "High Confidence": high,
            "Medium Confidence": medium,
            "Low Confidence": low,
            "Existing Planned This Week": planned_count(
                target_week_start, week_end
            ),
            "Existing Planned Next Week": planned_count(next_start, next_end),
            "Existing Planned Next 3 Weeks": planned_count(
                next_start, three_week_end
            ),
            "Future Pipeline Sites": future_count,
            "Future Drawing Pipeline": future_drawing,
            "Future Plan Drafting": future_plan_drafting,
            "Future Under Preparation": future_under_preparation,
            "Future Work Done": future_work_done,
            "Future Plan Drafting Work Done": future_plan_drafting_work_done,
            "Future Parent Work Done": future_parent_work_done,
            "Future GAM Awaiting Release": future_gam_awaiting,
            "Suggested Anchor Reserve": reserve,
            "Endgame Risk": endgame_risk,
            "Endgame Reason": endgame_reason,
            "Sample Buildings": " | ".join(sample_buildings),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    return summary.sort_values(
        ["Eligible This Week", "Planning Hours Available", "Total Portfolio Sites"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def deterministic_cluster_choices(
    cluster_summary: pd.DataFrame,
    max_sites_for_google: int,
) -> List[dict]:
    """
    Safe fallback when AI cluster selection is disabled/unavailable.
    """
    choices = []
    remaining = int(max_sites_for_google)

    for _, row in cluster_summary.iterrows():
        available = int(row.get("Eligible This Week", 0) or 0)
        if available <= 0 or remaining <= 0:
            continue

        target = min(available, remaining)
        choices.append({
            "cluster": str(row["Cluster"]),
            "priority": 50,
            "target_sites": target,
            "decision": "consider_this_week",
            "reason": "Deterministic fallback: high eligible site volume.",
        })
        remaining -= target

    return choices


def _site_sort_frame(group: pd.DataFrame, target_week_start: date) -> pd.DataFrame:
    result = group.copy()

    if "Planned Start" in result.columns:
        result["_planned"] = result["Planned Start"].apply(_parse_date)
        week_end = target_week_start + timedelta(days=6)
        result["_planned_in_week"] = result["_planned"].apply(
            lambda d: 0 if d is not None and target_week_start <= d <= week_end else 1
        )
        result["_planned_distance"] = result["_planned"].apply(
            lambda d: abs((d - target_week_start).days) if d is not None else 9999
        )
    else:
        result["_planned_in_week"] = 1
        result["_planned_distance"] = 9999

    confidence_rank = {"high": 0, "medium": 1, "low": 2, "none": 3}
    result["_confidence_rank"] = (
        result.get(
            "Prediction Confidence",
            pd.Series("none", index=result.index),
        )
        .astype(str)
        .str.lower()
        .map(confidence_rank)
        .fillna(3)
    )

    result["_duration"] = pd.to_numeric(
        result.get(
            "Planning Duration (Minutes)",
            pd.Series(index=result.index, dtype=float),
        ),
        errors="coerce",
    ).fillna(9999)

    # Endgame anchors are deliberately placed last inside their cluster. If a
    # target asks for fewer than all released sites, the most useful future
    # anchors are therefore the sites left behind. Existing planned-this-week
    # work still takes precedence over the anchor preference.
    result["_endgame_anchor"] = (
        result.get(
            "Endgame Anchor Candidate",
            pd.Series(False, index=result.index),
        ).astype(bool).astype(int)
    )

    return result.sort_values(
        [
            "_planned_in_week",
            "_endgame_anchor",
            "_planned_distance",
            "_confidence_rank",
            "_duration",
        ],
        ascending=True,
    )


def shortlist_sites(
    portfolio: pd.DataFrame,
    cluster_choices: Sequence[dict],
    max_sites_for_google: int,
    target_week_start: date,
) -> pd.DataFrame:
    """
    Convert strategic cluster choices into the much smaller site list that
    Google Routes is allowed to see.
    """
    eligible = portfolio[
        portfolio["Eligible for Selected Week"] == True
    ].copy()

    selected_parts = []
    selected_indices = set()
    remaining_capacity = int(max_sites_for_google)

    ordered_choices = sorted(
        list(cluster_choices),
        key=lambda x: float(x.get("priority", 50)),
        reverse=True,
    )

    for choice in ordered_choices:
        if remaining_capacity <= 0:
            break

        cluster = str(choice.get("cluster", "")).strip()
        if not cluster:
            continue

        group = eligible[
            eligible["Postcode Cluster"].astype(str) == cluster
        ].copy()
        group = group[~group.index.isin(selected_indices)]
        if group.empty:
            continue

        requested = choice.get("target_sites")
        try:
            requested = int(requested)
        except Exception:
            requested = len(group)

        take = min(max(1, requested), len(group), remaining_capacity)
        ranked = _site_sort_frame(group, target_week_start).head(take)

        ranked["AI Cluster Priority"] = int(choice.get("priority", 50))
        ranked["AI Cluster Decision"] = str(
            choice.get("decision", "consider_this_week")
        )
        ranked["AI Cluster Reason"] = str(choice.get("reason", ""))

        selected_parts.append(ranked)
        selected_indices.update(ranked.index)
        remaining_capacity -= len(ranked)

    # If AI returned too few sites, fill spare capacity from the strongest
    # remaining eligible clusters deterministically. This protects the weekly
    # schedule from an overly narrow LLM choice without exposing all sites to Google.
    if remaining_capacity > 0:
        remaining = eligible[~eligible.index.isin(selected_indices)].copy()
        if not remaining.empty:
            counts = (
                remaining.groupby("Postcode Cluster")
                .size()
                .sort_values(ascending=False)
            )
            for cluster, _ in counts.items():
                if remaining_capacity <= 0:
                    break
                group = remaining[
                    remaining["Postcode Cluster"] == cluster
                ]
                ranked = _site_sort_frame(
                    group, target_week_start
                ).head(remaining_capacity)
                if ranked.empty:
                    continue
                ranked["AI Cluster Priority"] = 40
                ranked["AI Cluster Decision"] = "filler"
                ranked["AI Cluster Reason"] = (
                    "Added deterministically to ensure enough candidates "
                    "for Google routing."
                )
                selected_parts.append(ranked)
                selected_indices.update(ranked.index)
                remaining_capacity -= len(ranked)

    if not selected_parts:
        return eligible.head(0)

    result = pd.concat(selected_parts).drop_duplicates()
    helper_cols = [c for c in result.columns if c.startswith("_")]
    return result.drop(columns=helper_cols, errors="ignore").head(
        int(max_sites_for_google)
    )


def build_drawing_priority_queue(
    portfolio: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    selected_cluster_choices: Sequence[dict],
    target_week_start: date,
) -> pd.DataFrame:
    """
    Order every Needs Drawing site without Google routing.

    Near-term selected clusters come first because they are the areas being
    prepared for the target survey week. Remaining sites are ordered by a cheap
    cluster score using existing planned dates, ready-site support, drawing
    volume and available survey workload.
    """
    needs = portfolio[
        portfolio["Normalised Drawing Status"]
        .astype(str)
        .str.lower()
        .eq("needs drawing")
    ].copy()

    if needs.empty:
        return needs

    selected_map = {}
    for rank, choice in enumerate(selected_cluster_choices, start=1):
        cluster = str(choice.get("cluster", "")).strip()
        if not cluster:
            continue
        selected_map[cluster] = {
            "priority": int(choice.get("priority", 50)),
            "reason": str(choice.get("reason", "")),
            "selected_rank": rank,
        }

    summary_map = {}
    if not cluster_summary.empty:
        for _, row in cluster_summary.iterrows():
            cluster = str(row.get("Cluster", "")).strip()
            if not cluster:
                continue

            next_week = float(row.get("Existing Planned Next Week", 0) or 0)
            next_3 = float(row.get("Existing Planned Next 3 Weeks", 0) or 0)
            ready = float(row.get("Ready Sites", 0) or 0)
            drawing = float(row.get("Needs Drawing Sites", 0) or 0)
            hours = float(row.get("Planning Hours Available", 0) or 0)

            cheap_score = (
                next_week * 20.0
                + next_3 * 5.0
                + ready * 4.0
                + drawing * 1.0
                + hours * 0.5
            )

            summary_map[cluster] = {
                "cheap_score": cheap_score,
                "next_week": next_week,
                "next_3": next_3,
                "ready": ready,
                "drawing": drawing,
                "hours": hours,
            }

    planned_dates = (
        needs["Planned Start"].apply(_parse_date)
        if "Planned Start" in needs.columns
        else pd.Series([None] * len(needs), index=needs.index)
    )

    rows = []
    for idx, row in needs.iterrows():
        cluster = str(row.get("Postcode Cluster", "")).strip()
        selected = selected_map.get(cluster)
        summary = summary_map.get(cluster, {})

        if selected:
            tier = "Priority - target week cluster"
            tier_rank = 0
            cluster_priority = int(selected["priority"])
            cluster_rank = int(selected["selected_rank"])
            reason = (
                "Selected for the target survey week. "
                + selected["reason"]
            ).strip()
        else:
            tier = "Future cluster"
            tier_rank = 1
            cluster_priority = 0
            cluster_rank = 9999
            reason = (
                "Future drawing queue ordered by cluster workload and "
                "existing planned-date signals."
            )

        planned = planned_dates.loc[idx]
        planned_sort = (
            abs((planned - target_week_start).days)
            if planned is not None
            else 99999
        )

        rows.append({
            "_source_index": idx,
            "_tier_rank": tier_rank,
            "_selected_cluster_rank": cluster_rank,
            "_selected_priority_sort": -cluster_priority,
            "_future_score_sort": -float(summary.get("cheap_score", 0)),
            "_planned_sort": planned_sort,
            "Drawing Priority Tier": tier,
            "Drawing Cluster Priority": cluster_priority,
            "Drawing Priority Reason": reason,
            "Cluster Future Score": round(
                float(summary.get("cheap_score", 0)), 1
            ),
        })

    ranking = pd.DataFrame(rows).set_index("_source_index")
    ranked = needs.join(ranking, how="left")

    ranked = ranked.sort_values(
        [
            "_tier_rank",
            "_selected_cluster_rank",
            "_selected_priority_sort",
            "_future_score_sort",
            "_planned_sort",
            "Postcode Cluster",
            "Customer Reference"
            if "Customer Reference" in ranked.columns
            else "Building Name",
        ],
        ascending=True,
    ).copy()

    ranked.insert(0, "Drawing Order", range(1, len(ranked) + 1))
    ranked["Target Survey Week"] = target_week_start.isoformat()

    helper_cols = [c for c in ranked.columns if c.startswith("_")]
    ranked = ranked.drop(columns=helper_cols, errors="ignore")
    return ranked.reset_index(drop=True)
