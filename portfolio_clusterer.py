from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from scheduler import postcode_district


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

        eligible = target_week_start >= monday_of(earliest)

        earliest_dates.append(earliest)
        eligibility.append(bool(eligible))
        reasons.append(reason)

    result["Calculated Earliest Survey Date"] = earliest_dates
    result["Eligible for Selected Week"] = eligibility
    result["Eligibility Reason"] = reasons
    result["Postcode Cluster"] = result["Postcode"].apply(postcode_district)

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

    return result.sort_values(
        [
            "_planned_in_week",
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
