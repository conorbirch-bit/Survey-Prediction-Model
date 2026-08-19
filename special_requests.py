from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from portfolio_clusterer import _site_sort_frame


@dataclass
class SpecialRequestResult:
    raw_note: str
    surveyor_name: str
    requested_date: str
    location: str
    status: str
    target_cluster: str = ""
    anchor_to_cluster_minutes: Optional[float] = None
    reason: str = ""


def all_cluster_representatives(
    portfolio: pd.DataFrame,
    target_week_start,
) -> List[dict]:
    """One representative eligible site per postcode district."""
    eligible = portfolio[
        portfolio["Eligible for Selected Week"] == True
    ].copy()

    reps = []
    for cluster, group in eligible.groupby("Postcode Cluster"):
        cluster = str(cluster or "").strip()
        if not cluster:
            continue
        ranked = _site_sort_frame(group.copy(), target_week_start)
        if ranked.empty:
            continue
        row = ranked.iloc[0]
        building = str(row.get("Building Name", "")).strip()
        postcode = str(row.get("Postcode", "")).strip()
        route_location = (
            f"{building}, {postcode}" if building else postcode
        )
        reps.append({
            "cluster": cluster,
            "route_location": route_location,
            "building_name": building or postcode,
            "postcode": postcode,
        })
    return reps


def choose_nearby_cluster(
    router,
    request_location: str,
    requested_date,
    start_clock,
    timezone,
    representatives: Sequence[dict],
    max_minutes: float,
) -> Tuple[Optional[dict], Optional[float], List[dict]]:
    """
    Use Google only inside the selected week to measure the request anchor
    against one representative per eligible postcode cluster.
    """
    if not representatives:
        return None, None, []

    departure = datetime.combine(
        requested_date,
        start_clock,
        tzinfo=timezone,
    )
    destinations = [r["route_location"] for r in representatives]
    minutes = router.one_to_many(
        request_location,
        destinations,
        departure,
    )

    ranked = []
    for rep, mins in zip(representatives, minutes):
        if mins is None:
            continue
        ranked.append({
            **rep,
            "anchor_minutes": float(mins),
        })

    ranked.sort(key=lambda x: x["anchor_minutes"])
    if not ranked:
        return None, None, []

    best = ranked[0]
    if best["anchor_minutes"] > float(max_minutes):
        return None, float(best["anchor_minutes"]), ranked

    return best, float(best["anchor_minutes"]), ranked


def scheduled_reference_set(team_results: Dict[str, object], exclude_name: str = "") -> set:
    refs = set()
    for name, result in team_results.items():
        if name == exclude_name or result is None:
            continue
        for day in result.days:
            for item in day.items:
                if item.customer_reference:
                    refs.add(str(item.customer_reference))
    return refs


def build_trial_dataframe(
    portfolio: pd.DataFrame,
    current_shortlist: pd.DataFrame,
    target_cluster: str,
    surveyor_name: str,
    requested_date,
    raw_note: str,
    excluded_refs: set,
    candidate_cap: int,
    target_candidate_count: int,
    target_week_start,
) -> pd.DataFrame:
    """
    Build a capped trial pool. The request-area sites are added first, while
    preserving existing accepted-note candidates and then the normal shortlist.
    """
    eligible = portfolio[
        portfolio["Eligible for Selected Week"] == True
    ].copy()

    target = eligible[
        eligible["Postcode Cluster"].astype(str) == str(target_cluster)
    ].copy()

    if "Customer Reference" in target.columns and excluded_refs:
        target = target[
            ~target["Customer Reference"].astype(str).isin(excluded_refs)
        ].copy()

    target = _site_sort_frame(target, target_week_start)
    target = target.head(max(1, int(target_candidate_count))).copy()

    if target.empty:
        return current_shortlist.head(0).copy()

    target["Special Request Date"] = requested_date
    target["Special Request Text"] = raw_note
    target["Special Request Target Cluster"] = str(target_cluster)
    target["Assigned Surveyor"] = surveyor_name

    existing = current_shortlist.copy()
    if (
        not existing.empty
        and "Customer Reference" in existing.columns
        and excluded_refs
    ):
        existing = existing[
            ~existing["Customer Reference"].astype(str).isin(excluded_refs)
        ].copy()

    if not existing.empty:
        if "Special Request Date" not in existing.columns:
            existing["Special Request Date"] = pd.NaT
        if "Special Request Text" not in existing.columns:
            existing["Special Request Text"] = ""
        if "Special Request Target Cluster" not in existing.columns:
            existing["Special Request Target Cluster"] = ""

    combined = pd.concat([target, existing], axis=0)

    # Deduplicate by reference when possible; otherwise use building+postcode.
    if "Customer Reference" in combined.columns:
        nonblank = combined["Customer Reference"].astype(str).str.strip().ne("")
        with_ref = combined[nonblank].drop_duplicates(
            subset=["Customer Reference"], keep="first"
        )
        without_ref = combined[~nonblank].drop_duplicates(
            subset=[c for c in ["Building Name", "Postcode"] if c in combined.columns],
            keep="first",
        )
        combined = pd.concat([with_ref, without_ref], axis=0)
    else:
        combined = combined.drop_duplicates()

    return combined.head(int(candidate_cap)).copy()


def request_results_dataframe(results: Sequence[SpecialRequestResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([asdict(r) for r in results])
