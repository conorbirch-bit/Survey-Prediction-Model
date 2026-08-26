from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from portfolio_clusterer import _site_sort_frame
from coordinate_clustering import haversine_km


# ============================================================================
# SURVEYOR HOME-FIT RULES
# ============================================================================
#
# These are the station coordinates supplied for the five regular surveyors.
# They are used only as a cheap geographic sanity check during cluster
# allocation. Google remains the public-transport source of truth.
#
# No extra Google requests are introduced by this logic.
SURVEYOR_HOME_COORDINATES = {
    "conor birch": (51.81489, -0.35194),       # Harpenden Train Station
    "harrison grice": (51.441346, 0.366643),  # Gravesend Station
    "joe reynolds": (51.742, -0.491),          # Hemel Hempstead
    "rod harrison": (52.379, -1.250),          # Rugby
    "toby lawal": (51.567956, 0.129558),       # Chadwell Heath
}

# If an available surveyor is more than this many Google-transit minutes worse
# than the best available surveyor for a cluster, workload balancing cannot
# steal the cluster from the better-positioned surveyor.
HOME_FIT_TRAVEL_BAND_MINUTES = 30.0

# Geographic sanity check. A surveyor more than this many straight-line km
# farther from the cluster centre than the nearest available surveyor is not
# allowed to win purely because of workload balancing while a natural-fit
# surveyor still has capacity.
HOME_FIT_DISTANCE_BAND_KM = 25.0


def _normalise_name(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _surveyor_home_coordinates(surveyor: "SurveyorConfig"):
    return SURVEYOR_HOME_COORDINATES.get(
        _normalise_name(surveyor.name)
    )


@dataclass
class SurveyorConfig:
    name: str
    start_location: str
    available_dates: object = None


@dataclass
class ClusterAllocation:
    surveyor_name: str
    cluster: str
    target_sites: int
    home_to_cluster_minutes: float
    cluster_priority: int
    cluster_reason: str
    estimated_candidate_minutes: float


def representative_sites(
    portfolio: pd.DataFrame,
    cluster_choices: Sequence[dict],
    target_week_start,
) -> List[dict]:
    """
    Pick one real site to represent each selected geographic planning cluster.

    Google is used only for these tiny home -> cluster-representative comparisons
    before the more detailed per-surveyor routing begins.
    """
    eligible = portfolio[
        portfolio["Eligible for Selected Week"] == True
    ].copy()

    reps = []
    for choice in cluster_choices:
        cluster = str(choice.get("cluster", "")).strip()
        group = eligible[
            eligible["Planning Cluster"].astype(str) == cluster
        ].copy()
        if group.empty:
            continue

        ranked = _site_sort_frame(group, target_week_start)
        row = ranked.iloc[0]
        building = str(row.get("Building Name", "")).strip()
        postcode = str(row.get("Postcode", "")).strip()
        route_location = (
            f"{building}, {postcode}" if building else postcode
        )

        centre_latitude = row.get(
            "Cluster Centre Latitude",
            row.get("Latitude Clean", row.get("Latitude")),
        )
        centre_longitude = row.get(
            "Cluster Centre Longitude",
            row.get("Longitude Clean", row.get("Longitude")),
        )

        reps.append({
            "cluster": cluster,
            "route_location": route_location,
            "building_name": building or postcode,
            "postcode": postcode,
            "latitude": centre_latitude,
            "longitude": centre_longitude,
            "priority": int(choice.get("priority", 50)),
            "target_sites": int(choice.get("target_sites", 1)),
            "reason": str(choice.get("reason", "")),
        })

    return reps


def home_to_cluster_matrix(
    router,
    surveyors: Sequence[SurveyorConfig],
    representatives: Sequence[dict],
    departure_time: datetime,
) -> Dict[Tuple[str, str], float]:
    """
    Very small Google matrix: one origin per active surveyor and one destination
    per selected strategic cluster.
    """
    if not representatives:
        return {}

    destinations = [r["route_location"] for r in representatives]
    result = {}

    # Cheap geographic metadata for the allocator. These calculations happen
    # locally and do not create any Google API usage.
    for surveyor in surveyors:
        home_coordinates = _surveyor_home_coordinates(surveyor)
        if home_coordinates is None:
            continue

        home_latitude, home_longitude = home_coordinates
        for rep in representatives:
            distance_km = haversine_km(
                home_latitude,
                home_longitude,
                rep.get("latitude"),
                rep.get("longitude"),
            )
            if distance_km is not None:
                result[
                    ("__home_distance_km__", surveyor.name, rep["cluster"])
                ] = float(distance_km)

    for surveyor in surveyors:
        surveyor_departure = departure_time
        if surveyor.available_dates:
            first_available = min(surveyor.available_dates)
            surveyor_departure = departure_time.replace(
                year=first_available.year,
                month=first_available.month,
                day=first_available.day,
            )

        durations = router.one_to_many(
            surveyor.start_location,
            destinations,
            surveyor_departure,
        )
        for rep, minutes in zip(representatives, durations):
            if minutes is not None:
                result[(surveyor.name, rep["cluster"])] = float(minutes)

    return result


def _cluster_average_minutes(
    cluster_summary: pd.DataFrame,
    cluster: str,
) -> float:
    match = cluster_summary[
        cluster_summary["Cluster"].astype(str) == str(cluster)
    ]
    if match.empty:
        return 75.0
    value = match.iloc[0].get("Average Planning Minutes")
    try:
        value = float(value)
    except Exception:
        value = 75.0
    if pd.isna(value) or value <= 0:
        value = 75.0
    return value


def allocate_cluster_targets(
    cluster_choices: Sequence[dict],
    cluster_summary: pd.DataFrame,
    surveyors: Sequence[SurveyorConfig],
    travel_matrix: Dict[Tuple[str, str], float],
    max_sites_per_surveyor: int,
) -> List[ClusterAllocation]:
    """
    Split selected cluster capacity across the active team.

    The assignment uses:
    - actual Google home -> cluster representative transit time;
    - supplied surveyor station coordinates as a cheap geographic sanity check;
    - workload balancing;
    - a small bonus for keeping a surveyor in a cluster already allocated to them.

    Clear home-fit advantages are protected before workload balancing. A large
    cluster can still be shared, and a more distant surveyor can still be used
    after the naturally placed surveyors run out of capacity.
    """
    if not surveyors:
        return []

    capacities = {
        s.name: max(
            1,
            round(
                int(max_sites_per_surveyor)
                * min(5, len(s.available_dates or []))
                / 5
            ),
        )
        for s in surveyors
    }
    assigned_sites = {s.name: 0 for s in surveyors}
    assigned_minutes = {s.name: 0.0 for s in surveyors}
    clusters_by_surveyor = {s.name: set() for s in surveyors}

    # Use manageable chunks so one large cluster can feed more than one person.
    chunk_size = max(6, min(15, int(max_sites_per_surveyor) // 3 or 6))

    chunks = []
    for choice in sorted(
        list(cluster_choices),
        key=lambda c: int(c.get("priority", 50)),
        reverse=True,
    ):
        cluster = str(choice.get("cluster", "")).strip()
        if not cluster:
            continue
        try:
            total = int(choice.get("target_sites", 1))
        except Exception:
            total = 1
        total = max(1, total)

        while total > 0:
            take = min(chunk_size, total)
            chunks.append((choice, take))
            total -= take

    allocations: List[ClusterAllocation] = []

    for choice, requested_chunk in chunks:
        cluster = str(choice.get("cluster", "")).strip()
        avg_minutes = _cluster_average_minutes(
            cluster_summary,
            cluster,
        )

        candidates = []
        for surveyor in surveyors:
            capacity_left = (
                capacities[surveyor.name]
                - assigned_sites[surveyor.name]
            )
            if capacity_left <= 0:
                continue

            travel = travel_matrix.get(
                (surveyor.name, cluster)
            )
            if travel is None:
                # If Google could not route this home -> cluster pair,
                # do not force the allocation to this surveyor.
                continue

            home_distance_km = travel_matrix.get(
                (
                    "__home_distance_km__",
                    surveyor.name,
                    cluster,
                )
            )

            chunk = min(requested_chunk, capacity_left)
            projected_minutes = (
                assigned_minutes[surveyor.name]
                + chunk * avg_minutes
            )

            # Rough target only for balancing candidate workload. Detailed
            # time feasibility is still enforced later by each weekly router.
            total_selected_minutes = 0.0
            for c in cluster_choices:
                c_cluster = str(c.get("cluster", "")).strip()
                total_selected_minutes += (
                    max(1, int(c.get("target_sites", 1)))
                    * _cluster_average_minutes(
                        cluster_summary,
                        c_cluster,
                    )
                )
            total_available_days = max(
                1,
                sum(
                    min(5, len(s.available_dates or []))
                    for s in surveyors
                ),
            )
            surveyor_day_share = (
                min(5, len(surveyor.available_dates or []))
                / total_available_days
            )
            target_minutes = max(
                1.0,
                total_selected_minutes * surveyor_day_share,
            )

            load_penalty = (
                projected_minutes / target_minutes
            ) * 50.0

            continuity_bonus = (
                12.0
                if cluster in clusters_by_surveyor[surveyor.name]
                else 0.0
            )

            # Give commute efficiency slightly more weight than before. This
            # does not reduce candidate capacity; it simply makes a cluster more
            # likely to go to the surveyor whose home journey is cheaper, leaving
            # more of the fixed survey window available for actual jobs.
            commute_efficiency_penalty = float(travel) * 0.50
            score = (
                float(travel)
                + commute_efficiency_penalty
                + load_penalty
                - continuity_bonus
            )

            candidates.append(
                (
                    score,
                    surveyor,
                    chunk,
                    float(travel),
                    projected_minutes,
                    (
                        float(home_distance_km)
                        if home_distance_km is not None
                        else None
                    ),
                )
            )

        if not candidates:
            continue

        # Home-fit protection comes BEFORE workload balancing.
        #
        # If one or more available surveyors are clearly well positioned for
        # this cluster, only that natural-fit group competes for the chunk.
        # Workload balancing still operates normally inside the group.
        #
        # If those natural-fit surveyors later run out of capacity they are no
        # longer present in `candidates`, so the cluster can still fall back to
        # another available surveyor rather than being left unscheduled.
        best_google_travel = min(candidate[3] for candidate in candidates)
        geo_distances = [
            candidate[5]
            for candidate in candidates
            if candidate[5] is not None
        ]
        best_geo_distance = (
            min(geo_distances)
            if geo_distances
            else None
        )

        natural_fit_candidates = []
        for candidate in candidates:
            google_fit = (
                candidate[3]
                <= best_google_travel
                + float(HOME_FIT_TRAVEL_BAND_MINUTES)
            )
            geo_distance = candidate[5]
            geographic_fit = (
                best_geo_distance is None
                or geo_distance is None
                or geo_distance
                <= best_geo_distance
                + float(HOME_FIT_DISTANCE_BAND_KM)
            )
            if google_fit and geographic_fit:
                natural_fit_candidates.append(candidate)

        if natural_fit_candidates:
            candidates = natural_fit_candidates

        candidates.sort(key=lambda x: x[0])
        _, chosen, chunk, travel, _, _ = candidates[0]

        estimated_minutes = chunk * avg_minutes
        allocations.append(
            ClusterAllocation(
                surveyor_name=chosen.name,
                cluster=cluster,
                target_sites=int(chunk),
                home_to_cluster_minutes=round(travel, 1),
                cluster_priority=int(choice.get("priority", 50)),
                cluster_reason=str(choice.get("reason", "")),
                estimated_candidate_minutes=round(
                    estimated_minutes, 1
                ),
            )
        )

        assigned_sites[chosen.name] += int(chunk)
        assigned_minutes[chosen.name] += estimated_minutes
        clusters_by_surveyor[chosen.name].add(cluster)

    return allocations


def build_team_shortlists(
    portfolio: pd.DataFrame,
    allocations: Sequence[ClusterAllocation],
    surveyors: Sequence[SurveyorConfig],
    target_week_start,
    max_sites_per_surveyor: int,
) -> Dict[str, pd.DataFrame]:
    """
    Hand distinct sites from each selected cluster to the allocated surveyors.
    No site can appear in two surveyor shortlists.
    """
    eligible = portfolio[
        portfolio["Eligible for Selected Week"] == True
    ].copy()

    by_cluster = {}
    for cluster, group in eligible.groupby("Planning Cluster"):
        by_cluster[str(cluster)] = _site_sort_frame(
            group.copy(),
            target_week_start,
        )

    used_indices = set()
    output = {
        s.name: eligible.head(0).copy() for s in surveyors
    }

    # Preserve strategic priority, then lower home travel.
    ordered = sorted(
        list(allocations),
        key=lambda a: (
            -int(a.cluster_priority),
            float(a.home_to_cluster_minutes),
        ),
    )

    parts = {s.name: [] for s in surveyors}
    counts = {s.name: 0 for s in surveyors}

    for allocation in ordered:
        surveyor_cfg = next(
            s for s in surveyors
            if s.name == allocation.surveyor_name
        )
        effective_capacity = max(
            1,
            round(
                int(max_sites_per_surveyor)
                * min(5, len(surveyor_cfg.available_dates or []))
                / 5
            ),
        )
        remaining_capacity = (
            effective_capacity
            - counts[allocation.surveyor_name]
        )
        if remaining_capacity <= 0:
            continue

        group = by_cluster.get(allocation.cluster)
        if group is None or group.empty:
            continue

        group = group[~group.index.isin(used_indices)]
        if group.empty:
            continue

        take = min(
            int(allocation.target_sites),
            len(group),
            remaining_capacity,
        )
        chosen = group.head(take).copy()

        chosen["Assigned Surveyor"] = (
            allocation.surveyor_name
        )
        chosen["Home to Cluster (Minutes)"] = (
            allocation.home_to_cluster_minutes
        )
        chosen["AI Cluster Priority"] = (
            allocation.cluster_priority
        )
        chosen["AI Cluster Reason"] = (
            allocation.cluster_reason
        )

        parts[allocation.surveyor_name].append(chosen)
        used_indices.update(chosen.index)
        counts[allocation.surveyor_name] += len(chosen)

    for surveyor in surveyors:
        if parts[surveyor.name]:
            df = pd.concat(
                parts[surveyor.name],
                ignore_index=False,
            ).drop_duplicates()
            helper = [
                c for c in df.columns if c.startswith("_")
            ]
            effective_capacity = max(
                1,
                round(
                    int(max_sites_per_surveyor)
                    * min(5, len(surveyor.available_dates or []))
                    / 5
                ),
            )
            output[surveyor.name] = df.drop(
                columns=helper,
                errors="ignore",
            ).head(effective_capacity)

    return output


def allocations_dataframe(
    allocations: Sequence[ClusterAllocation],
) -> pd.DataFrame:
    if not allocations:
        return pd.DataFrame()
    return pd.DataFrame([asdict(a) for a in allocations])
