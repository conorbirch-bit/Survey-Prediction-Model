from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re
from difflib import SequenceMatcher

import pandas as pd

from coordinate_clustering import (
    NO_GOOGLE_RADIUS_KM,
    estimate_local_transfer_minutes,
    haversine_km,
    should_bypass_google_between_sites,
)


# ============================================================================
# SAME-ROAD TEAM TRAVEL LEEWAY RULE
# ============================================================================
SAME_ROAD_COORDINATE_FALLBACK_KM = 0.20
_ROAD_SUFFIX_CANONICAL = {
    "road":"road", "rd":"road", "street":"street", "st":"street",
    "avenue":"avenue", "ave":"avenue", "lane":"lane", "ln":"lane",
    "way":"way", "close":"close", "crescent":"crescent", "cres":"crescent",
    "drive":"drive", "dr":"drive", "gardens":"gardens", "garden":"gardens",
    "grove":"grove", "hill":"hill", "park":"park", "place":"place", "pl":"place",
    "square":"square", "terrace":"terrace", "walk":"walk", "row":"row",
    "mews":"mews", "parade":"parade", "rise":"rise", "vale":"vale", "view":"view",
    "boulevard":"boulevard", "approach":"approach", "green":"green", "common":"common",
    "broadway":"broadway", "circus":"circus", "causeway":"causeway", "bank":"bank",
    "quay":"quay", "wharf":"wharf", "yard":"yard",
}
_UK_POSTCODE_AT_END_RE = re.compile(r"\b(?:GIR\s?0AA|(?:[A-Z]{1,2}\d[A-Z\d]?)\s?\d[A-Z]{2})\s*$", re.IGNORECASE)
_ROAD_PHRASE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z'&.\-]*(?:\s+[A-Za-z][A-Za-z'&.\-]*){0,5})\s+(" +
    "|".join(sorted((re.escape(v) for v in _ROAD_SUFFIX_CANONICAL), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
def _normalise_road_text(value: str) -> str:
    value=str(value or "").lower().replace("&"," and ")
    value=re.sub(r"[^a-z0-9'\s-]+"," ",value)
    return re.sub(r"\s+"," ",value).strip()
def infer_road_name_from_building_name(building_name: str):
    raw=str(building_name or "").strip()
    if not raw: return None
    parts=[part.strip() for part in re.split(r"[|,]",raw) if part and part.strip()]
    for part in reversed(parts):
        candidate=_UK_POSTCODE_AT_END_RE.sub("",part).strip()
        candidate=re.sub(r"^\s*(?:[A-Z](?:-[A-Z])?\s*)?(?:\d+[A-Z]?(?:\s*[-/]\s*\d+[A-Z]?)?\s*)+","",candidate,flags=re.IGNORECASE).strip()
        matches=list(_ROAD_PHRASE_RE.finditer(candidate))
        if not matches: continue
        m=matches[-1]; root=_normalise_road_text(m.group(1)); suffix=_ROAD_SUFFIX_CANONICAL.get(m.group(2).lower())
        if not root or not suffix: continue
        if root in {"block","house","court","building","community","estate"}: continue
        return f"{root} {suffix}"
    return None
def should_remove_team_travel_leeway(current_site: dict, next_site: dict, fallback_radius_km: float=SAME_ROAD_COORDINATE_FALLBACK_KM) -> bool:
    current_road=infer_road_name_from_building_name(current_site.get("building_name",""))
    next_road=infer_road_name_from_building_name(next_site.get("building_name",""))
    if current_road is not None and next_road is not None:
        return current_road == next_road
    distance_km=haversine_km(current_site.get("latitude"),current_site.get("longitude"),next_site.get("latitude"),next_site.get("longitude"))
    return distance_km is not None and distance_km <= float(fallback_radius_km)


# ============================================================================
# LOCAL PROXIMITY GRAPH ORDERING
# ============================================================================
# The strategic Planning Cluster is unchanged. Within it, sites connected by
# successive <= NO_GOOGLE_RADIUS_KM coordinate hops form one local group.


def _planning_cluster_key(site: dict) -> str:
    return str(
        site.get("planning_cluster", "")
        or postcode_district(site.get("postcode", ""))
    )


def _coordinate_edge_distance_km(
    site_a: dict,
    site_b: dict,
    radius_km: float = NO_GOOGLE_RADIUS_KM,
):
    distance = haversine_km(
        site_a.get("latitude"), site_a.get("longitude"),
        site_b.get("latitude"), site_b.get("longitude"),
    )
    if distance is None or distance > float(radius_km):
        return None
    return float(distance)


def _connected_components_for_indices(
    sites: List[dict],
    indices: Sequence[int],
    radius_km: float = NO_GOOGLE_RADIUS_KM,
) -> List[List[int]]:
    """Connected local groups inside the existing strategic cluster."""
    indices = list(indices)
    adjacency = {idx: set() for idx in indices}

    for pos, idx_a in enumerate(indices):
        cluster_a = _planning_cluster_key(sites[idx_a])
        for idx_b in indices[pos + 1:]:
            if cluster_a != _planning_cluster_key(sites[idx_b]):
                continue

            linked = (
                _coordinate_edge_distance_km(
                    sites[idx_a], sites[idx_b], radius_km
                )
                is not None
            )

            # Preserve the old exact-postcode / legacy-campus collapse signals.
            if not linked:
                linked = bool(
                    should_bypass_google_between_sites(
                        sites[idx_a], sites[idx_b], radius_km=radius_km
                    )[0]
                )
            if not linked:
                try:
                    linked = same_campus(
                        sites[idx_a].get("building_name", ""),
                        sites[idx_a].get("postcode", ""),
                        sites[idx_b].get("building_name", ""),
                        sites[idx_b].get("postcode", ""),
                    )
                except Exception:
                    linked = False

            if linked:
                adjacency[idx_a].add(idx_b)
                adjacency[idx_b].add(idx_a)

    components = []
    unseen = set(indices)
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in sorted(adjacency[node], reverse=True):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))

    components.sort(key=lambda values: values[0])
    return components


def _coordinate_shortest_paths_from_current(
    current_site: dict,
    sites: List[dict],
    candidate_indices: Sequence[int],
    radius_km: float = NO_GOOGLE_RADIUS_KM,
) -> Dict[int, float]:
    """Shortest chained local distance from current site to connected sites."""
    current_cluster = _planning_cluster_key(current_site)
    if not current_cluster:
        return {}

    eligible = [
        idx
        for idx in candidate_indices
        if _planning_cluster_key(sites[idx]) == current_cluster
    ]
    nodes = [-1] + eligible
    node_site = {-1: current_site}
    node_site.update({idx: sites[idx] for idx in eligible})
    adjacency = {node: [] for node in nodes}

    for pos, node_a in enumerate(nodes):
        for node_b in nodes[pos + 1:]:
            edge = _coordinate_edge_distance_km(
                node_site[node_a], node_site[node_b], radius_km
            )
            if edge is not None:
                adjacency[node_a].append((node_b, edge))
                adjacency[node_b].append((node_a, edge))

    distances = {-1: 0.0}
    visited = set()
    while True:
        choices = [
            (distance, node)
            for node, distance in distances.items()
            if node not in visited
        ]
        if not choices:
            break
        distance, node = min(choices)
        visited.add(node)
        for neighbour, edge in adjacency[node]:
            candidate_distance = distance + edge
            if candidate_distance < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate_distance

    return {
        idx: float(distance)
        for idx, distance in distances.items()
        if idx >= 0
    }


def _same_confident_road(site_a: dict, site_b: dict) -> bool:
    road_a = infer_road_name_from_building_name(
        site_a.get("building_name", "")
    )
    road_b = infer_road_name_from_building_name(
        site_b.get("building_name", "")
    )
    return (
        road_a is not None
        and road_b is not None
        and road_a == road_b
    )

def postcode_district(postcode: str) -> str:
    text = str(postcode or "").strip().upper()
    if not text:
        return ""
    # UK outward code, e.g. HA8, NW9, SW1V, SG14.
    # Standard formatted UK postcodes place a space before the inward code.
    parts = text.split()
    if len(parts) >= 2:
        return parts[0]
    # Fallback for compact postcodes: remove the final three-character inward code.
    compact = re.sub(r"\s+", "", text)
    return compact[:-3] if len(compact) > 3 else compact


def _normalise_postcode(postcode: str) -> str:
    return re.sub(r"\s+", "", str(postcode or "").upper())


def _normalise_site_name(name: str, postcode: str = "") -> str:
    """
    Reduce Salesforce building/address text to a comparable campus signature.

    Examples:
      "102831 | A-B, 155 Cambridge Street SW1V 4QB"
      "101100 | 157 Cambridge Street SW1V 4QB"

    both become close to "cambridge street".
    """
    text = str(name or "").lower()

    # Remove a leading Salesforce / asset reference before the pipe.
    if "|" in text:
        text = text.split("|", 1)[1]

    pc = _normalise_postcode(postcode).lower()
    compact_text = re.sub(r"\s+", "", text)
    if pc and pc in compact_text:
        # Remove postcode in a spacing-insensitive way by first replacing the
        # normally formatted postcode variants.
        formatted = str(postcode or "").lower().strip()
        if formatted:
            text = text.replace(formatted, " ")

    # Remove flat numbers, street numbers and punctuation so nearby blocks with
    # the same underlying estate/address wording compare well.
    text = re.sub(r"\b\d+[a-z]?\b", " ", text)
    text = re.sub(r"\b[a-z]\s*-\s*[a-z]\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    # These tokens commonly vary between blocks on the same campus but do not
    # materially identify a different site.
    weak_tokens = {
        "block", "blocks", "building", "buildings",
        "flat", "flats", "car", "park", "parking",
        "entrance", "wing", "tower",
    }

    tokens = [
        token
        for token in text.split()
        if len(token) > 1 and token not in weak_tokens
    ]
    return " ".join(tokens)


def same_campus(
    name_a: str,
    postcode_a: str,
    name_b: str,
    postcode_b: str,
    similarity_threshold: float = 0.62,
) -> bool:
    """
    Treat two buildings as the same campus/site group when:
      1) their full postcodes match; and
      2) their normalised building/address names are similar.

    This is intentionally conservative: postcode match is mandatory.
    """
    pc_a = _normalise_postcode(postcode_a)
    pc_b = _normalise_postcode(postcode_b)
    if not pc_a or pc_a != pc_b:
        return False

    a = _normalise_site_name(name_a, postcode_a)
    b = _normalise_site_name(name_b, postcode_b)
    if not a or not b:
        return False

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    # Strong signal: one address signature is contained within the other.
    if tokens_a.issubset(tokens_b) or tokens_b.issubset(tokens_a):
        if len(tokens_a & tokens_b) >= 2:
            return True

    jaccard = (
        len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        if (tokens_a | tokens_b)
        else 0.0
    )
    sequence = SequenceMatcher(None, a, b).ratio()

    return max(jaccard, sequence) >= similarity_threshold


@dataclass
class ScheduledSurvey:
    sequence: int
    customer_reference: str
    building_name: str
    postcode: str
    cluster: str
    depart_previous: datetime
    travel_minutes: float
    arrive_site: datetime
    survey_minutes: float
    survey_start: datetime
    survey_end: datetime
    predicted_minutes: float
    confidence: str
    model_used: str
    building_height: Optional[float] = None
    flats: Optional[float] = None
    ground_floor_area: Optional[float] = None


@dataclass
class DailyScheduleResult:
    items: List[ScheduledSurvey]
    start_location: str
    start_time: datetime
    return_location: str
    return_departure: datetime
    return_travel_minutes: float
    return_time: datetime
    latest_return: datetime
    unscheduled_count: int
    first_survey_target: Optional[datetime] = None
    latest_survey_finish: Optional[datetime] = None
    lunch_start: Optional[datetime] = None
    lunch_end: Optional[datetime] = None
    lunch_location: str = ""
    lunch_after_sequence: Optional[int] = None
    routing_stats: Optional[dict] = None

    @property
    def survey_minutes(self) -> float:
        return round(sum(i.survey_minutes for i in self.items), 1)

    @property
    def travel_minutes(self) -> int:
        return round(
            sum(i.travel_minutes for i in self.items)
            + self.return_travel_minutes
        )

    def to_dataframe(self) -> pd.DataFrame:
        rows = []

        def lunch_row():
            return {
                "Sequence": "LUNCH",
                "Customer Reference": "",
                "Building Name": (
                    f"Lunch break — {self.lunch_location}"
                    if self.lunch_location
                    else "Lunch break"
                ),
                "Postcode": "",
                "Cluster": "",
                "Depart Previous": "",
                "Travel (Minutes)": "",
                "Arrive": "",
                "Survey Start": (
                    self.lunch_start.strftime("%H:%M")
                    if self.lunch_start
                    else ""
                ),
                "Survey End": (
                    self.lunch_end.strftime("%H:%M")
                    if self.lunch_end
                    else ""
                ),
                "Planning Survey Duration (Minutes)": 30,
                "Raw Predicted Duration (Minutes)": "",
                "Building Height": "",
                "Sovereign Flat": "",
                "Internal Ground Floor Area (m2)": "",
                "Prediction Confidence": "",
                "Prediction Model": "Lunch break",
            }

        lunch_inserted = False
        if (
            self.lunch_start is not None
            and self.lunch_after_sequence == 0
        ):
            rows.append(lunch_row())
            lunch_inserted = True

        for item in self.items:
            rows.append({
                "Sequence": item.sequence,
                "Customer Reference": item.customer_reference,
                "Building Name": item.building_name,
                "Postcode": item.postcode,
                "Cluster": item.cluster,
                "Depart Previous": item.depart_previous.strftime("%H:%M"),
                "Travel (Minutes)": round(item.travel_minutes),
                "Arrive": item.arrive_site.strftime("%H:%M"),
                "Survey Start": item.survey_start.strftime("%H:%M"),
                "Survey End": item.survey_end.strftime("%H:%M"),
                "Planning Survey Duration (Minutes)": item.survey_minutes,
                "Raw Predicted Duration (Minutes)": item.predicted_minutes,
                "Building Height": item.building_height,
                "Sovereign Flat": item.flats,
                "Internal Ground Floor Area (m2)": item.ground_floor_area,
                "Prediction Confidence": item.confidence,
                "Prediction Model": item.model_used,
            })

            if (
                self.lunch_start is not None
                and not lunch_inserted
                and self.lunch_after_sequence == item.sequence
            ):
                rows.append(lunch_row())
                lunch_inserted = True

        if self.lunch_start is not None and not lunch_inserted:
            rows.append(lunch_row())

        rows.append({
            "Sequence": "RETURN",
            "Customer Reference": "",
            "Building Name": self.return_location,
            "Postcode": "",
            "Cluster": "",
            "Depart Previous": self.return_departure.strftime("%H:%M"),
            "Travel (Minutes)": round(self.return_travel_minutes),
            "Arrive": self.return_time.strftime("%H:%M"),
            "Survey Start": "",
            "Survey End": "",
            "Planning Survey Duration (Minutes)": "",
            "Raw Predicted Duration (Minutes)": "",
            "Building Height": "",
            "Sovereign Flat": "",
            "Internal Ground Floor Area (m2)": "",
            "Prediction Confidence": "",
            "Prediction Model": "",
        })
        return pd.DataFrame(rows)



@dataclass
class WeeklyScheduleResult:
    days: List[DailyScheduleResult]
    unscheduled_sites: List[dict]

    @property
    def total_surveys(self) -> int:
        return sum(len(day.items) for day in self.days)

    @property
    def total_survey_minutes(self) -> float:
        return sum(day.survey_minutes for day in self.days)

    @property
    def total_travel_minutes(self) -> int:
        return sum(day.travel_minutes for day in self.days)

    @property
    def routing_stats(self) -> dict:
        """Aggregate reporting-only routing counters across the final week."""
        keys = {
            "candidate_evaluations",
            "google_matrix_logical_calls",
            "google_matrix_destinations",
            "same_postcode_bypasses",
            "coordinate_radius_bypasses",
            "legacy_campus_bypasses",
            "collapsed_nearby_destinations",
        }
        totals = {key: 0 for key in keys}
        for day in self.days:
            for key, value in (day.routing_stats or {}).items():
                if key in totals:
                    totals[key] += int(value or 0)
        totals["local_bypasses"] = (
            totals["same_postcode_bypasses"]
            + totals["coordinate_radius_bypasses"]
            + totals["legacy_campus_bypasses"]
        )
        totals["routing_calculations_avoided"] = (
            totals["local_bypasses"]
            + totals["collapsed_nearby_destinations"]
        )
        denominator = (
            totals["google_matrix_destinations"]
            + totals["routing_calculations_avoided"]
        )
        totals["site_to_site_avoidance_pct"] = (
            100.0 * totals["routing_calculations_avoided"] / denominator
            if denominator
            else 0.0
        )
        return totals

    def summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for day in self.days:
            clusters = []
            for item in day.items:
                if item.cluster and item.cluster not in clusters:
                    clusters.append(item.cluster)

            service_dt = day.first_survey_target or day.start_time
            first_actual = day.items[0].survey_start if day.items else None
            last_actual = day.items[-1].survey_end if day.items else None
            rows.append({
                "Date": service_dt.strftime("%A %d %B %Y"),
                "Surveys": len(day.items),
                "Clusters": " → ".join(clusters),
                "Leave Harpenden": day.start_time.strftime("%H:%M"),
                "First Survey": first_actual.strftime("%H:%M") if first_actual else "",
                "Last Survey Finish": last_actual.strftime("%H:%M") if last_actual else "",
                "Return Harpenden": day.return_time.strftime("%H:%M"),
                "Survey Time (Minutes)": day.survey_minutes,
                "Travel Time (Minutes)": day.travel_minutes,
            })

        return pd.DataFrame(rows)

    def full_schedule_dataframe(self) -> pd.DataFrame:
        frames = []
        for day in self.days:
            df = day.to_dataframe().copy()
            service_dt = day.first_survey_target or day.start_time
            df.insert(0, "Date", service_dt.date().isoformat())
            df.insert(1, "Day", service_dt.strftime("%A"))
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)


class DailyTransitScheduler:
    """
    Greedy, time-dependent public-transport scheduler.

    Latitude/Longitude provide the cheap geographic layer before Google.
    Buildings within the configured no-Google radius, or with the same full
    postcode, bypass Google for their site-to-site move and use a conservative
    local walking/transfer estimate instead. Inside each existing strategic
    Planning Cluster, successive local coordinate links form connected groups,
    which are traversed nearest-next before leaving for another local group.

    Google remains the final public-transport authority for meaningful journeys,
    first-site travel and return-home feasibility. Hard time/lunch rules are
    unchanged.
    """

    def __init__(
        self,
        router,
        home_location: str = "Harpenden Station",
        max_candidate_checks: int = 8,
        same_postcode_transfer_minutes: int = 5,
        travel_leeway_minutes: int = 5,
        pre_survey_buffer_minutes: int = 5,
        post_survey_buffer_minutes: int = 5,
        ai_priority_weight_minutes: float = 20.0,
        lunch_minutes: int = 30,
        lunch_window_start_clock=time(11, 45),
        lunch_latest_start_clock=time(13, 0),
    ):
        self.router = router
        self.home_location = home_location
        self.max_candidate_checks = max_candidate_checks
        self.same_postcode_transfer_minutes = same_postcode_transfer_minutes
        self.travel_leeway_minutes = travel_leeway_minutes
        self.pre_survey_buffer_minutes = pre_survey_buffer_minutes
        self.post_survey_buffer_minutes = post_survey_buffer_minutes
        self.ai_priority_weight_minutes = ai_priority_weight_minutes
        self.lunch_minutes = lunch_minutes
        self.lunch_window_start_clock = lunch_window_start_clock
        self.lunch_latest_start_clock = lunch_latest_start_clock

    @staticmethod
    def _same_postcode(a: str, b: str) -> bool:
        normalise = lambda x: re.sub(r"\s+", "", str(x or "").upper())
        return bool(normalise(a)) and normalise(a) == normalise(b)

    def _site_to_site_leeway_minutes(self, current_site: dict, next_site: dict) -> float:
        """Apply the same-road / <=200 m exception only to site-to-site travel."""
        if should_remove_team_travel_leeway(current_site, next_site):
            return 0.0
        return float(self.travel_leeway_minutes)

    def build_day(
        self,
        sites: List[dict],
        first_survey_start: datetime,
        latest_survey_finish: datetime,
        latest_return: datetime,
    ) -> DailyScheduleResult:
        remaining = [dict(site) for site in sites]
        scheduled: List[ScheduledSurvey] = []

        # Reporting only. Counts the routing work the existing algorithm performs.
        routing_stats = {
            "candidate_evaluations": 0,
            "google_matrix_logical_calls": 0,
            "google_matrix_destinations": 0,
            "same_postcode_bypasses": 0,
            "coordinate_radius_bypasses": 0,
            "legacy_campus_bypasses": 0,
            "collapsed_nearby_destinations": 0,
        }

        current_location = self.home_location
        current_postcode = ""
        current_building_name = ""
        current_latitude = None
        current_longitude = None
        current_planning_cluster = ""

        # For the first leg, Google needs a departure time to price/rank transit.
        # Probe shortly before the requested first-survey start, then back-calculate
        # the actual home departure for the chosen first site. This stops a long
        # commute from consuming the user's survey-work window.
        current_time = first_survey_start - timedelta(minutes=90)
        home_departure_time = current_time

        lunch_taken = False
        lunch_start = None
        lunch_end = None
        lunch_location = ""
        lunch_after_sequence = None
        lunch_window_start = datetime.combine(
            first_survey_start.date(),
            self.lunch_window_start_clock,
            tzinfo=first_survey_start.tzinfo,
        )
        lunch_latest_start = datetime.combine(
            first_survey_start.date(),
            self.lunch_latest_start_clock,
            tzinfo=first_survey_start.tzinfo,
        )

        while remaining:
            # Take lunch at the first natural between-survey boundary from 11:45.
            # The break must START by 13:00. It is a hard constraint, not an AI
            # preference, and therefore cannot be skipped to fit another survey.
            if (
                not lunch_taken
                and lunch_window_start <= current_time <= lunch_latest_start
            ):
                lunch_start = current_time
                lunch_end = lunch_start + timedelta(minutes=self.lunch_minutes)
                lunch_location = current_location
                lunch_after_sequence = len(scheduled)
                current_time = lunch_end
                lunch_taken = True

            # This should only be reachable if no feasible pre-13:00 lunch could
            # be protected. Do not schedule additional work after missing lunch.
            if not lunch_taken and current_time > lunch_latest_start:
                break
            # Coordinate-first Google reduction.
            #
            # After the first site is reached, a remaining building is NOT sent
            # to the Google Route Matrix for the site-to-site move when:
            #   - it shares the exact full postcode; OR
            #   - its coordinates are within NO_GOOGLE_RADIUS_KM of the current site.
            # The old same-campus name check remains as an additional fallback.
            #
            # External destinations are also collapsed when they are physically
            # close to one another. Google therefore sees one representative for
            # a micro-area instead of many nearly identical nearby buildings.
            routing_stats["candidate_evaluations"] += len(remaining)
            travel_by_index = {}
            external_indices = []

            current_site = {
                "building_name": current_building_name,
                "postcode": current_postcode,
                "latitude": current_latitude,
                "longitude": current_longitude,
                "planning_cluster": current_planning_cluster,
            }

            connected_path_km = (
                _coordinate_shortest_paths_from_current(
                    current_site,
                    remaining,
                    range(len(remaining)),
                    radius_km=NO_GOOGLE_RADIUS_KM,
                )
                if scheduled
                else {}
            )

            for idx, site in enumerate(remaining):
                bypass = False
                distance = None
                bypass_reason = ""
                if scheduled:
                    bypass, distance, bypass_reason = (
                        should_bypass_google_between_sites(
                            current_site,
                            site,
                            radius_km=NO_GOOGLE_RADIUS_KM,
                        )
                    )
                    if not bypass and current_postcode:
                        try:
                            bypass = same_campus(
                                current_building_name,
                                current_postcode,
                                site.get("building_name", ""),
                                site.get("postcode", ""),
                            )
                            if bypass:
                                bypass_reason = "legacy same-campus rule"
                        except Exception:
                            bypass = False

                if bypass:
                    if bypass_reason == "same full postcode":
                        routing_stats["same_postcode_bypasses"] += 1
                    elif bypass_reason == "legacy same-campus rule":
                        routing_stats["legacy_campus_bypasses"] += 1
                    else:
                        routing_stats["coordinate_radius_bypasses"] += 1

                    travel_by_index[idx] = estimate_local_transfer_minutes(
                        distance,
                        minimum_minutes=float(self.same_postcode_transfer_minutes),
                    )
                elif idx in connected_path_km:
                    # Reachable through a chain of <=radius local coordinate hops.
                    routing_stats["collapsed_nearby_destinations"] += 1
                    travel_by_index[idx] = estimate_local_transfer_minutes(
                        connected_path_km[idx],
                        minimum_minutes=float(self.same_postcode_transfer_minutes),
                    )
                else:
                    external_indices.append(idx)

            # Google now sees one representative per CONNECTED local component
            # rather than one representative-radius pocket.
            proximity_groups = []
            for component in _connected_components_for_indices(
                remaining,
                external_indices,
                radius_km=NO_GOOGLE_RADIUS_KM,
            ):
                representative_idx = component[0]
                proximity_groups.append({
                    "indices": component,
                    "representative_idx": representative_idx,
                    "route_location": remaining[representative_idx]["route_location"],
                })

            google_representative_indices = {
                group["representative_idx"]
                for group in proximity_groups
            }
            google_group_member_indices = {
                idx
                for group in proximity_groups
                for idx in group["indices"]
            }

            if proximity_groups:
                # Every extra member of a proximity group is one destination
                # calculation that the coordinate/postcode logic avoided.
                routing_stats["collapsed_nearby_destinations"] += sum(
                    max(0, len(group["indices"]) - 1)
                    for group in proximity_groups
                )

                google_destinations = [
                    group["route_location"]
                    for group in proximity_groups
                ]
                routing_stats["google_matrix_logical_calls"] += 1
                routing_stats["google_matrix_destinations"] += len(
                    google_destinations
                )

                google_matrix = self.router.one_to_many(
                    current_location,
                    google_destinations,
                    current_time,
                )

                for group, minutes in zip(
                    proximity_groups,
                    google_matrix,
                ):
                    for idx in group["indices"]:
                        travel_by_index[idx] = minutes

            ranked = []
            for idx, site in enumerate(remaining):
                minutes = travel_by_index.get(idx)

                if minutes is None:
                    continue

                # Strategic cluster first; inside it, finish the connected local
                # group nearest-next and favour a continuous same-road sequence.
                cluster_tier = 0
                local_group_tier = 1
                same_road_tier = 1
                local_distance_sort = float("inf")
                postcode_tier = 0
                google_representative_tier = 0

                if scheduled:
                    next_cluster = str(
                        site.get("planning_cluster", "")
                        or postcode_district(site.get("postcode"))
                    )
                    cluster_tier = (
                        0
                        if current_planning_cluster
                        and current_planning_cluster == next_cluster
                        else 1
                    )
                    postcode_tier = (
                        0
                        if self._same_postcode(
                            current_postcode, site.get("postcode", "")
                        )
                        else 1
                    )

                    if idx in connected_path_km:
                        local_group_tier = 0
                        direct_distance = haversine_km(
                            current_latitude,
                            current_longitude,
                            site.get("latitude"),
                            site.get("longitude"),
                        )
                        local_distance_sort = (
                            float(direct_distance)
                            if direct_distance is not None
                            else float(connected_path_km[idx])
                        )
                        if _same_confident_road(current_site, site):
                            same_road_tier = 0
                    else:
                        direct_bypass, direct_distance, _ = (
                            should_bypass_google_between_sites(
                                current_site,
                                site,
                                radius_km=NO_GOOGLE_RADIUS_KM,
                            )
                        )
                        if direct_bypass:
                            local_group_tier = 0
                            if direct_distance is not None:
                                local_distance_sort = float(direct_distance)
                            if _same_confident_road(current_site, site):
                                same_road_tier = 0
                elif idx in google_group_member_indices:
                    google_representative_tier = (
                        0 if idx in google_representative_indices else 1
                    )

                # AI priority is advisory: Google transit remains dominant.
                # A 100-point AI priority can improve the candidate score by at
                # most ai_priority_weight_minutes; a 0-point priority adds no bonus.
                try:
                    ai_priority = float(site.get("ai_priority", 50))
                except Exception:
                    ai_priority = 50.0
                ai_priority = max(0.0, min(100.0, ai_priority))
                # Centre the AI effect at 50. High-priority sites become
                # somewhat more attractive; low-priority/deferred sites become
                # somewhat less attractive. The absolute effect remains capped
                # so AI cannot overpower real transit geography.
                ai_adjustment = (
                    (ai_priority - 50.0) / 50.0
                ) * self.ai_priority_weight_minutes

                defer_penalty = (
                    15.0
                    if str(site.get("ai_decision", "")).lower() == "defer"
                    else 0.0
                )

                # Weekly notes are a preference layer only. A preferred site is
                # fully reserved until its requested date, then strongly favoured
                # on that date. The normal feasibility/return-home checks below
                # still decide whether it can actually be scheduled.
                special_request_adjustment = 0.0
                preferred_date = site.get("special_request_date")
                if preferred_date:
                    try:
                        if hasattr(preferred_date, "date"):
                            preferred_date = preferred_date.date()
                        current_day = first_survey_start.date()
                        if current_day < preferred_date:
                            # Do not consume a request-area site on an earlier day.
                            # If the note trial later fails, the entire trial is
                            # discarded and the baseline schedule is restored.
                            continue
                        elif current_day == preferred_date:
                            special_request_adjustment -= float(
                                site.get("special_request_bonus_minutes", 75.0)
                            )
                    except Exception:
                        pass

                score = (
                    float(minutes)
                    - ai_adjustment
                    + defer_penalty
                    + special_request_adjustment
                )
                ranked.append(
                    (
                        cluster_tier,
                        local_group_tier,
                        same_road_tier,
                        local_distance_sort,
                        postcode_tier,
                        google_representative_tier,
                        score,
                        idx,
                        float(minutes),
                    )
                )

            if not ranked:
                break

            # Coordinate-cluster tier first, then real transit economics.
            #
            # To preserve robustness, always keep up to two out-of-cluster
            # fallback candidates in the feasibility pool. That means an
            # impossible local job cannot prematurely end the day, while a
            # distant hop cannot beat feasible nearby work merely because its
            # transit time happens to be a few minutes shorter.
            ranked.sort(
                key=lambda x: (
                    x[0], x[1], x[2], x[3], x[4], x[5], x[6]
                )
            )

            if scheduled:
                same_cluster = [r for r in ranked if r[0] == 0]
                other_cluster = [r for r in ranked if r[0] != 0]

                fallback_slots = (
                    min(2, len(other_cluster))
                    if same_cluster
                    else min(self.max_candidate_checks, len(other_cluster))
                )
                local_slots = max(
                    0,
                    self.max_candidate_checks - fallback_slots,
                )
                candidate_pool = (
                    same_cluster[:local_slots]
                    + other_cluster[:fallback_slots]
                )
            else:
                candidate_pool = ranked[:self.max_candidate_checks]

            chosen = None
            lunch_blocked_candidate = False

            for (
                _, _, _, _, _, _, _, idx, travel_minutes
            ) in candidate_pool:
                site = remaining[idx]

                is_first_survey = len(scheduled) == 0

                if is_first_survey:
                    # Back-calculate home departure so the first survey begins at
                    # the selected target rather than after an arbitrary fixed
                    # "leave home" time. Use one exact Compute Routes call only for
                    # the shortlisted candidate being feasibility-tested.
                    estimated_departure = first_survey_start - timedelta(
                        minutes=(
                            travel_minutes
                            + self.travel_leeway_minutes
                            + self.pre_survey_buffer_minutes
                        )
                    )
                    try:
                        outbound_route = self.router.compute_route(
                            self.home_location,
                            site["route_location"],
                            estimated_departure,
                        )
                    except Exception:
                        continue

                    buffered_travel_minutes = (
                        float(outbound_route.duration_minutes)
                        + self.travel_leeway_minutes
                    )
                    # Re-anchor the displayed/operational home departure using
                    # the exact route duration we just received so the first
                    # survey starts at the selected time rather than merely
                    # "not before" it. This does not add another Google call.
                    depart_previous = first_survey_start - timedelta(
                        minutes=(
                            buffered_travel_minutes
                            + self.pre_survey_buffer_minutes
                        )
                    )
                    arrive = depart_previous + timedelta(
                        minutes=buffered_travel_minutes
                    )
                    survey_start = first_survey_start
                else:
                    site_to_site_leeway = self._site_to_site_leeway_minutes(
                        current_site, site
                    )
                    buffered_travel_minutes = travel_minutes + site_to_site_leeway
                    depart_previous = current_time
                    arrive = current_time + timedelta(
                        minutes=buffered_travel_minutes
                    )
                    survey_start = arrive + timedelta(
                        minutes=self.pre_survey_buffer_minutes
                    )

                survey_end = survey_start + timedelta(
                    minutes=float(site["planning_minutes"])
                )

                # Independent hard deadline: the final survey itself cannot end
                # after the user's selected cut-off, regardless of the journey home.
                if survey_end > latest_survey_finish:
                    continue

                ready_to_leave = survey_end + timedelta(
                    minutes=self.post_survey_buffer_minutes
                )

                # If we have not yet had lunch and taking this job would keep the
                # surveyor occupied beyond the latest allowed lunch START (13:00),
                # do not commit it. If no shorter feasible candidate exists, the
                # scheduler will reserve lunch at 11:45 and then re-route.
                if (
                    not lunch_taken
                    and current_time < lunch_window_start
                    and ready_to_leave > lunch_latest_start
                ):
                    lunch_blocked_candidate = True
                    continue

                # If this survey finishes during the lunch-start window, include
                # the 30-minute break in the return-home feasibility calculation.
                # This prevents a site being accepted only because lunch was
                # silently omitted from the hard return deadline.
                return_departure_for_check = ready_to_leave
                if (
                    not lunch_taken
                    and lunch_window_start <= ready_to_leave <= lunch_latest_start
                ):
                    return_departure_for_check = (
                        ready_to_leave
                        + timedelta(minutes=self.lunch_minutes)
                    )

                # Hard feasibility check using a real public-transport route home
                # at the time this particular survey (and, where required, lunch)
                # would finish.
                try:
                    return_route = self.router.compute_route(
                        site["route_location"],
                        self.home_location,
                        return_departure_for_check,
                    )
                except Exception:
                    continue

                return_time = return_departure_for_check + timedelta(
                    minutes=return_route.duration_minutes
                    + self.travel_leeway_minutes
                )

                if return_time <= latest_return:
                    chosen = (
                        idx,
                        site,
                        buffered_travel_minutes,
                        arrive,
                        survey_start,
                        survey_end,
                        ready_to_leave,
                        depart_previous,
                    )
                    break

            if chosen is None:
                # All otherwise-attractive candidates would cause lunch to be
                # missed. Reserve lunch at 11:45, then re-run routing at 12:15.
                if (
                    lunch_blocked_candidate
                    and not lunch_taken
                    and current_time < lunch_window_start
                ):
                    lunch_start = lunch_window_start
                    lunch_end = lunch_start + timedelta(minutes=self.lunch_minutes)
                    lunch_location = current_location
                    lunch_after_sequence = len(scheduled)
                    current_time = lunch_end
                    lunch_taken = True
                    continue
                break

            (
                idx,
                site,
                travel_minutes,
                arrive,
                survey_start,
                survey_end,
                ready_to_leave,
                depart_previous,
            ) = chosen

            if not scheduled:
                home_departure_time = depart_previous

            scheduled.append(
                ScheduledSurvey(
                    sequence=len(scheduled) + 1,
                    customer_reference=str(site.get("customer_reference", "")),
                    building_name=str(site.get("building_name", "")),
                    postcode=str(site.get("postcode", "")),
                    cluster=str(
                        site.get("planning_cluster", "")
                        or postcode_district(site.get("postcode", ""))
                    ),
                    depart_previous=depart_previous,
                    travel_minutes=travel_minutes,
                    arrive_site=arrive,
                    survey_minutes=float(site["planning_minutes"]),
                    survey_start=survey_start,
                    survey_end=survey_end,
                    predicted_minutes=float(site["predicted_minutes"]),
                    confidence=str(site.get("confidence", "")),
                    model_used=str(site.get("model_used", "")),
                    building_height=site.get("building_height"),
                    flats=site.get("flats"),
                    ground_floor_area=site.get("ground_floor_area"),
                )
            )

            current_location = site["route_location"]
            current_postcode = site.get("postcode", "")
            current_building_name = site.get("building_name", "")
            current_latitude = site.get("latitude")
            current_longitude = site.get("longitude")
            current_planning_cluster = str(
                site.get("planning_cluster", "")
                or postcode_district(site.get("postcode", ""))
            )
            current_time = ready_to_leave
            remaining.pop(idx)

        # If the final survey finishes during the lunch window, take the break
        # before the final journey home. A day that genuinely finishes before
        # 11:45 does not need a lunch break inserted.
        if (
            scheduled
            and not lunch_taken
            and lunch_window_start <= current_time <= lunch_latest_start
        ):
            lunch_start = current_time
            lunch_end = lunch_start + timedelta(minutes=self.lunch_minutes)
            lunch_location = current_location
            lunch_after_sequence = len(scheduled)
            current_time = lunch_end
            lunch_taken = True

        # Always calculate the final journey home from the last site.
        if scheduled:
            final_route = self.router.compute_route(
                current_location,
                self.home_location,
                current_time,
            )
            return_minutes = (
                final_route.duration_minutes + self.travel_leeway_minutes
            )
            return_time = current_time + timedelta(minutes=return_minutes)
        else:
            return_minutes = 0.0
            return_time = first_survey_start
            home_departure_time = first_survey_start

        return DailyScheduleResult(
            items=scheduled,
            start_location=self.home_location,
            start_time=home_departure_time,
            return_location=self.home_location,
            return_departure=current_time,
            return_travel_minutes=return_minutes,
            return_time=return_time,
            latest_return=latest_return,
            unscheduled_count=len(remaining),
            first_survey_target=first_survey_start,
            latest_survey_finish=latest_survey_finish,
            lunch_start=lunch_start,
            lunch_end=lunch_end,
            lunch_location=lunch_location,
            lunch_after_sequence=lunch_after_sequence,
            routing_stats=routing_stats,
        )

    def build_week(
        self,
        sites: List[dict],
        dates: Sequence,
        first_survey_start_clock,
        latest_survey_finish_clock,
        latest_return_clock,
        timezone,
    ) -> WeeklyScheduleResult:
        """
        Build a multi-day schedule.

        Each day uses the same live transit-routing logic as build_day.
        Once a site is scheduled it is removed from the candidate pool before
        the next day is planned.
        """
        remaining = [dict(site) for site in sites]
        days: List[DailyScheduleResult] = []

        for day_date in dates:
            if not remaining:
                break

            first_survey_dt = datetime.combine(
                day_date,
                first_survey_start_clock,
                tzinfo=timezone,
            )
            latest_survey_finish_dt = datetime.combine(
                day_date,
                latest_survey_finish_clock,
                tzinfo=timezone,
            )
            return_deadline_dt = datetime.combine(
                day_date,
                latest_return_clock,
                tzinfo=timezone,
            )

            day_result = self.build_day(
                sites=remaining,
                first_survey_start=first_survey_dt,
                latest_survey_finish=latest_survey_finish_dt,
                latest_return=return_deadline_dt,
            )
            days.append(day_result)

            scheduled_refs = {
                item.customer_reference
                for item in day_result.items
                if item.customer_reference
            }

            # Use reference code where available, otherwise building+postcode.
            scheduled_fallback = {
                (item.building_name, item.postcode)
                for item in day_result.items
            }

            new_remaining = []
            for site in remaining:
                ref = str(site.get("customer_reference", ""))
                fallback = (
                    str(site.get("building_name", "")),
                    str(site.get("postcode", "")),
                )

                if ref and ref in scheduled_refs:
                    continue
                if fallback in scheduled_fallback:
                    continue
                new_remaining.append(site)

            # If no progress is possible on this date, retain all sites and move on.
            remaining = new_remaining

        return WeeklyScheduleResult(
            days=days,
            unscheduled_sites=remaining,
        )

