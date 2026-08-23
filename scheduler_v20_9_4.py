from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re
from difflib import SequenceMatcher

import pandas as pd


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

    At every stop it asks Google for current-location -> remaining sites that
    are not already recognised as the same campus. Same-postcode + similar-name
    internal moves bypass Google completely and use the configured fixed transfer
    time. It then tests the best candidates to ensure the survey plus a transit
    journey home still finishes before the hard return deadline.

    Shorter inter-site public-transport journeys are naturally favoured, so the
    chosen route tends to form a geographic/transit cluster without hardcoding
    London areas.
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

    def build_day(
        self,
        sites: List[dict],
        first_survey_start: datetime,
        latest_survey_finish: datetime,
        latest_return: datetime,
    ) -> DailyScheduleResult:
        remaining = [dict(site) for site in sites]
        scheduled: List[ScheduledSurvey] = []

        current_location = self.home_location
        current_postcode = ""
        current_building_name = ""

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
            # Same-campus candidates are deliberately EXCLUDED from the Google
            # Route Matrix. They receive the fixed internal-transfer time instead.
            #
            # Remaining destinations are also collapsed by campus before Google
            # is called. If six candidate buildings share the same full postcode
            # and similar address/name text, Google sees one representative
            # destination and that journey time is reused for all six candidates.
            travel_by_index = {}
            external_indices = []

            for idx, site in enumerate(remaining):
                if (
                    current_postcode
                    and same_campus(
                        current_building_name,
                        current_postcode,
                        site.get("building_name", ""),
                        site.get("postcode", ""),
                    )
                ):
                    travel_by_index[idx] = float(
                        self.same_postcode_transfer_minutes
                    )
                else:
                    external_indices.append(idx)

            campus_groups = []
            for idx in external_indices:
                site = remaining[idx]
                placed = False

                for group in campus_groups:
                    representative = remaining[group["indices"][0]]
                    if same_campus(
                        representative.get("building_name", ""),
                        representative.get("postcode", ""),
                        site.get("building_name", ""),
                        site.get("postcode", ""),
                    ):
                        group["indices"].append(idx)
                        placed = True
                        break

                if not placed:
                    campus_groups.append({
                        "indices": [idx],
                        "route_location": site["route_location"],
                    })

            if campus_groups:
                google_destinations = [
                    group["route_location"]
                    for group in campus_groups
                ]
                google_matrix = self.router.one_to_many(
                    current_location,
                    google_destinations,
                    current_time,
                )

                for group, minutes in zip(
                    campus_groups,
                    google_matrix,
                ):
                    for idx in group["indices"]:
                        travel_by_index[idx] = minutes

            ranked = []
            for idx, site in enumerate(remaining):
                minutes = travel_by_index.get(idx)

                if minutes is None:
                    continue

                # Density-first day routing.
                #
                # Once the day has entered a postcode district, remaining jobs in
                # that district are considered before jobs in a different district.
                # Within the district, the exact same full postcode is preferred
                # again. Google transit time still ranks candidates INSIDE those
                # geographic tiers, and the normal hard feasibility checks still
                # decide whether a job can actually be fitted.
                district_tier = 0
                postcode_tier = 0
                if scheduled:
                    current_district = postcode_district(current_postcode)
                    next_district = postcode_district(site.get("postcode"))
                    district_tier = (
                        0
                        if current_district
                        and current_district == next_district
                        else 1
                    )
                    postcode_tier = (
                        0
                        if self._same_postcode(
                            current_postcode,
                            site.get("postcode", ""),
                        )
                        else 1
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
                        district_tier,
                        postcode_tier,
                        score,
                        idx,
                        float(minutes),
                    )
                )

            if not ranked:
                break

            # Geographic tier first, then real transit economics.
            #
            # To preserve robustness, always keep up to two out-of-district
            # fallback candidates in the feasibility pool. That means an
            # impossible local job cannot prematurely end the day, while a
            # distant hop cannot beat a feasible local job merely because its
            # transit time happens to be a few minutes shorter.
            ranked.sort(key=lambda x: (x[0], x[1], x[2]))

            if scheduled:
                same_district = [r for r in ranked if r[0] == 0]
                other_district = [r for r in ranked if r[0] != 0]

                fallback_slots = (
                    min(2, len(other_district))
                    if same_district
                    else min(self.max_candidate_checks, len(other_district))
                )
                local_slots = max(
                    0,
                    self.max_candidate_checks - fallback_slots,
                )
                candidate_pool = (
                    same_district[:local_slots]
                    + other_district[:fallback_slots]
                )
            else:
                candidate_pool = ranked[:self.max_candidate_checks]

            chosen = None
            lunch_blocked_candidate = False

            for _, _, _, idx, travel_minutes in candidate_pool:
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
                    buffered_travel_minutes = (
                        travel_minutes + self.travel_leeway_minutes
                    )
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
                    cluster=postcode_district(site.get("postcode", "")),
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

