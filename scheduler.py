from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re

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
    survey_minutes: int
    survey_start: datetime
    survey_end: datetime
    predicted_minutes: float
    confidence: str
    model_used: str


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

    @property
    def survey_minutes(self) -> int:
        return sum(i.survey_minutes for i in self.items)

    @property
    def travel_minutes(self) -> int:
        return round(
            sum(i.travel_minutes for i in self.items)
            + self.return_travel_minutes
        )

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
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
                "Prediction Confidence": item.confidence,
                "Prediction Model": item.model_used,
            })

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
            "Prediction Confidence": "",
            "Prediction Model": "",
        })
        return pd.DataFrame(rows)


class DailyTransitScheduler:
    """
    Greedy, time-dependent public-transport scheduler.

    At every stop it asks Google for current-location -> all remaining sites,
    then tests the best candidates to ensure the survey plus a transit journey
    home still finishes before the hard return deadline.

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
    ):
        self.router = router
        self.home_location = home_location
        self.max_candidate_checks = max_candidate_checks
        self.same_postcode_transfer_minutes = same_postcode_transfer_minutes
        self.travel_leeway_minutes = travel_leeway_minutes
        self.pre_survey_buffer_minutes = pre_survey_buffer_minutes
        self.post_survey_buffer_minutes = post_survey_buffer_minutes

    @staticmethod
    def _same_postcode(a: str, b: str) -> bool:
        normalise = lambda x: re.sub(r"\s+", "", str(x or "").upper())
        return bool(normalise(a)) and normalise(a) == normalise(b)

    def build_day(
        self,
        sites: List[dict],
        start_time: datetime,
        latest_return: datetime,
    ) -> DailyScheduleResult:
        remaining = [dict(site) for site in sites]
        scheduled: List[ScheduledSurvey] = []

        current_location = self.home_location
        current_postcode = ""
        current_time = start_time

        while remaining:
            destinations = [s["route_location"] for s in remaining]

            # If already at the same postcode as a candidate, avoid a pointless
            # transit lookup for that site and treat it as a short internal move.
            matrix = self.router.one_to_many(
                current_location,
                destinations,
                current_time,
            )

            ranked = []
            for idx, (site, minutes) in enumerate(zip(remaining, matrix)):
                if self._same_postcode(current_postcode, site.get("postcode")):
                    minutes = float(self.same_postcode_transfer_minutes)

                if minutes is None:
                    continue

                # The transit duration is the main clustering score.
                # A small same-district bonus helps keep adjacent surveys together
                # when timings are otherwise close.
                district_bonus = 0
                if scheduled:
                    current_district = postcode_district(current_postcode)
                    next_district = postcode_district(site.get("postcode"))
                    if current_district and current_district == next_district:
                        district_bonus = 5

                score = float(minutes) - district_bonus
                ranked.append((score, idx, float(minutes)))

            if not ranked:
                break

            ranked.sort(key=lambda x: x[0])

            chosen = None
            for _, idx, travel_minutes in ranked[:self.max_candidate_checks]:
                site = remaining[idx]

                buffered_travel_minutes = travel_minutes + self.travel_leeway_minutes
                arrive = current_time + timedelta(minutes=buffered_travel_minutes)

                survey_start = arrive + timedelta(
                    minutes=self.pre_survey_buffer_minutes
                )
                survey_end = survey_start + timedelta(
                    minutes=int(site["planning_minutes"])
                )
                ready_to_leave = survey_end + timedelta(
                    minutes=self.post_survey_buffer_minutes
                )

                # Hard feasibility check using a real public-transport route home
                # at the time this particular survey would finish.
                try:
                    return_route = self.router.compute_route(
                        site["route_location"],
                        self.home_location,
                        ready_to_leave,
                    )
                except Exception:
                    continue

                return_time = ready_to_leave + timedelta(
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
                    )
                    break

            if chosen is None:
                break

            (
                idx,
                site,
                travel_minutes,
                arrive,
                survey_start,
                survey_end,
                ready_to_leave,
            ) = chosen

            scheduled.append(
                ScheduledSurvey(
                    sequence=len(scheduled) + 1,
                    customer_reference=str(site.get("customer_reference", "")),
                    building_name=str(site.get("building_name", "")),
                    postcode=str(site.get("postcode", "")),
                    cluster=postcode_district(site.get("postcode", "")),
                    depart_previous=current_time,
                    travel_minutes=travel_minutes,
                    arrive_site=arrive,
                    survey_minutes=int(site["planning_minutes"]),
                    survey_start=survey_start,
                    survey_end=survey_end,
                    predicted_minutes=float(site["predicted_minutes"]),
                    confidence=str(site.get("confidence", "")),
                    model_used=str(site.get("model_used", "")),
                )
            )

            current_location = site["route_location"]
            current_postcode = site.get("postcode", "")
            current_time = ready_to_leave
            remaining.pop(idx)

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
            return_time = start_time

        return DailyScheduleResult(
            items=scheduled,
            start_location=self.home_location,
            start_time=start_time,
            return_location=self.home_location,
            return_departure=current_time,
            return_travel_minutes=return_minutes,
            return_time=return_time,
            latest_return=latest_return,
            unscheduled_count=len(remaining),
        )
