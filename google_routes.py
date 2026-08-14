from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo

import requests


LONDON_TZ = ZoneInfo("Europe/London")


class GoogleRoutesError(RuntimeError):
    pass


@dataclass
class TransitRoute:
    origin: str
    destination: str
    departure_time: datetime
    duration_minutes: float
    distance_meters: Optional[int] = None

    @property
    def arrival_time(self) -> datetime:
        return self.departure_time + timedelta(minutes=self.duration_minutes)


class GoogleTransitRouter:
    """
    Minimal Google Maps Platform Routes API wrapper for TRANSIT routing.

    Uses:
      - Compute Routes for an individual time-dependent journey
      - Compute Route Matrix for one-to-many candidate ranking

    API key is held in memory only. It is not written to disk.
    """

    COMPUTE_ROUTES_URL = (
        "https://routes.googleapis.com/directions/v2:computeRoutes"
    )
    MATRIX_URL = (
        "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
    )

    def __init__(
        self,
        api_key: str,
        transit_preference: Optional[str] = None,
        timeout_seconds: int = 30,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("A Google Maps Platform API key is required.")
        self.api_key = api_key.strip()
        self.transit_preference = transit_preference
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _rfc3339(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=LONDON_TZ)
        return dt.isoformat(timespec="seconds")

    @staticmethod
    def _waypoint(address: str) -> dict:
        # Add UK context to postcode-only locations.
        text = str(address).strip()
        if not text:
            raise ValueError("Route waypoint cannot be blank.")
        if "united kingdom" not in text.lower() and ", uk" not in text.lower():
            text = f"{text}, UK"
        return {"address": text}

    @staticmethod
    def _seconds(duration_text: str) -> float:
        if not duration_text:
            raise GoogleRoutesError("Google did not return a route duration.")
        return float(str(duration_text).rstrip("s"))

    def _transit_preferences(self) -> Optional[dict]:
        if not self.transit_preference:
            return None
        return {"routingPreference": self.transit_preference}

    def compute_route(
        self,
        origin: str,
        destination: str,
        departure_time: datetime,
    ) -> TransitRoute:
        body = {
            "origin": self._waypoint(origin),
            "destination": self._waypoint(destination),
            "travelMode": "TRANSIT",
            "departureTime": self._rfc3339(departure_time),
            "languageCode": "en-GB",
            "regionCode": "uk",
            "units": "METRIC",
        }

        prefs = self._transit_preferences()
        if prefs:
            body["transitPreferences"] = prefs

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters",
        }

        response = requests.post(
            self.COMPUTE_ROUTES_URL,
            headers=headers,
            json=body,
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise GoogleRoutesError(
                f"Google Compute Routes error {response.status_code}: "
                f"{response.text[:500]}"
            )

        data = response.json()
        routes = data.get("routes") or []
        if not routes:
            raise GoogleRoutesError(
                f"No public-transport route found from {origin} to {destination}."
            )

        route = routes[0]
        minutes = self._seconds(route["duration"]) / 60.0
        return TransitRoute(
            origin=origin,
            destination=destination,
            departure_time=departure_time,
            duration_minutes=minutes,
            distance_meters=route.get("distanceMeters"),
        )

    def one_to_many(
        self,
        origin: str,
        destinations: Sequence[str],
        departure_time: datetime,
    ) -> List[Optional[float]]:
        """
        Return transit minutes from one origin to each destination.

        Google limits TRANSIT matrices to 100 elements and address/place-ID
        origins+destinations to 50 per request. With one origin, batches of 49
        destinations stay inside both limits.
        """
        if not destinations:
            return []

        results: List[Optional[float]] = []
        batch_size = 49

        for start in range(0, len(destinations), batch_size):
            batch = list(destinations[start:start + batch_size])

            body = {
                "origins": [{"waypoint": self._waypoint(origin)}],
                "destinations": [
                    {"waypoint": self._waypoint(d)} for d in batch
                ],
                "travelMode": "TRANSIT",
                "departureTime": self._rfc3339(departure_time),
                "languageCode": "en-GB",
                "regionCode": "uk",
                "units": "METRIC",
            }

            prefs = self._transit_preferences()
            if prefs:
                body["transitPreferences"] = prefs

            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": (
                    "originIndex,destinationIndex,status,condition,duration"
                ),
            }

            response = requests.post(
                self.MATRIX_URL,
                headers=headers,
                json=body,
                timeout=self.timeout_seconds,
            )
            if not response.ok:
                raise GoogleRoutesError(
                    f"Google Compute Route Matrix error {response.status_code}: "
                    f"{response.text[:500]}"
                )

            elements = response.json()
            batch_values: List[Optional[float]] = [None] * len(batch)

            for element in elements:
                idx = element.get("destinationIndex")
                if idx is None or idx >= len(batch):
                    continue
                if element.get("condition") != "ROUTE_EXISTS":
                    continue
                status = element.get("status") or {}
                if status.get("code", 0) not in (0, None):
                    continue
                if element.get("duration"):
                    batch_values[idx] = self._seconds(
                        element["duration"]
                    ) / 60.0

            results.extend(batch_values)

        return results
