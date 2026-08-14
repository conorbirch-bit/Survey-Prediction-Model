from __future__ import annotations

from datetime import date
from typing import Optional
import requests


class MetOfficeClient:
    """
    Thin Weather DataHub adapter.

    MET_OFFICE_GLOBAL_SPOT_URL should be the user's subscribed Global Spot
    hourly endpoint. This avoids hard-coding a product/version path that may
    differ by Weather DataHub subscription.
    """

    def __init__(
        self,
        api_key: str,
        endpoint_url: str,
        timeout_seconds: int = 20,
    ):
        self.api_key = api_key.strip()
        self.endpoint_url = endpoint_url.strip()
        self.timeout_seconds = timeout_seconds

    def forecast_summary(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> str:
        if not self.api_key or not self.endpoint_url:
            return "Met Office forecast not configured."

        # Common Weather DataHub deployments accept an API key header; because
        # accounts/products differ, both Authorization and apikey are supplied.
        headers = {
            "accept": "application/json",
            "apikey": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }
        params = {
            "latitude": latitude,
            "longitude": longitude,
        }

        try:
            r = requests.get(
                self.endpoint_url,
                headers=headers,
                params=params,
                timeout=self.timeout_seconds,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            return f"Met Office forecast lookup unavailable: {exc}"

        # Keep compact, but retain enough raw forecast context for reasoning.
        features = data.get("features", []) if isinstance(data, dict) else []
        if not features:
            return "Met Office returned no forecast features."

        props = features[0].get("properties", {})
        time_series = (
            props.get("timeSeries")
            or props.get("timeseries")
            or props.get("time_series")
            or []
        )
        if not time_series:
            return "Met Office forecast was returned but no hourly time series was found."

        relevant = []
        for row in time_series:
            timestamp = str(
                row.get("time")
                or row.get("validityTime")
                or row.get("timestamp")
                or ""
            )
            # Simple date filtering if ISO timestamps are present.
            if timestamp:
                d = timestamp[:10]
                if d < start_date.isoformat() or d > end_date.isoformat():
                    continue

            relevant.append({
                "time": timestamp,
                "temperature": row.get("screenTemperature")
                    or row.get("temperature"),
                "precipitation_probability": row.get("probOfPrecipitation")
                    or row.get("precipitationProbability"),
                "precipitation_rate": row.get("precipitationRate"),
                "wind_speed": row.get("windSpeed10m")
                    or row.get("windSpeed"),
                "wind_gust": row.get("windGustSpeed10m")
                    or row.get("windGust"),
                "weather_code": row.get("significantWeatherCode")
                    or row.get("weatherCode"),
            })

        if not relevant:
            return "No Met Office forecast rows fell inside the selected week."

        # Limit prompt size.
        return str(relevant[:120])
