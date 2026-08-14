from __future__ import annotations

from datetime import date
from typing import Any, Dict, Iterable, List, Optional
import requests


class MetOfficeClient:
    """
    Met Office Weather DataHub Global Spot DAILY client.

    Expected endpoint:
      https://data.hub.api.metoffice.gov.uk/sitespecific/v0/point/daily

    Authentication:
      apikey header

    Required request parameter:
      dataSource=BD1
    """

    WEATHER_CODES = {
        0: "Clear night",
        1: "Sunny day",
        2: "Partly cloudy night",
        3: "Partly cloudy day",
        5: "Mist",
        6: "Fog",
        7: "Cloudy",
        8: "Overcast",
        9: "Light rain shower night",
        10: "Light rain shower day",
        11: "Drizzle",
        12: "Light rain",
        13: "Heavy rain shower night",
        14: "Heavy rain shower day",
        15: "Heavy rain",
        16: "Sleet shower night",
        17: "Sleet shower day",
        18: "Sleet",
        19: "Hail shower night",
        20: "Hail shower day",
        21: "Hail",
        22: "Light snow shower night",
        23: "Light snow shower day",
        24: "Light snow",
        25: "Heavy snow shower night",
        26: "Heavy snow shower day",
        27: "Heavy snow",
        28: "Thunder shower night",
        29: "Thunder shower day",
        30: "Thunder",
    }

    def __init__(
        self,
        api_key: str,
        endpoint_url: str,
        timeout_seconds: int = 20,
    ):
        self.api_key = api_key.strip()
        self.endpoint_url = endpoint_url.strip()
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _normalise_key(key: str) -> str:
        return "".join(ch.lower() for ch in str(key) if ch.isalnum())

    @classmethod
    def _first_value(
        cls,
        row: Dict[str, Any],
        exact_names: Iterable[str] = (),
        contains_groups: Iterable[Iterable[str]] = (),
    ):
        # Exact aliases first.
        for name in exact_names:
            if name in row and row[name] is not None:
                return row[name]

        normalised = {
            cls._normalise_key(k): v
            for k, v in row.items()
            if v is not None
        }

        for name in exact_names:
            value = normalised.get(cls._normalise_key(name))
            if value is not None:
                return value

        # Fuzzy fallback: all words in a group must appear in the key.
        for group in contains_groups:
            words = [cls._normalise_key(w) for w in group]
            for key, value in normalised.items():
                if all(word in key for word in words):
                    return value

        return None

    @classmethod
    def _weather_text(cls, value) -> Optional[str]:
        if value is None:
            return None
        try:
            code = int(float(value))
            text = cls.WEATHER_CODES.get(code)
            return f"{text} (code {code})" if text else f"Weather code {code}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _fmt(value, suffix=""):
        if value is None:
            return "not supplied"
        try:
            number = float(value)
            if number.is_integer():
                return f"{int(number)}{suffix}"
            return f"{number:.1f}{suffix}"
        except (TypeError, ValueError):
            return f"{value}{suffix}"

    @classmethod
    def _parse_daily_row(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = cls._first_value(
            row,
            exact_names=("time", "validityTime", "timestamp"),
        )

        max_temp = cls._first_value(
            row,
            exact_names=(
                "dayMaxScreenTemperature",
                "maxScreenAirTemp",
                "maxScreenTemperature",
                "dayMaximumTemperature",
                "maximumTemperature",
            ),
            contains_groups=(
                ("day", "max", "temperature"),
                ("max", "screen", "temperature"),
                ("maximum", "temperature"),
            ),
        )

        min_temp = cls._first_value(
            row,
            exact_names=(
                "nightMinScreenTemperature",
                "minScreenAirTemp",
                "minScreenTemperature",
                "nightMinimumTemperature",
                "minimumTemperature",
            ),
            contains_groups=(
                ("night", "min", "temperature"),
                ("min", "screen", "temperature"),
                ("minimum", "temperature"),
            ),
        )

        day_rain = cls._first_value(
            row,
            exact_names=(
                "dayProbabilityOfRain",
                "dayProbabilityOfPrecipitation",
                "dayPrecipitationProbability",
                "probabilityOfRainDay",
            ),
            contains_groups=(
                ("day", "probability", "rain"),
                ("day", "probability", "precipitation"),
                ("day", "precipitation", "probability"),
            ),
        )

        night_rain = cls._first_value(
            row,
            exact_names=(
                "nightProbabilityOfRain",
                "nightProbabilityOfPrecipitation",
                "nightPrecipitationProbability",
                "probabilityOfRainNight",
            ),
            contains_groups=(
                ("night", "probability", "rain"),
                ("night", "probability", "precipitation"),
                ("night", "precipitation", "probability"),
            ),
        )

        day_wind = cls._first_value(
            row,
            exact_names=(
                "midday10MWindSpeed",
                "dayWindSpeed10m",
                "dayWindSpeed",
            ),
            contains_groups=(
                ("midday", "wind", "speed"),
                ("day", "wind", "speed"),
            ),
        )

        night_wind = cls._first_value(
            row,
            exact_names=(
                "midnight10MWindSpeed",
                "nightWindSpeed10m",
                "nightWindSpeed",
            ),
            contains_groups=(
                ("midnight", "wind", "speed"),
                ("night", "wind", "speed"),
            ),
        )

        day_gust = cls._first_value(
            row,
            exact_names=(
                "dayMax10MWindGust",
                "dayMaxWindGust",
                "maximumDayWindGust",
            ),
            contains_groups=(
                ("day", "max", "wind", "gust"),
                ("day", "wind", "gust"),
            ),
        )

        night_gust = cls._first_value(
            row,
            exact_names=(
                "nightMax10MWindGust",
                "nightMaxWindGust",
                "maximumNightWindGust",
            ),
            contains_groups=(
                ("night", "max", "wind", "gust"),
                ("night", "wind", "gust"),
            ),
        )

        day_weather = cls._first_value(
            row,
            exact_names=(
                "daySignificantWeatherCode",
                "dayWeatherCode",
                "significantWeatherCodeDay",
            ),
            contains_groups=(
                ("day", "significant", "weather", "code"),
                ("day", "weather", "code"),
            ),
        )

        night_weather = cls._first_value(
            row,
            exact_names=(
                "nightSignificantWeatherCode",
                "nightWeatherCode",
                "significantWeatherCodeNight",
            ),
            contains_groups=(
                ("night", "significant", "weather", "code"),
                ("night", "weather", "code"),
            ),
        )

        return {
            "time": timestamp,
            "max_temperature_c": max_temp,
            "min_temperature_c": min_temp,
            "day_rain_probability_pct": day_rain,
            "night_rain_probability_pct": night_rain,
            "day_wind_speed": day_wind,
            "night_wind_speed": night_wind,
            "day_max_wind_gust": day_gust,
            "night_max_wind_gust": night_gust,
            "day_weather": cls._weather_text(day_weather),
            "night_weather": cls._weather_text(night_weather),
        }

    @classmethod
    def _daily_summary_text(cls, parsed_rows: List[Dict[str, Any]]) -> str:
        blocks = []
        for row in parsed_rows:
            raw_time = str(row.get("time") or "")
            date_text = raw_time[:10] if raw_time else "Unknown date"

            lines = [
                date_text,
                (
                    "Temperature: max "
                    f"{cls._fmt(row.get('max_temperature_c'), '°C')}, min "
                    f"{cls._fmt(row.get('min_temperature_c'), '°C')}"
                ),
                (
                    "Rain probability: day "
                    f"{cls._fmt(row.get('day_rain_probability_pct'), '%')}, "
                    "night "
                    f"{cls._fmt(row.get('night_rain_probability_pct'), '%')}"
                ),
                (
                    "Wind: day "
                    f"{cls._fmt(row.get('day_wind_speed'))}, night "
                    f"{cls._fmt(row.get('night_wind_speed'))}; "
                    "max gust day "
                    f"{cls._fmt(row.get('day_max_wind_gust'))}, night "
                    f"{cls._fmt(row.get('night_max_wind_gust'))}"
                ),
                (
                    "Weather: day "
                    f"{row.get('day_weather') or 'not supplied'}, night "
                    f"{row.get('night_weather') or 'not supplied'}"
                ),
            ]
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    def forecast_summary(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> str:
        if not self.api_key or not self.endpoint_url:
            return "Met Office forecast not configured."

        headers = {
            "accept": "application/json",
            "apikey": self.api_key,
        }
        params = {
            "dataSource": "BD1",
            "latitude": latitude,
            "longitude": longitude,
        }

        try:
            response = requests.get(
                self.endpoint_url,
                headers=headers,
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return f"Met Office forecast lookup unavailable: {exc}"

        features = data.get("features", []) if isinstance(data, dict) else []
        if not features:
            return "Met Office returned no forecast features."

        properties = features[0].get("properties", {})
        time_series = (
            properties.get("timeSeries")
            or properties.get("timeseries")
            or properties.get("time_series")
            or []
        )

        if not time_series:
            return "Met Office forecast was returned but no daily time series was found."

        parsed = []
        for row in time_series:
            if not isinstance(row, dict):
                continue

            parsed_row = self._parse_daily_row(row)
            timestamp = str(parsed_row.get("time") or "")
            if timestamp:
                day_text = timestamp[:10]
                if (
                    day_text < start_date.isoformat()
                    or day_text > end_date.isoformat()
                ):
                    continue
            parsed.append(parsed_row)

        if not parsed:
            return "No Met Office daily forecast rows fell inside the selected period."

        # If every weather field is still absent, return the actual API keys to
        # make future schema changes obvious rather than silently showing None.
        weather_fields = [
            "max_temperature_c",
            "min_temperature_c",
            "day_rain_probability_pct",
            "night_rain_probability_pct",
            "day_wind_speed",
            "night_wind_speed",
            "day_max_wind_gust",
            "night_max_wind_gust",
            "day_weather",
            "night_weather",
        ]
        if all(
            all(row.get(field) is None for field in weather_fields)
            for row in parsed
        ):
            sample_keys = sorted(time_series[0].keys())
            return (
                "Met Office daily forecast received, but its parameter names "
                "did not match the known schema. Actual keys returned were: "
                + ", ".join(sample_keys)
            )

        return self._daily_summary_text(parsed)
