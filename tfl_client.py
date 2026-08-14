from __future__ import annotations

from datetime import date
from typing import Optional
import requests


class TfLClient:
    BASE = "https://api.tfl.gov.uk"

    def __init__(self, app_key: str = "", timeout_seconds: int = 20):
        self.app_key = app_key.strip()
        self.timeout_seconds = timeout_seconds

    def _params(self, extra=None):
        params = dict(extra or {})
        if self.app_key:
            params["app_key"] = self.app_key
        return params

    def disruption_summary(self, start_date: date, end_date: date) -> str:
        """
        Query major TfL line status across the requested dates.
        The response is deliberately summarised into compact text for the AI.
        """
        modes = "tube,dlr,overground,elizabeth-line"
        url = f"{self.BASE}/Line/Mode/{modes}/Status"
        params = self._params({
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "detail": "true",
        })
        try:
            r = requests.get(url, params=params, timeout=self.timeout_seconds)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            return f"TfL disruption lookup unavailable: {exc}"

        issues = []
        for line in data:
            name = line.get("name", "Unknown line")
            for status in line.get("lineStatuses", []):
                desc = status.get("statusSeverityDescription", "")
                reason = status.get("reason", "")
                if desc and desc.lower() not in {"good service", "special service"}:
                    entry = f"{name}: {desc}"
                    if reason:
                        entry += f" — {reason[:300]}"
                    issues.append(entry)

        if not issues:
            return "No material TfL rail/tube disruption returned for the selected period."
        return "\n".join(issues[:30])
