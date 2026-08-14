from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from openai import OpenAI


@dataclass
class AIPlanDecision:
    customer_reference: str
    priority: int
    decision: str
    reason: str


class OpenAISchedulePlanner:
    """
    AI decision layer.

    It does NOT invent travel times and it does NOT bypass the deterministic
    schedule constraints. It reviews trusted inputs and decides whether each
    site is attractive for the current week or better deferred for a future
    cluster.
    """

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing.")
        if not model:
            raise ValueError("OPENAI_MODEL is missing.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def rank_sites(
        self,
        sites: List[dict],
        week_label: str,
        tfl_summary: str,
        weather_summary: str,
        future_cluster_summary: str,
    ) -> List[AIPlanDecision]:

        compact_sites = []
        for site in sites:
            compact_sites.append({
                "customer_reference": site.get("customer_reference", ""),
                "building_name": site.get("building_name", ""),
                "postcode": site.get("postcode", ""),
                "postcode_district": site.get("postcode_district", ""),
                "planning_minutes": site.get("planning_minutes"),
                "prediction_confidence": site.get("confidence", ""),
                "existing_planned_start": site.get("planned_start", ""),
                "same_district_this_week": site.get("same_district_this_week", 0),
                "same_district_next_week": site.get("same_district_next_week", 0),
                "same_district_next_3_weeks": site.get("same_district_next_3_weeks", 0),
            })

        instructions = """
You are a planning decision layer for UK site surveys.

You are NOT a routing engine. Never invent travel times. Google Routes will
calculate and validate all actual public-transport journeys after your output.

Your job is to rank candidate sites for the requested working week. Consider:
- keeping geographically coherent postcode-district clusters together;
- whether a site becomes more valuable to defer because several nearby sites
  appear in a future week;
- the site's existing planned date, where present;
- predicted survey duration and prediction confidence;
- weather risk;
- TfL disruption risk;
- avoiding unnecessarily risky/tight days.

A site should only be marked "defer" when there is a concrete planning reason,
especially a stronger future cluster. Do not defer merely because another week
exists.

Priority must be an integer from 0 to 100:
- 80-100: strongly favour this week
- 50-79: reasonable this week
- 20-49: weak this week / useful filler
- 0-19: deliberately defer if practical

Return JSON only using this structure:
{
  "decisions": [
    {
      "customer_reference": "...",
      "priority": 0,
      "decision": "schedule_this_week" | "defer" | "neutral",
      "reason": "brief operational reason"
    }
  ],
  "week_strategy": "brief explanation"
}
"""

        payload = {
            "week": week_label,
            "sites": compact_sites,
            "tfl_disruption_summary": tfl_summary,
            "weather_summary": weather_summary,
            "future_cluster_summary": future_cluster_summary,
        }

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
        )

        text = response.output_text.strip()
        # Tolerate fenced JSON if the model adds fences despite instruction.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        data = json.loads(text)
        decisions = []
        for item in data.get("decisions", []):
            try:
                priority = int(item.get("priority", 50))
            except Exception:
                priority = 50
            priority = max(0, min(100, priority))
            decisions.append(
                AIPlanDecision(
                    customer_reference=str(item.get("customer_reference", "")),
                    priority=priority,
                    decision=str(item.get("decision", "neutral")),
                    reason=str(item.get("reason", "")),
                )
            )

        return decisions, str(data.get("week_strategy", ""))
