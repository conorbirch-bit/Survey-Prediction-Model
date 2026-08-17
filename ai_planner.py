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


@dataclass
class AIClusterDecision:
    cluster: str
    priority: int
    target_sites: int
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



    def select_clusters(
        self,
        cluster_summary: list,
        week_label: str,
        working_days: int,
        max_sites_for_google: int,
        tfl_summary: str,
        weather_summary: str,
        team_size: int = 1,
    ):
        """
        Strategic portfolio filter.

        This runs BEFORE Google Routes. It receives postcode-cluster summaries,
        not thousands of individual route candidates, and chooses which clusters
        are worth precise transit routing for the requested week.
        """

        instructions = """
You are the strategic portfolio-filter layer for UK site-survey scheduling.

You are NOT a routing engine. Do not estimate or invent journey times. Google
Routes will only be called AFTER your cluster shortlist.

You are given cheap postcode-district summaries across a potentially very large
portfolio. Choose a small set of clusters that are genuinely worth considering
for the requested survey week.

Objectives:
- create geographically coherent survey days;
- prefer clusters with enough eligible survey work to sustain useful days;
- respect existing planned work where it creates sensible continuity;
- consider prediction confidence and total available survey hours;
- use TfL disruption and weather only as risk/context, not as invented routing;
- avoid selecting isolated clusters when a stronger cluster can fill the week;
- remember that sites awaiting drawing may become useful in later weeks, but
  only "Eligible This Week" sites can enter the Google shortlist now;
- keep Google cost low by not selecting unnecessary clusters.

The total target_sites across selected clusters should normally be close to,
but never exceed, max_sites_for_google. Select enough candidates that the
deterministic Google scheduler has choice, but do not send the whole portfolio.

For one surveyor, roughly 2-6 clusters for a five-day week is normally enough.
For a team, select enough coherent clusters to feed the active surveyors without
creating unnecessary geographic spread. A team of 3-4 may reasonably need more
clusters than a single surveyor. For a single day, usually 1-2 clusters per
active surveyor is enough.

Priority is 0-100. Return JSON only:
{
  "selected_clusters": [
    {
      "cluster": "NW9",
      "priority": 92,
      "target_sites": 18,
      "decision": "consider_this_week",
      "reason": "brief reason"
    }
  ],
  "deferred_clusters": [
    {
      "cluster": "SW9",
      "reason": "brief reason"
    }
  ],
  "strategy": "brief strategic explanation"
}
"""

        payload = {
            "week": week_label,
            "working_days": int(working_days),
            "team_size": int(team_size),
            "max_sites_for_google": int(max_sites_for_google),
            "cluster_summary": cluster_summary,
            "tfl_disruption_summary": tfl_summary,
            "weather_summary": weather_summary,
        }

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False, default=str),
        )

        text = response.output_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        data = json.loads(text)

        decisions = []
        total_requested = 0
        for item in data.get("selected_clusters", []):
            cluster = str(item.get("cluster", "")).strip()
            if not cluster:
                continue

            try:
                priority = int(item.get("priority", 50))
            except Exception:
                priority = 50
            priority = max(0, min(100, priority))

            try:
                target_sites = int(item.get("target_sites", 1))
            except Exception:
                target_sites = 1
            target_sites = max(1, target_sites)

            # Hard cap: AI can never ask Google to see more than the UI limit.
            remaining = max(0, int(max_sites_for_google) - total_requested)
            if remaining <= 0:
                break
            target_sites = min(target_sites, remaining)
            total_requested += target_sites

            decisions.append(
                AIClusterDecision(
                    cluster=cluster,
                    priority=priority,
                    target_sites=target_sites,
                    decision=str(
                        item.get("decision", "consider_this_week")
                    ),
                    reason=str(item.get("reason", "")),
                )
            )

        deferred = data.get("deferred_clusters", [])
        return decisions, deferred, str(data.get("strategy", ""))

    def summarise_week(
        self,
        week_summary: list,
        full_schedule: list,
        ai_site_decisions: list,
        unscheduled_sites: list,
        tfl_summary: str,
        weather_summary: str,
        future_cluster_summary: str,
    ) -> str:
        """
        Produce a concise, conversational explanation of the final schedule.
        This is explanation/review only; it does not change the schedule.
        """

        instructions = """
You are explaining a completed UK site-survey weekly schedule to an operations
manager.

Write in a natural, conversational but professional tone. Do not sound like a
formal report and do not overstate certainty.

Explain:
- the overall strategy of the week;
- why particular geographic/postcode clusters were grouped together;
- why some sites were left for later if future clustering made that sensible;
- where the week has more or less slack;
- any material TfL disruption or weather considerations;
- any lower-confidence survey-duration predictions that make a day less certain.

Do not invent travel times, disruptions, weather, or operational facts.
Only use the supplied data.

Keep it to roughly 4-7 short paragraphs. Mention actual weekdays and areas when
they are present. Finish with a brief "Overall" sentence describing how robust
the week looks.

Do not use JSON. Return normal prose only.
"""

        payload = {
            "week_summary": week_summary,
            "full_schedule": full_schedule,
            "ai_site_decisions": ai_site_decisions,
            "unscheduled_sites": unscheduled_sites,
            "tfl_disruption_summary": tfl_summary,
            "weather_summary": weather_summary,
            "future_cluster_summary": future_cluster_summary,
        }

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False, default=str),
        )

        return response.output_text.strip()

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
