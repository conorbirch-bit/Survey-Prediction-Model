"""Optional, transactional final pass: spread real visits, then add visible allowance.

Normal planning and gap filling must run first. This module never adds a new
building, changes predicted survey time, or invents journey time. Every changed
day is rebuilt with the regular time-dependent router and accepted in full.
"""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta

import pandas as pd

from coordinate_clustering import haversine_km
from google_routes import GoogleRoutesError
from scheduler_v20_10 import (
    WeeklyScheduleResult, survey_clocks_for_date, _site_allowed_today,
    _site_allowed_for_surveyor, _site_identity_for_sequence, _requested_retry_day,
    _saturday_small_flat_preferred,
)


def _identity(item):
    return _site_identity_for_sequence(vars(item))


def _present(value):
    return value is not None and str(value).strip().lower() not in {"", "nan", "none", "<na>"}


def _remaining_minutes(day):
    if not day.items:
        return (day.latest_survey_finish - day.first_survey_target).total_seconds() / 60
    return max(0.0, min(
        (day.latest_survey_finish - day.items[-1].survey_end).total_seconds() / 60,
        (day.latest_return - day.return_time).total_seconds() / 60,
    ))


def _groups(items, sites, split_local=False):
    """Keep explicit access groups intact; prefer whole 400 m local groups."""
    groups = [[item] for item in items]
    def connected(a, b):
        left, right = sites[_identity(a)], sites[_identity(b)]
        ga, gb = left.get("visit_group"), right.get("visit_group")
        if _present(ga) or _present(gb):
            return _present(ga) and ga == gb
        if split_local:
            return False
        distance = haversine_km(left.get("latitude"), left.get("longitude"),
                                right.get("latitude"), right.get("longitude"))
        return distance is not None and distance <= 0.4
    for i in range(len(groups) - 1, -1, -1):
        for j in range(i):
            if any(connected(a, b) for a in groups[i] for b in groups[j]):
                groups[j].extend(groups.pop(i))
                break
    return [sorted(group, key=lambda item: item.sequence) for group in groups]


def rebuild_day(scheduler, ordered_sites, first, finish, deadline, *,
                buffer_minutes=0, saturday_pool=None):
    """Re-time an exact order, using the ordinary scheduler for each extension.

    Allowance precedes departure for the next building, so route lookups use
    that later departure. There is none before the first or after the last visit.
    No partial rebuild is returned: an infeasible trial leaves its caller's
    original day intact, including lunch, visit order and appointment times.
    """
    old_ratio, old_diagnostics = scheduler.minimum_survey_to_travel_ratio, scheduler.first_visit_diagnostics
    try:
        scheduler.minimum_survey_to_travel_ratio = 0.0
        scheduler.first_visit_diagnostics = None
        result = None
        for index, site in enumerate(ordered_sites):
            prefix = result
            if prefix is not None:
                prefix = replace(prefix, return_departure=prefix.return_departure
                                 + timedelta(minutes=buffer_minutes))
            trial = scheduler.build_day(
                [site], first, finish, deadline, resume_from=prefix,
                resume_site=ordered_sites[index - 1] if index else None,
                saturday_candidate_pool=[*(saturday_pool or []), *ordered_sites[index:]],
            )
            if len(trial.items) != index + 1 or _identity(trial.items[-1]) != _site_identity_for_sequence(site):
                return None
            trial.items[-1] = replace(trial.items[-1], balancing_buffer_minutes=buffer_minutes if index else 0)
            result = trial
        if result is None:
            result = scheduler.build_day([], first, finish, deadline)
        if result.return_time > deadline or any(i.survey_end > finish or i.survey_start < first for i in result.items):
            return None
        if result.lunch_start is not None:
            start = result.lunch_start.time().replace(tzinfo=None)
            if not scheduler.lunch_window_start_clock <= start <= scheduler.lunch_latest_start_clock:
                return None
            if (result.lunch_end - result.lunch_start).total_seconds() < scheduler.lunch_minutes * 60:
                return None
        return result
    finally:
        scheduler.minimum_survey_to_travel_ratio = old_ratio
        scheduler.first_visit_diagnostics = old_diagnostics


def rebalance_team_workload(
    team_results, surveyors, sites_by_surveyor, scheduler_factory,
    first_survey_clock, latest_survey_clock, latest_return_clock, timezone,
    *, reserve_sites=None, saturday_time_window=None, enabled=False,
    max_buffer_minutes=30, buffer_step_minutes=5,
):
    """Return results, moved-site assignment audit, and before/after day review.

    Cover empty selected days first, then reduce variance in survey workload
    relative to each day's window. Whole local groups are tried before smaller
    moves. Explicit visit groups and accepted weekly notes stay protected.
    A strictly improving score makes the finite relocation search terminate.
    """
    if not enabled:
        return team_results, pd.DataFrame(), pd.DataFrame()
    if buffer_step_minutes <= 0 or not 0 <= max_buffer_minutes <= 30:
        raise ValueError("Balancing buffers must use a positive step and a maximum of 0–30 minutes.")
    people = {person.name: person for person in surveyors}
    sites = {_site_identity_for_sequence(site): dict(site) for site in reserve_sites or []}
    for name, pool in sites_by_surveyor.items():
        for site in pool:
            identity = _site_identity_for_sequence(site)
            if identity not in sites or not _present(sites[identity].get("special_request_date")):
                sites[identity] = dict(site)
    baseline_counts = Counter(_identity(item) for result in team_results.values() if result
                              for day in result.days for item in day.items)
    if any(count != 1 for count in baseline_counts.values()):
        raise ValueError("Cannot balance a schedule with duplicate building visits.")
    if set(baseline_counts) - sites.keys():
        raise ValueError("Cannot balance without source locations and access rules for every scheduled building.")
    booked = set(baseline_counts)
    schedulers = {name: scheduler_factory(person) for name, person in people.items()}
    slots, clocks = {}, {}
    for name, person in people.items():
        result = team_results.get(name)
        existing = {(day.first_survey_target or day.start_time).date(): day for day in result.days} if result else {}
        for day_date in sorted(set(person.available_dates or [])):
            clocks[name, day_date] = tuple(datetime.combine(day_date, clock, tzinfo=timezone) for clock in
                survey_clocks_for_date(day_date, first_survey_clock, latest_survey_clock,
                                      latest_return_clock, saturday_time_window))
            slots[name, day_date] = existing.get(day_date) or rebuild_day(schedulers[name], [], *clocks[name, day_date])
        if set(existing) - set(person.available_dates or []):
            raise ValueError("Scheduled dates do not match selected contractor availability.")
    before = dict(slots)
    initial_owner = {_identity(item): key[0] for key, day in slots.items() for item in day.items}
    errors = {key: set() for key in slots}
    trial_cache = {}

    def pending_small(day_date):
        return [site for identity, site in sites.items() if identity not in booked
                and _saturday_small_flat_preferred(site, day_date) and _site_allowed_today(site, day_date)
                and any(day_date in (person.available_dates or [])
                        and _site_allowed_for_surveyor(site, person.name) for person in surveyors)]

    def trial(key, items, buffer=0):
        cache_key = (key, tuple(_identity(item) for item in items), buffer)
        if cache_key in trial_cache:
            return trial_cache[cache_key]
        try:
            result = rebuild_day(schedulers[key[0]], [sites[_identity(item)] for item in items],
                               *clocks[key], buffer_minutes=buffer,
                               saturday_pool=pending_small(key[1]))
        except GoogleRoutesError as exc:
            errors[key].add(f"Routing trial failed: {type(exc).__name__}; original retained.")
            result = None
        trial_cache[cache_key] = result
        return result

    def window(key):
        return max(1, (clocks[key][1] - clocks[key][0]).total_seconds() / 60)

    def movable(items, source, target):
        for item in items:
            site = sites[_identity(item)]
            if _present(site.get("special_request_date")):
                return False  # Keep the user's accepted date/owner note intact.
            if not _site_allowed_today(site, target[1]) or not _site_allowed_for_surveyor(site, target[0]):
                return False
            if _requested_retry_day(site, source[1]) and not _requested_retry_day(site, target[1]):
                return False
        return True

    # Only survey work enters this score: longer journeys cannot masquerade as
    # better balance. Half-day Saturdays use their own shorter denominator.
    while True:
        accepted = False
        for split_local in (False, True):
            candidates = []
            for target, target_day in slots.items():
                if _remaining_minutes(target_day) < buffer_step_minutes:
                    continue
                for source, source_day in slots.items():
                    if source == target or len(source_day.items) < 2:
                        continue
                    for group in _groups(source_day.items, sites, split_local=split_local):
                        if len(group) == len(source_day.items) or not movable(group, source, target):
                            continue
                        minutes = sum(item.survey_minutes for item in group)
                        if minutes + target_day.survey_minutes > window(target):
                            continue
                        a, b = source_day.survey_minutes, target_day.survey_minutes
                        gain = (a / window(source)) ** 2 + (b / window(target)) ** 2 - (
                            ((a - minutes) / window(source)) ** 2 + ((b + minutes) / window(target)) ** 2)
                        covers_empty = not target_day.items
                        if not covers_empty and gain <= 1e-8:
                            continue
                        candidates.append((not covers_empty, -gain, source[0] != target[0], source, target, group))
            candidates.sort(key=lambda row: row[:5])
            for _, _, _, source, target, group in candidates:
                ids = {_identity(item) for item in group}
                source_items = [item for item in slots[source].items if _identity(item) not in ids]
                donor = trial(source, source_items)
                if donor is None:
                    continue
                # Insert the whole group, trying nearby boundaries first. The
                # existing order is otherwise unchanged (including first visits).
                target_items = slots[target].items
                def distance(position):
                    if not target_items:
                        return 0
                    neighbour = sites[_identity(target_items[min(position, len(target_items) - 1)])]
                    first_site = sites[_identity(group[0])]
                    value = haversine_km(neighbour.get("latitude"), neighbour.get("longitude"),
                                         first_site.get("latitude"), first_site.get("longitude"))
                    return value if value is not None else 0
                recipient = None
                for position in sorted(range(len(target_items) + 1), key=distance):
                    ordered = target_items[:position] + group + target_items[position:]
                    # Keep first visits ahead of ordinary retries on weekdays.
                    if target[1].weekday() < 5:
                        def first(item):
                            s = sites[_identity(item)]
                            return str(s.get("never_visited", False)).lower() in {"true", "1", "1.0"} and str(s.get("is_retry", False)).lower() not in {"true", "1", "1.0"}
                        if any(first(item) for item in ordered):
                            ordered = sorted(ordered, key=lambda item: not first(item))
                    recipient = trial(target, ordered)
                    if recipient is not None:
                        break
                if recipient is not None:
                    slots[source], slots[target] = donor, recipient
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            break

    # Buffer trials are independent of relocation and only run on short days.
    # The original day is retained if retiming loses even one visit.
    for key, day in list(slots.items()):
        if len(day.items) < 2 or _remaining_minutes(day) < buffer_step_minutes:
            continue
        best = day
        for buffer in range(buffer_step_minutes, max_buffer_minutes + 1, buffer_step_minutes):
            candidate = trial(key, day.items, buffer)
            if candidate is None:
                # Transit changes can be non-monotonic; a later departure may fit.
                continue
            if _remaining_minutes(candidate) < _remaining_minutes(best):
                best = candidate
            if _remaining_minutes(best) < buffer_step_minutes:
                break
        slots[key] = best

    final_counts = Counter(_identity(item) for day in slots.values() for item in day.items)
    if final_counts != baseline_counts:
        raise ValueError("Balancing must retain every original building exactly once.")
    results = dict(team_results)
    for name in people:
        days = [day for (person, _), day in sorted(slots.items()) if person == name]
        original = team_results.get(name)
        results[name] = WeeklyScheduleResult(days=days, unscheduled_sites=list(original.unscheduled_sites) if original else [])
    moves = []
    for key, day in slots.items():
        for item in day.items:
            source = initial_owner[_identity(item)]
            if source != key[0]:
                moves.append({"Original Surveyor": source, "Surveyor": key[0],
                              "Customer Reference": item.customer_reference, "Building Name": item.building_name,
                              "Postcode": item.postcode, "Date": key[1].isoformat()})
    review = []
    for key, day in sorted(slots.items()):
        original = before[key]
        remaining = round(_remaining_minutes(day), 1)
        buffer = sum(item.balancing_buffer_minutes for item in day.items)
        status = ("No feasible work to move here under current rules" if not day.items else
                  "Filled within 5 minutes of survey/return limit" if remaining < buffer_step_minutes else
                  "Shortfall remains after feasible moves and buffer trials")
        review.append({"Surveyor": key[0], "Date": key[1].isoformat(),
                       "Before Visits": len(original.items), "After Visits": len(day.items),
                       "Before Survey Minutes": original.survey_minutes, "Survey Minutes": day.survey_minutes,
                       "Before Last Survey": original.items[-1].survey_end.strftime("%H:%M") if original.items else "",
                       "After Last Survey": day.items[-1].survey_end.strftime("%H:%M") if day.items else "",
                       "Before Return Home": original.return_time.strftime("%H:%M") if original.items else "",
                       "After Return Home": day.return_time.strftime("%H:%M") if day.items else "",
                       "Buildings Moved In": ", ".join(sorted({_identity(i) for i in day.items} - {_identity(i) for i in original.items})),
                       "Buildings Moved Out": ", ".join(sorted({_identity(i) for i in original.items} - {_identity(i) for i in day.items})),
                       "Travel Minutes": day.travel_minutes,
                       "Additional Balancing Buffer (Minutes)": buffer,
                       "Minutes Still Available": remaining, "Outcome": status,
                       "Details": " ".join(sorted(errors[key]))})
    return results, pd.DataFrame(moves), pd.DataFrame(review)
