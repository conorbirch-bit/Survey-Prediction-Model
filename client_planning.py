"""Read-only, weekday-only access scenario. No booking or approval writes."""
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from io import BytesIO
from zoneinfo import ZoneInfo

import pandas as pd

from access_rules import requested_access_days
from audited_decisions import outstanding_bookings, text
from google_routes import GoogleRoutesError
from retry_planner import _candidate_row_from_retry
from scheduler_v20_10 import CachedRunRouter, DailyTransitScheduler, _site_allowed_today
from coordinate_clustering import haversine_km
from team_scheduler import SURVEYOR_HOME_COORDINATES

TZ = ZoneInfo('Europe/London')
UNPLACED_COLUMNS = ['Customer Reference', 'Building Name', 'Reason']


def client_candidates(plan, master=None):
    """No replacement SA required for a scenario, but completed/booked work is out."""
    rows = []
    master = pd.DataFrame() if master is None else master
    source = plan.decisions.loc[plan.decisions['Decision'].eq('CLIENT_ACCESS_REQUIRED')]
    for decision in source.to_dict('records'):
        wo = decision['Work Order Number']
        appointments = plan.appointment_history
        group = appointments.loc[appointments['Work Order Number'].eq(wo)]
        if 'Building_Id' in group:
            ids = set(group['Building_Id'].map(text)) - {''}
            group = appointments.loc[appointments['Work Order Number'].eq(wo) | appointments['Building_Id'].map(text).isin(ids)]
        failed = set(group.loc[group['_sa_status'].eq('cannot complete'), 'Service Appointment ID']) | {decision.get('Old Service Appointment ID', '')}
        if group['_sa_status'].eq('completed').any() or not outstanding_bookings(group, failed).empty or decision.get('Booking Excluded') is True:
            continue
        row = _candidate_row_from_retry(pd.Series(decision))
        if not master.empty and 'Customer Reference' in master:
            matches = master.loc[master['Customer Reference'].map(text).eq(text(decision['Customer Reference']))]
            if not matches.empty:
                for key, value in matches.iloc[0].items():
                    if text(value):
                        row[key] = value
        # This is not a bookable appointment. Never expose a Salesforce upload here.
        row['Service Appointment ID'] = ''
        row['Primary Service Appointment: Service Appointment ID'] = ''
        row['Planned Start'] = None
        row['Is Retry'] = True
        row['Never Visited'] = False
        preferred, required = requested_access_days(decision.get('Failure Reason', ''))
        row['Retry Required Weekdays'] = required
        row['Retry Preferred Weekdays'] = ', '.join(preferred)
        row['Client Help Required'] = text(decision.get('Recommended Client Action')) or text(decision.get('Decision Reason'))
        row['Surveyor Description'] = text(decision.get('Surveyor Original Description'))
        row['Route Address Override'] = text(decision.get('Route Address Override'))
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates('Customer Reference') if rows else pd.DataFrame()


@dataclass
class ClientProgramme:
    schedule: pd.DataFrame
    unplaced: pd.DataFrame
    summary: dict


def build_client_programme(sites, surveyors, start_week, router, *, first=time(9), finish=time(15),
                           home_by=time(17), scheduler_factory=DailyTransitScheduler,
                           pre_buffer=5, post_buffer=5, travel_leeway=5, progress=None):
    """Pack available weekdays until done. Stop only on no progress or routing failure.

    Availability is a recurring Mon–Fri pattern, not a finite list of dates.
    No balancing pass, artificial allowance or fixed week cap is used.
    """
    if not first < finish <= home_by:
        raise ValueError('Choose a survey start before the finish, and a return deadline at or after the finish.')
    active = [s for s in surveyors if set(s['weekdays']) & set(range(5))]
    if not active:
        raise ValueError('Select at least one surveyor with Monday–Friday availability.')
    names = [s['name'].strip() for s in active]
    if len(set(names)) != len(names) or any(not n for n in names) or any(not s['home'].strip() for s in active):
        raise ValueError('Each selected surveyor needs a unique name and a home/start location.')
    week = start_week - timedelta(days=start_week.weekday())
    original_week = week
    remaining, unplaced = {}, []
    seen_references = set()
    available_days = set().union(*(set(s['weekdays']) & set(range(5)) for s in active))
    for source in sites:
        site = dict(source)
        ref = text(site.get('customer_reference'))
        if not ref or ref in seen_references:
            raise ValueError('Client candidates must have unique, nonblank customer references.')
        seen_references.add(ref)
        required = site.get('retry_required_weekdays') or []
        if isinstance(required, str):
            required = [v.strip() for v in required.split(',') if v.strip()]
        site['is_retry'], site['never_visited'] = True, False
        site['retry_forbidden_weekday'] = None  # Access is assumed arranged for the scenario.
        site['retry_reason_category'] = 'CLIENT_PLANNING'
        if required and not any((week + timedelta(days=d)).strftime('%A') in required for d in available_days):
            unplaced.append([ref, site['building_name'], 'Required access day is outside the selected Monday–Friday availability.'])
        elif not site.get('route_location'):
            unplaced.append([ref, site['building_name'], 'Missing routing address/postcode.'])
        elif not 0 < float(site.get('planning_minutes') or 0) <= (datetime.combine(week, finish)-datetime.combine(week, first)).total_seconds()/60:
            unplaced.append([ref, site['building_name'], 'Survey duration is missing or exceeds the selected daily survey window.'])
        else:
            remaining[ref] = site
    shared_router = CachedRunRouter(router)
    engines = {s['name']: scheduler_factory(shared_router, home_location=s['home'], surveyor_name=s['name'],
                pre_survey_buffer_minutes=pre_buffer, post_survey_buffer_minutes=post_buffer,
                travel_leeway_minutes=travel_leeway, minimum_survey_to_travel_ratio=0.0,
                ai_priority_weight_minutes=0.0) for s in active}
    frames, worked_days, survey_minutes, travel_minutes = [], set(), 0, 0
    stopped_reason = ''
    while remaining:
        before = len(remaining)
        week_number = (week-original_week).days//7 + 1
        if progress:
            progress(week_number, before)
        try:
            for offset in range(5):
                day = week + timedelta(days=offset)
                # Best geographic home fit gets first choice; no artificial team spreading.
                def fit(person):
                    home = SURVEYOR_HOME_COORDINATES.get(person['name'].casefold())
                    if not home:
                        return float('inf')
                    distances = [haversine_km(*home, x.get('latitude'), x.get('longitude')) for x in remaining.values()]
                    return min((d for d in distances if d is not None), default=float('inf'))
                people = sorted((s for s in active if offset in s['weekdays']), key=lambda s: (fit(s), s['name']))
                for person in people:
                    pool = [x for x in remaining.values() if _site_allowed_today(x, day)]
                    if not pool:
                        continue
                    result = engines[person['name']].build_day(pool,
                        datetime.combine(day, first, TZ), datetime.combine(day, finish, TZ), datetime.combine(day, home_by, TZ))
                    if not result.items:
                        continue
                    frame = result.to_dataframe()
                    frame.insert(0, 'Week', week_number)
                    frame.insert(1, 'Date', day.isoformat())
                    frame.insert(2, 'Surveyor', person['name'])
                    frame['Client Help Required'] = frame['Customer Reference'].map(lambda ref: remaining.get(text(ref), {}).get('client_help', ''))
                    frame['Surveyor Description'] = frame['Customer Reference'].map(lambda ref: remaining.get(text(ref), {}).get('surveyor_description', ''))
                    frame['Scenario'] = 'Provisional — assumes client-arranged access'
                    frames.append(frame)
                    worked_days.add((day, person['name']))
                    survey_minutes += result.survey_minutes
                    travel_minutes += result.travel_minutes
                    for item in result.items:
                        ref = text(item.customer_reference)
                        if ref not in remaining:
                            raise ValueError('Duplicate or unknown building returned by client routing.')
                        remaining.pop(ref)
        except GoogleRoutesError as exc:
            stopped_reason = f'Routing stopped: {exc}. These buildings have not been placed; retry after resolving the routing problem.'
            break
        if len(remaining) == before:
            # Future release dates may legitimately leave a whole week empty.
            future = [pd.to_datetime(s['special_request_date']).date() for s in remaining.values()
                      if s.get('special_request_date') and pd.to_datetime(s['special_request_date']).date() > week+timedelta(days=4)]
            if future:
                next_date = min(future)
                week = max(week + timedelta(days=7), next_date - timedelta(days=next_date.weekday()))
                continue
            stopped_reason = 'No feasible weekday visit was found in a full week using the selected team, access days, survey hours and return deadlines. Check journey availability and these constraints.'
            break
        week += timedelta(days=7)
    unplaced.extend([ref, s['building_name'], stopped_reason] for ref, s in remaining.items())
    schedule = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    placed = schedule.loc[schedule['Customer Reference'].map(text).ne('')] if not schedule.empty else pd.DataFrame()
    summary = {'Buildings in scenario': len(sites), 'Buildings placed': len(placed), 'Not scheduled': len(unplaced),
               'Surveyor-days': len(worked_days), 'Calendar weeks': max((int(f['Week'].iloc[0]) for f in frames), default=0),
               'Survey hours': round(survey_minutes/60, 1), 'Travel hours': round(travel_minutes/60, 1),
               'Assumption': 'Client access arranged. Monday–Friday only. No artificial spacing or workload spreading.'}
    assert len(placed) + len(unplaced) == len(sites)
    return ClientProgramme(schedule, pd.DataFrame(unplaced, columns=UNPLACED_COLUMNS), summary)


def export_client_programme(programme, settings=None):
    # Reuse the application's existing Excel report writer and styling.
    from openpyxl import Workbook
    from access_report import table_sheet
    wb = Workbook(); wb.remove(wb.active)
    table_sheet(wb, 'Summary', ['Measure', 'Value'], list(programme.summary.items()), {1:32, 2:95})
    if not programme.schedule.empty:
        table_sheet(wb, 'Provisional Programme', list(programme.schedule), programme.schedule.values.tolist(),
                    {i+1:55 if c in ('Building Name', 'Client Help Required', 'Surveyor Description', 'Scenario') else 23 for i,c in enumerate(programme.schedule)})
    table_sheet(wb, 'Not Scheduled', UNPLACED_COLUMNS, programme.unplaced.values.tolist(), {1:25, 2:55, 3:95})
    if settings:
        table_sheet(wb, 'Planning Settings', ['Setting', 'Value'], list(settings.items()), {1:35, 2:95})
    output = BytesIO(); wb.save(output); return output.getvalue()
