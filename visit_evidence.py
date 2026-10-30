"""Appointment-level categories: never copy a primary note to later visits."""
import pandas as pd
from audited_decisions import text

TECHNICAL = {'metro failure', 'drawing or planstudio issue'}
CATEGORY_REASONS = {
    'no answer': 'No answer at door',
    'resident refused access': 'Resident refused access',
    'appointment required': 'Appointment required before visiting',
    'escort or staff required': 'Staff escort required',
    'key / fob / code issue': 'Key or fob does not work',
    'building or entrance identification': 'Cannot find the building or entrance',
    'drawing or planstudio issue': 'Drawing or PlanStudio issue',
    'metro failure': 'Metro unable to attend',
    'building works': 'Building works prevent access',
}


def failed_visits(appointments, history):
    by_id = {}
    for _, row in history.iterrows():
        identity = text(row.get('Old Service Appointment ID'))
        if identity:
            by_id[identity] = {'id': identity, 'issue': text(row.get('AI Cannot Complete Issue')),
                               'reason': text(row.get('Failure Reason')),
                               'when': row.get('_event_dt'), 'resource': ''}
    for _, row in appointments.iterrows():
        identity = text(row.get('Service Appointment ID'))
        if row.get('_sa_status') != 'cannot complete' and identity not in by_id:
            continue
        item = by_id.setdefault(identity, {'id': identity, 'issue': '', 'reason': '', 'when': None, 'resource': ''})
        item['issue'] = text(row.get('AI Cannot Complete Issue')) or item['issue']
        if pd.notna(row.get('_actual_start_dt')):
            item['when'] = row['_actual_start_dt']
        for col in ('Assigned Service Resource: Name', 'Resource Name: Name', 'Resource Name'):
            if text(row.get(col)):
                item['resource'] = text(row[col]); break
    return sorted(by_id.values(), key=lambda v: str(v['when'] or ''))


def latest_issue_rule(visits, classify_access, customer_count, metro_count, reference):
    if not visits or not visits[-1]['issue']:
        return None
    issue = visits[-1]['issue'].casefold()
    reason = CATEGORY_REASONS.get(issue)
    if not reason:
        return None
    rule = classify_access(reason, customer_count, metro_count, reference)
    if rule:
        rule = dict(rule)
        rule['Decision Source'] = 'Latest failed appointment AI issue'
    return rule


def evidenced_saturday_retry(visits, base):
    """Return True/False for AI-category histories, None for the legacy fallback."""
    if not any(v['issue'] for v in visits):
        return None
    if base['Customer Failure Count'] != 2:
        return False
    customers = [v for v in visits if v['issue'].casefold() not in TECHNICAL]
    # Unknown categories do not prove no-answer; audited approvals are separate.
    return len(customers) == 2 and all(v['issue'].casefold() == 'no answer' for v in customers)
