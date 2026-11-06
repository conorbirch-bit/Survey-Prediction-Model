"""Portfolio coverage, independent of whether any selected days are empty."""
from collections import Counter
from datetime import timedelta

import pandas as pd

from audited_decisions import text


def programme_completion(portfolio, team_results, week_start, retry_decisions=None):
    """Do not announce completion after a prediction, mapping or routing omission.

    The input is the full retry-gated portfolio, including missing postcodes and
    prediction failures, with calculated earliest dates. Actual completion in
    Salesforce is deliberately distinct from allocation in this programme.
    """
    expected = set()
    unidentified = 0
    for row in portfolio.to_dict('records'):
        earliest = pd.to_datetime(row.get('Calculated Earliest Survey Date'), errors='coerce')
        if pd.notna(earliest) and earliest.date() > week_start + timedelta(days=6):
            continue
        ref = text(row.get('Customer Reference')).casefold()
        if ref:
            expected.add(ref)
        else:
            unidentified += 1
    mapping_holds = set()
    if retry_decisions is not None:
        for row in retry_decisions.to_dict('records'):
            if (row.get('Decision') in {'RETRY', 'RETRY_WITH_CONSTRAINT'}
                    and str(row.get('Booking Excluded')).lower() not in {'true','1'}
                    and str(row.get('Retry Eligible')).lower() not in {'true','1'}):
                mapping_holds.add(text(row.get('Customer Reference')).casefold() or text(row.get('Work Order Number')))
    items = [i for result in team_results.values() if result for day in result.days for i in day.items]
    counts = Counter(text(i.customer_reference).casefold() for i in items)
    remaining = expected - counts.keys()
    finished = bool(expected) and not remaining and not unidentified and not mapping_holds and all(counts[r] == 1 for r in expected)
    last = max((i for i in items if text(i.customer_reference).casefold() in expected), key=lambda i:i.survey_end, default=None)
    return {'finished': finished, 'eligible': len(expected), 'remaining': len(remaining)+unidentified+len(mapping_holds),
            'last_building': last.building_name if finished and last else '',
            'last_finish': last.survey_end if finished and last else None}
