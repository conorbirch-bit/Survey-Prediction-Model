"""Persistent call-ahead flags; separate from bookings and audit approvals."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from audited_decisions import text

STATUSES = ['To call', 'Access confirmed', 'Rearrangement needed']
COLUMNS = ['Customer Reference', 'Building Name', 'Week Commencing', 'Proposed Date',
           'Proposed Time', 'Surveyor', 'Phone', 'Access Instructions', 'Status', 'Contact Notes']


def state_path():
    root = Path(os.environ.get('SCHEDULER_STATE_DIR', Path(__file__).with_name('.scheduler_state')))
    root.mkdir(parents=True, exist_ok=True)
    return root / 'calls.sqlite3'


@contextmanager
def connect(path=None):
    db = sqlite3.connect(str(path or state_path()), timeout=15)
    db.execute('CREATE TABLE IF NOT EXISTS calls (reference TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    try:
        with db:
            yield db
    finally:
        db.close()


def call_rows(schedule, decisions):
    if schedule.empty or decisions.empty:
        return pd.DataFrame(columns=COLUMNS)
    approved = {text(r.get('Customer Reference')): r for r in decisions.to_dict('records')
                if r.get('Call Ahead Required') is True and r.get('Retry Eligible') is True}
    rows = []
    for item in schedule.to_dict('records'):
        ref = text(item.get('Customer Reference'))
        if ref not in approved:
            continue
        row = approved[ref]
        when = pd.to_datetime(item['Date']).date()
        rows.append(dict(zip(COLUMNS, [ref, row['Building Name'],
            (when - timedelta(days=when.weekday())).isoformat(), when.isoformat(),
            item['Survey Start'], item['Surveyor'], row.get('Call Phone', ''),
            row.get('Audit Instruction', ''), 'To call', ''])))
    return pd.DataFrame(rows, columns=COLUMNS).drop_duplicates('Customer Reference')


def save_proposals(frame, path=None):
    """Keep confirmation only while the proposed date/time/surveyor is unchanged."""
    with connect(path) as db:
        for row in frame.to_dict('records'):
            old = db.execute('SELECT payload FROM calls WHERE reference=?', (row['Customer Reference'],)).fetchone()
            if old:
                old = json.loads(old[0])
                if all(row[k] == old[k] for k in ('Proposed Date', 'Proposed Time', 'Surveyor')):
                    row.update({k: old.get(k, '') for k in ('Status', 'Contact Notes')})
            db.execute('INSERT OR REPLACE INTO calls VALUES (?,?)', (row['Customer Reference'], json.dumps(row)))


def read_calls(path=None):
    with connect(path) as db:
        rows = [json.loads(r[0]) for r in db.execute('SELECT payload FROM calls ORDER BY reference')]
    return pd.DataFrame(rows, columns=COLUMNS)


def update_calls(frame, path=None):
    with connect(path) as db:
        for row in frame.to_dict('records'):
            if row['Status'] not in STATUSES:
                raise ValueError('Invalid call status.')
            old = db.execute('SELECT payload FROM calls WHERE reference=?', (row['Customer Reference'],)).fetchone()
            if not old:
                continue
            saved = json.loads(old[0])
            if any(saved[k] != row[k] for k in ('Proposed Date', 'Proposed Time', 'Surveyor')):
                raise ValueError('The proposal changed in another session. Refresh the call list before saving.')
            saved.update({k: text(row[k]) for k in ('Status', 'Contact Notes')})
            db.execute('UPDATE calls SET payload=? WHERE reference=?', (json.dumps(saved), row['Customer Reference']))


def add_call_notes(schedule, calls):
    result = schedule.copy()
    if result.empty:
        return result
    by_ref = {r['Customer Reference']: r for r in calls.to_dict('records')}
    result['Access Confirmation'] = result['Customer Reference'].map(
        lambda ref: by_ref.get(text(ref), {}).get('Status', ''))
    return result


def render_call_panel(st, retry_bytes=None):
    st.subheader('Calls required this week')
    calls = read_calls()
    if retry_bytes and not calls.empty:
        from retry_planner import _detect_retry_tabs, _prepare_sa_mapping
        from audited_decisions import load_decisions, linked_appointments
        appointments = _prepare_sa_mapping(_detect_retry_tabs(retry_bytes)[0])
        records = load_decisions()
        closed = []
        for ref in calls['Customer Reference']:
            record = records.get(ref.casefold())
            if not record:
                continue
            linked = linked_appointments(record, appointments)
            later_failure = set(linked.loc[linked['_sa_status'].eq('cannot complete'), 'Service Appointment ID']) - set(record['failed_appointment_ids'])
            if linked['_sa_status'].eq('completed').any() or later_failure:
                closed.append(ref)
        calls = calls.loc[~calls['Customer Reference'].isin(closed)]
    if calls.empty:
        st.caption('Call-ahead visits will appear here after the normal weekly schedule is generated.')
        return
    weeks = sorted(calls['Week Commencing'].unique())
    week = st.selectbox('Call list week commencing', weeks, index=len(weeks)-1, key='call_list_week')
    frame = calls.loc[calls['Week Commencing'].eq(week)].copy()
    st.warning(f"{len(frame.loc[frame['Status'].ne('Access confirmed')])} visit(s) still need access confirmation. Proposed slots remain provisional.")
    edited = st.data_editor(frame, hide_index=True, use_container_width=True, key=f'call_flags_{week}',
        disabled=[c for c in COLUMNS if c not in ('Status', 'Contact Notes')],
        column_config={'Status': st.column_config.SelectboxColumn(options=STATUSES, required=True)})
    if st.button('Save call updates', key='save_call_updates'):
        update_calls(edited)
        st.success('Call updates saved. Changed proposal dates or times will require fresh confirmation.')
    current = read_calls()
    current = current.loc[current['Customer Reference'].isin(frame['Customer Reference'])]
    st.download_button('Download current call flags (CSV)', current.to_csv(index=False).encode('utf-8-sig'),
                       file_name='Calls_Required.csv', mime='text/csv', key='calls_csv')
