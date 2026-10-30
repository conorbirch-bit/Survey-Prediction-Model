"""Separate Streamlit tab; imported without side effects for offline testing."""
import hashlib
import json
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from client_planning import client_candidates, build_client_programme, export_client_programme, UNPLACED_COLUMNS
from google_routes import GoogleTransitRouter
from retry_planner import build_retry_plan
from salesforce_master import read_salesforce_or_standard_excel
from audited_decisions import AUDIT_FILE


def render_client_planning(st, predict, normalise, convert_sites, get_secret, default_retry=None, default_master=None, model_fingerprint=''):
    st.subheader('Client Access Planning')
    st.write('Provisional workload programme assuming the client arranges access. Monday–Friday only, continuing across as many weeks as needed. This does not create bookings or change the normal schedule.')
    upload = st.file_uploader('Current appointment and cannot-complete workbook', type=['xlsx','xls'], key='client_history_upload')
    master_upload = st.file_uploader('Master portfolio for additional building data (optional)', type=['xlsx','xls'], key='client_master_upload')
    source = upload if upload is not None else default_retry
    master = master_upload if master_upload is not None else default_master
    if source is default_retry and source is not None:
        st.caption('Using the appointment workbook currently uploaded in Weekly scheduling.')
    today = datetime.now(ZoneInfo('Europe/London')).date()
    monday = today - timedelta(days=today.weekday()) + timedelta(days=7)
    start = st.date_input('Client programme start week', value=monday, min_value=today, key='client_start_week')
    start = start - timedelta(days=start.weekday())
    if start < today:
        start += timedelta(days=7)
    cols = st.columns(3)
    options = [time(h,m) for h in range(6,22) for m in (0,30)]
    first = cols[0].selectbox('Client first survey', options, index=options.index(time(9)), key='client_first', format_func=lambda v:v.strftime('%H:%M'))
    finish = cols[1].selectbox('Client last survey finish', options, index=options.index(time(15)), key='client_finish', format_func=lambda v:v.strftime('%H:%M'))
    home_by = cols[2].selectbox('Client return home by', options, index=options.index(time(17)), key='client_home_by', format_func=lambda v:v.strftime('%H:%M'))
    defaults = [('Conor Birch','Harpenden Station'), ('Rod Harrison','Rugby Station'),
                ('Toby Lawal','Grove Park Station, Baring Road, London SE12 0DZ'),
                ('Joe Reynolds','Hemel Hempstead Station'), ('Harrison Grice','Gravesend Station')]
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    availability = pd.DataFrame([{'Name':n,'Start / Finish Location':h, **{d:n!='Harrison Grice' for d in days}} for n,h in defaults])
    st.caption('This availability pattern repeats each week. Travel and survey times are retained; no added spacing or final-week spreading is applied.')
    edited = st.data_editor(availability, hide_index=True, num_rows='dynamic', key='client_availability',
                            column_config={d:st.column_config.CheckboxColumn(d) for d in days})
    buffers = st.columns(3)
    pre = buffers[0].number_input('Client before-survey buffer (min)', 0,30,5, key='client_pre')
    post = buffers[1].number_input('Client after-survey buffer (min)', 0,30,5, key='client_post')
    leeway = buffers[2].number_input('Client travel leeway (min)', 0,30,5, key='client_leeway')
    people = [{'name':str(r['Name'] or '').strip(), 'home':str(r['Start / Finish Location'] or '').strip(),
               'weekdays':[i for i,d in enumerate(days) if r[d] is True or r[d] == 1]}
              for r in edited.fillna('').to_dict('records')]
    settings = {'Start week':start.isoformat(), 'First survey':str(first), 'Last survey finish':str(finish),
                'Return home by':str(home_by), 'Before survey minutes':pre, 'After survey minutes':post,
                'Travel leeway minutes':leeway, 'Recurring availability':json.dumps(people),
                'Mode':'Client planning only; no bookings, Salesforce upload, added spacing or spreading.'}
    fingerprint = hashlib.sha256((source.getvalue() if source else b'') + (master.getvalue() if master else b'') + json.dumps(settings,sort_keys=True).encode() + str(model_fingerprint).encode() + AUDIT_FILE.read_bytes()).hexdigest()
    if st.button('Generate client planning schedule', key='generate_client_programme', type='primary'):
        st.session_state.pop('client_programme_result', None)
        if source is None:
            st.error('Upload current appointment history so completed and already-booked buildings can be excluded.')
        elif not get_secret('GOOGLE_MAPS_API_KEY'):
            st.error('Configure GOOGLE_MAPS_API_KEY to calculate realistic client-planning journeys.')
        else:
            try:
                with st.spinner('Preparing client-help buildings and routing consecutive working weeks...'):
                    plan = build_retry_plan(source.getvalue(), excluded_week_starts=[],
                            openai_api_key=get_secret('OPENAI_API_KEY'), openai_model=get_secret('OPENAI_MODEL','gpt-5.6'))
                    data = normalise(read_salesforce_or_standard_excel(master.getvalue())[0]) if master else pd.DataFrame()
                    candidates = client_candidates(plan, data)
                    if candidates.empty:
                        st.info('No outstanding, unscheduled client-help buildings remain in this upload.')
                    else:
                        predictions = predict(candidates)
                        valid = predictions['Prediction Status'].eq('Predicted') & predictions['Postcode'].fillna('').astype(str).str.strip().ne('')
                        bad = predictions.loc[~valid]
                        sites = convert_sites(predictions.loc[valid])
                        details = predictions.set_index('Customer Reference').to_dict('index')
                        for site in sites:
                            row = details[site['customer_reference']]
                            site['client_help'] = row['Client Help Required']
                            site['surveyor_description'] = row['Surveyor Description']
                            earliest = pd.to_datetime(row.get('Earliest Survey Date'), errors='coerce')
                            if pd.notna(earliest):
                                site['special_request_date'] = earliest.date()
                        status = st.empty()
                        result = build_client_programme(sites, people, start, GoogleTransitRouter(get_secret('GOOGLE_MAPS_API_KEY')),
                            first=first, finish=finish, home_by=home_by, pre_buffer=pre, post_buffer=post, travel_leeway=leeway,
                            progress=lambda week,left:status.info(f'Planning week {week}: {left} buildings remaining'))
                        status.empty()
                        if not bad.empty:
                            failed = pd.DataFrame([[r['Customer Reference'],r['Building Name'],
                                'Missing postcode/routing address.' if not str(r.get('Postcode') or '').strip() else 'Duration prediction failed: '+str(r['Prediction Error'])]
                                for r in bad.to_dict('records')], columns=UNPLACED_COLUMNS)
                            result.unplaced = pd.concat([result.unplaced, failed],ignore_index=True)
                            result.summary['Buildings in scenario'] += len(bad)
                            result.summary['Not scheduled'] += len(bad)
                        st.session_state['client_programme_result'] = (fingerprint,result,export_client_programme(result,settings))
            except Exception as exc:
                st.error(f'Could not build client planning programme: {exc}')
    saved = st.session_state.get('client_programme_result')
    if saved and saved[0] == fingerprint:
        _, result, export = saved
        cols = st.columns(4)
        for c,label in zip(cols,['Buildings placed','Surveyor-days','Calendar weeks','Not scheduled']):
            c.metric(label,result.summary[label])
        st.caption('Calendar weeks are based on the selected recurring team availability and assumed access. Routing is a feasible geographic heuristic, not a guaranteed mathematical minimum.')
        if not result.unplaced.empty:
            st.warning('The programme is incomplete for the buildings listed below. No-progress or routing failures are reported, rather than creating endless empty weeks.')
            st.dataframe(result.unplaced,hide_index=True,use_container_width=True)
        st.dataframe(result.schedule,hide_index=True,use_container_width=True)
        st.download_button('Download provisional client programme',export,
                           file_name=f'Client_Access_Programme_{start.isoformat()}.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',key='client_programme_download')
    elif saved:
        st.info('Inputs have changed. Generate the client programme again to update the estimate.')
