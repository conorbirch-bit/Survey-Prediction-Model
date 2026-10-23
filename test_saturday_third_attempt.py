from datetime import date
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from access_report import action_group
from retry_planner import build_retry_plan, apply_retry_plan_to_portfolio
from scheduler_v20_10 import _site_allowed_today, DailyTransitScheduler
from test_booking_exclusions import reports
from test_cohesive_access_groups import app_sites
from test_scheduler_regressions import site, Router, TZ


class SaturdayThirdAttemptTests(TestCase):
    def plan(self, flats=4, reasons=('No answer at door', 'No response'),
             rollup=1, completed=False, booked=False, replacement=True,
             extra_failed=False, reason_code='Customer', duplicate=False, latest_only=False):
        appointments, failures = reports([dict(customer=rollup, metro=0)])
        appointments['Status.1'] = ['Cannot Complete', 'Completed' if completed else 'New']
        failures['Resource Name: Name'] = 'Toby Lawal'
        failures['Sovereign Flat'] = flats
        failures['Reason Not Complete'] = reason_code
        failures['Primary Service Appointment: Reason Description'] = reasons[0]
        failures['Cancelation Reason Description'] = None
        original_a, original_f = appointments.iloc[0].copy(), failures.iloc[0].copy()
        for number, reason in enumerate(reasons[1:], 2):
            a, f = original_a.copy(), original_f.copy()
            identity = f'08pVISIT{number:010d}'
            a['Service Appointment ID'] = identity
            f['Primary Service Appointment: Service Appointment ID'] = identity
            a['Actual Start'] = f'{19 + number}/10/2026, 10:00'
            f['Actual Start'] = a['Actual Start']
            f['Primary Service Appointment: Reason Description'] = reason
            appointments = pd.concat([appointments, pd.DataFrame([a])], ignore_index=True)
            failures = pd.concat([failures, pd.DataFrame([f])], ignore_index=True)
        if booked:
            appointments.loc[1, 'Scheduled Start'] = '27/10/2026, 10:00'
        if not replacement:
            appointments = appointments.drop(index=1)
        if extra_failed:
            a = original_a.copy(); a['Service Appointment ID'] = '08pOTHERFAIL000001'
            appointments = pd.concat([appointments, pd.DataFrame([a])], ignore_index=True)
        if duplicate:
            failures = pd.concat([failures, failures], ignore_index=True)
        if latest_only:
            failures = failures.tail(1).copy()
            failures['Failed Visits (Customer)'] = len(reasons)
        with patch('retry_planner._detect_retry_tabs', return_value=(appointments, failures)):
            return build_retry_plan(b'fixture', excluded_week_starts=[date(2026, 10, 26)])

    def test_two_evidenced_no_answers_open_small_saturday_trial(self):
        plan = self.plan()
        row = plan.decisions.iloc[0]
        self.assertTrue(row['Retry Eligible'])
        self.assertEqual(row['Reason Category'], 'SATURDAY_THIRD_NO_ANSWER')
        self.assertEqual(row['Required Weekdays'], ['Saturday'])
        self.assertEqual(row['Customer Failure Count'], 2)
        self.assertEqual(action_group(row), 'Third attempt — Saturday only')
        self.assertTrue(plan.client_access_required.empty)

    def test_blank_descriptions_count_as_no_answer(self):
        self.assertTrue(self.plan(reasons=(None, '')).decisions.iloc[0]['Retry Eligible'])

    def test_size_boundary_and_unknowns(self):
        for flats in [1, 5]:
            self.assertTrue(self.plan(flats=flats).decisions.iloc[0]['Retry Eligible'])
        for flats in [0, 6, 20, None, float('nan'), 'unknown']:
            self.assertFalse(self.plan(flats=flats).decisions.iloc[0]['Retry Eligible'])

    def test_refusals_barriers_and_generic_no_access_do_not_qualify(self):
        for reason in ['Resident refused access', 'No answer; intercom broken',
                       'No answer; need appointment', 'No access']:
            for reasons in [(reason, 'No answer at door'), ('No answer at door', reason)]:
                self.assertFalse(self.plan(reasons=reasons).decisions.iloc[0]['Retry Eligible'])
        self.assertFalse(self.plan(reason_code='Resident refused access').decisions.iloc[0]['Retry Eligible'])

    def test_no_fourth_visit_or_unexplained_additional_failure(self):
        for kwargs in [dict(rollup=3), dict(reasons=('No answer',)*3), dict(extra_failed=True)]:
            self.assertFalse(self.plan(**kwargs).decisions.iloc[0]['Retry Eligible'])

    def test_rollup_alone_cannot_establish_both_reasons(self):
        self.assertFalse(self.plan(reasons=('No answer',), rollup=2).decisions.iloc[0]['Retry Eligible'])

    def test_salesforce_rollup_with_two_failed_appointments_and_latest_reason(self):
        row = self.plan(latest_only=True).decisions.iloc[0]
        self.assertTrue(row['Retry Eligible'])
        self.assertIn("other visit's description is unavailable", row['Decision Reason'])
        self.assertFalse(self.plan(latest_only=True, reasons=('No answer', 'Resident refused access')).decisions.iloc[0]['Retry Eligible'])
        self.assertFalse(self.plan(latest_only=True, extra_failed=True).decisions.iloc[0]['Retry Eligible'])

    def test_rollup_does_not_require_note_to_refer_to_latest_failed_visit(self):
        appointments, failures = reports([dict(customer=2, metro=0)])
        appointments['Status.1'] = ['Cannot Complete', 'New']
        extra = appointments.iloc[0].copy()
        extra['Service Appointment ID'] = '08pLATER000000001'
        extra['Actual Start'] = '21/10/2026, 10:00'
        appointments = pd.concat([appointments, pd.DataFrame([extra])], ignore_index=True)
        failures['Sovereign Flat'] = 2
        failures['Primary Service Appointment: Reason Description'] = 'No answer'
        with patch('retry_planner._detect_retry_tabs', return_value=(appointments, failures)):
            plan = build_retry_plan(b'fixture')
        self.assertTrue(plan.decisions.iloc[0]['Retry Eligible'])
        self.assertIn('only one failed-visit description', plan.decisions.iloc[0]['Decision Reason'])

    def test_duplicate_rows_are_not_extra_visits(self):
        row = self.plan(duplicate=True).decisions.iloc[0]
        self.assertEqual(row['Customer Failure Count'], 2)
        self.assertTrue(row['Retry Eligible'])

    def test_completed_and_booked_stay_excluded(self):
        self.assertEqual(self.plan(completed=True).decisions.iloc[0]['Decision'], 'RESOLVED')
        plan = self.plan(booked=True)
        self.assertTrue(plan.decisions.empty)
        self.assertEqual(len(plan.booking_exclusions), 1)

    def test_valid_replacement_still_required(self):
        row = self.plan(replacement=False).decisions.iloc[0]
        self.assertEqual(row['Reason Category'], 'SATURDAY_THIRD_NO_ANSWER')
        self.assertFalse(row['Retry Eligible'])

    def test_required_weekday_conflict_stays_with_client(self):
        row = self.plan(reasons=('No answer; office open Monday only', 'No answer')).decisions.iloc[0]
        self.assertFalse(row['Retry Eligible'])

    def test_engine_size_and_saturday_gate_cannot_be_bypassed_by_metadata(self):
        sat = date(2026, 10, 31)
        for flats in [1, 5, 6, None]:
            candidate = site(1, is_retry=True, flats=flats, retry_reason_category='SATURDAY_THIRD_NO_ANSWER')
            self.assertEqual(_site_allowed_today(candidate, sat), flats in (1, 5))
            self.assertFalse(_site_allowed_today(candidate, date(2026, 10, 30)))

    def test_portfolio_and_export_carry_saturday_label(self):
        plan = self.plan()
        pool, _ = apply_retry_plan_to_portfolio(pd.DataFrame(), plan.decisions)
        pool['Planning Duration (Minutes)'] = 20
        pool['Predicted Survey Duration (Minutes)'] = 20
        pool['Prediction Confidence'] = 'High'
        pool['Prediction Model Used'] = 'Fixture'
        # Use the application's real converter, extracted without launching UI.
        sites = app_sites(pool)
        self.assertEqual(sites[0]['retry_reason_category'], 'SATURDAY_THIRD_NO_ANSWER')
        from datetime import datetime, time
        sat = date(2026, 10, 31)
        result = DailyTransitScheduler(Router()).build_day(
            sites, *[datetime.combine(sat, t, tzinfo=TZ) for t in (time(10), time(13), time(14))])
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.to_dataframe().iloc[0]['Visit Note'], 'Third attempt — Saturday only')
