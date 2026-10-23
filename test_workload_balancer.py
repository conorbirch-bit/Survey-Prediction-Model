import ast
from datetime import datetime, time, timedelta
from pathlib import Path
from unittest import TestCase

from google_routes import GoogleRoutesError
from scheduler_v20_10 import DailyTransitScheduler, WeeklyScheduleResult
from team_scheduler import SurveyorConfig, SURVEYOR_HOME_COORDINATES
from workload_balancer import rebalance_team_workload, rebuild_day, _groups
from test_scheduler_regressions import Router, site, DAY, TZ


class WorkloadBalancerTests(TestCase):
    def setup_week(self, rows, days=2, finish=time(10), router=None, second_person=False):
        self.router = router or Router()
        self.finish = finish
        self.people = [SurveyorConfig('A', 'Home', [DAY + timedelta(days=i) for i in range(days)])]
        if second_person:
            self.people.append(SurveyorConfig('B', 'Home', list(self.people[0].available_dates)))
        self.rows = rows
        self.pool = {'A': rows}
        self.results = {'A': self.factory(self.people[0]).build_week(rows, [DAY], time(8), finish, time(17), TZ)}
        if second_person:
            self.pool['B'] = []
            self.results['B'] = None
        return self.results

    def factory(self, person):
        return DailyTransitScheduler(self.router, home_location=person.start_location,
            surveyor_name=person.name, travel_leeway_minutes=0,
            pre_survey_buffer_minutes=0, post_survey_buffer_minutes=0,
            same_postcode_transfer_minutes=0, minimum_survey_to_travel_ratio=0)

    def balance(self, **kwargs):
        return rebalance_team_workload(self.results, self.people, self.pool, self.factory,
            time(8), self.finish, time(17), TZ, reserve_sites=self.rows, enabled=True, **kwargs)

    def test_disabled_is_exact_noop_without_router_calls(self):
        results = self.setup_week([site(i) for i in range(4)])
        calls = len(self.router.routes)
        out, moves, review = rebalance_team_workload(results, self.people, self.pool, self.factory,
            time(8), time(10), time(17), TZ)
        self.assertIs(out, results)
        self.assertTrue(moves.empty and review.empty)
        self.assertEqual(calls, len(self.router.routes))

    def test_covers_all_days_retains_buildings_and_originals(self):
        original = self.setup_week([site(i, 30) for i in range(6)], days=3, finish=time(11))
        before = original['A'].full_schedule_dataframe().copy()
        out, _, review = self.balance()
        self.assertEqual([len(day.items) for day in out['A'].days], [2, 2, 2])
        self.assertEqual(sorted(i.customer_reference for d in out['A'].days for i in d.items), [str(i) for i in range(6)])
        self.assertTrue(before.equals(original['A'].full_schedule_dataframe()))
        self.assertTrue((review['Additional Balancing Buffer (Minutes)'] == 30).all())
        self.assertTrue(review['Outcome'].str.contains('Shortfall').all())

    def test_full_day_is_unchanged_and_has_no_added_allowance(self):
        original = self.setup_week([site(1, 60), site(2, 60)], days=1)
        out, _, _ = self.balance()
        self.assertIs(out['A'].days[0], original['A'].days[0])
        self.assertEqual(sum(i.balancing_buffer_minutes for i in out['A'].days[0].items), 0)

    def test_buffer_only_between_visits_and_stop_when_full(self):
        self.setup_week([site(1, 30), site(2, 30), site(3, 30)], days=1)
        out, _, review = self.balance()
        day = out['A'].days[0]
        self.assertEqual([i.balancing_buffer_minutes for i in day.items], [0, 15, 15])
        self.assertEqual([i.survey_minutes for i in day.items], [30, 30, 30])
        self.assertEqual(day.items[-1].survey_end.hour, 10)
        self.assertEqual(review.iloc[0]['Minutes Still Available'], 0)
        self.assertEqual(day.travel_minutes, 20)
        self.assertEqual(day.to_dataframe()['Additional Balancing Buffer (Minutes)'].sum(), 30)

    def test_one_building_day_is_not_padded(self):
        self.setup_week([site(1, 30)], days=1)
        out, _, review = self.balance()
        self.assertEqual(out['A'].days[0].items[0].survey_end.hour, 8)
        self.assertEqual(review.iloc[0]['Additional Balancing Buffer (Minutes)'], 0)
        self.assertIn('Shortfall', review.iloc[0]['Outcome'])

    def test_access_day_and_requested_day_remain_protected(self):
        for field in ['retry_required_weekdays', 'retry_preferred_weekdays']:
            self.setup_week([site(i, 30, is_retry=True, **{field: ['Monday']}) for i in range(4)])
            out, _, _ = self.balance()
            self.assertEqual([len(day.items) for day in out['A'].days], [4, 0])

    def test_accepted_weekly_note_stays_on_date_and_owner(self):
        self.setup_week([site(i, 30, special_request_date=DAY) for i in range(4)], second_person=True)
        out, moves, _ = self.balance()
        self.assertEqual([len(day.items) for day in out['A'].days], [4, 0])
        self.assertEqual(out['B'].total_surveys, 0)
        self.assertTrue(moves.empty)

    def test_cross_person_moves_report_final_owner_once(self):
        self.setup_week([site(i, 30) for i in range(4)], days=1, second_person=True)
        out, moves, _ = self.balance()
        self.assertEqual(out['A'].total_surveys, 2)
        self.assertEqual(out['B'].total_surveys, 2)
        self.assertEqual(len(moves), 2)
        self.assertEqual(set(moves['Original Surveyor']), {'A'})
        self.assertEqual(set(moves['Surveyor']), {'B'})

    def test_original_failed_reviewer_cannot_receive_moved_work(self):
        self.setup_week([site(i, 30, is_retry=True, retry_review_original_surveyor='B') for i in range(4)],
                        days=1, second_person=True)
        out, _, _ = self.balance()
        self.assertEqual(out['B'].total_surveys, 0)

    def test_explicit_groups_move_whole_and_keep_other_groups_together(self):
        rows = [site(i, 30, visit_group='G1' if i < 2 else 'G2') for i in range(4)]
        self.setup_week(rows)
        out, _, _ = self.balance()
        for day in out['A'].days:
            ids = {i.customer_reference for i in day.items}
            self.assertIn(ids, [{'0', '1'}, {'2', '3'}])

    def test_local_group_preferred_but_can_split_for_empty_days(self):
        rows = [site(i, 30, latitude=51.5, longitude=-0.1) for i in range(4)]
        self.setup_week(rows)
        groups = _groups(self.results['A'].days[0].items, {str(i): s for i, s in enumerate(rows)})
        self.assertEqual(len(groups), 1)
        out, _, _ = self.balance()
        self.assertEqual([len(day.items) for day in out['A'].days], [2, 2])

    def test_saturday_never_receives_fresh_buildings(self):
        self.setup_week([site(i, 30) for i in range(4)], days=1)
        self.people[0].available_dates.append(DAY + timedelta(days=5))
        out, _, _ = self.balance(saturday_time_window=(time(10), time(13), time(14)))
        self.assertEqual(len(out['A'].days[1].items), 0)

    def test_pending_small_blocks_large_saturday_transfer(self):
        self.setup_week([site(i, 30, is_retry=True, flats=8) for i in range(4)], days=1)
        self.people[0].available_dates.append(DAY + timedelta(days=5))
        self.rows = self.rows + [site(99, 300, is_retry=True, flats=2, retry_required_weekdays=['Saturday'])]
        out, _, _ = self.balance(saturday_time_window=(time(10), time(13), time(14)))
        self.assertEqual(len(out['A'].days[1].items), 0)

    def test_saturday_half_day_and_small_retries(self):
        self.setup_week([site(i, 30, is_retry=True, flats=2) for i in range(4)], days=1)
        self.people[0].available_dates.append(DAY + timedelta(days=5))
        out, _, _ = self.balance(saturday_time_window=(time(10), time(11), time(12)))
        sat = out['A'].days[1]
        self.assertTrue(sat.items)
        self.assertGreaterEqual(sat.items[0].survey_start.time(), time(10))
        self.assertLessEqual(sat.items[-1].survey_end.time(), time(11))
        self.assertLessEqual(sat.return_time.time(), time(12))

    def test_routing_error_retains_original_day_and_reports_it(self):
        original = self.setup_week([site(i, 30) for i in range(3)], days=1)
        self.router.outage = GoogleRoutesError('offline')
        out, _, review = self.balance()
        self.assertEqual([i.customer_reference for i in out['A'].days[0].items], ['0', '1', '2'])
        self.assertEqual(out['A'].days[0].items[-1].survey_end, original['A'].days[0].items[-1].survey_end)
        self.assertIn('Routing trial failed', review.iloc[0]['Details'])

    def test_return_limit_rejects_extra_allowance_without_dropping_visit(self):
        rows = [site(1, 30), site(2, 30)]
        self.setup_week(rows, days=1)
        first, finish, deadline = [datetime.combine(DAY, t, tzinfo=TZ) for t in [time(8), time(10), time(9, 25)]]
        scheduler = self.factory(self.people[0])
        self.assertIsNone(rebuild_day(scheduler, rows, first, finish, deadline, buffer_minutes=30))
        valid = rebuild_day(scheduler, rows, first, finish, deadline, buffer_minutes=15)
        self.assertEqual(len(valid.items), 2)
        self.assertLessEqual(valid.return_time, deadline)

    def test_buffer_requeries_transit_at_later_departure_and_keeps_duration(self):
        rows = [site(1, 30, local=False), site(2, 30, local=False)]
        self.setup_week(rows, days=1)
        values = [datetime.combine(DAY, t, tzinfo=TZ) for t in [time(8), time(11), time(12)]]
        result = rebuild_day(self.factory(self.people[0]), rows, *values, buffer_minutes=20)
        matrix = [call for call in self.router.matrices if call[0] == 'Site 1'][-1]
        self.assertEqual(matrix[2].time(), time(8, 50))
        self.assertEqual(result.items[1].travel_minutes, 10)
        self.assertEqual(result.items[1].balancing_buffer_minutes, 20)

    def test_lunch_still_fits_when_buffers_cross_midday(self):
        rows = [site(i, 45) for i in range(4)]
        self.setup_week(rows, days=1, finish=time(15))
        values = [datetime.combine(DAY, t, tzinfo=TZ) for t in [time(10), time(15), time(16)]]
        result = rebuild_day(self.factory(self.people[0]), rows, *values, buffer_minutes=20)
        self.assertEqual(len(result.items), 4)
        self.assertGreaterEqual(result.lunch_start.time(), time(11, 45))
        self.assertLessEqual(result.lunch_start.time(), time(13))
        self.assertEqual((result.lunch_end - result.lunch_start).total_seconds(), 1800)

    def test_first_visit_stays_before_retry_after_moving(self):
        self.setup_week([site(i, 30, never_visited=i < 2, is_retry=i >= 2) for i in range(4)])
        out, _, _ = self.balance()
        for day in out['A'].days:
            ids = [int(i.customer_reference) for i in day.items]
            self.assertEqual([i < 2 for i in ids], sorted([i < 2 for i in ids], reverse=True))

    def test_app_runs_optional_balancing_after_filling_before_exports(self):
        source = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
        tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        balance = next(n for n in calls if n.func.id == 'rebalance_team_workload')
        fills = [n.lineno for n in calls if n.func.id == 'fill_team_gaps']
        review = next(n for n in calls if n.func.id == 'build_first_visit_review')
        self.assertGreater(balance.lineno, max(fills))
        self.assertLess(balance.lineno, review.lineno)
        self.assertIn('sheet_name="Workload Balancing"', source)
        self.assertIn('Grove Park Station, Baring Road, London SE12 0DZ', source)
        self.assertEqual(SURVEYOR_HOME_COORDINATES['toby lawal'], (51.43044575, 0.022578266))
