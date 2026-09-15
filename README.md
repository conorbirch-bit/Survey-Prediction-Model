# Site Survey Scheduling Agent — Version 20.11.2

This is a focused update to the uploaded 20.11 code. The older version notes below are retained as history.

## Install and run

Extract the ZIP and copy the contents of `scheduling_agent` into your existing project folder, replacing matching files. Filenames are restored to their importable names: `app.py`, `team_scheduler.py`, etc., without the download suffix `(1)`.

Keep your existing API secrets and `Predictive Model.xlsx`. No secrets or real survey data are included. If the training workbook is not alongside `app.py`, upload your completed-surveys workbook through the sidebar.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For an existing Streamlit deployment, replace the project files and reboot the app. The page should show **Version 20.11.2**. The active scheduler module remains `scheduler_v20_10.py`; the older scheduler modules are retained but are not imported by the app.

## Version 20.11.2 changes

- When the current site has no coordinates, coordinate sequencing ranks are disabled, allowing measured Google times to choose between otherwise equally preferred candidates. Unknown destination coordinates do not receive invented nearest-neighbour ranks. Existing local and same-road preferences remain in place. This fixes a reproduced 30-minute move being selected ahead of an available 2-minute move.
- After initial home-fit allocation, workload repair now considers everyone whose predicted candidate workload is below their share of available days, including people with a small nonzero allocation. Transfers come from colleagues' surplus while retaining their targets. The target is capped at the selected survey window after lunch, plus the existing 25% candidate reserve. It is not a cap on the number of candidate sites retained.
- Transfers prefer lower home transit times and continuity, use the existing home-to-cluster matrix, and add no allocation-stage Google calls. They change candidate pools; the day scheduler continues to enforce lunch, survey/return deadlines, availability and retry restrictions. A candidate allocation is not a promise that all jobs fit the final schedule.
- Surveyors with no selected dates do not receive allocations.
- The app displays retry coordinate coverage. An added `Run Settings` worksheet records the version, selected time windows and coordinate counts to make future output checks reproducible.

## Uploaded report check

The report `15.09.2026 8.08pm(1).xlsx` contains usable coordinates and an unambiguous replacement appointment for all 361 Work Orders. The importer already supports this layout; no workbook changes were needed. An offline replay reused the existing AI access decisions from `Team_Survey_Week_2026-09-21.xlsx` to avoid new AI calls: all 274 approved retries retained their coordinates, and all 685 portfolio rows had valid coordinates after clustering. Planning clusters changed from 75 to 65 when the coordinate-based layer was recomputed.

Separately, applying workload repair to the OLD allocations and their saved home matrix increased Rod's candidate allocation from 22 to 75 sites (roughly 202 to 1,003 estimated survey minutes), retaining 666 distinct candidate rows across the team. That isolates the allocation repair; it is not a forecast of the new live schedule, since corrected coordinates will change clusters and require fresh routing.

## Retained Version 20.11.1 fixes

- The daily scheduler checks all already-routed candidates before concluding that no more work fits. Previously it examined only the first eight.
- When the first geographic batch has no feasible work, it checks the next batch. Each batch still contains at most eight geographic representatives. Rejected candidates are not re-queried at the same position/time; a survey or lunch starts a fresh search state.
- Cheap duration checks reject jobs that cannot fit before requesting exact outbound/return routes. Fallback searches can use additional Google calls when needed to find usable work; this is not a guarantee of unchanged API cost.
- Google service, quota and permission failures propagate to the app's error handler instead of silently appearing as infeasible sites. A successful request with no public-transport route still allows another site to be tried. Matrix element errors are surfaced even when the HTTP response succeeds.
- If a later AI retry-triage batch fails, completed decisions are retained. Cases with no valid decision remain excluded and a warning identifies the partial triage.
- If lunch must occur before a long first survey, the first survey starts after lunch rather than being incorrectly reset to the original morning time.

The duration models, far-cluster efficiency threshold, weekly notes, Salesforce export format and retry business rules retain the uploaded implementation. Workload repair occurs before daily routing; there is no additional reassignment of leftovers after the individual weeks have been built.

The final AI narrative is generated after the schedules: failure of that summary does not itself alter the schedules. An earlier triage failure can reduce the retry pool; a Google routing failure now stops the run visibly.

## Verification

```bash
python -m unittest discover -s tests -v
```

23 offline regression checks passed, including the new missing-coordinate routing and workload repair cases. These cover candidate search beyond eight, geographic-batch exhaustion, route failures, retry overlap and missing-building insertion, completed AI batch retention, lunch, return deadlines, retry weekday exclusion, and a local route with a rejected inefficient remote detour. The first three regression cases were reproduced as failures against the uploaded code before applying the fixes. All Python modules compiled. Streamlit startup passed with synthetic duration-training data.

No live Google/OpenAI calls were made. Import/clustering and allocation were checked against the supplied files as described above. Startup used synthetic training data. An actual full-week rerun with your completed-surveys training workbook and current input reports is still required to measure survey/travel time and final daily utilisation. An underfilled day may remain where hard constraints or practical travel prevent further work.

## Historical release notes


## Version 20 duration-model change

Duration prediction is now split into independent model families based on Sovereign Flat count:

- **Garage:** 0 flats. Garage history is kept separate from residential history. Height/area are used when enough garage rows exist; otherwise the historical mean garage duration is used.
- **Small residential:** 1–6 flats inclusive. This family has its own regressions trained only on completed 1–6-flat buildings.
- **Standard/larger residential:** 7+ flats. This family has its own regressions trained only on completed 7+-flat buildings.
- **Flat count missing:** the existing all-residential Height + Area fallback is retained because the size family cannot be identified.

The existing rule remains: when Ground Floor Area is unavailable for a residential building with a known flat count, prediction uses that segment's **flats-only** equation. Planning Survey Duration remains identical to Raw Predicted Duration.

# Site Survey Scheduling Agent — Version 19

This version keeps the Version 16 weekly notes and same-campus routing behaviour, with these duration/schedule updates:

- Drawing priority is no longer shown on the Streamlit page, but remains in the downloaded workbook.
- Planning survey duration equals the raw predicted duration (no percentage uplift / 5-minute rounding).
- If ground-floor area is unavailable, the duration model uses Sovereign flats only. If flats are missing but area is present, height + area remains available as a fallback.
- 0-flat records are treated as garages and excluded from the residential model training; missing flat counts are retained.
- Weekly scheduling shows a per-building prediction-reliability table using training rows, LOOCV MAE/RMSE, MAE as a percentage of the prediction and an MAE reference band.
- Schedule tables include Building Height, Sovereign Flat and Internal Ground Floor Area (m2).

All other Version 16 routing, clustering, weekly-note hard-rule protection, Google-cost controls and workbook outputs are retained.


## Version 18 completed-file import
The completed-surveys training upload now accepts both a normal row-1 table and Salesforce-style completed-surveys reports with report title/filter rows above the real header row. The model and scheduling rules are otherwise unchanged from Version 17.


## Version 18.1 fix
The completed-surveys upload parser now lives in `app.py`, avoiding deployment/cache mismatches where an older `DurationPredictor` class did not yet expose `read_training_excel`. Both original and Salesforce report formats remain supported.


## Version 19 To Do / master portfolio import
The weekly scheduling master-portfolio upload now accepts the Salesforce To Do report layout used in `21.08.2026 - 12.17 Chat.xlsx`, including report title/filter rows above the real table header, grouped Work Type/Status rows, Salesforce Total/Count footer rows, and the `Ground Floor Area (m2)` field name. `Ground Floor Area (m2)` is mapped to the existing canonical `Internal Ground Floor Area (m2)` predictor input. All Version 18.1 scheduling, prediction, notes, routing and export behaviour is otherwise unchanged.


## Version 20.2 breakpoint
The segmented duration breakpoint is now Garage = 0 flats, Small residential = 1–6 flats, Standard/larger residential = 7+ flats. All other Version 20.1 behaviour is unchanged.


## Version 20.3 — Released-only weekly scheduling
The selected-week schedule has a hard Salesforce release gate. A site is routable/schedulable only when `Work Type Name` is exactly `Geospatial Asset Mapping` and `Status` is exactly `Released` (case/whitespace insensitive). Other portfolio rows are retained for drawing priority and portfolio exports but cannot enter the week's Google shortlist or schedule. All Version 20.2 duration-model behavior is unchanged.

## Version 20.4 — Endgame-aware rolling-horizon planning
The weekly cluster pre-filter now looks across the whole remaining master portfolio before Google routing so the project does not simply consume the strongest clusters first and leave inefficient geographic orphans at the end.

- The hard weekly gate is unchanged: only `Geospatial Asset Mapping` + `Released` rows can be scheduled this week.
- Future pipeline is identified cheaply from Plan Drafting / Needs Drawing rows and Geospatial Asset Mapping rows that are not yet Released.
- Each postcode-district summary now shows Future Pipeline Sites, Future Drawing Pipeline, Future GAM Awaiting Release, Suggested Anchor Reserve, Endgame Risk and Endgame Reason.
- Where useful, a small number of currently Released sites are marked as future geographic anchor candidates. Same-campus future support is preferred first, then exact-postcode support, then postcode-district support.
- The strategic target can deliberately shortlist fewer than all Released sites in an area, leaving anchor sites for future work.
- A deterministic endgame guardrail replaces preserved anchors with non-anchor candidates elsewhere where possible. If there is not enough alternative candidate capacity, anchors are released again so the current week is not left under-supplied.
- Endgame planning is portfolio-only and makes no future-week Google calls. Google remains restricted to the one selected week.
- A new on-page `Endgame / orphan-risk planning` table and an `Endgame Plan` workbook sheet make the decisions auditable.

All Version 20.3 duration prediction, Salesforce imports, weekly notes, same-campus logic, buffers, surveyor availability, Released-only eligibility and workbook outputs remain unchanged.


## Version 20.5 — Default surveyor team
The weekly surveyor table is pre-filled with Conor Birch (Harpenden Station), Rod Harrison (Rugby Station), Toby Lawal (Chadwell Heath Station), Harrison Grice (Gravesend Station), and Joe Reynolds (Hemel Hempstead Station), plus two blank spare surveyor slots. Only Conor is available by default; other surveyors generate no Google routing until availability dates are ticked.
\n\n## Version 20.6 — Lunch break\n- Every active surveyor gets one 30-minute lunch break per working day.\n- Lunch is taken at the first sensible between-survey boundary from 11:45.\n- 13:00 is the latest permitted lunch start.\n- Lunch is included in hard return-home feasibility checks.\n- The lunch break appears as a LUNCH row in daily/full schedule outputs.\n- Days genuinely finishing before 11:45 do not receive an artificial lunch row.\n

## Version 20.7 — Salesforce copy restored
- Restores the Salesforce Copy table on the Streamlit results page.
- Restores the Salesforce Copy worksheet in the downloaded weekly workbook.
- Only genuine scheduled surveys are included.
- LUNCH and RETURN rows are excluded from the Salesforce copy.
- Salesforce date/time format is DD/MM/YYYY, HH:MM.
- All Version 20.6 scheduling, lunch, prediction, endgame, notes, Released-only and routing logic is otherwise unchanged.


## Version 20.8 — exact Salesforce upload worksheet
- Keeps all Version 20.7 scheduling/prediction/endgame/lunch behaviour.
- Salesforce Copy is workbook-only; it is no longer displayed on the Streamlit page.
- The worksheet exactly follows the supplied 10-column Salesforce Field Service upload layout.
- Conor Birch -> Harpenden -> 0HnR50000005RlxKAE.
- Rod Harrison -> Rugby -> 0Hn4L0000000Yy8SAE.
- Toby Lawal -> Chadwell Heath -> 0HnR50000005S6vKAE.
- Harrison Grice, Joe Reynolds and spare surveyors are excluded until Salesforce resource IDs are known.
- Work Order / Primary Service Appointment / Service Appointment ID / Customer Reference / Building Name are joined from the uploaded To Do portfolio.
- LUNCH and RETURN rows are excluded.
- Scheduled timestamps use YYYY-MM-DDTHH:MM:00.000+0000 as in the supplied template.


## Version 20.9 — Full-portfolio planning + three time windows
- Strategic/endgame planning now keeps every postcode-bearing portfolio row, including Plan Drafting, Work Request, Geospatial Under Preparation and rows without a duration prediction.
- Current-week routing remains hard-gated to Geospatial Asset Mapping + Released + a valid duration prediction.
- New 15-minute dropdowns: First survey starts at; Last survey finishes no later than; Return home no later than.
- The first home departure is back-calculated from the first-site transit estimate so long commutes happen before the survey window rather than consuming it.
- Each candidate must satisfy both the last-survey-finish deadline and the return-home deadline.
- Team cluster allocation gives extra weight to commute efficiency to reduce wasted travel while keeping candidate capacity unchanged.
- Lunch, notes, endgame anchors, Salesforce Copy format, duration model and all other Version 20.8 behaviour remain unchanged.


## Version 20.9.1 — scheduler compatibility fix
- No scheduling/business-rule changes from Version 20.9.
- The new time-window scheduler now lives in `scheduler_v20_9_1.py` so Streamlit cannot reuse an older cached `scheduler` module.
- `app.py` validates the required `build_week()` arguments at startup.
- Stale `__pycache__` files are not included in the package.
\n\n## Version 20.9.2 — Work Done future pipeline\n- No weekly eligibility/routing changes from Version 20.9.1.\n- All Plan Drafting rows remain part of future planning.\n- Main Status = Work Done is explicitly recognised as future pipeline.\n- Parent Work Order: Status = Work Done is now also recognised.\n- Released GAM rows are not double-counted as future pipeline.\n- Cluster/endgame tables expose Future Work Done, Future Plan Drafting Work Done and Future Parent Work Done counts.\n- AI treats Work Done as a stronger near-term pipeline signal, but it still cannot schedule a site until GAM + Released.\n

## Version 20.9.3 — all portfolio rows considered for weekly scheduling
- Removes the Geospatial Asset Mapping + Released hard gate.
- Plan Drafting, Work Request, Work Done, Under Preparation, Released and other Work Type/Status combinations can all enter the actual weekly schedule.
- A row still needs a usable postcode, date/drawing eligibility and a duration prediction before it can be routed.
- The one-week drawing lead-time rule is unchanged.
- Rows that are not yet eligible remain in future-cluster/endgame planning.
- Google is still only run for the selected week.
- All Version 20.9.2 time-window, lunch, endgame, notes, surveyor, duration and Salesforce-export behaviour is otherwise unchanged.
\n\n## Version 20.9.4 — density-first shortlist and day routing\n- No eligibility, prediction, time-window, lunch, notes, endgame, surveyor or Salesforce changes from Version 20.9.3.\n- Inside a selected postcode district, full-postcode density now outranks an existing Planned Start when deciding which sites reach the Google shortlist.\n- All eligible Work Types/Statuses are treated equally; Plan Drafting can therefore strengthen a dense micro-cluster rather than being pushed out by sparse pre-planned rows.\n- Once a survey day enters a postcode district, feasible remaining jobs in that district are tried before a different-district jump.\n- Exact same-postcode jobs are preferred again within the district.\n- Up to two out-of-district fallback candidates remain available so the day can continue if the local jobs genuinely cannot fit the hard rules.\n

## Version 20.9.5 — Work Request status support
- Everything else is unchanged from Version 20.9.4.
- Explicitly recognises Salesforce Status = Work Request.
- Also accepts the spelling Work Requested and normalises it to Work Request.
- Work Request rows can enter the weekly schedule under the same operational rules as other statuses.
- Density-first daily routing is unchanged.
- Endgame anchor preservation is unchanged: eligible buildings that better support known future same-campus / same-postcode / postcode-district work can still be held for later where practical.


## Version 20.9.6 — use every selected working day
- Everything else is unchanged from Version 20.9.5.
- The existing Google candidate-cap input is now a minimum rather than a hard proportional weekly limit.
- Candidate capacity expands automatically from the selected First Survey and Last Survey Finish window plus actual predicted survey durations.
- This prevents a three-day surveyor from receiving only 24 candidates when those 24 can all be completed in the first two days.
- Candidate expansion is capped at 25 per selected day to keep Google cost controlled.
- AI cluster choices are topped up deterministically when they do not supply enough candidates to use all selected days.
- Density-first routing is unchanged.
- Endgame preservation remains active: non-anchor alternatives are used before a building reserved for a stronger future cluster is released.


## Version 20.9.6.1 — math import fix
- No business logic changes from Version 20.9.6.
- Adds the missing `import math` required by the time-aware candidate-cap calculation.
