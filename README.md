
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
