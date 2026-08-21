# Site Survey Scheduling Agent — Version 17

This version keeps the Version 16 weekly notes and same-campus routing behaviour, with these duration/schedule updates:

- Drawing priority is no longer shown on the Streamlit page, but remains in the downloaded workbook.
- Planning survey duration equals the raw predicted duration (no percentage uplift / 5-minute rounding).
- If ground-floor area is unavailable, the duration model uses Sovereign flats only. If flats are missing but area is present, height + area remains available as a fallback.
- 0-flat records are treated as garages and excluded from the residential model training; missing flat counts are retained.
- Weekly scheduling shows a per-building prediction-reliability table using training rows, LOOCV MAE/RMSE, MAE as a percentage of the prediction and an MAE reference band.
- Schedule tables include Building Height, Sovereign Flat and Internal Ground Floor Area (m2).

All other Version 16 routing, clustering, weekly-note hard-rule protection, Google-cost controls and workbook outputs are retained.
