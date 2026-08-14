# Site Survey Duration Agent

This project predicts the duration of a site survey using:

- Building Height
- Internal Ground Floor Area (m2)
- Sovereign Flat
- Primary Service Appointment: Actual Duration (Minutes) as the training target

There is **no surveyor multiplier**.

## Missing data

The predictor trains all usable feature combinations. If a new building is
missing one or two inputs, it automatically falls back to a model that can use
the information that is present.

For example:

- Floor count + area + flats -> three-input model
- Area + flats -> two-input fallback
- Floor count + area -> two-input fallback
- Flats only -> one-input fallback

A prediction is only impossible if all three predictor fields are missing.

## Aborted / suspicious visits

By default, historical rows with an actual duration below 6 minutes are excluded
from training. This prevents obvious 1–5 minute visits from distorting the
duration model.

The threshold can be changed in the Streamlit sidebar.

## Planning duration

The app reports:

1. **Predicted duration** — model estimate.
2. **Planning duration** — prediction plus a configurable scheduling buffer
   (default 15%), rounded up to the nearest 5 minutes.

Use the planning duration in the future scheduling agent.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Updating the model

Upload a fresh completed-surveys Excel export in the Streamlit sidebar.
The predictor retrains immediately from the latest data.

For the eventual scheduling agent, import the class directly:

```python
from duration_predictor import DurationPredictor

predictor = DurationPredictor().load_excel("Predictive Model.xlsx")

result = predictor.predict(
    building_height=24,
    ground_floor_area=520,
    flats=None,
)

print(result.predicted_minutes)
print(result.planning_minutes)
print(result.confidence)
```

The `planning_minutes` value is the one intended to be combined with travel time
and daily start/finish constraints.
