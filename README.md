# Site Survey Scheduling Agent

This app combines the survey-duration model with Google Maps Platform public-
transport routing.

## What it does

1. Trains the duration model from completed surveys.
2. Uploads `Future Surveys.xlsx`.
3. Predicts each building's survey duration using:
   - Building Height
   - Internal Ground Floor Area (m2)
   - Sovereign Flat
4. Falls back to the available inputs when data is missing.
5. Adds a survey-planning buffer (default 15%).
6. Uses Google Maps Platform Routes API in `TRANSIT` mode.
7. Builds a single-day route that:
   - leaves Harpenden Station at 07:50 by default;
   - favours short public-transport journeys between successive surveys;
   - keeps same-postcode buildings together;
   - checks the time-dependent public-transport journey home after each
     candidate survey;
   - rejects any addition that would return after 16:00.

Walking segments that form part of a Google transit route are included in the
transit duration returned by Google.

## Google Maps setup

In Google Cloud:

1. Create/select a project.
2. Add billing to the project.
3. Enable **Routes API**.
4. Create an API key.
5. Restrict the key to the Routes API where practical.

You can either paste the key into the Streamlit app or set:

```bash
GOOGLE_MAPS_API_KEY=your_key_here
```

Do **not** commit the API key to GitHub.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Current routing algorithm

This version intentionally does not ask an LLM to invent or estimate journey
times. Google Routes supplies the transit timings.

At each survey stop the scheduler:
- obtains transit times from the current location to the remaining candidates;
- ranks the shortest journeys first;
- checks the best candidates against a real transit journey back to Harpenden
  after the proposed survey ends;
- schedules the first feasible option;
- repeats until no additional survey fits.

This naturally forms a transit-time cluster while respecting the hard return
deadline.

This is the first single-day optimiser. A later version can add multi-day
allocation / global optimisation after the routing data has been validated.
