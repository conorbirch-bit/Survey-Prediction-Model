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


## Operational buffers

Version 6 adds three configurable schedule buffers:

- **Travel leeway per journey** — default 5 minutes. Added to every Google
  public-transport journey, including the final journey home.
- **Before each survey** — default 5 minutes. Allows time to find the correct
  entrance, get bearings and prepare equipment.
- **After each survey** — default 5 minutes. Allows time to pack equipment,
  finish notes, orientate and leave the site.

These are separate from the duration model's percentage survey buffer.


## Full-week scheduling

Version 7 adds a `Full working week` planning mode.

Choose a week commencing Monday and the working days to use. The scheduler runs
the existing live, time-dependent Google Transit routing logic for each day.
Sites scheduled on Monday are removed before Tuesday is planned, and so on.

Every selected day independently respects:
- start at Harpenden Station;
- default departure 07:50;
- latest return 16:00;
- travel leeway;
- pre-survey setup buffer;
- post-survey pack-up buffer;
- predicted survey planning duration.

The Excel export contains:
- Week Summary
- Full Week Schedule
- one sheet per scheduled day
- Unscheduled Sites
- Candidate Sites


## Google API key — Streamlit Secrets

Version 8 reads the Google Maps API key automatically from Streamlit Secrets.

### Streamlit Community Cloud

Open the deployed app's settings and add this to **Secrets**:

```toml
GOOGLE_MAPS_API_KEY = "YOUR_REAL_GOOGLE_API_KEY"
```

Save the settings and restart/reboot the app if required.

The API key no longer needs to be pasted into the app interface.

### Running locally

Create this file inside the project:

```text
.streamlit/secrets.toml
```

Put this inside it:

```toml
GOOGLE_MAPS_API_KEY = "YOUR_REAL_GOOGLE_API_KEY"
```

Then restart Streamlit:

```bash
streamlit run app.py
```

Do not commit `.streamlit/secrets.toml` to GitHub. This project includes it in
`.gitignore`. A safe `secrets.toml.example` file is provided as a template.


## Version 9 — AI planning layer

The OpenAI model is used as a decision layer, not as a source of journey times.

Inputs provided to the AI include:
- each site's predicted/planning survey duration;
- prediction confidence;
- postcode district;
- existing Planned Start;
- how many sites in the same postcode district appear this week;
- how many appear next week;
- how many appear in the next three weeks;
- TfL disruption context;
- Met Office forecast context.

The AI returns:
- a 0–100 planning priority;
- `schedule_this_week`, `neutral`, or `defer`;
- a brief reason;
- an overall week strategy.

A deliberate `defer` is penalised in the route optimiser. This allows the
planner to recognise cases such as:

> One HA8 site is available this week, but five HA8 sites are already planned
> for next week. Unless the single site is urgent, keep it for next week's HA8
> cluster rather than creating a separate journey this week.

### Safety / reliability boundary

OpenAI does **not** fabricate transit duration and cannot override the hard
working-day constraints.

- Google Routes = actual public-transport journey calculations
- Duration model = survey duration estimate
- TfL = disruption context
- Met Office Weather DataHub = forecast context
- OpenAI = planning judgment / future-cluster reasoning
- Python scheduler = hard constraint validation

The AI priority can only modify candidate attractiveness by a capped amount.
A geographically poor journey cannot become valid simply because the AI likes
the site.

## Secrets

Create `.streamlit/secrets.toml` locally or add these in Streamlit Community
Cloud Secrets:

```toml
GOOGLE_MAPS_API_KEY = "..."
OPENAI_API_KEY = "..."
OPENAI_MODEL = "gpt-5.6"
TFL_API_KEY = "..."
MET_OFFICE_API_KEY = "..."
MET_OFFICE_GLOBAL_SPOT_URL = "..."
```

The Met Office endpoint is configurable because Weather DataHub product/version
delivery URLs can differ by subscription. Use the Global Spot hourly endpoint
shown in your Weather DataHub product/API documentation.

Never commit the real `secrets.toml` file to GitHub.
