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


## Version 9.1 weather fix

The Met Office integration now targets the Global Spot **daily** endpoint
directly and sends the required `dataSource=BD1` parameter.

The parser recognises daily day/night fields rather than hourly fields and
turns them into readable text for the OpenAI planning layer.

If an unexpected Met Office schema is returned, the client reports the actual
field names so the issue can be diagnosed without exposing the API key.


## Version 10 — conversational schedule explanation

After the deterministic weekly scheduler finishes, OpenAI now performs a
second, explanation-only review of the completed schedule.

It receives:
- the final weekday-by-weekday schedule;
- AI site priorities and defer reasons;
- unscheduled sites;
- future postcode-cluster counts;
- TfL disruption context;
- Met Office weather context;
- duration-prediction confidence.

It then produces a natural-language explanation of the week: why sites were
grouped together, why some were deferred, which days look tighter, and what
weather/disruption or low-confidence predictions are worth watching.

This second AI call cannot modify the schedule. It is deliberately separated
from the optimisation step so the explanation cannot override Google travel
times or the hard working-day constraints.


## Version 11 — cluster pre-filter before Google Routes

This version changes the cost architecture.

### Old flow

```text
All eligible sites
→ Google Route Matrix
→ schedule
```

With a large portfolio, Google could repeatedly compare the current location
against hundreds or thousands of sites.

### New flow

```text
Full portfolio
→ drawing / earliest-date eligibility
→ free postcode-district aggregation
→ OpenAI reviews cluster summaries
→ small strategic site shortlist
→ Google Route Matrix
→ validated weekly schedule
```

Google Routes only receives the number set in **Maximum sites sent to Google
Routes** (default 60 for a full week and 25 for a single day).

The app displays:
- total portfolio sites;
- sites eligible for the selected week;
- sites actually sent to Google;
- percentage filtered before Google;
- cluster summaries;
- AI-selected clusters;
- the exact Google shortlist.

### Drawing-status rule

The master spreadsheet can optionally contain:

```text
Drawing Status
Earliest Survey Date
```

Recognised drawing states include values such as `Ready`, `Drawn`,
`Drawing Complete`, `Needs Drawing`, `Undrawn`, etc.

If a site is marked **Needs Drawing**, the app gives the drawing team a one-week
lead time:

- current-week scheduling: the site is excluded;
- next Monday onwards: the site can be considered using its available duration
  inputs, even if Ground Floor Area is still missing.

If `Earliest Survey Date` is supplied, that explicit date takes precedence.

This means a future master portfolio can contain both:
- existing survey-ready sites;
- sites that still need drawings.

### Duration accuracy

Sites that only have Building Height + Sovereign Flats still receive a fallback
duration prediction. Once Internal Ground Floor Area is added, the duration
model automatically uses the richer three-input model.

### AI / Google responsibility split

- **Postcode logic** cheaply creates the portfolio cluster summaries.
- **OpenAI** decides which clusters are strategically worth considering.
- **OpenAI site reasoning** only sees the reduced shortlist.
- **Google Routes** calculates actual public-transport journeys only for the
  shortlist.
- **Python** continues to enforce the hard start/return constraints.

The AI cannot send more sites to Google than the UI maximum.

### Excel output

The schedule workbook now includes:
- schedule sheets;
- Portfolio;
- Cluster Summary;
- Selected Clusters;
- Google Shortlist.

This makes the filtering decision auditable.


## Version 12 — multi-surveyor weekly scheduling

Version 12 keeps the Version 11 cost-control architecture and adds a dedicated
**Team weekly schedule** tab.

### Team flow

```text
Full portfolio
→ drawing / earliest-date eligibility
→ cheap postcode-district cluster summary
→ AI strategic cluster selection
→ tiny Google home-to-cluster assignment matrix
→ split cluster workload across active surveyors
→ capped, non-overlapping shortlist for each surveyor
→ Google transit scheduler for each person's own shortlist
→ combined team workbook
```

### Surveyor setup

The team editor contains four rows by default:

- Conor Birch
- Surveyor 2
- Surveyor 3
- Surveyor 4

Each row has:
- Name
- Active
- Start / Finish Location
- Active From

Only active surveyors whose start date falls within/before the selected week are
routed. An inactive or future-starting surveyor creates no Google routing spend.

The currently selected team working days and leave/latest-return times are shared
across the team. Individual start locations and active dates are supported.

### Cost control

The new control **Max Google candidates per surveyor** defaults to 40.

For a three-person week this therefore caps the detailed candidate pool at about:

```text
3 surveyors × 40 = 120 candidate sites
```

rather than exposing the full portfolio to every surveyor.

Before detailed routing, Google is used once for a deliberately tiny allocation
matrix:

```text
active surveyor homes × selected cluster representatives
```

For example, 3 surveyors and 6 selected clusters creates only 18 home-to-cluster
elements. This gives the team allocator real transit evidence about which
clusters are sensible from each person's home, without matrixing the full
portfolio.

Large clusters may be split across more than one surveyor. Site assignment is
non-overlapping: the same building cannot be placed in two surveyors' candidate
pools.

### Team workbook

The downloaded team workbook contains:
- Team Summary
- Full Team Schedule
- Cluster Summary
- Selected Clusters
- Team Allocations
- Home Cluster Matrix
- Google Shortlists
- Portfolio
- one full-week schedule sheet per active surveyor

### Long-range planning

Version 12 still uses detailed Google transit routing only for the week being
created. The wider portfolio remains represented by cheap cluster summaries for
strategic planning.


## Version 13 — direct Salesforce master report support

Version 13 can read the Salesforce report format used by the current master list
without requiring a manual clean-up step.

### Salesforce import handling

The importer now:
- detects the real table header even when report titles and Salesforce filters
  occupy the first rows of the workbook;
- strips Salesforce sort arrows from headers;
- carries grouped `Work Type Name` and `Status` values down to each building;
- removes report footer / totals rows;
- extracts a UK postcode from the end of `Building Name` when there is no
  standalone Postcode column.

### Drawing-status business rule

The following combination is automatically interpreted as a drawing job:

```text
Work Type Name = Plan Drafting
AND
Status = Work Request
→ Drawing Status = Needs Drawing
```

`Geospatial Asset Mapping` rows are treated as `Ready` unless an explicit
Drawing Status already exists.

The existing one-week lead-time rule then applies:
- Needs Drawing rows are excluded from the current survey week;
- they can become provisionally eligible from the following Monday.

### Drawing Priority queue

The Team weekly schedule now also creates a full numbered `Drawing Priority`
queue for every `Needs Drawing` site.

Priority logic:
1. sites belonging to clusters selected for the target survey week;
2. remaining sites ordered cheaply by cluster workload / planned-date signals.

No Google route matrix is required to order the long-term remainder.

The downloaded team workbook includes a `Drawing Priority` sheet containing the
full list, not just the first rows shown in Streamlit.


## Version 14 — unified weekly scheduling

The separate `Upcoming surveys + routing` and `Team weekly schedule` workflows
have been merged into one **Weekly scheduling** tab.

The three top-level tabs are now:
- Predict one building
- Weekly scheduling
- Model diagnostics

The weekly scheduler works for one, two, three or four surveyors. If only one
person has availability selected, it behaves as a one-person weekly scheduler.

### Exact surveyor availability

After choosing a Monday week commencing date, the app displays the five dates
for that week as checkbox columns beside every surveyor, for example:

```text
Name       Start / Finish       Mon 24 Aug  Tue 25 Aug  Wed 26 Aug ...
Conor      Harpenden Station        ✓           ✓           ✓
Surveyor 2 Croydon Station          ✓           ✓
Surveyor 3 Watford Junction                                 ✓
```

A surveyor with no dates selected creates no Google routing calls.

The detailed Google candidate allowance also scales with availability. With the
default 40-candidate five-day cap:
- 5 available days → up to 40 candidates
- 3 available days → about 24 candidates
- 2 available days → about 16 candidates
- 1 available day → about 8 candidates

This prevents a part-week surveyor from receiving the same routing workload and
API spend as someone working all five days.

### Hard Google horizon

Google Routes is only used for the single selected week.

Paid Google calls are limited to:
1. each available surveyor's home/start location to the small set of selected
   cluster representatives, dated on that person's first available day;
2. that surveyor's own capped candidate shortlist on their actual available
   dates in the selected week.

No Google routing is performed for week +1, week +2, week +3, etc.

Future weeks may still influence:
- AI cluster strategy;
- future-cluster counts;
- drawing priority.

Those future signals use spreadsheet/postcode/AI analysis only, not Google
journey calculations.


## Version 15 — same-campus Google bypass

Version 15 reduces Google Route Matrix usage for sites that are effectively on
the same campus / estate.

Two buildings are treated as the same campus when:
1. their **full postcode is identical**; and
2. their normalised building/address names are sufficiently similar.

The name comparison strips Salesforce asset IDs, street/flat numbers,
postcodes and weak words such as `block`, `building`, `car park`, etc.

Examples that are recognised as the same campus:

```text
102831 | A-B, 155 Cambridge Street SW1V 4QB
101100 | 157 Cambridge Street SW1V 4QB
101101 | 159 Cambridge Street SW1V 4QB
```

and:

```text
Arlidge House EC1N 8TW
Car Park Arlidge House EC1N 8TW
```

### Google cost behaviour

Before calling Google, remaining candidates are grouped into same-campus groups.

If six candidate buildings are on the same campus:

```text
Home
→ Google sees ONE representative campus destination
→ first campus building
→ fixed internal transfer
→ second campus building
→ fixed internal transfer
→ ...
```

The fixed internal transfer is controlled by **Same-campus transfer (min)**,
default 5 minutes.

Google is still used where it matters:
- travelling from a surveyor's start location to a campus;
- travelling between genuinely different campuses/locations;
- validating the journey home against the hard return deadline.

The existing Version 14 rule remains unchanged: Google is only used for the one
selected scheduling week, never for future-week strategic clustering.

## Version 16 — weekly notes with hard-rule protection

Version 16 adds an optional **Weekly notes / special requests** section to the
unified weekly scheduler.

Example:

```text
Keep Conor as close to Kilburn as possible for Thursday
```

### Safety / rule hierarchy

Weekly notes are deliberately below the existing hard scheduling rules.
They can never override:
- drawing / survey eligibility;
- the surveyor's selected availability dates;
- no duplicate building assignments across surveyors;
- survey duration and operational buffers;
- the hard latest-return time;
- Google routing being restricted to the one selected week.

The notes layer is transactional:

```text
normal valid weekly schedule
        ↓
try special request in a separate route trial
        ↓
request satisfied + all hard rules still valid?
        ↓                         ↓
      YES                        NO
       ↓                          ↓
commit trial               reject request
                         keep baseline schedule
```

A rejected note therefore does not partially influence or corrupt the normal
schedule.

### Supported note type

This version intentionally supports one narrow request type reliably:

**named surveyor + requested area + specific day/date**

Examples:
- `Keep Conor as close to Kilburn as possible for Thursday`
- `Try to keep Toby around Wembley on Tuesday`

OpenAI parses the note into a structured request. Ambiguous or unsupported notes
are rejected rather than guessed.

### How location notes are validated

For a supported request, Google is used only inside the selected week to compare
the requested area against **one representative per eligible postcode cluster**.
The closest eligible cluster must fall within the configurable
`Maximum distance from requested area` threshold (default 30 transit minutes).

The scheduler then creates a small trial candidate pool for that surveyor,
reserves the request-area candidates until the requested date, and reruns only
that surveyor's selected week. The request is accepted only if the requested
cluster is actually scheduled on that date and the normal hard rules remain
valid.

The request results are shown as **Accepted** or **Rejected** with a reason and
are exported to a `Weekly Notes` sheet in the team workbook.

### API horizon

The original cost rule remains unchanged. Notes may add:
- one small requested-area → cluster-representative matrix; and
- one trial reroute for the affected surveyor.

Both use dates in the selected week only. No future week is sent to Google.
