# UK Schools Finder

Interactive map tool for analyzing UK schools by A-level / SATs performance and local crime statistics. Generates a standalone web app that works on any device with no server required.

**[Live Demo: Web App →](http://web.andico.org/school_finder.html)**

<img src="schoolfinder.png" alt="School Finder Screenshot" width="400">

---

## Overview

Two data pipelines produce the school data:

- **KS5 (secondary schools)** — ranked by 16-18 A-level performance
- **KS2 (primary schools, state only)** — ranked by SATs results

Both share the geocoding (AWS Location Services) and optional crime data (data.police.uk) infrastructure. `combined_create_standalone_app.py` reads the processed output of both pipelines and emits a single standalone HTML — `school_finder.html` — with primary and secondary schools on the same map (different marker shapes, shared reticle, pooled crime percentile). This is what the live demo serves.

---

## Prerequisites

### Install Dependencies

```bash
pip install pandas boto3 tqdm numpy httpx
```

### Configure AWS

```bash
aws configure
```

Create a Place Index in Amazon Location Service and update `config.json` (KS5) / `ks2_config.json` (KS2) with its name:

```json
{
  "geocoding": {
    "index_name": "your-place-index-name",
    "region_name": "eu-west-2"
  }
}
```

---

## KS5 Pipeline (secondary schools)

### 1. Download School Data

```bash
python3 ks5_download_data.py
```

This fetches 16-18 final results from the DfE via the Explore Education Statistics API.

**EES availability:** the school-level summary file was introduced in the 2023/24 EES release. The script downloads automatically for 2023-24 onwards and prints manual instructions for earlier years.

For 2022-23 and earlier, download manually from [compare-school-performance.service.gov.uk/download-data](https://www.compare-school-performance.service.gov.uk/download-data/): select the year → "All of England" → "16-18 results (final)" → CSV, then:

```bash
gzip 2022-2023_england_ks5final.csv
gzip 2021-2022_england_ks5final.csv
```

The script is idempotent — re-running skips files already present and complete.

Optional flags:
```bash
python3 ks5_download_data.py --years 2023-2024        # single year only
python3 ks5_download_data.py --output-dir data/       # write to a subdirectory
```

### 2. Process the Data

```bash
python3 generate_school_data.py      # 5-15 mins, generates processed_school_data.csv + caches
```

This produces `processed_school_data.csv` plus the geocoding and crime caches. The map itself is built by `combined_create_standalone_app.py` once both pipelines have run — see [Building the Map](#building-the-map).

### KS5 Configuration

Edit `config.json` to customize:

```json
{
  "filtering": {
    "percentile": 0.90,           // Top 10% of schools
    "min_age_threshold": 7         // Secondary schools only
  },
  "crime": {
    "school_crime_radius_km": 3,
    "excluded_crime_types": ["Shoplifting", "Bicycle theft", ...]
  },
  "grading": {
    "a_star_threshold": 50,        // TB3PTSE ≥50 = A* grade
    "a_threshold": 40              // TB3PTSE ≥40 = A grade
  }
}
```

**Color Coding:**
- Independent schools: Red (A*), Orange (A), Yellow (≤B)
- State schools: Navy (A*), Blue (A), Light Blue (≤B)

---

## KS2 Pipeline (primary schools)

A parallel set of scripts (all prefixed `ks2_`) generates an equivalent interactive map for **state primary schools**, ranked by KS2 SATs results (percentage of pupils meeting the expected standard in combined reading, writing and maths).

### Key differences from the KS5 map

| | KS5 (secondary) | KS2 (primary) |
|---|---|---|
| Data portal | compare-school-performance.service.gov.uk | explore-education-statistics.service.gov.uk |
| File format | Wide CSV, one row per school per year | Long CSV, one row per school × subject × year → needs pivot |
| Address source | In the performance CSV | Separate GIAS download |
| School types | State + independent | State only (independents don't sit mandatory SATs) |
| Metric | TB3PTSE (avg best 3 A-levels) | % meeting expected standard (R+W+M combined) |
| Colour scheme | Navy/Blue/Royal Blue (state), Red/Orange/Yellow (independent) | Dark green / Green / Light green |

### 1. Download KS2 Data

Run the download script — it fetches all three years of EES data and the GIAS establishment list automatically:

```bash
python3 ks2_download_data.py
```

This produces:
- `ks2_school_attainment_202324.csv` — EES 2023/24 (via content API)
- `ks2_school_attainment_202223.csv` — EES 2022/23
- `ks2_school_attainment_202122.csv` — EES 2021/22
- `gias_establishments.csv` — GIAS all-establishments list

The script is idempotent: re-running it skips files that are already present and complete. Interrupted downloads leave no partial files (writes to `.tmp` then renames atomically).

Optional flags:

```bash
python3 ks2_download_data.py --skip-gias          # skip GIAS if already downloaded
python3 ks2_download_data.py --skip-ees           # skip EES if files already present
python3 ks2_download_data.py --years 2023-24       # download a single year only
python3 ks2_download_data.py --output-dir data/    # write files to a subdirectory
```

If automatic download fails, the script prints manual instructions. EES data is also available at [explore-education-statistics.service.gov.uk](https://explore-education-statistics.service.gov.uk/find-statistics/key-stage-2-attainment) and GIAS at [get-information-schools.service.gov.uk/Downloads](https://get-information-schools.service.gov.uk/Downloads).

### 2. Process the Data

```bash
python3 ks2_generate_school_data.py      # 5-15 mins, generates ks2_processed_school_data.csv
```

The map is built by `combined_create_standalone_app.py` once both pipelines have run — see [Building the Map](#building-the-map).

### KS2 Configuration (`ks2_config.json`)

```json
{
  "filtering": {
    "percentile": 0.95,
    "subject": "Reading, writing and maths",
    "breakdown": "Total"
  },
  "grading": {
    "high_threshold": 85,
    "expected_threshold": 75
  }
}
```

**Colour coding** (all state schools):
- Dark green: ≥85% meeting expected standard
- Green: ≥75%
- Light green: above selected percentile threshold but <75%

---

## Building the Map

Once both pipelines have been run, build the standalone web app:

```bash
python3 combined_create_standalone_app.py     # generates school_finder.html
```

The script reads the processed CSVs and cache files from both pipelines, emits one standalone HTML, and re-ranks the crime index over the pooled cohort so primaries and secondaries are directly comparable.

Marker shapes distinguish the two school types:
- **KS5 (secondary):** teardrop pin, labelled with TB3PTSE
- **KS2 (primary):** circle, labelled with % meeting expected standard

`school_finder.html` is a single mobile-friendly file that:
- Searches by location (GPS or address) with auto-zoom on address searches
- Filters by an adaptive crosshair/reticle that scales radius with zoom (5km max, 500m min, max 60% of screen) and updates as you pan
- Shows a distance scale ruler and works offline after the first load
- Shows both school types at once — there is no phase toggle

Optional flags override any of the input paths or the output file (see `--help`).

---

## Times Parent Power ranks (Optional)

`combined_create_standalone_app.py` adds a school's *Times / Sunday Times Parent Power* league-table rank to its info window when a match is found. Only the rank is shown — the A-level, GCSE and SATs figures already come from the DfE data.

> **Terms of use:** the Parent Power tables are © Times Newspapers Ltd. Accessing and reusing the ranking data may be subject to their terms of use. The extraction described here is for personal analysis only; check the current terms before fetching or redistributing the data.

The ranks are read from two extracted CSVs:

- `best_schools_2025_combined_secondary.csv` — joint state + private secondary ranking (matched to secondaries)
- `best_schools_2025_top500_state_primary.csv` — state primary ranking (matched to primaries)

`times_ranking.py` does the matching: a school is matched on a normalised name, with the town used to break ties; a token-subset fallback (also gated on town agreement) recovers the Times/DfE naming differences, e.g. *King's College School* → *King's College School, Wimbledon* or *Bancroft's* → *Bancroft's School*. Schools absent from the Times list simply show no rank. The CSVs are optional: if they are missing the generator logs a warning and omits ranks.

### Where the data lives

The tables come from the *Parent Power 2025* league tables behind <https://www.thetimes.com/best-schools-league-table>. The public page only embeds a top-10 teaser, but the full ranked tables are served by the same Parent Power app:

```
https://dlv.tnl-parent-power.gcpp.io/2025?filterId=<FILTER_ID>
```

That endpoint is fully server-rendered (a plain `curl` returns every row — no JavaScript, login or subscriber paywall) and is not behind the "Verifying Device" interstitial that guards `www.thetimes.com`. The `filterId` selects the list; the path segment is the year. Known filters include:

- `the-top-combined-secondary-schools` (joint state + private)
- `the-top-state-secondary-schools`
- `the-top-independent-secondary-schools`
- `the-top-500-english-state-primary-schools`

The table markup uses the CSS class `pp-main-table-2023`; the extractor reads the header row from the page rather than hard-coding it, so any `filterId` works without code changes.

### Extracting the tables

`extract_times_table.py` uses the Python standard library only (it shells out to `curl` to fetch):

```bash
# fetch + parse straight from the live endpoint
python3 extract_times_table.py the-top-combined-secondary-schools best_schools_2025_combined_secondary.csv
python3 extract_times_table.py the-top-500-english-state-primary-schools best_schools_2025_top500_state_primary.csv

# or parse a previously saved page
curl -s "https://dlv.tnl-parent-power.gcpp.io/2025?filterId=the-top-state-secondary-schools" -o state.html
python3 extract_times_table.py the-top-state-secondary-schools state.csv --html state.html

# a different year
python3 extract_times_table.py the-top-state-secondary-schools out.csv --year 2024
```

`best_schools_2025_combined_secondary.csv` is the joint state-and-private secondary ranking (St Paul's School at rank 1). The primary file is labelled "top 500" in the app but the table actually returns ~1000 rows.

### Column schemas

Headers differ by table and are read from the page's header row.

**Secondary** (11 columns):
`Rank, School, Location, Type, A-level (% A*), A-level (% A*/A), A-level (% A*/B), A-level rank, GCSE (% 9/8/7), GCSE rank, Entry gender`

**Primary** (8 columns):
`Rank, School, Location, Reading (averaged scaled score), Grammar (averaged scaled score), Maths (averaged scaled score), Total, Entry gender`

(The secondary header in the source has a typo, "GSCE", preserved verbatim.)

### Parsing notes

- **Rank** cells contain a `rank-number` span plus a year-on-year movement marker; only the span text is kept. Ties show as `6=`, `995=`, etc.
- **Entry gender** is rendered as icons. A `male-icon` and `female-icon` together means Mixed; a single icon means Boys or Girls. State primaries come back entirely Mixed (they are co-ed).
- **Type** is only populated for state schools (Comprehensive, Selective (100%), Partially selective (N%)); independent schools show `-`.
- All cell whitespace is collapsed and HTML entities are unescaped.

Retrieved from the 2025–26 edition ("Parent Power 2025"), captured June 2026.

---

## Crime Data (Optional, Shared)

Crime data adds local safety statistics (3km radius around each school) and is used by both pipelines.

### Download

- Visit https://data.police.uk/data/
- Select "Custom download"
- Choose time period (e.g., last 12 months)
- Check "all forces" and "Include crime data"
- Download and extract the ZIP file

### Consolidate

```bash
python3 consolidate_crime_data.py --crime-data-dir /path/to/extracted/data
```

This script:
- Recursively finds all crime CSV files in the extracted directory
- Extracts columns: Month, Longitude, Latitude, Crime type
- Excludes outcome files (not needed for statistics)
- Combines all files into `combined_crimes.csv.gz`
- Typical output: 5-10 million crime records → ~200MB compressed file

### Configuration

You can configure crime processing in `config.json`:

```json
{
  "crime": {
    "crime_data_file": "combined_crimes.csv.gz",
    "source_crime_data_dir": "/path/to/extracted/data",
    "excluded_outcomes": true
  },
  "crime_processing": {
    "columns": ["Month", "Longitude", "Latitude", "Crime type"],
    "compress_output": true
  }
}
```

### Usage

**Command line:**
```bash
# Specify directory containing extracted crime data
python3 consolidate_crime_data.py --crime-data-dir ~/Downloads/police-data-2023

# Custom output file
python3 consolidate_crime_data.py --crime-data-dir ~/Downloads/police-data-2023 --output my_crimes.csv.gz
```

**Via config:**
```bash
# Add source_crime_data_dir to config.json, then run without arguments
python3 consolidate_crime_data.py
```

### Crime Statistics Used

The school finder calculates crime statistics within 3km of each school:
- **Excluded types**: Shoplifting, Bicycle theft, Anti-social behaviour, Drugs, etc.
- **Included types**: Violence, Burglary, Robbery, Vehicle crime, etc.
- **Crime index**: Percentile ranking (0-1, lower is safer)

Crime filtering is configurable via `config.json` → `crime.excluded_crime_types`.

---

## Address Crime Lookup (Standalone)

`crime_lookup.py` is a small standalone tool for aggregating crime counts around one or more free-form UK addresses. It reuses the same Amazon Location Service place index and `combined_crimes.csv.gz` as the school pipelines — useful for direct address-by-address comparisons (e.g., evaluating individual properties).

Crimes are counted across **all** crime types (the `excluded_crime_types` filter from `config.json` is not applied here — the goal is a complete picture, broken down by type).

### Usage

Interactive prompt — enter addresses one per line, type `STOP` or `END` to finish:

```bash
python3 crime_lookup.py
```

Batch input — read one address per line from a text file (blank lines and `#` comments are ignored, no prompting):

```bash
python3 crime_lookup.py --input addresses.txt
```

Default radius is 500 m. Override with a comma-separated list of metres:

```bash
python3 crime_lookup.py --radius 250,500,1000
```

Output path (default `crime_lookup_output.csv`):

```bash
python3 crime_lookup.py --output my_lookup.csv
```

### Output format

One row per address (full normalised address from AWS in column A). Two header rows: the radius (in metres) for each column, then the crime type. With one radius the radius row is repeated; with multiple radii the columns cycle through the crime types per radius:

```csv
,500,500,500,500
Address,Burglary,Robbery,Theft,Violence
"10 Downing Street, London SW1A 2AA, UK",2,0,0,2
"Buckingham Palace, London SW1A 1AA, UK",0,1,0,1
...
```

Crime types are listed alphabetically and zero-filled when absent, so columns line up across addresses for side-by-side comparison.

Load with pandas using a 2-row header and the address column as index:

```python
import pandas as pd
df = pd.read_csv("crime_lookup_output.csv", header=[0, 1], index_col=0)
```

---

## State Secondary Ethnicity Report (Standalone)

`grammar_ethnicity_report.py` generates a per-school ethnicity breakdown for the **Times / Sunday Times Parent Power state-secondary league table**, as a Markdown report sorted by Parent Power rank (rank 1 first). It is independent of the map pipelines — it does not use AWS, crime data or the geocoding caches.

The cohort comes from the Parent Power state-secondary CSV (every school by default); the ethnicity figures come from the **DfE school census**. It uses the *school-level underlying data* file from "Schools, pupils and their characteristics" (one row per school, **unsuppressed**), which carries the full detailed ethnic-group split per school (White British, Indian, Pakistani, Bangladeshi, Chinese, Black African, Mixed, etc.) plus EAL.

Each Parent Power school is matched to a census URN at runtime by name + location (a token-overlap score with religious/suffix normalisation, e.g. "St" ≈ "Saint", "RC" ≈ "Catholic", trailing "Academy" ignored; location breaks ties). Schools that can't be matched are listed in the report — almost all are in **Northern Ireland or Wales**, which the England-only census does not cover.

**State schools only.** Independent schools submit an aggregate census return with no ethnicity breakdown, so per-school ethnicity does not exist for them and they cannot be included.

### Usage

First produce the Parent Power state-secondary CSV (see [Times Parent Power ranks](#times-parent-power-ranks-optional)):

```bash
python3 extract_times_table.py the-top-state-secondary-schools best_schools_2025_state_secondary.csv
```

Then run the report:

```bash
python3 grammar_ethnicity_report.py                 # download census + write grammar_ethnicity_report.md
python3 grammar_ethnicity_report.py --top 100        # exactly 100 matched schools (over-fetches the table; 0/omit = all)
python3 grammar_ethnicity_report.py --times-csv f.csv # use a different Parent Power CSV
python3 grammar_ethnicity_report.py --csv local.csv # use a local census CSV instead of downloading
python3 grammar_ethnicity_report.py -o out.md        # choose output path
python3 grammar_ethnicity_report.py --force          # re-download the census even if cached
```

The census file (~22 MB) is downloaded once and cached as `spc_school_level.csv` (gitignored). No Python package dependencies. The download uses `curl` when available (the EES CDN serves over HTTP/2 and supports no resume, which the stdlib HTTP/1.1 client handles unreliably) and falls back to a `urllib` retry loop otherwise. The script handles the file's Windows-1252 encoding and quoted, comma-containing school names.

To roll to a newer census, update `CSV_URL` and `CENSUS_LABEL` to the new EES release's file link. To refresh the cohort, regenerate the Parent Power CSV with `extract_times_table.py`.

### Output

The report (`grammar_ethnicity_report.md`) contains three tables, each carrying the Parent Power rank and sorted by it (rank 1 first):
- **Summary** — % ethnic minority (= 100 − % White British), % White British, % EAL
- **Group-by-group breakdown** — every ethnic group as a % of pupils
- **Largest single minority group, by school** — auto-derived per school

A closing note lists any Parent Power schools that could not be matched to the census. Percentages are whole-school headcount (not sixth-form/intake specific).

### PDF

A GitHub-faithful PDF (`grammar_ethnicity_report.pdf`) is produced by `build_grammar_pdf.sh`, which pipes the Markdown through `cmark-gfm` → HTML (inline CSS) → `wkhtmltopdf`, rendered A4 **landscape** with auto-sized table columns so the wide group-by-group table fits without stretching the School column:

```bash
./build_grammar_pdf.sh                                  # defaults: report .md -> .pdf, 11px font
./build_grammar_pdf.sh in.md out.pdf 10                  # custom input/output and font size
```

---

## Deployment

### Local

```bash
open school_finder.html
```

### AWS S3

```bash
# Create and configure bucket
aws s3 mb s3://uk-schools-finder
aws s3 website s3://uk-schools-finder --index-document index.html

# Upload
aws s3 cp school_finder.html s3://uk-schools-finder/index.html \
  --content-type "text/html" --acl public-read

# Set bucket policy for public access
aws s3api put-bucket-policy --bucket uk-schools-finder --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::uk-schools-finder/*"
  }]
}'
```

Access at: `http://uk-schools-finder.s3-website-REGION.amazonaws.com`

**Note:** For HTTPS (required for GPS location), add CloudFront.

### Updating Data

When new school year data is released (2023-24 onwards downloads automatically):

```bash
# Download new year from EES (skips years already present)
python3 ks5_download_data.py --years 2024-2025

# Regenerate (existing schools use cached geocoding)
python3 generate_school_data.py
python3 combined_create_standalone_app.py    # rebuilds school_finder.html

# Redeploy
aws s3 cp school_finder.html s3://your-bucket/index.html --acl public-read
```

---

## Architecture

### KS5 data pipeline

- **`ks5_download_data.py`** — Downloads KS5 16-18 final results CSVs from DfE (idempotent)
- **`school_data_lib.py`** — Shared library with all KS5 data processing functions
- **`consolidate_crime_data.py`** — Consolidates downloaded crime CSV files (shared)
- **`generate_school_data.py`** — Data pipeline: load → filter → geocode → crime stats → save `processed_school_data.csv`

### KS2 data pipeline

- **`ks2_download_data.py`** — Downloads KS2 performance data from EES and GIAS establishment list
- **`ks2_school_data_lib.py`** — KS2 data processing: load long-format EES data, pivot, join GIAS, consolidate across years
- **`ks2_generate_school_data.py`** — Pipeline: load → filter → geocode → crime stats → save `ks2_processed_school_data.csv`

### Map generator

- **`combined_create_standalone_app.py`** — Reads both pipelines' processed CSVs + caches and emits `school_finder.html` with KS2 + KS5 markers on one map, pooled crime percentile, and Times ranks
- **`times_ranking.py`** — Matches schools to their Times Parent Power rank (used by the generator)

### Data Flow

```
ks5_download_data.py → 20XX-20XX_england_ks5final.csv.gz (×3)
                                  ↓
Crime CSVs → consolidate_crime_data.py → combined_crimes.csv.gz
                                  ↓
generate_school_data.py: Consolidate → Filter by percentile → Geocode (AWS) → Crime stats → Cache
                                  ↓
                       processed_school_data.csv ───────────────┐
                                                                │
ks2_school_attainment_20XXYY.csv (×3) ─┐                        │
gias_establishments.csv ───────────────┤                        │
   (ks2_download_data.py)               ↓                        │
        ks2_generate_school_data.py: pivot + join + filter       │
        + geocode + crime stats                                 │
                                  ↓                             │
                       ks2_processed_school_data.csv ───────────┤
                                                                ↓
                       combined_create_standalone_app.py (+ times_ranking.py)
                                  ↓
                       school_finder.html  (KS2 + KS5 on one map)
```

### Key Features

- **Smart caching**: Geocoding and crime data cached with SHA256 validation
- **Clean addresses**: NaN values filtered at source during geocoding
- **Pooled crime index**: Crime percentile re-ranked over the combined cohort so primaries and secondaries are comparable
- **Standalone output**: Single self-contained HTML file

---

## Technical Details

### Data Sources
- **School Performance**: UK Government (compare-school-performance.service.gov.uk)
  - TB3PTSE: Average of best 3 A-levels
  - TALLPPE_ALEV_1618: Average per A-level
- **Crime Data**: data.police.uk (3km radius, serious crimes only)

### Processing Pipeline
1. Load & consolidate multi-year CSVs
2. Calculate per-school averages across years
3. Filter by percentile (default: top 10%)
4. Clean addresses (remove NaN values)
5. Geocode with AWS Location Services (cached)
6. Calculate crime statistics (cached with SHA256 validation)
7. Save processed data and caches

### Caching
- **Geocoding**: Never invalidated (addresses don't change), saves ~$0.004/lookup
- **Crime**: Invalidated on data/config changes, ~100x speedup on hits

### Performance
- **Cached geocoding/crime**: ~100x speedup on re-runs
- **Web app**: 1-3s load time on 4G, instant client-side search

---

## File Structure

```
map_uk_schools/
│
├── KS5 data pipeline
│   ├── ks5_download_data.py           # KS5 data downloader (idempotent)
│   ├── school_data_lib.py             # KS5 processing library
│   ├── generate_school_data.py        # Data processor
│   ├── config.json                    # KS5 settings
│   ├── 2023-2024_england_ks5final.csv.gz  # Downloaded KS5 data
│   ├── 2022-2023_england_ks5final.csv.gz
│   ├── 2021-2022_england_ks5final.csv.gz
│   ├── processed_school_data.csv      # Generated output
│   ├── geocoding_cache.json
│   └── crime_cache.json
│
├── KS2 data pipeline
│   ├── ks2_download_data.py               # Data downloader (idempotent)
│   ├── ks2_school_data_lib.py             # KS2 processing library
│   ├── ks2_generate_school_data.py        # Pipeline script
│   ├── ks2_config.json                    # KS2 settings
│   ├── ks2_school_attainment_202324.csv   # Downloaded from EES (2023/24)
│   ├── ks2_school_attainment_202223.csv   # Downloaded from EES (2022/23)
│   ├── ks2_school_attainment_202122.csv   # Downloaded from EES (2021/22)
│   ├── gias_establishments.csv            # Downloaded from GIAS
│   ├── ks2_processed_school_data.csv      # Generated output
│   ├── ks2_geocoding_cache.json
│   └── ks2_crime_cache.json
│
├── Map generator
│   ├── combined_create_standalone_app.py  # Reads both pipelines → single map
│   ├── times_ranking.py                    # Times Parent Power rank matcher
│   ├── extract_times_table.py              # Times league-table extractor
│   ├── best_schools_2025_combined_secondary.csv   # Times secondary ranks
│   ├── best_schools_2025_top500_state_primary.csv # Times primary ranks
│   └── school_finder.html                  # Generated output (KS2 + KS5)
│
├── Shared
│   ├── consolidate_crime_data.py      # Crime data consolidator
│   ├── combined_crimes.csv.gz         # Generated crime data
│   └── crime_lookup.py                # Address-based crime aggregator (standalone)
│
└── State secondary ethnicity report (standalone)
    ├── grammar_ethnicity_report.py             # Ethnicity report (Parent Power cohort × DfE census)
    ├── build_grammar_pdf.sh                     # Markdown -> landscape PDF builder
    ├── best_schools_2025_state_secondary.csv   # Times state-secondary cohort + ranks
    ├── grammar_ethnicity_report.md             # Generated Markdown report
    ├── grammar_ethnicity_report.pdf            # Generated PDF (cmark-gfm + wkhtmltopdf, landscape)
    └── spc_school_level.csv                    # Downloaded census cache (gitignored)
```

---

## License

- School data: UK Government Open Data License
- Crime data: data.police.uk Open Data License
- Map tiles: © OpenStreetMap contributors
- Code: MIT License
