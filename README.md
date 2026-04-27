# UK Schools Finder

Interactive map tool for analyzing UK schools by A-level performance and local crime statistics. Generates a standalone web app that works on any device with no server required.

**[Live Demo: Web App →](http://web.andico.org/school_finder.html)**

<img src="schoolfinder.png" alt="School Finder Screenshot" width="400">

**[Live Demo: Static Cluster Map →](http://web.andico.org/schools_map.html)**

![Schools Map Screenshot](schools_map.png)

---

## Overview

Two parallel pipelines:

- **KS5 (secondary schools)** — ranked by 16-18 A-level performance
- **KS2 (primary schools, state only)** — ranked by SATs results

Both share the geocoding (AWS Location Services) and optional crime data (data.police.uk) infrastructure, and both produce a single standalone HTML file with a mobile-friendly search interface.

A third script, `combined_create_standalone_app.py`, merges the outputs of both pipelines into a single HTML so primary and secondary schools appear on the same map (different marker shapes, shared reticle, pooled crime percentile). This is what the live demo link above serves.

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

### 2. Generate Outputs

Run the data pipeline once, then render whichever outputs you want:

```bash
python3 generate_school_data.py      # 5-15 mins, generates processed_school_data.csv + caches
python3 create_standalone_app.py     # seconds, generates school_finder.html (mobile web app)
python3 plot_schools.py              # seconds, generates schools_map.html (static cluster map)
```

Both rendering scripts are independent and read from `processed_school_data.csv` + the cache files, so you can regenerate either HTML at any time without re-running the data pipeline.

`create_standalone_app.py` reads `processed_school_data.csv` and the cache files. Optional flags:
```bash
python3 create_standalone_app.py --input processed_school_data.csv --output school_finder.html
```

### Outputs

#### Static Cluster Map (`schools_map.html`)
- Self-contained Leaflet map (same rendering stack as the web app)
- Shows geographic clusters of high-performing schools with translucent cluster-area circles and numbered cluster-centre markers
- Configurable radius: `python3 plot_schools.py --radius 10 --min-schools 3`
- Use case: Finding dense areas for home purchases

#### Web App (`school_finder.html`)
- Mobile-friendly standalone HTML
- Location-based search (GPS or address)
- Dynamic radius filtering with automatic updates when scrolling map
- Adaptive crosshair/reticle that adjusts radius based on zoom level (5km max, 500m min, max 60% of screen)
- Auto-zoom to appropriate level for address searches
- Distance scale ruler for reference
- Works offline after initial load
- Use case: On-the-go school searches

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

### 2. Process and Generate

```bash
python3 ks2_generate_school_data.py      # 5-15 mins, generates ks2_processed_school_data.csv
python3 ks2_create_standalone_app.py     # seconds, generates ks2_school_finder.html
```

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

## Combined Map (KS2 + KS5)

Once both pipelines have been run, generate a single HTML containing both cohorts:

```bash
python3 combined_create_standalone_app.py     # generates combined_school_finder.html
```

The script reads the processed CSVs and cache files from both pipelines, emits one standalone HTML, and re-ranks the crime index over the pooled cohort so primaries and secondaries are directly comparable.

Marker shapes distinguish the two school types:
- **KS5 (secondary):** teardrop pin, labelled with TB3PTSE
- **KS2 (primary):** circle, labelled with % meeting expected standard

The legend, reticle and search behaviour match the KS5 web app. There is no toggle — both school types are always visible.

Optional flags override any of the input paths or the output file (see `--help`).

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

A per-address legend at the top of the CSV (one row each: full normalised address from AWS + short code), followed by the column header row and the data:

```csv
"10 Downing Street, London SW1A 2AA, UK",A1
"Buckingham Palace, London SW1A 1AA, UK",A2
Radius_m,Crime type,A1,A2
500,Burglary,2,0
500,Robbery,0,1
500,Violence,2,1
...
```

Crime types are listed alphabetically and zero-filled when absent so columns line up across addresses for side-by-side comparison.

To load just the data table with pandas, skip the legend rows:

```python
import pandas as pd
n_addresses = 2  # equal to the number of legend rows in the file
df = pd.read_csv("crime_lookup_output.csv", header=n_addresses)
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
python3 create_standalone_app.py
python3 plot_schools.py              # optional: refresh schools_map.html too

# Redeploy
aws s3 cp school_finder.html s3://your-bucket/index.html --acl public-read
```

---

## Architecture

### KS5 Scripts

- **`ks5_download_data.py`** — Downloads KS5 16-18 final results CSVs from DfE (idempotent)
- **`school_data_lib.py`** (~885 lines) - Shared library with all data processing functions
- **`consolidate_crime_data.py`** (~150 lines) - Consolidates downloaded crime CSV files
- **`generate_school_data.py`** (~120 lines) - Data pipeline: load → filter → geocode → crime stats → save
- **`create_standalone_app.py`** (~1100 lines) - Builds `school_finder.html` (mobile web app) with embedded data
- **`plot_schools.py`** (~430 lines) - Builds `schools_map.html` (static cluster map) with embedded data; uses a vectorised haversine distance matrix for clustering

### KS2 Scripts

- **`ks2_download_data.py`** — Downloads KS2 performance data from EES and GIAS establishment list
- **`ks2_school_data_lib.py`** (~530 lines) — KS2 data processing: load long-format EES data, pivot, join GIAS, consolidate across years
- **`ks2_generate_school_data.py`** (~120 lines) — Full pipeline: load → filter → geocode → crime stats → save
- **`ks2_create_standalone_app.py`** (~800 lines) — Builds standalone `ks2_school_finder.html` with embedded data

### Combined Script

- **`combined_create_standalone_app.py`** (~940 lines) — Reads both pipelines' outputs and emits `combined_school_finder.html` with KS2 + KS5 markers on one map, pooled crime percentile

### KS5 Data Flow

```
ks5_download_data.py → 20XX-20XX_england_ks5final.csv.gz (×3)
                                           ↓
Crime CSVs → consolidate_crime_data.py → combined_crimes.csv.gz
                                                    ↓
School CSVs → Consolidate → Filter by percentile → Geocode (AWS) → Calculate crime stats → Cache
                                                                                              ↓
                                           generate_school_data.py
                                                              ↓
                                           processed_school_data.csv + cache files
                                                              ↓
                                           ├── create_standalone_app.py → school_finder.html (web app)
                                           └── plot_schools.py          → schools_map.html  (cluster map)
```

### KS2 Data Flow

```
ks2_school_attainment_20XXYY.csv (×3, long format) ─────────────────────────────┐
                                                                                 │
gias_establishments.csv ────┐                                                    │
   (ks2_download_data.py)   ↓                                                    ↓
                     ks2_school_data_lib.py: pivot + join + consolidate across years
                                                    ↓
                      Filter by percentile → Geocode (AWS) → Crime stats → Cache
                                                    ↓
                              ks2_generate_school_data.py → ks2_processed_school_data.csv
                                                    ↓
                              ks2_create_standalone_app.py → ks2_school_finder.html
```

### Key Features

- **Single source of truth**: All common code in `school_data_lib.py`
- **Smart caching**: Geocoding and crime data cached with SHA256 validation
- **Clean addresses**: NaN values filtered at source during geocoding
- **Fast clustering**: BallTree spatial indexing (O(n log n))
- **Standalone output**: Web app is single HTML file (~450KB)

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
- **BallTree clustering**: O(n log n) vs O(n²)
- **Vectorized haversine**: ~100x faster than loops
- **Web app**: 1-3s load time on 4G, instant client-side search

---

## File Structure

```
map_uk_schools/
│
├── KS5 pipeline
│   ├── ks5_download_data.py           # KS5 data downloader (idempotent)
│   ├── school_data_lib.py             # Shared library
│   ├── plot_schools.py                # Static map generator
│   ├── generate_school_data.py        # Data processor
│   ├── create_standalone_app.py       # Web app generator
│   ├── config.json                    # KS5 settings
│   ├── 2023-2024_england_ks5final.csv.gz  # Downloaded KS5 data
│   ├── 2022-2023_england_ks5final.csv.gz
│   ├── 2021-2022_england_ks5final.csv.gz
│   ├── processed_school_data.csv      # Generated output
│   ├── geocoding_cache.json
│   ├── crime_cache.json
│   ├── schools_map.html
│   └── school_finder.html
│
├── KS2 pipeline
│   ├── ks2_download_data.py               # Data downloader (idempotent)
│   ├── ks2_school_data_lib.py             # KS2 processing library
│   ├── ks2_generate_school_data.py        # Pipeline script
│   ├── ks2_create_standalone_app.py       # Web app generator
│   ├── ks2_config.json                    # KS2 settings
│   ├── ks2_school_attainment_202324.csv   # Downloaded from EES (2023/24)
│   ├── ks2_school_attainment_202223.csv   # Downloaded from EES (2022/23)
│   ├── ks2_school_attainment_202122.csv   # Downloaded from EES (2021/22)
│   ├── gias_establishments.csv            # Downloaded from GIAS
│   ├── ks2_processed_school_data.csv      # Generated output
│   ├── ks2_geocoding_cache.json
│   ├── ks2_crime_cache.json
│   └── ks2_school_finder.html
│
├── Combined
│   ├── combined_create_standalone_app.py  # Merged map generator
│   └── combined_school_finder.html        # Generated output (KS2 + KS5)
│
└── Shared
    ├── consolidate_crime_data.py      # Crime data consolidator
    ├── combined_crimes.csv.gz         # Generated crime data
    └── crime_lookup.py                # Address-based crime aggregator (standalone)
```

---

## License

- School data: UK Government Open Data License
- Crime data: data.police.uk Open Data License
- Map tiles: © OpenStreetMap contributors
- Code: MIT License
