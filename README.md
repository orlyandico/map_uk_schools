# UK Schools Finder

Interactive map visualization tool for analyzing UK schools based on A-level performance and local crime statistics. Creates a standalone web application that works on any device with no server required.

**[Web App →](http://web.andico.org/school_finder.html)**

**[Static Map with School Clusters →](http://web.andico.org/schools_map.html)**

![Schools Map Screenshot](schools_map.png)

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Data Sources](#data-sources)
- [Requirements](#requirements)
- [Usage](#usage)
- [Web Application Features](#web-application-features)
- [Deployment](#deployment)
- [Updating Data](#updating-data)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

This project provides tools to:
1. **Process** UK school performance data across multiple years
2. **Geocode** school addresses using AWS Location Services
3. **Analyze** local crime statistics around each school
4. **Generate** visualizations:
   - Interactive standalone web app (location-based search)
   - Static cluster map (for finding dense areas of high-performing schools)

### Key Features

- ✅ **Standalone Web App** - Single HTML file, no server required
- ✅ **Mobile-Friendly** - Works on iOS and Android browsers
- ✅ **Location-Based Search** - GPS or address/postcode with 5km radius
- ✅ **Cluster Analysis** - Static map showing geographic clusters of schools
- ✅ **Crime Statistics** - Percentile-ranked safety data
- ✅ **Multi-Year Averaging** - Consolidates performance across years
- ✅ **S3-Ready** - Deploy to AWS S3 for static hosting

### Two Visualization Options

1. **Standalone Web App** (`school_finder.html`)
   - Mobile-friendly, location-based searching
   - Find schools near a specific address
   - Perfect for individual school searches

2. **Static Cluster Map** (`schools_map.html`)
   - Shows all schools with geographic clustering
   - Identifies dense areas of high-performing schools
   - Useful for finding optimal home locations
   - Great for area comparison and planning

---

## Quick Start

### Step 1: Download School Performance Data

Visit https://www.compare-school-performance.service.gov.uk/download-data/

1. Select year (2021-2022, 2022-2023, or 2023-2024)
2. Select "All of England"
3. Select "16-18 results (final)"
4. Select "CSV" format
5. Download and gzip the files:

```bash
gzip 2021-2022_england_ks5final.csv
gzip 2022-2023_england_ks5final.csv
gzip 2023-2024_england_ks5final.csv
```

### Step 2: Download Crime Data (Optional)

Visit https://data.police.uk/data/

1. Select "Custom download" tab
2. Choose time period and check "all forces"
3. Check "Include crime data"
4. Download ZIP and extract to a directory
5. Run consolidation script:

```bash
python3 consolidate_crime_data.py --crime-data-dir /path/to/extracted/data
```

This creates `combined_crimes.csv.gz`.

### Step 3: Configure AWS Credentials

```bash
aws configure
# Enter your AWS Access Key ID and Secret Access Key
```

Create a Place Index in Amazon Location Service and update `config.json`:

```json
{
  "geocoding": {
    "index_name": "your-place-index-name",
    "region_name": "eu-west-2"
  }
}
```

### Step 4: Process School Data & Generate Visualizations

**Option A: Generate Both Visualizations (Recommended)**

```bash
python3 plot_schools.py
```

**Time:** 5-15 minutes
**Output:**
- `processed_school_data.csv` - Processed data
- `schools_map.html` - Static cluster map with Folium
- `geocoding_cache.json` - Geocoding cache
- `crime_cache.json` - Crime statistics cache

**Option B: Just Process Data (for web app only)**

```bash
python3 generate_school_data.py
```

**Time:** 5-15 minutes (same processing, no map generation)
**Output:** Same as above except no `schools_map.html`

### Step 5: Generate Standalone Web App

```bash
python3 create_standalone_app.py
```

**Time:** ~5 seconds
**Output:** `school_finder.html` (~450KB)

### Step 6: Use the App

**Local Use:**
```bash
# Just double-click the file or:
open school_finder.html
```

**Deploy to S3:**
```bash
aws s3 cp school_finder.html \
  s3://your-bucket/index.html \
  --content-type "text/html" \
  --acl public-read
```

---

## Data Sources

### School Performance Data

- **Source:** UK Government Compare School Performance service
- **URL:** https://www.compare-school-performance.service.gov.uk/download-data/
- **Format:** CSV (gzipped)
- **Years:** 2021-2024 (only these have "16-18 results (final)")
- **Key Metrics:**
  - TB3PTSE: Average of best 3 A-levels
  - TALLPPE_ALEV_1618: Average per A-level

### Crime Data

- **Source:** data.police.uk
- **URL:** https://data.police.uk/data/
- **Format:** CSV files (consolidated and gzipped)
- **Coverage:** All police forces in England
- **Radius:** 3km around each school
- **Filtered Crimes:** Excludes shoplifting, bicycle theft, drugs, anti-social behavior, etc.

---

## Requirements

### Python Packages

```bash
pip install pandas boto3 tqdm folium geopy scikit-learn numpy
```

### AWS Services

- AWS Account with Location Services enabled
- Place Index created in Amazon Location Service
- IAM credentials with location:SearchPlaceIndexForText permission

### System Requirements

- Python 3.7+
- 2GB+ RAM
- Internet connection for initial data processing
- Modern web browser for viewing app

---

## Usage

### Complete Workflow

**For Both Visualizations:**

```bash
# 1. Process data and generate static cluster map
python3 plot_schools.py

# 2. Generate standalone web app
python3 create_standalone_app.py

# 3. View cluster map (desktop browser)
open schools_map.html

# 4. Use web app (any device)
open school_finder_standalone.html

# 5. Deploy web app to cloud (optional)
aws s3 cp school_finder_standalone.html s3://your-bucket/index.html --acl public-read
```

**For Web App Only (skip static map):**

```bash
# 1. Process data only
python3 generate_school_data.py

# 2. Generate standalone web app
python3 create_standalone_app.py

# 3. Use or deploy
open school_finder.html
```

### Updating with New Data

When new school data is released:

```bash
# Download and gzip new CSV files
gzip 2024-2025_england_ks5final.csv

# Regenerate everything
python3 plot_schools.py              # Generates static map + caches
python3 create_standalone_app.py     # Generates web app

# Or just process data without static map
python3 generate_school_data.py      # Just caches, no map
python3 create_standalone_app.py     # Generates web app

# Redeploy web app
aws s3 cp school_finder.html s3://your-bucket/index.html --acl public-read
```

**Caching:** Existing schools use cached geocoding and crime data (fast!)

**Score Distribution Analysis:** Both scripts now output a distribution table showing TB3PTSE scores by decile, helping you understand where to set the percentile filter. Example output:

```
============================================================
TB3PTSE Score Distribution by Decile
============================================================
Percentile   Score      Grade    Schools
------------------------------------------------------------
Min            25.00    ≤B           10132
P10            35.50    ≤B            9119
P20            38.75    ≤B            8106
P30            40.25    A             7093
...
P90            54.25    A*            1015
Max            62.50    A*               1
------------------------------------------------------------
Marker color thresholds (for visualization): A* ≥50, A ≥40, ≤B <40
Note: Scores reflect average points across best 3 A-levels (A*=60, A=50, B=40)
```

This helps you decide which percentile filter to use based on the grade distribution.

### Use Cases

**Static Cluster Map (`schools_map.html`):**
- Planning home purchases - find dense clusters of good schools
- Area comparison - which neighborhoods have most options
- Investment analysis - identify high-performing school regions
- Desktop viewing with detailed cluster information

**Standalone Web App (`school_finder.html`):**
- On-the-go school searches from mobile devices
- Quick lookup near specific addresses
- Sharing with family/friends
- Embedding in websites
- Real estate listings integration

---

## Static Cluster Map Features

The `schools_map.html` file generated by `plot_schools.py` provides:

### Clustering Algorithm

- **BallTree Spatial Indexing** - Efficient O(n log n) geographic clustering
- **Haversine Distance** - Accurate great-circle distance calculations
- **Geographic Centroids** - Cluster centers calculated using Cartesian coordinates
- **Configurable Radius** - Default 5km, adjustable via command line
- **Minimum School Count** - Default 2 schools per cluster, adjustable

### Visualization

**Cluster Circles:**
- Blue circles showing cluster boundaries
- Radius visualization (default 5km)
- Cluster center markers with IDs

**School Markers:**
- Same color coding as web app (by type and performance)
- Displays TB3PTSE score on marker
- Popup with full school details

**Legend:**
- Color coding explanation
- Independent vs. State schools
- Grade thresholds

### Use Cases

**Finding Dense Areas:**
- Identify neighborhoods with multiple high-performing schools
- Compare different regions of England
- Plan optimal home location for school choice

**Area Analysis:**
- See which cities/towns have school clusters
- Evaluate catchment area overlaps
- Investment and planning decisions

**Command Line Options:**
```bash
# Custom cluster radius (e.g., 10km)
python3 plot_schools.py --radius 10

# Minimum schools per cluster (e.g., 3)
python3 plot_schools.py --min-schools 3

# Both
python3 plot_schools.py --radius 8 --min-schools 4
```

---

## Web Application Features

### User Interface

- **Clean, modern design** with floating controls
- **Mobile-optimized** with touch-friendly buttons
- **Real-time status** updates and error messages
- **Interactive legend** showing color coding
- **Loading indicators** for async operations

### Search Methods

1. **GPS Location**
   - Click "📍 Use My Location"
   - Automatic permission request
   - Centers map on your current position

2. **Address Search**
   - Enter postcode, address, or landmark
   - Uses OpenStreetMap Nominatim geocoding (free)
   - Automatic UK location bias

### Map Features

**Markers:**
- Color-coded by school type and performance grade
- Display TB3PTSE score directly on marker
- Custom SVG pins for professional look

**Radius Circle:**
- Visual 5km radius indicator
- Automatically filters schools
- Updates with each search

**Popups:**
Each school marker shows:
- School name and full address
- Phone number (clickable on mobile)
- Admission policy (Independent/State/Selective)
- Gender and age range
- Student count (16-18)
- Average performance scores with grade badges
- Year-by-year score breakdown
- Crime statistics: count, percentile index, top 3 crime types

### Color Coding

**Independent Schools:**
- 🔴 Red: A* grade (TB3PTSE ≥ 50)
- 🟠 Orange: A grade (TB3PTSE 40-49)
- 🟡 Yellow: ≤B grade (TB3PTSE < 40)

**State Schools:**
- 🔵 Navy: A* grade (TB3PTSE ≥ 50)
- 🔵 Blue: A grade (TB3PTSE 40-49)
- 🔵 Light Blue: ≤B grade (TB3PTSE < 40)

### Crime Index

- **Scale:** 0.00 to 1.00 (percentile ranking)
- **Interpretation:** Higher = more crimes relative to other areas
- **Example:** 0.85 = higher crime than 85% of locations
- **Radius:** 3km around school address
- **Filtered:** Excludes low-impact crimes

---

## Deployment

### AWS S3 (Recommended)

**Quick Deploy:**
```bash
# Create bucket
aws s3 mb s3://uk-schools-finder

# Enable static hosting
aws s3 website s3://uk-schools-finder \
  --index-document index.html

# Upload file
aws s3 cp school_finder.html \
  s3://uk-schools-finder/index.html \
  --content-type "text/html" \
  --acl public-read

# Make public
aws s3api put-bucket-policy \
  --bucket uk-schools-finder \
  --policy '{
    "Version": "2012-10-17",
    "Statement": [{
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::uk-schools-finder/*"
    }]
  }'
```

**Access at:** `http://uk-schools-finder.s3-website-REGION.amazonaws.com`

**Note:** For HTTPS support (required for geolocation features), add CloudFront in front of S3.

---

## Updating Data

### When to Update

- New school year data released (annually)
- Crime data updated (monthly/quarterly)
- Configuration changes

### Update Process

```bash
# 1. Process new data
python3 generate_school_data.py

# 2. Regenerate app
python3 create_standalone_app.py

# 3. Deploy
aws s3 cp school_finder.html s3://your-bucket/index.html --acl public-read
```

---

## Configuration

All settings in `config.json`:

### Key Settings

```json
{
  "filtering": {
    "percentile": 0.90,
    "min_age_threshold": 7
  },
  "crime": {
    "school_crime_radius_km": 3,
    "excluded_crime_types": [
      "Shoplifting",
      "Bicycle theft",
      "Other theft",
      "Drugs",
      "Anti-social behaviour"
    ]
  },
  "grading": {
    "a_star_threshold": 50,
    "a_threshold": 40
  },
  "colors": {
    "independent": {
      "a_star": "#FF0000",
      "a": "#FFA500",
      "b_or_below": "#FFD700"
    },
    "state": {
      "a_star": "#000080",
      "a": "#0000FF",
      "b_or_below": "#4169E1"
    }
  }
}
```

### Customization

**Change filtering:**
- `percentile`: 0.90 = top 10% of schools
- `min_age_threshold`: 7 = includes secondary schools only

**Change crime radius:**
- `school_crime_radius_km`: 3 = 3km radius (default)

**Change grading thresholds:**
- `a_star_threshold`: 50 = A* grade cutoff
- `a_threshold`: 40 = A grade cutoff

---

## File Structure

```
map_uk_schools/
├── school_data_lib.py             # Shared library with common functions
├── plot_schools.py                # Full processing + static cluster map
├── generate_school_data.py        # Data processing only (no map)
├── create_standalone_app.py       # Standalone web app generator
├── consolidate_crime_data.py      # Crime data consolidation
├── config.json                    # Configuration settings
├── README.md                      # This file
│
├── 2021-2022_england_ks5final.csv.gz  # School data (download)
├── 2022-2023_england_ks5final.csv.gz
├── 2023-2024_england_ks5final.csv.gz
├── combined_crimes.csv.gz             # Crime data (generated)
│
├── processed_school_data.csv      # Processed output
├── geocoding_cache.json           # Geocoding cache
├── crime_cache.json               # Crime statistics cache
├── plot_schools.log               # Processing log
│
├── schools_map.html               # Static cluster map (Folium)
└── school_finder.html             # Standalone web app (deploy!)
```

### Script Comparison

| Feature | `school_data_lib.py` | `plot_schools.py` | `generate_school_data.py` |
|---------|----------------------|-------------------|---------------------------|
| Role | Shared library | Full pipeline + map | Data processing only |
| Process CSVs | ✅ (functions) | ✅ (uses lib) | ✅ (uses lib) |
| Geocoding | ✅ (functions) | ✅ (uses lib) | ✅ (uses lib) |
| Crime stats | ✅ (functions) | ✅ (uses lib) | ✅ (uses lib) |
| Clustering | ❌ | ✅ (native) | ❌ |
| Static map | ❌ | ✅ (`schools_map.html`) | ❌ |
| Speed | N/A | Slower (generates map) | Faster (data only) |
| Lines of code | ~878 | ~724 | ~120 |
| Use case | Code reuse | Desktop analysis | Web app prep |

---

## Code Architecture

### Shared Library Design

The project uses a DRY (Don't Repeat Yourself) architecture with a shared library:

**`school_data_lib.py`** - Core data processing library (~878 lines)
- Configuration management (load, validate, defaults)
- Geocoding with AWS Location Services
- Crime statistics calculation and caching
- School data loading and consolidation
- Filtering and percentile calculations
- Cache management with SHA256 validation

**`plot_schools.py`** - Static map generation (~724 lines)
- Imports `school_data_lib` for all data processing
- Adds clustering algorithm (BallTree spatial indexing)
- Folium map generation with markers and clusters
- Command-line interface for cluster parameters

**`generate_school_data.py`** - Data-only pipeline (~120 lines)
- Imports `school_data_lib` for all data processing
- Streamlined for fast CSV generation
- No clustering or visualization overhead

**Benefits:**
- ~1000+ lines of duplicate code eliminated
- Single source of truth for data processing
- Bug fixes and improvements made once
- Both scripts guaranteed to use identical logic
- Independent testing of core functionality

---

## Technical Details

### Data Processing Pipeline

1. **Load & Consolidate** - Combines multi-year CSV files
2. **Calculate Averages** - Per-school averages across years
3. **Filter by Percentile** - Keeps top-performing schools
4. **Clean & Geocode Addresses** - Filters out NaN/empty values, then geocodes with AWS Location Services
5. **Calculate Crime Stats** - 3km radius analysis with percentile ranking
6. **Save Outputs** - CSV + JSON caches with clean data

### Caching System

**Geocoding Cache:**
- Key: Full address string (cleaned, no NaN values)
- Value: (latitude, longitude)
- Never invalidated (addresses don't change)
- Saves ~$0.004 per lookup
- Address cleaning: Filters out pd.isna(), empty strings, and "nan" literals

**Crime Cache:**
- Key: `lat,lon,radius` (e.g., "51.5074,-0.1278,3")
- Value: Crime statistics dict
- Invalidated when: crime data changes, radius changes, excluded types change
- Includes SHA256 file hash for validation
- ~100x speedup on cache hits

### Algorithms

**Haversine Distance:**
```python
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dLat = (lat2 - lat1) * π / 180
    dLon = (lon2 - lon1) * π / 180
    a = sin(dLat/2)² + cos(lat1) * cos(lat2) * sin(dLon/2)²
    c = 2 * atan2(√a, √(1-a))
    return R * c
```

**Percentile Ranking:**
```python
crime_index = crime_count.rank(pct=True)
# Returns 0.00 to 1.00
```

### Web App Architecture

- **Frontend:** Pure JavaScript ES6 (no frameworks)
- **Mapping:** Leaflet.js 1.9.4
- **Tiles:** OpenStreetMap (free)
- **Geocoding:** Nominatim (free, no API key)
- **Data:** Embedded JSON in HTML file
- **Compatibility:** iOS Safari 12+, Android Chrome 80+, all modern desktop browsers

### Performance

- **File Size:** ~450KB compressed with gzip during transfer
- **Load Time:** 1-3 seconds on 4G
- **Search:** Instant (client-side filtering)
- **Markers:** Renders 50-100 schools instantly

---

## Troubleshooting

### Data Processing Issues

**"No school data files found"**
- Check CSV files are in current directory
- Verify filenames match pattern: `20*ks5final*.csv.gz`
- Ensure files are gzipped

**"AWS Error geocoding"**
- Verify AWS credentials: `aws configure list`
- Check Place Index exists in Location Service
- Verify IAM permissions include `location:SearchPlaceIndexForText`

**"Crime data file not found"**
- Run `consolidate_crime_data.py` first
- Check `combined_crimes.csv.gz` exists
- Crime data is optional (app works without it)

**"Configuration Error"**
- Check `config.json` is valid JSON
- Verify all required keys present
- Use defaults if needed (code has fallbacks)

### Web App Issues

**"Failed to load school data"**
- Not applicable to standalone version (data is embedded)
- Ensure you ran `create_standalone_app.py` to generate the HTML file

**Location permission denied**
- Click location icon in browser address bar
- Select "Allow" for location access
- Note: Requires HTTPS for production (add CloudFront in front of S3)

**No schools showing**
- Zoom out to see if schools nearby
- Try different location (e.g., "London")
- Check browser console (F12) for errors

**Map not loading**
- Check internet connection
- Verify access to unpkg.com (Leaflet CDN)
- Verify access to openstreetmap.org (tiles)

### Deployment Issues

**File downloads instead of opening**
- Set content-type: `--content-type "text/html"`
- Verify S3 static hosting enabled

**403 Forbidden on S3**
- Check bucket policy allows public access
- Verify "Block public access" settings are off
- Ensure file has public-read ACL

**Old version still showing**
- Clear browser cache (Cmd+Shift+R / Ctrl+Shift+R)
- Check upload succeeded (S3 console)

---

## Recent Improvements

### 2025 Updates

**Code Refactoring (January 2025):**
- Created shared library `school_data_lib.py` with all common functions
- Eliminated ~1000+ lines of duplicate code between scripts
- `plot_schools.py` reduced from 1631 to 724 lines (~56% reduction)
- `generate_school_data.py` reduced from 1605 to 120 lines (~93% reduction)
- Single source of truth for all data processing logic
- Improved maintainability and consistency
- **NaN address handling fixed at source**: Library filters out NaN values during geocoding, preventing "nan" strings in all downstream code and caches

**Standalone Web App:**
- All data embedded in single HTML file
- No external dependencies (except CDN)
- Works without web server
- Perfect for S3 static hosting
- Fixed popup layout: consistent 9pt font, two-column grid design
- Simplified JavaScript code: removed redundant NaN checks (data pre-cleaned by Python)

**Simplified Workflow:**
- Two processing options: full (`plot_schools.py`) or data-only (`generate_school_data.py`)
- `plot_schools.py` generates static cluster map for area analysis
- `generate_school_data.py` for faster processing when map not needed
- Separated web app generation into `create_standalone_app.py`
- Clearer pipeline: process data → generate app → deploy

**Performance:**
- BallTree spatial indexing for clustering (5-10x faster)
- Vectorized haversine distance calculations (~100x faster)
- SHA256-based cache validation
- Intelligent cache invalidation

**Documentation:**
- Consolidated all guides into single README
- Clear step-by-step instructions
- Troubleshooting section
- Deployment options

---

## License

School performance data: UK Government Open Data License
Crime data: data.police.uk Open Data License
Code: MIT License (or your chosen license)

Map tiles: © OpenStreetMap contributors

---

## Support

**Issues:** Report at project repository
**Data:** UK Government school performance service
**Crime Data:** data.police.uk

---

**Built with:** Python, Pandas, Boto3, Leaflet.js, OpenStreetMap
