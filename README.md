# Map UK Schools

An interactive visualization tool for analyzing UK schools based on A-level performance data across multiple years. The tool geocodes school addresses, clusters nearby schools geographically, and overlays local crime statistics to provide comprehensive insights for school selection.

**[View the interactive map →](http://web.andico.org/schools_map.html)**

![Schools Map Screenshot](schools_map.png)

## Data Sources

### School Performance Data
Download CSV files from https://www.compare-school-performance.service.gov.uk/download-data/:

- select desired year (e.g., 2021-2022, 2022-2023, 2023-2024)
- select all of England
- select 16-18 results (final)
- select CSV format

Note that only the 2021-2022, 2022-2023, and 2023-2024 years have the options for "16-18 results (final)" - 2024-2025 results are not yet available as of October 2025, and previous years have a different schema/format.

The downloaded files will have the correct naming format. Gzip them after download:
```bash
gzip 2021-2022_england_ks5final.csv
gzip 2022-2023_england_ks5final.csv
gzip 2023-2024_england_ks5final.csv
```

### Crime Data (Optional)
Download police/crime data from https://data.police.uk/data/:
- Select the "Custom download" tab, your desired time period and the "all forces" checkbox
- Select the "Include crime data" checkbox (outcomes data is optional and excluded by default)
- Download the ZIP file and extract it to a directory
- Run `consolidate_crime_data.py` with the path to that directory to create `combined_crimes.csv.gz`:
  ```bash
  python consolidate_crime_data.py --crime-data-dir /path/to/extracted/data
  ```

This adds local crime statistics to each school's popup showing serious crimes within a 3km radius. The crime index is a percentile ranking (0.00-1.00 scale) showing how the area compares to all other locations in the dataset. For example, a crime index of 0.95 means the radius around that school is at the 95th percentile (higher crime than 95% of locations).

**Crime Filtering**: Only serious crimes are counted in the safety index. The following crime types are excluded as they are considered lower-impact or less relevant to school safety:
- Shoplifting
- Bicycle theft  
- Other theft
- Other crime
- Drugs
- Anti-social behaviour
- Criminal damage and arson

The remaining crimes (such as burglary, robbery, violence, sexual offences, vehicle crime, etc.) are included in the crime count and percentile calculation.

## Requirements

```bash
pip install pandas boto3 tqdm folium geopy scikit-learn
```

AWS credentials must be configured (via `aws configure`) for Amazon Location Services geocoding. You'll need to create a Place Index in Amazon Location Service and update the `index_name` parameter in `config.json` (default: "your-place-index-name").

## Scripts

### plot_schools.py
Main script that performs end-to-end processing and visualization:

**Data Loading & Processing:**
- Auto-discovers all `20*ks5final*.csv.gz` files in current directory
- Consolidates multi-year data per school using address+postcode as unique key
- Calculates average TB3PTSE and TALLPPE_ALEV_1618 scores across available years
- Filters schools by configurable percentile threshold (default 90th percentile)
- Filters by minimum age threshold to include only schools with younger students

**Geocoding & Crime Analysis:**
- Geocodes addresses using AWS Location Services with persistent caching
- Calculates crime statistics within 3km radius of each school location
- Uses intelligent caching with SHA256 file hashing to detect crime data changes
- Generates percentile-based crime index (0.00-1.00 scale)
- Excludes low-impact crimes (shoplifting, bicycle theft, drugs, anti-social behaviour, etc.)

**Clustering Algorithm:**
- Uses BallTree spatial indexing with haversine metric for accurate geographic distances
- 5-10x faster than O(n²) approaches for large datasets
- Identifies schools within specified radius and clusters them
- Calculates geographic centroids using Cartesian coordinate conversion
- Reverse geocodes cluster centers to obtain postcodes

**Output Files:**
- `schools_map.html` - Interactive folium map with markers, clusters, and popups
- `processed_school_data.csv` - Filtered and geocoded school data sorted by performance
- `geocoding_cache.json` - Cached geocoding results to avoid redundant API calls
- `crime_cache.json` - Cached crime statistics with comprehensive validation
- `plot_schools.log` - Detailed logging of all operations

Usage:
```bash
python plot_schools.py [--radius 5] [--min-schools 2]
```

**Map Features:**
- Color-coded markers by performance (A*/A/≤B) and school type (independent/state)
- Markers display rounded TB3PTSE score
- Cluster circles with radius visualization
- Cluster center markers with school counts and postcodes

**Popup Information:**
- School name, address, phone, admissions policy, age range, gender, student count
- Average TB3PTSE and TALLPPE_ALEV_1618 scores with grade equivalents
- Year-by-year score breakdown
- Crime statistics: count of serious crimes within 3km radius
- Crime index (percentile ranking from 0.00 to 1.00)
- Top 3 crime types in the area

### consolidate_crime_data.py
Utility script that consolidates multiple crime CSV files from https://data.police.uk/data/ into a single compressed file.

**Features:**
- Recursively walks through directory structure to find all crime CSV files
- Extracts essential columns: Month, Longitude, Latitude, Crime type
- Optionally excludes outcome files (enabled by default)
- Outputs compressed gzipped CSV for efficient storage
- Configurable through `config.json` or command-line arguments

Usage:
```bash
python consolidate_crime_data.py --crime-data-dir /path/to/extracted/crime/data --output combined_crimes.csv.gz
```

Or configure in `config.json`:
```json
{
  "crime": {
    "source_crime_data_dir": "/path/to/extracted/crime/data",
    "crime_data_file": "combined_crimes.csv.gz"
  }
}
```

**Note:** This script only needs to be run once after downloading and extracting crime data. The output file is then used by `plot_schools.py` for all crime analysis.

## Configuration

All configuration is now managed through `config.json`. Key settings include:

- **Percentile filtering**: Default 90th percentile
- **Clustering**: Default 5km radius, minimum 2 schools (can override via command line)
- **Crime analysis**: 3km radius per school, with configurable excluded crime types
- **Geocoding**: AWS Location Services region and index name
- **Colors**: Customizable color schemes for independent/state schools and clusters
- **Grading thresholds**: A* and A grade point score thresholds

Edit `config.json` to customize these settings without modifying code.

## Output Files

- `schools_map.html` - Interactive folium map with markers, clusters, crime overlays, and legend
- `processed_school_data.csv` - Filtered and geocoded school data sorted by TB3PTSE score
- `geocoding_cache.json` - Persistent cache of address → (lat, lon) mappings to avoid redundant AWS API calls
- `crime_cache.json` - Persistent cache of crime statistics with metadata for validation (file hash, excluded crime types, radius)
- `plot_schools.log` - Detailed log file with timestamps, info, warnings, and errors from script execution

## Recent Improvements

### Configuration Management (2025)
All hardcoded constants moved to `config.json` for easier customization:
- Single configuration file for all settings
- Graceful fallback to defaults if config file is missing
- Easy modification of colors, thresholds, and parameters

### Performance Optimization (2025)
**Clustering Algorithm**: Replaced O(n²) nested loop approach with BallTree spatial indexing
- **5-10x faster** clustering for large datasets
- O(n log n) complexity instead of O(n²)
- Uses haversine metric with Earth radius for accurate geographic distance calculations
- BallTree queries all nearby schools within radius in a single operation
- Geographic centroid calculation uses Cartesian coordinate conversion for accuracy
- For 500 schools: ~5,000 calculations vs ~250,000 in old version

**Crime Statistics Caching**: Added intelligent caching system with robust invalidation
- Dual caching: geocoding cache for coordinates, crime cache for statistics
- Uses SHA256 file hashing to detect any content changes in crime data file
- Crime cache metadata includes: file hash, excluded crime types, and crime radius
- Automatically invalidates cache when:
  - Crime data file content changes (detected via hash, not just timestamp)
  - Excluded crime types list changes in config.json
  - Crime radius parameter changes in config.json
- Provides clear feedback about why cache was invalidated
- Significantly reduces processing time for repeated runs (typical: ~100x speedup with warm cache)
- Vectorized haversine distance calculations using NumPy for crime filtering (~100x faster than loops)

### Code Quality (2025)
- Added comprehensive configuration validation with detailed error messages
- Proper error handling for configuration loading with graceful fallback to defaults
- Improved modularity: main() function orchestrates pipeline through well-defined helper functions
- Extensive inline documentation explaining algorithms and optimization techniques
- Logging system with both file output (`plot_schools.log`) and console output
- ITU-R BT.601 luma coefficients for perceptually-correct marker text color selection

### Progress Bar Clarity (2025)
**Problem**: The progress bar always showed "Geocoding" even when addresses were cached but crime statistics needed recalculation, making it unclear what work was being done.

**Solution**: Enhanced progress bar to show exactly what's happening:
- Changed description from "Geocoding" to "Processing (geocoding and crime stats)"
- Added separate counters for geocoding hits/misses and crime calculations
- Progress bar now shows: `[geo_hits=95, geo_miss=5, crime_calc=42]`
- Summary message split into clear sections: "Geocoding: 95 cached, 5 new | Crime stats: 42 calculated"

**Example scenarios:**
- All cached: `geo_hits=100, geo_miss=0` (no crime_calc shown - super fast!)
- Crime cache invalidated: `geo_hits=100, geo_miss=0, crime_calc=100` (recalculating crime stats)
- New schools: `geo_hits=85, geo_miss=15, crime_calc=15` (geocoding and analyzing new schools)
