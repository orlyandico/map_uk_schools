# Map UK Schools

Use UK government data to filter and plot UK schools, filtering by A-level results across multiple years.

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
- Select the "Custom download tab," your desired time period and the "all forces" checkbox
- Select the "Include crime data" and "include outcomes data" checkboxes
- Download the ZIP file and unzip it to a directory
- Run `consolidate_crime_data.py` with the path to that directory (hard-coded in the file) to create `combined_crimes.csv.gz`

This adds local crime statistics to each school's popup showing serious crimes within a 3km radius (the count of crimes and percentile are included, across all crimes in the police data). For example, a crime index of 0.95 means the radius around that particular school address is at the 95th percentile for those crimes in the police data in England.

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
Main script that:
- Automatically finds all `20*ks5final*.csv.gz` files in current directory
- Consolidates data across multiple years per school (includes schools from ANY year)
- Calculates average performance scores across all available years
- Geocodes school addresses using Amazon Location Services with caching
- Creates interactive map with crime statistics (if available)
- Clusters schools within specified radius using geographic centroids
- Outputs: `processed_school_data.csv`, `schools_map.html`, and `geocoding_cache.json`

Usage:
```bash
python plot_schools.py [--radius 5] [--min-schools 2]
```

Map markers show average scores across all years. Popups display:
- School details and contact information
- Average scores across all available years
- Year-by-year breakdown (each year on separate line)
- Local crime statistics within 3km radius (with percentile-based crime index)

### consolidate_crime_data.py
Utility script to merge multiple crime CSV files from https://data.police.uk/data/ into a single `combined_crimes.csv` file. Edit the script to set your crime data directory path.

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

- `schools_map.html` - Interactive map with clustered schools
- `processed_school_data.csv` - Filtered and geocoded school data
- `geocoding_cache.json` - Cached geocoding results to avoid re-processing
- `crime_cache.json` - Cached crime statistics to speed up subsequent runs

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
- Uses haversine metric for accurate geographic distance calculations
- For 500 schools: ~5,000 calculations vs ~250,000 in old version

**Crime Statistics Caching**: Added intelligent caching system with robust invalidation
- Caches crime calculations per location with comprehensive validation
- Uses SHA256 file hashing to detect any content changes in crime data
- Invalidates cache when crime data file content changes (not just timestamp)
- Invalidates cache when excluded crime types list changes in config.json
- Invalidates cache when crime radius changes in config.json
- Provides clear feedback about why cache was invalidated
- Significantly reduces processing time for repeated runs

### Code Quality
- Added proper error handling for configuration loading
- Improved modularity with config-based parameter passing
- Better documentation and inline comments explaining optimizations

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
