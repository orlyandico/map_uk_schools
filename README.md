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

This adds local crime statistics to each school's popup showing serious crimes within a 3km radius (the count of crimes and decile are included, across all crimes in the police data).

## Requirements

```bash
pip install pandas boto3 tqdm folium geopy
```

AWS credentials must be configured (via `aws configure`) for Amazon Location Services geocoding. You'll need to create a Place Index in Amazon Location Service and update the `index_name` parameter in the geocoding functions (default: "your-place-index-name").

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
- Local crime statistics within 3km radius

### consolidate_crime_data.py
Utility script to merge multiple crime CSV files from https://data.police.uk/data/ into a single `combined_crimes.csv` file. Edit the script to set your crime data directory path.

## Configuration

- **Percentile filtering**: Default 90th percentile (modify `PERCENTILE` constant)
- **Clustering**: Default 5km radius, minimum 2 schools (via command line arguments)
- **Crime analysis**: 3km radius per school (modify `SCHOOL_CRIME_RADIUS_KM` constant)
- **Geocoding**: Uses EU-West-2 region by default
- **Crime file**: Expects `combined_crimes.csv.gz` (modify `CRIME_DATA_FILE` constant)

## Output Files

- `schools_map.html` - Interactive map with clustered schools
- `processed_school_data.csv` - Filtered and geocoded school data
- `geocoding_cache.json` - Cached geocoding results to avoid re-processing
