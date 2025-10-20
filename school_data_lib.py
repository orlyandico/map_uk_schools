"""
Shared library for UK school data processing

Contains common functions used by plot_schools.py and generate_school_data.py
for loading, processing, geocoding, and analyzing school performance data.
"""

import os
import re
import json
import math
import hashlib
import logging

import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm


# Constants
EARTH_RADIUS_KM = 6371.0  # Earth's radius in kilometers for haversine calculations
KM_PER_DEGREE_LATITUDE = 111.32  # Approximate km per degree of latitude


# Module-level caches
geocoding_cache = {}
crime_cache = {}


def update_geocoding_progress(pbar, hits_counter, misses_counter, crime_calc_counter=None):
    """
    Update progress bar with geocoding and crime calculation statistics

    Args:
        pbar: tqdm progress bar instance (or None)
        hits_counter: List containing cache hit count [int]
        misses_counter: List containing cache miss count [int]
        crime_calc_counter: Optional list containing crime calculation count [int]
    """
    if not pbar:
        return

    postfix = {
        "geo_hits": hits_counter[0],
        "geo_miss": misses_counter[0]
    }
    if crime_calc_counter is not None and crime_calc_counter[0] > 0:
        postfix["crime_calc"] = crime_calc_counter[0]

    pbar.set_postfix(postfix)


def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found. Using default values.")
        return get_default_config()
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing config file: {e}. Using default values.")
        return get_default_config()


def get_default_config():
    """Return default configuration if config file is missing"""
    return {
        "clustering": {"default_cluster_radius_km": 5, "default_min_schools": 2},
        "crime": {
            "school_crime_radius_km": 3,
            "crime_data_file": "combined_crimes.csv.gz",
            "excluded_crime_types": [
                "Shoplifting",
                "Bicycle theft",
                "Other theft",
                "Other crime",
                "Drugs",
                "Anti-social behaviour",
                "Criminal damage and arson",
            ],
        },
        "geocoding": {"index_name": "your-place-index-name", "region_name": "eu-west-2"},
        "caching": {
            "geocoding_cache_file": "geocoding_cache.json",
            "crime_cache_file": "crime_cache.json",
        },
        "filtering": {"percentile": 0.90, "min_age_threshold": 7},
        "grading": {"a_star_threshold": 50, "a_threshold": 40},
        "colors": {
            "independent": {"a_star": "#FF0000", "a": "#FFA500", "b_or_below": "#FFD700"},
            "state": {"a_star": "#000080", "a": "#0000FF", "b_or_below": "#4169E1"},
            "cluster": {"circle": "#1E90FF", "center_marker": "#FF4500"},
        },
        "data": {
            "school_data_pattern": "20*ks5final*.csv.gz",
            "required_columns": [
                "SCHNAME",
                "ADDRESS1",
                "TOWN",
                "PCODE",
                "TELNUM",
                "ADMPOL_PT",
                "GEND1618",
                "AGERANGE",
                "TPUP1618",
                "TALLPPE_ALEV_1618",
                "TB3PTSE",
                "YEAR",
            ],
            "numeric_columns": ["TB3PTSE", "TALLPPE_ALEV_1618"],
        },
        "output": {
            "processed_csv": "processed_school_data.csv",
            "map_filename": "schools_map.html",
        },
        "map": {
            "default_zoom": 8,
            "marker_size": 30,
            "cluster_marker_size": 20,
            "popup_max_width": 350,
        },
    }


def validate_config(config):
    """
    Validate configuration values are reasonable and within acceptable ranges

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If any configuration value is invalid

    Returns:
        bool: True if validation passes
    """
    errors = []

    # Clustering validation
    try:
        cluster_radius = config["clustering"]["default_cluster_radius_km"]
        if cluster_radius <= 0:
            errors.append("Cluster radius must be positive (got {})".format(cluster_radius))
    except KeyError as e:
        errors.append(f"Missing clustering config key: {e}")

    try:
        min_schools = config["clustering"]["default_min_schools"]
        if min_schools < 1:
            errors.append("Minimum schools per cluster must be at least 1 (got {})".format(min_schools))
    except KeyError as e:
        errors.append(f"Missing clustering config key: {e}")

    # Crime validation
    try:
        crime_radius = config["crime"]["school_crime_radius_km"]
        if crime_radius <= 0:
            errors.append("Crime radius must be positive (got {})".format(crime_radius))
    except KeyError as e:
        errors.append(f"Missing crime config key: {e}")

    # Filtering validation
    try:
        percentile = config["filtering"]["percentile"]
        if not (0 <= percentile <= 1):
            errors.append("Percentile must be between 0 and 1 (got {})".format(percentile))
    except KeyError as e:
        errors.append(f"Missing filtering config key: {e}")

    try:
        min_age = config["filtering"]["min_age_threshold"]
        if min_age < 0 or min_age > 18:
            errors.append("Min age threshold should be between 0 and 18 (got {})".format(min_age))
    except KeyError as e:
        errors.append(f"Missing filtering config key: {e}")

    # Grading validation
    try:
        a_star_threshold = config["grading"]["a_star_threshold"]
        a_threshold = config["grading"]["a_threshold"]
        if a_star_threshold <= a_threshold:
            errors.append(
                "A* threshold ({}) must be greater than A threshold ({})".format(
                    a_star_threshold, a_threshold
                )
            )
        if a_threshold < 0 or a_star_threshold < 0:
            errors.append("Grade thresholds must be non-negative")
    except KeyError as e:
        errors.append(f"Missing grading config key: {e}")

    # Map validation
    try:
        zoom = config["map"]["default_zoom"]
        if zoom < 1 or zoom > 20:
            errors.append("Map zoom must be between 1 and 20 (got {})".format(zoom))
    except KeyError as e:
        errors.append(f"Missing map config key: {e}")

    # If there are errors, raise with all messages
    if errors:
        error_msg = "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        raise ValueError(error_msg)

    logging.info("✓ Configuration validated successfully")
    return True


def get_file_hash(filepath, chunk_size=8192):
    """
    Calculate SHA256 hash of a file for cache validation.

    Args:
        filepath: Path to file to hash
        chunk_size: Size of chunks to read (default 8KB for memory efficiency)

    Returns:
        Hexadecimal hash string, or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        return None

    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        logging.error(f"Error hashing file {filepath}: {e}")
        return None


def load_crime_cache(config):
    """
    Load crime cache from file if it exists and is valid.

    Cache is considered valid only if ALL of these match:
    - Crime data file hash matches (detects content changes)
    - Excluded crime types match (config parameter)
    - Crime radius matches (config parameter)
    """
    global crime_cache
    crime_cache_file = config["caching"]["crime_cache_file"]
    crime_data_file = config["crime"]["crime_data_file"]

    if not os.path.exists(crime_cache_file) or not os.path.exists(crime_data_file):
        crime_cache = {}
        return

    try:
        with open(crime_cache_file, "r") as f:
            cache_data = json.load(f)

        cache_meta = cache_data.get("_metadata", {})

        # Calculate current file hash and config parameters
        current_hash = get_file_hash(crime_data_file)
        current_excluded = sorted(config["crime"]["excluded_crime_types"])
        current_radius = config["crime"]["school_crime_radius_km"]

        # Validate all cache parameters
        cached_hash = cache_meta.get("crime_file_hash")
        cached_excluded = cache_meta.get("excluded_crimes", [])
        cached_radius = cache_meta.get("school_crime_radius_km")

        # Check if cache is still valid
        if (current_hash and
            cached_hash == current_hash and
            cached_excluded == current_excluded and
            cached_radius == current_radius):
            crime_cache = {k: v for k, v in cache_data.items() if not k.startswith("_")}
            logging.info(f"Loaded {len(crime_cache)} cached crime calculations")
        else:
            # Provide helpful feedback about what changed
            if cached_hash != current_hash:
                logging.info("Crime data file changed, invalidating cache")
            elif cached_excluded != current_excluded:
                logging.info("Crime exclusion filters changed in config, invalidating cache")
            elif cached_radius != current_radius:
                logging.info(f"Crime radius changed ({cached_radius}km → {current_radius}km), invalidating cache")
            crime_cache = {}
    except Exception as e:
        logging.error(f"Error loading crime cache: {e}")
        crime_cache = {}


def save_crime_cache(config):
    """
    Save crime cache to file with comprehensive metadata for validation.
    """
    crime_cache_file = config["caching"]["crime_cache_file"]
    crime_data_file = config["crime"]["crime_data_file"]

    try:
        cache_data = dict(crime_cache)
        if os.path.exists(crime_data_file):
            file_hash = get_file_hash(crime_data_file)
            cache_data["_metadata"] = {
                "crime_file_hash": file_hash,
                "excluded_crimes": sorted(config["crime"]["excluded_crime_types"]),
                "school_crime_radius_km": config["crime"]["school_crime_radius_km"],
                "cached_at": pd.Timestamp.now().isoformat(),
            }

        with open(crime_cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        logging.info(f"Saved {len(crime_cache)} crime calculations to cache")
    except Exception as e:
        logging.error(f"Error saving crime cache: {e}")


def load_geocoding_cache(config):
    """Load geocoding cache from file if it exists"""
    global geocoding_cache
    cache_file = config["caching"]["geocoding_cache_file"]

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                geocoding_cache = json.load(f)
            logging.info(f"Loaded {len(geocoding_cache)} cached geocoded addresses")
        except Exception as e:
            logging.error(f"Error loading geocoding cache: {e}")
            geocoding_cache = {}
    else:
        geocoding_cache = {}


def save_geocoding_cache(config):
    """Save geocoding cache to file"""
    cache_file = config["caching"]["geocoding_cache_file"]

    try:
        with open(cache_file, "w") as f:
            json.dump(geocoding_cache, f, indent=2)
        logging.info(f"Saved {len(geocoding_cache)} unique addresses to cache")
    except Exception as e:
        logging.error(f"Error saving geocoding cache: {e}")


def point_score_to_grade(point_score, config):
    """Convert point score to grade"""
    a_star_threshold = config["grading"]["a_star_threshold"]
    a_threshold = config["grading"]["a_threshold"]

    if pd.isna(point_score) or point_score == 0:
        return "N/A"
    elif point_score >= a_star_threshold:
        return "A*"
    elif point_score >= a_threshold:
        return "A"
    else:
        return "≤B"


def geocode_address(
    address,
    config,
    pbar=None,
    hits_counter=None,
    misses_counter=None,
    crime_df=None,
    crime_calc_counter=None,
):
    """Geocode address using Amazon Location Services with caching"""
    global geocoding_cache, crime_cache

    index_name = config["geocoding"]["index_name"]
    region_name = config["geocoding"]["region_name"]
    school_crime_radius_km = config["crime"]["school_crime_radius_km"]

    if not address or pd.isna(address):
        if pbar:
            pbar.update(1)
        return pd.Series({"Latitude": None, "Longitude": None, "crime_stats": None})

    # Check cache first
    if address in geocoding_cache:
        lat, lon = geocoding_cache[address]

        # Check if we have crime data cached for this location
        crime_cache_key = (
            f"{lat:.6f},{lon:.6f},{school_crime_radius_km}" if lat else None
        )
        crime_stats = crime_cache.get(crime_cache_key) if crime_cache_key else None

        # If no crime stats cached but we have coordinates and crime_df, calculate now
        if lat and crime_df is not None and not crime_stats:
            crime_stats = get_crime_stats_for_location(
                lat, lon, school_crime_radius_km, crime_df, config
            )
            if crime_calc_counter is not None:
                crime_calc_counter[0] += 1

        if pbar:
            update_geocoding_progress(pbar, hits_counter, misses_counter, crime_calc_counter)
            hits_counter[0] += 1
            pbar.update(1)
        return pd.Series(
            {"Latitude": lat, "Longitude": lon, "crime_stats": crime_stats}
        )

    try:
        if pbar:
            update_geocoding_progress(pbar, hits_counter, misses_counter, crime_calc_counter)
            misses_counter[0] += 1

        location_client = boto3.client("location", region_name=region_name)
        response = location_client.search_place_index_for_text(
            IndexName=index_name, Text=address, MaxResults=1
        )

        if response["Results"]:
            coordinates = response["Results"][0]["Place"]["Geometry"]["Point"]
            lat, lon = coordinates[1], coordinates[0]

            # Calculate crime stats if crime_df is available
            crime_stats = None
            if crime_df is not None:
                crime_stats = get_crime_stats_for_location(
                    lat, lon, school_crime_radius_km, crime_df, config
                )
                if crime_calc_counter is not None:
                    crime_calc_counter[0] += 1

            # Store in cache
            geocoding_cache[address] = (lat, lon)
            if pbar:
                pbar.update(1)
            return pd.Series(
                {"Latitude": lat, "Longitude": lon, "crime_stats": crime_stats}
            )

        # Cache negative results
        geocoding_cache[address] = (None, None)
        if pbar:
            pbar.update(1)
        return pd.Series({"Latitude": None, "Longitude": None, "crime_stats": None})

    except ClientError as e:
        logging.error(
            f"AWS Error geocoding {address}: {e.response['Error']['Code']} - {e.response['Error']['Message']}"
        )
        if pbar:
            pbar.update(1)
        return pd.Series({"Latitude": None, "Longitude": None, "crime_stats": None})
    except Exception as e:
        logging.error(f"Error geocoding {address}: {e}")
        if pbar:
            pbar.update(1)
        return pd.Series({"Latitude": None, "Longitude": None, "crime_stats": None})


def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    """
    Vectorized haversine distance calculation for multiple points
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2_array)
    lon2_rad = np.radians(lon2_array)

    # Calculate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return c * EARTH_RADIUS_KM


def calculate_circle_bounding_box(center_lat, center_lon, radius_km):
    """Calculate the bounding box that contains a circle with given center and radius"""
    lat_change = radius_km / KM_PER_DEGREE_LATITUDE
    lon_change = radius_km / (KM_PER_DEGREE_LATITUDE * math.cos(math.radians(center_lat)))

    return (
        center_lat - lat_change,  # min_lat
        center_lat + lat_change,  # max_lat
        center_lon - lon_change,  # min_lon
        center_lon + lon_change,  # max_lon
    )


def get_crime_stats_for_location(center_lat, center_lon, radius_km, crime_df, config):
    """Get crime statistics for a location within specified radius"""
    global crime_cache

    # Create cache key
    cache_key = f"{center_lat:.6f},{center_lon:.6f},{radius_km}"

    # Check cache first
    if cache_key in crime_cache:
        return crime_cache[cache_key]

    crime_data_file = config["crime"]["crime_data_file"]

    if crime_df is None:
        try:
            crime_df = pd.read_csv(crime_data_file)
        except FileNotFoundError:
            result = {
                "total_crimes": 0,
                "crime_types": {},
                "error": "No crime data available",
            }
            crime_cache[cache_key] = result
            return result

    # Filter out low-impact crimes
    excluded_crimes = set(config["crime"]["excluded_crime_types"])
    crime_df_filtered = crime_df[~crime_df["Crime type"].isin(excluded_crimes)]

    # Use bounding box for initial filtering
    min_lat, max_lat, min_lon, max_lon = calculate_circle_bounding_box(
        center_lat, center_lon, radius_km
    )

    bbox_filtered = crime_df_filtered[
        (crime_df_filtered["Latitude"].between(min_lat, max_lat))
        & (crime_df_filtered["Longitude"].between(min_lon, max_lon))
    ]

    if len(bbox_filtered) == 0:
        result = {"total_crimes": 0, "crime_types": {}, "radius_km": radius_km}
        crime_cache[cache_key] = result
        return result

    # Calculate actual distances for remaining points (vectorized for performance)
    bbox_filtered = bbox_filtered.copy()
    bbox_filtered["distance"] = haversine_vectorized(
        center_lat,
        center_lon,
        bbox_filtered["Latitude"].values,
        bbox_filtered["Longitude"].values
    )

    # Final filter by actual radius
    final_filtered = bbox_filtered[bbox_filtered["distance"] <= radius_km]

    total_crimes = len(final_filtered)
    crime_types = (
        final_filtered["Crime type"].value_counts().to_dict()
        if total_crimes > 0
        else {}
    )

    result = {
        "total_crimes": total_crimes,
        "crime_types": crime_types,
        "radius_km": radius_km,
    }

    # Cache the result
    crime_cache[cache_key] = result
    return result


def filter_by_age_range(df, min_age_threshold=7):
    """Filter schools based on minimum age threshold"""

    def extract_min_age(age_range):
        if pd.isna(age_range):
            return None
        numbers = re.findall(r"\d+", str(age_range).strip())
        return int(numbers[0]) if numbers else None

    min_ages = df["AGERANGE"].apply(extract_min_age)
    valid_mask = min_ages.notna() & (min_ages <= min_age_threshold)
    return df[valid_mask]


def load_crime_data(crime_data_file):
    """
    Load crime data from file

    Args:
        crime_data_file: Path to crime data CSV file

    Returns:
        pd.DataFrame or None: Crime data, or None if file not found
    """
    logging.info("Loading crime data...")
    try:
        crime_df = pd.read_csv(crime_data_file)
        logging.info(f"Loaded {len(crime_df)} crime records")
        return crime_df
    except FileNotFoundError:
        logging.warning(
            f"Crime data file {crime_data_file} not found. Crime statistics will not be available."
        )
        return None


def load_school_data_files(school_data_pattern):
    """
    Find and load school data files matching pattern

    Args:
        school_data_pattern: Glob pattern for school data files

    Returns:
        pd.DataFrame: Combined school data from all years, or None if no files found
    """
    import glob

    csv_files = glob.glob(school_data_pattern)
    if not csv_files:
        logging.error(
            f"No school data files found matching pattern '{school_data_pattern}'"
        )
        return None

    logging.info(f"Found {len(csv_files)} data files: {csv_files}")

    # Load and combine all years
    all_dfs = []
    for csv_file in sorted(csv_files):
        # Extract year from filename with validation
        try:
            filename = os.path.basename(csv_file)
            year_candidate = filename.split("_")[0]

            # Validate that it looks like a year (4 digits)
            if year_candidate.isdigit() and len(year_candidate) == 4:
                year = year_candidate
            else:
                # Fallback: try to find a 4-digit year anywhere in the filename
                year_match = re.search(r'\b(20\d{2})\b', filename)
                if year_match:
                    year = year_match.group(1)
                else:
                    # Last resort: use filename without extension
                    year = os.path.splitext(filename)[0]
                    logging.warning(f"Could not extract year from {filename}, using '{year}'")
        except Exception as e:
            logging.error(f"Error extracting year from {csv_file}: {e}, using 'unknown'")
            year = "unknown"

        try:
            df_year = pd.read_csv(csv_file, low_memory=False, dtype={"TELNUM": str})
            df_year["YEAR"] = year
            all_dfs.append(df_year)
            logging.info(f"Loaded {len(df_year)} rows from {csv_file} (year: {year})")
        except Exception as e:
            logging.error(f"Error loading {csv_file}: {e}")
            continue

    if not all_dfs:
        logging.error("No data files could be loaded")
        return None

    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Combined total rows: {len(df)}")
    return df


def process_and_consolidate_schools(df, config):
    """
    Process school data: select columns, consolidate by school, calculate averages

    Args:
        df: Raw combined school dataframe
        config: Configuration dictionary

    Returns:
        pd.DataFrame: Processed and consolidated school data, or None on error
    """
    # Select and process columns
    columns_to_keep = config["data"]["required_columns"]

    try:
        df_selected = df[columns_to_keep].copy()
    except KeyError as e:
        logging.error(f"Missing columns in data file: {e}")
        return None

    # Convert numeric columns
    numeric_columns = config["data"]["numeric_columns"]
    for col in numeric_columns:
        df_selected[col] = pd.to_numeric(df_selected[col], errors="coerce").fillna(0)

    # Calculate average scores per school across all years (group by address+postcode)
    df_selected["school_key"] = (
        df_selected["ADDRESS1"].astype(str) + "|" + df_selected["PCODE"].astype(str)
    )
    school_groups = df_selected.groupby("school_key")

    # Create consolidated dataframe with averages and year-specific data
    consolidated_data = []
    for school_key, group_df in school_groups:
        # Calculate averages (only for non-zero scores)
        valid_scores = group_df[group_df["TB3PTSE"] > 0]
        if len(valid_scores) == 0:
            continue

        avg_tb3ptse = valid_scores["TB3PTSE"].mean()
        avg_tallppe = valid_scores["TALLPPE_ALEV_1618"].mean()

        # Get the most recent row for other fields
        latest_row = group_df.iloc[-1].copy()
        latest_row["TB3PTSE"] = avg_tb3ptse
        latest_row["TALLPPE_ALEV_1618"] = avg_tallppe

        # Create year-specific scores string for popup
        year_scores = []
        for _, year_row in valid_scores.iterrows():
            year_scores.append(f"{year_row['YEAR']}: {year_row['TB3PTSE']:.2f}")
        latest_row["YEAR_SCORES"] = "<br>".join(year_scores)

        consolidated_data.append(latest_row)

    df_selected = pd.DataFrame(consolidated_data)
    logging.info(f"Consolidated to {len(df_selected)} unique schools")

    return df_selected


def filter_schools_by_percentile(df_selected, percentile, config):
    """
    Filter schools by performance percentile

    Args:
        df_selected: Consolidated school dataframe
        percentile: Percentile threshold (0-1)
        config: Configuration dictionary

    Returns:
        pd.DataFrame: Filtered schools
    """
    # Calculate and display score distribution by deciles
    logging.info("\n" + "="*60)
    logging.info("TB3PTSE Score Distribution by Decile")
    logging.info("="*60)

    deciles = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    a_star_threshold = config["grading"]["a_star_threshold"]
    a_threshold = config["grading"]["a_threshold"]

    logging.info(f"{'Percentile':<12} {'Score':<10} {'Grade':<8} {'Schools':<10}")
    logging.info("-" * 60)

    for i, p in enumerate(deciles):
        score = df_selected["TB3PTSE"].quantile(p)
        grade = point_score_to_grade(score, config)

        # Count schools at or above this percentile
        schools_count = len(df_selected[df_selected["TB3PTSE"] >= score])

        percentile_label = f"P{int(p*100)}" if p > 0 else "Min"
        if p == 1.0:
            percentile_label = "Max"

        logging.info(f"{percentile_label:<12} {score:>7.2f}    {grade:<8} {schools_count:>6}")

    logging.info("-" * 60)
    logging.info(f"Marker color thresholds (for visualization): A* ≥{a_star_threshold}, A ≥{a_threshold}, ≤B <{a_threshold}")
    logging.info(f"Note: Scores reflect average points across best 3 A-levels (A*=60, A=50, B=40)")
    logging.info("="*60 + "\n")

    # Now filter based on specified percentile
    df_filtered = df_selected[
        df_selected["TB3PTSE"] >= df_selected["TB3PTSE"].quantile(percentile)
    ]
    logging.info(f"Filtering at P{percentile*100:.0f} (score ≥{df_selected['TB3PTSE'].quantile(percentile):.2f}): {len(df_filtered)} schools selected")
    return df_filtered


def geocode_and_enrich_schools(df_selected, crime_df, config):
    """
    Geocode school addresses and calculate crime statistics

    Args:
        df_selected: Filtered school dataframe
        crime_df: Crime data dataframe (or None)
        config: Configuration dictionary

    Returns:
        pd.DataFrame: Schools with coordinates and crime stats
    """
    # Create full address for geocoding, handling NaN values
    def build_address(row):
        """Build address string from row, filtering out NaN/empty values"""
        parts = []
        for field in ['ADDRESS1', 'TOWN', 'PCODE']:
            value = row[field]
            if pd.notna(value) and str(value).strip() and str(value).lower() != 'nan':
                parts.append(str(value))
        return ", ".join(parts) if parts else None

    df_selected["full_address"] = df_selected.apply(build_address, axis=1)

    # Geocode addresses with progress tracking
    total_addresses = len(df_selected)
    cache_hits = [0]
    cache_misses = [0]
    crime_calculations = [0]

    logging.info(f"Processing {total_addresses} addresses (geocoding and crime stats)...")
    with tqdm(total=total_addresses, desc="Processing", unit="address") as pbar:
        geocoded_results = df_selected["full_address"].apply(
            lambda addr: geocode_address(
                addr,
                config,
                pbar=pbar,
                hits_counter=cache_hits,
                misses_counter=cache_misses,
                crime_df=crime_df,
                crime_calc_counter=crime_calculations,
            )
        )

    df_selected["Latitude"] = geocoded_results["Latitude"]
    df_selected["Longitude"] = geocoded_results["Longitude"]
    df_selected["crime_stats"] = geocoded_results["crime_stats"]

    logging.info(
        f"Processing complete - Geocoding: {cache_hits[0]} cached, {cache_misses[0]} new | "
        f"Crime stats: {crime_calculations[0]} calculated"
    )

    # Save caches
    save_geocoding_cache(config)
    save_crime_cache(config)

    # Remove schools without coordinates
    df_selected = df_selected.dropna(subset=["Latitude", "Longitude"])
    logging.info(f"Schools with valid coordinates: {len(df_selected)}")

    return df_selected


def extract_and_index_crime_data(df_selected, school_crime_radius_km):
    """
    Extract crime statistics and calculate crime indices

    Args:
        df_selected: Schools with crime_stats column
        school_crime_radius_km: Radius for school crime calculations

    Returns:
        tuple: (df_selected with crime_count and crime_index, school_crime_stats list)
    """
    logging.info("Extracting crime statistics from geocoding cache...")

    def extract_crime_count(crime_stats):
        """Extract total crime count from stats dict"""
        if crime_stats:
            return crime_stats["total_crimes"]
        return 0

    # Vectorized extraction of crime counts
    df_selected["crime_count"] = df_selected["crime_stats"].apply(extract_crime_count)

    # Build school_crime_stats list with fallback for missing data
    school_crime_stats = []
    for crime_stat in df_selected["crime_stats"]:
        if crime_stat:
            school_crime_stats.append(crime_stat)
        else:
            # Fallback for missing crime data
            school_crime_stats.append({
                "total_crimes": 0,
                "crime_types": {},
                "radius_km": school_crime_radius_km,
            })

    # Save crime cache after processing
    # Note: This is handled by the caller now

    # Calculate crime indices using percentile ranking
    crime_counts = df_selected["crime_count"].copy()
    df_selected["crime_index"] = crime_counts.rank(pct=True)

    return df_selected, school_crime_stats
