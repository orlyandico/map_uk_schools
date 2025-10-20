import os
import re
import json
import math
import hashlib
from collections import defaultdict

import pandas as pd
import numpy as np
import folium
from folium.features import DivIcon
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
from geopy.distance import geodesic
from sklearn.neighbors import BallTree


# Load configuration
def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using default values.")
        return get_default_config()
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}. Using default values.")
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


# Load configuration at module level
config = load_config()

# Geocoding cache
geocoding_cache = {}

# Crime cache
crime_cache = {}


def get_file_hash(filepath, chunk_size=8192):
    """
    Calculate SHA256 hash of a file for cache validation.

    This is more reliable than file size + mtime because:
    - Detects any content changes, even if file size stays the same
    - Not affected by file system operations that preserve timestamps
    - Produces consistent results across different systems

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
        print(f"Error hashing file {filepath}: {e}")
        return None


def load_crime_cache():
    """
    Load crime cache from file if it exists and is valid.

    Cache is considered valid only if ALL of these match:
    - Crime data file hash matches (detects content changes)
    - Excluded crime types match (config parameter)
    - Crime radius matches (config parameter)

    This ensures cache is invalidated when data OR configuration changes.
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
            print(f"Loaded {len(crime_cache)} cached crime calculations")
        else:
            # Provide helpful feedback about what changed
            if cached_hash != current_hash:
                print("Crime data file changed, invalidating cache")
            elif cached_excluded != current_excluded:
                print("Crime exclusion filters changed in config, invalidating cache")
            elif cached_radius != current_radius:
                print(f"Crime radius changed ({cached_radius}km → {current_radius}km), invalidating cache")
            crime_cache = {}
    except Exception as e:
        print(f"Error loading crime cache: {e}")
        crime_cache = {}


def save_crime_cache():
    """
    Save crime cache to file with comprehensive metadata for validation.

    Metadata includes:
    - File hash: Detects any content changes in crime data
    - Excluded crimes: Config parameter that affects calculations
    - Radius: Config parameter that affects calculations
    - Timestamp: When cache was created
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
        print(f"Saved {len(crime_cache)} crime calculations to cache")
    except Exception as e:
        print(f"Error saving crime cache: {e}")


def load_geocoding_cache():
    """Load geocoding cache from file if it exists"""
    global geocoding_cache
    cache_file = config["caching"]["geocoding_cache_file"]

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                geocoding_cache = json.load(f)
            print(f"Loaded {len(geocoding_cache)} cached geocoded addresses")
        except Exception as e:
            print(f"Error loading geocoding cache: {e}")
            geocoding_cache = {}
    else:
        geocoding_cache = {}


def save_geocoding_cache():
    """Save geocoding cache to file"""
    cache_file = config["caching"]["geocoding_cache_file"]

    try:
        with open(cache_file, "w") as f:
            json.dump(geocoding_cache, f, indent=2)
        print(f"Saved {len(geocoding_cache)} unique addresses to cache")
    except Exception as e:
        print(f"Error saving geocoding cache: {e}")


def point_score_to_grade(point_score):
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


def reverse_geocode_location(lat, lon, index_name=None, region_name=None):
    """Reverse geocode coordinates to get postcode using Amazon Location Services"""
    if index_name is None:
        index_name = config["geocoding"]["index_name"]
    if region_name is None:
        region_name = config["geocoding"]["region_name"]

    try:
        location_client = boto3.client("location", region_name=region_name)
        response = location_client.search_place_index_for_position(
            IndexName=index_name,
            Position=[lon, lat],  # Amazon Location uses [lon, lat] order
            MaxResults=1,
        )

        if response["Results"]:
            place = response["Results"][0]["Place"]
            if "PostalCode" in place:
                return place["PostalCode"]
            elif "Label" in place:
                # Extract postcode from label using regex
                postcode_match = re.search(
                    r"[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}", place["Label"]
                )
                if postcode_match:
                    return postcode_match.group()

        return None

    except Exception as e:
        print(f"Error reverse geocoding ({lat:.4f}, {lon:.4f}): {e}")
        return None


def geocode_address(
    address,
    index_name=None,
    region_name=None,
    pbar=None,
    hits_counter=None,
    misses_counter=None,
    crime_df=None,
    crime_calc_counter=None,
):
    """Geocode address using Amazon Location Services with caching"""
    if index_name is None:
        index_name = config["geocoding"]["index_name"]
    if region_name is None:
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
                lat, lon, school_crime_radius_km, crime_df
            )
            if crime_calc_counter is not None:
                crime_calc_counter[0] += 1

        if pbar:
            postfix = {
                "geo_hits": hits_counter[0],
                "geo_miss": misses_counter[0]
            }
            if crime_calc_counter is not None and crime_calc_counter[0] > 0:
                postfix["crime_calc"] = crime_calc_counter[0]
            pbar.set_postfix(postfix)
            hits_counter[0] += 1
            pbar.update(1)
        return pd.Series(
            {"Latitude": lat, "Longitude": lon, "crime_stats": crime_stats}
        )

    try:
        if pbar:
            postfix = {
                "geo_hits": hits_counter[0],
                "geo_miss": misses_counter[0]
            }
            if crime_calc_counter is not None and crime_calc_counter[0] > 0:
                postfix["crime_calc"] = crime_calc_counter[0]
            pbar.set_postfix(postfix)
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
                    lat, lon, school_crime_radius_km, crime_df
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
        print(
            f"AWS Error geocoding {address}: {e.response['Error']['Code']} - {e.response['Error']['Message']}"
        )
        if pbar:
            pbar.update(1)
        return pd.Series({"Latitude": None, "Longitude": None, "crime_stats": None})
    except Exception as e:
        print(f"Error geocoding {address}: {e}")
        if pbar:
            pbar.update(1)
        return pd.Series({"Latitude": None, "Longitude": None, "crime_stats": None})


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in kilometers"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371  # Earth's radius in km


def calculate_circle_bounding_box(center_lat, center_lon, radius_km):
    """Calculate the bounding box that contains a circle with given center and radius"""
    lat_change = radius_km / 111.32  # km per degree latitude
    lon_change = radius_km / (111.32 * math.cos(math.radians(center_lat)))

    return (
        center_lat - lat_change,  # min_lat
        center_lat + lat_change,  # max_lat
        center_lon - lon_change,  # min_lon
        center_lon + lon_change,  # max_lon
    )


def get_crime_stats_for_location(center_lat, center_lon, radius_km, crime_df=None):
    """Get crime statistics for a location within specified radius"""
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

    # Calculate actual distances for remaining points
    bbox_filtered = bbox_filtered.copy()
    bbox_filtered["distance"] = bbox_filtered.apply(
        lambda row: haversine_distance(
            center_lat, center_lon, row["Latitude"], row["Longitude"]
        ),
        axis=1,
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


def format_crime_stats(crime_stats, crime_index=None):
    """Format crime statistics for display in popup"""
    if crime_stats.get("error"):
        return crime_stats["error"]

    total = crime_stats["total_crimes"]
    radius = crime_stats["radius_km"]

    if total == 0:
        crime_index_text = " (Crime Index: 0.00)" if crime_index is not None else ""
        return f"No serious crimes in {radius}km radius{crime_index_text}"

    # Get top 3 crime types
    top_crimes = sorted(
        crime_stats["crime_types"].items(), key=lambda x: x[1], reverse=True
    )[:3]

    crime_index_text = (
        f" (Crime Index: {crime_index:.2f})" if crime_index is not None else ""
    )
    lines = [f"<b>{total} serious crimes</b> in {radius}km radius{crime_index_text}:"]
    lines.extend([f"• {crime_type}: {count}" for crime_type, count in top_crimes])

    return "<br>".join(lines)


# This function is no longer needed as we calculate crime indices directly in the dataframe


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


def get_grade_color(score, is_independent):
    """Get color for school marker based on score and type"""
    a_star_threshold = config["grading"]["a_star_threshold"]
    a_threshold = config["grading"]["a_threshold"]
    colors = config["colors"]

    if is_independent:
        if score >= a_star_threshold:
            return colors["independent"]["a_star"]
        elif score >= a_threshold:
            return colors["independent"]["a"]
        else:
            return colors["independent"]["b_or_below"]
    else:
        if score >= a_star_threshold:
            return colors["state"]["a_star"]
        elif score >= a_threshold:
            return colors["state"]["a"]
        else:
            return colors["state"]["b_or_below"]


def get_text_color(background_color):
    """Determine appropriate text color based on background color brightness"""
    bg_color = background_color.lstrip("#")
    rgb = tuple(int(bg_color[i : i + 2], 16) for i in (0, 2, 4))
    brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
    return "#000000" if brightness > 128 else "#FFFFFF"


def cluster_schools(df, max_distance_km, min_schools=2):
    """
    Optimized school clustering algorithm using BallTree for spatial indexing

    This is significantly faster than the O(n²) nested loop approach.
    BallTree uses haversine distance metric for accurate geographic calculations.

    Returns:
        tuple: (labels array, cluster_centers_data list)
    """
    if len(df) == 0:
        return np.array([]), []

    print(f"Evaluating {len(df)} schools as potential cluster centers...")

    # Reset index to ensure consistent positioning
    df_reset = df.reset_index(drop=True)

    # Prepare coordinates for BallTree (needs radians for haversine metric)
    coords = np.radians(df_reset[["Latitude", "Longitude"]].values)

    # Build BallTree with haversine metric for geographic distances
    # Haversine accounts for Earth's curvature
    tree = BallTree(coords, metric='haversine')

    # Convert km to radians for BallTree query (Earth radius = 6371 km)
    radius_radians = max_distance_km / 6371.0

    # Find all schools within radius for each potential center
    # This is MUCH faster than nested loops: O(n log n) vs O(n²)
    indices_list = tree.query_radius(coords, r=radius_radians)

    # Find potential cluster centers
    cluster_centers = []
    for center_idx, nearby_indices in enumerate(indices_list):
        nearby_schools = nearby_indices.tolist()

        # Check if this forms a valid cluster
        if len(nearby_schools) >= min_schools:
            center_row = df_reset.iloc[center_idx]
            cluster_centers.append(
                {
                    "lat": center_row["Latitude"],
                    "lon": center_row["Longitude"],
                    "schools": nearby_schools,
                    "count": len(nearby_schools),
                    "center_school_idx": center_idx,
                }
            )

    print(f"Found {len(cluster_centers)} potential cluster centers")

    if not cluster_centers:
        return np.full(len(df_reset), -1), []

    # Remove redundant clusters (keep those with most unique schools)
    final_centers = []
    used_schools = set()

    # Sort by school count (descending)
    cluster_centers.sort(key=lambda x: x["count"], reverse=True)

    for center in cluster_centers:
        new_schools = set(center["schools"]) - used_schools
        if len(new_schools) >= min_schools:
            final_centers.append(center)
            used_schools.update(center["schools"])

    print(f"After removing redundancy: {len(final_centers)} cluster centers")

    # Assign schools to clusters
    labels = np.full(len(df_reset), -1)

    for cluster_id, center in enumerate(final_centers):
        for school_idx in center["schools"]:
            labels[school_idx] = cluster_id

    # Calculate geographic centroids for final cluster centers
    print("Calculating geographic centroids for each cluster...")
    refined_centers = []

    for cluster_id, center in enumerate(final_centers):
        cluster_schools = [
            pos for pos, label in enumerate(labels) if label == cluster_id
        ]

        if not cluster_schools:
            continue

        # Calculate centroid using Cartesian coordinates for accuracy
        x_coords, y_coords, z_coords = [], [], []

        for pos in cluster_schools:
            lat_rad = math.radians(df_reset.iloc[pos]["Latitude"])
            lon_rad = math.radians(df_reset.iloc[pos]["Longitude"])

            x = math.cos(lat_rad) * math.cos(lon_rad)
            y = math.cos(lat_rad) * math.sin(lon_rad)
            z = math.sin(lat_rad)

            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

        # Calculate mean and convert back to lat/lon
        mean_x = sum(x_coords) / len(x_coords)
        mean_y = sum(y_coords) / len(y_coords)
        mean_z = sum(z_coords) / len(z_coords)

        centroid_lon = math.atan2(mean_y, mean_x)
        centroid_lat = math.atan2(mean_z, math.sqrt(mean_x**2 + mean_y**2))

        centroid_lat_deg = math.degrees(centroid_lat)
        centroid_lon_deg = math.degrees(centroid_lon)

        # Get postcode for cluster center
        postcode = reverse_geocode_location(centroid_lat_deg, centroid_lon_deg)

        refined_centers.append(
            {
                "lat": centroid_lat_deg,
                "lon": centroid_lon_deg,
                "schools": cluster_schools,
                "count": len(cluster_schools),
                "cluster_id": cluster_id,
                "postcode": postcode,
            }
        )

    clustered_count = sum(1 for label in labels if label != -1)
    print(
        f"Clustered {clustered_count} of {len(df_reset)} schools into {len(refined_centers)} clusters"
    )

    return labels, refined_centers


def create_school_marker(
    row, label=-1, school_crime_stats=None, school_crime_index=None
):
    """Create a folium marker for a school"""
    is_independent = pd.isna(row["ADMPOL_PT"]) or row["ADMPOL_PT"].strip() == ""
    colour = get_grade_color(row["TB3PTSE"], is_independent)
    fontcolour = get_text_color(colour)
    admpol = "independent" if is_independent else row["ADMPOL_PT"]

    # Create address string
    address_parts = [
        str(row[field])
        for field in ["ADDRESS1", "TOWN", "PCODE"]
        if pd.notna(row[field]) and str(row[field]).strip()
    ]
    address_string = ", ".join(address_parts)

    # Create popup HTML
    popup_html = f"""
    <b>{row["SCHNAME"]}</b><br>
    {address_string}<br>
    Phone: {row["TELNUM"]}<br>
    Admissions Policy: {admpol}<br>
    Age Range: {row["AGERANGE"]}<br>
    Gender: {row["GEND1618"]}<br>
    Students (16-18): {row["TPUP1618"]}<br>
    <b>Average across all years:</b><br>
    Avg best 3 A-levels (TB3PTSE): {row["TB3PTSE"]:.2f} (<b>{point_score_to_grade(row["TB3PTSE"])}</b>)<br>
    Avg per A-level (TALLPPE_ALEV_1618): {row["TALLPPE_ALEV_1618"]:.2f} (<b>{point_score_to_grade(row["TALLPPE_ALEV_1618"])}</b>)<br>
    <b>Year-by-year TB3PTSE scores:</b><br>
    {row.get("YEAR_SCORES", "No year data available")}
    """
    if label != -1:
        popup_html += f"<br>Cluster: {label}"

    # Add school-specific crime information
    if school_crime_stats:
        popup_html += "<br><br><b>Local Area Safety (School-specific):</b><br>"
        popup_html += format_crime_stats(school_crime_stats, school_crime_index)

    # Create marker icon
    marker_size = config["map"]["marker_size"]
    popup_max_width = config["map"]["popup_max_width"]

    icon = DivIcon(
        icon_size=(marker_size, marker_size),
        icon_anchor=(marker_size // 2, marker_size),
        html=f'''
            <div style="width: {marker_size}px; height: {marker_size}px;">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 0C7.58 0 4 3.58 4 8c0 5.76 8 16 8 16s8-10.24 8-16c0-4.42-3.58-8-8-8z"
                        fill="{colour}"
                        stroke="black"
                        stroke-width="1"/>
                    <text x="12" y="9" font-family="Arial" font-size="8" font-weight="bold"
                          fill="{fontcolour}" text-anchor="middle" dy=".3em">{round(row["TB3PTSE"])}</text>
                </svg>
            </div>
        ''',
    )

    return folium.Marker(
        [row["Latitude"], row["Longitude"]],
        popup=folium.Popup(popup_html, max_width=popup_max_width),
        icon=icon,
    )


def create_legend():
    """Create HTML legend for the map"""
    colors = config["colors"]
    a_star_threshold = config["grading"]["a_star_threshold"]
    a_threshold = config["grading"]["a_threshold"]

    return f"""
    <div style="position: fixed;
                bottom: 10px; right: 10px;
                width: 140px;
                border:2px solid grey; z-index:9999;
                font-size:11px;
                background-color:white;
                padding: 8px;
                border-radius: 5px;
                max-height: 50vh;
                overflow-y: auto;
                ">
        <div style="font-weight: bold; margin-bottom: 8px; font-size:12px;">Legend</div>
        <div style="font-weight: bold; margin-bottom: 6px; font-size:10px;">Independent</div>
        <div style="margin-bottom: 3px;">
            <span style="display:inline-block; width: 10px; height: 10px; background-color: {colors["independent"]["a_star"]}; margin-right: 4px;"></span>A* (≥{a_star_threshold})
        </div>
        <div style="margin-bottom: 3px;">
            <span style="display:inline-block; width: 10px; height: 10px; background-color: {colors["independent"]["a"]}; margin-right: 4px;"></span>A ({a_threshold}-{a_star_threshold-1})
        </div>
        <div style="margin-bottom: 8px;">
            <span style="display:inline-block; width: 10px; height: 10px; background-color: {colors["independent"]["b_or_below"]}; margin-right: 4px;"></span>≤B (<{a_threshold})
        </div>
        <div style="font-weight: bold; margin-bottom: 6px; font-size:10px;">State</div>
        <div style="margin-bottom: 3px;">
            <span style="display:inline-block; width: 10px; height: 10px; background-color: {colors["state"]["a_star"]}; margin-right: 4px;"></span>A* (≥{a_star_threshold})
        </div>
        <div style="margin-bottom: 3px;">
            <span style="display:inline-block; width: 10px; height: 10px; background-color: {colors["state"]["a"]}; margin-right: 4px;"></span>A ({a_threshold}-{a_star_threshold-1})
        </div>
        <div>
            <span style="display:inline-block; width: 10px; height: 10px; background-color: {colors["state"]["b_or_below"]}; margin-right: 4px;"></span>≤B (<{a_threshold})
        </div>
    </div>
    """


def main(cluster_radius_km=None, min_schools=None):
    """Main function to process schools and create map"""
    # Use config defaults if not specified
    if cluster_radius_km is None:
        cluster_radius_km = config["clustering"]["default_cluster_radius_km"]
    if min_schools is None:
        min_schools = config["clustering"]["default_min_schools"]

    crime_data_file = config["crime"]["crime_data_file"]
    school_crime_radius_km = config["crime"]["school_crime_radius_km"]
    percentile = config["filtering"]["percentile"]
    school_data_pattern = config["data"]["school_data_pattern"]
    output_csv = config["output"]["processed_csv"]
    map_filename = config["output"]["map_filename"]

    # Load geocoding cache
    load_geocoding_cache()

    # Load crime cache
    load_crime_cache()

    # Load crime data
    print("Loading crime data...")
    try:
        crime_df = pd.read_csv(crime_data_file)
        print(f"Loaded {len(crime_df)} crime records")
    except FileNotFoundError:
        print(
            f"Warning: Crime data file {crime_data_file} not found. Crime statistics will not be available."
        )
        crime_df = None

    # Find and load all school data files
    import glob

    csv_files = glob.glob(school_data_pattern)
    if not csv_files:
        print(
            f"Error: No school data files found matching pattern '{school_data_pattern}'"
        )
        return

    print(f"Found {len(csv_files)} data files: {csv_files}")

    # Load and combine all years
    all_dfs = []
    for csv_file in sorted(csv_files):
        year = csv_file.split("_")[0]  # Extract year from filename
        try:
            df_year = pd.read_csv(csv_file, low_memory=False, dtype={"TELNUM": str})
            df_year["YEAR"] = year
            all_dfs.append(df_year)
            print(f"Loaded {len(df_year)} rows from {csv_file}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

    if not all_dfs:
        print("Error: No data files could be loaded")
        return

    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined total rows: {len(df)}")

    # Select and process columns
    columns_to_keep = config["data"]["required_columns"]

    try:
        df_selected = df[columns_to_keep].copy()
    except KeyError as e:
        print(f"Error: Missing columns in data file: {e}")
        return

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
    print(f"Consolidated to {len(df_selected)} unique schools")

    # Filter by percentile and age range
    df_selected = df_selected[
        df_selected["TB3PTSE"] >= df_selected["TB3PTSE"].quantile(percentile)
    ]
    print(f"Number of schools in P{percentile}: {len(df_selected)}")

    # Optionally enable age filtering by uncommenting these lines
    # min_age_threshold = config["filtering"]["min_age_threshold"]
    # df_selected = filter_by_age_range(df_selected, min_age_threshold)
    # print(f"Number of schools after age filtering: {len(df_selected)}")

    # Create full address for geocoding
    df_selected["full_address"] = df_selected.apply(
        lambda row: f"{row['ADDRESS1']}, {row['TOWN']}, {row['PCODE']}", axis=1
    )

    # Geocode addresses with progress tracking
    total_addresses = len(df_selected)
    cache_hits = [0]
    cache_misses = [0]
    crime_calculations = [0]

    print(f"Processing {total_addresses} addresses (geocoding and crime stats)...")
    with tqdm(total=total_addresses, desc="Processing", unit="address") as pbar:
        geocoded_results = df_selected["full_address"].apply(
            lambda addr: geocode_address(
                addr,
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

    print(
        f"Processing complete - Geocoding: {cache_hits[0]} cached, {cache_misses[0]} new | "
        f"Crime stats: {crime_calculations[0]} calculated"
    )

    # Save cache and remove schools without coordinates
    save_geocoding_cache()
    save_crime_cache()
    df_selected = df_selected.dropna(subset=["Latitude", "Longitude"])
    print(f"Schools with valid coordinates: {len(df_selected)}")

    # Save processed data
    df_selected.sort_values(by="TB3PTSE", ascending=False).to_csv(
        output_csv, index=False
    )
    print(f"Saved processed data to {output_csv}")

    # Create map
    center_lat = df_selected["Latitude"].mean()
    center_lon = df_selected["Longitude"].mean()
    default_zoom = config["map"]["default_zoom"]
    m = folium.Map(location=[center_lat, center_lon], zoom_start=default_zoom)

    # Add title
    title_html = f"""
        <h3 align="center" style="font-size:20px">
        <a href="https://www.compare-school-performance.service.gov.uk/download-data">
        Schools by A-Level Performance, Multi-Year Average ({percentile} Percentile)
        </a>
        <br>
        <span style="font-size:16px">Clustered with {cluster_radius_km}km radius (min {min_schools} schools per cluster)</span>
        <br>
        <span style="font-size:16px">Crimes in {school_crime_radius_km}km radius around school, index is percentile with 1.00 maximum</span>
        </h3>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Perform clustering
    labels, cluster_centers_data = cluster_schools(
        df_selected, cluster_radius_km, min_schools
    )

    # Extract crime data from cached results
    print("Extracting crime statistics from geocoding cache...")
    df_selected["crime_count"] = 0
    school_crime_stats = []

    for idx, row in df_selected.iterrows():
        if row["crime_stats"]:
            crime_stats = row["crime_stats"]
            df_selected.at[idx, "crime_count"] = crime_stats["total_crimes"]
            school_crime_stats.append(crime_stats)
        else:
            # Fallback for missing crime data
            crime_stats = {
                "total_crimes": 0,
                "crime_types": {},
                "radius_km": SCHOOL_CRIME_RADIUS_KM,
            }
            school_crime_stats.append(crime_stats)

    # Save crime cache after processing
    save_crime_cache()

    # Calculate crime indices using percentile ranking for better interpretability
    crime_counts = df_selected["crime_count"].copy()
    df_selected["crime_index"] = crime_counts.rank(pct=True)

    # Add cluster circles and center markers
    for center_data in cluster_centers_data:
        cluster_id = center_data["cluster_id"]
        center_lat, center_lon = center_data["lat"], center_data["lon"]
        cluster_size = center_data["count"]
        postcode = center_data.get("postcode", "Unknown")

        # Add cluster circle
        cluster_colors = config["colors"]["cluster"]
        folium.Circle(
            location=[center_lat, center_lon],
            radius=cluster_radius_km * 1000,  # Convert to meters
            color=cluster_colors["circle"],
            fill=True,
            fill_opacity=0.2,
            weight=2,
            popup=f"Cluster {cluster_id}: {cluster_size} schools",
        ).add_to(m)

        # Add center marker
        cluster_marker_size = config["map"]["cluster_marker_size"]
        cluster_icon = DivIcon(
            icon_size=(cluster_marker_size, cluster_marker_size),
            icon_anchor=(cluster_marker_size // 2, cluster_marker_size // 2),
            html=f"""
                <div style="width: {cluster_marker_size}px; height: {cluster_marker_size}px;">
                    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="12" r="10" fill="{cluster_colors["center_marker"]}" stroke="#000000" stroke-width="2"/>
                        <text x="12" y="12" font-family="Arial" font-size="10" font-weight="bold"
                              fill="#FFFFFF" text-anchor="middle" dy=".3em">{cluster_id}</text>
                    </svg>
                </div>
            """,
        )

        # Get postcode from cluster center data
        postcode = center_data.get("postcode", "Unknown")
        postcode_text = f" ({postcode})" if postcode and postcode != "Unknown" else ""

        popup_html = f"""
        <b>Cluster {cluster_id} Center{postcode_text}</b><br>
        Schools: {cluster_size}<br>
        Geographic Centroid<br>
        Coordinates: {center_lat:.4f}, {center_lon:.4f}<br>
        Radius: {cluster_radius_km}km
        """

        folium.Marker(
            [center_lat, center_lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=cluster_icon,
        ).add_to(m)

    # Add school markers with individual crime statistics
    for idx, row in df_selected.iterrows():
        # Find the position in the reset dataframe
        position = df_selected.index.get_loc(idx)
        label = labels[position] if position < len(labels) else -1

        # Get school-specific crime data
        school_crime_stat = (
            school_crime_stats[position] if position < len(school_crime_stats) else None
        )
        school_crime_idx = row["crime_index"] if "crime_index" in row else None

        if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]):
            marker = create_school_marker(
                row, label, school_crime_stat, school_crime_idx
            )
            marker.add_to(m)

    # Add legend
    m.get_root().html.add_child(folium.Element(create_legend()))

    # Print cluster summary
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            clusters[label].append(df_selected.iloc[idx])

    print(f"\nClusters within {cluster_radius_km} km (minimum {min_schools} schools):")
    print(f"Number of clusters: {len(clusters)}")
    unclustered_count = sum(1 for l in labels if l == -1)
    print(
        f"Unclustered schools: {unclustered_count} of {len(df_selected)} ({unclustered_count / len(df_selected):.1%})"
    )

    # Print detailed cluster information
    for cluster_id in sorted(clusters.keys()):
        schools = clusters[cluster_id]
        center_data = next(
            (c for c in cluster_centers_data if c["cluster_id"] == cluster_id), {}
        )
        postcode = center_data.get("postcode", "Unknown")
        postcode_text = f" ({postcode})" if postcode and postcode != "Unknown" else ""

        print(f"\nCluster {cluster_id}{postcode_text}:")

        # Calculate average school-level crime for this cluster
        cluster_school_positions = [
            df_selected.index.get_loc(school.name)
            for school in schools
            if school.name in df_selected.index
        ]
        cluster_school_crimes = [
            school_crime_stats[pos]["total_crimes"]
            for pos in cluster_school_positions
            if pos < len(school_crime_stats)
        ]

        if cluster_school_crimes:
            avg_school_crime = sum(cluster_school_crimes) / len(cluster_school_crimes)
            print(
                f"  Average school-level crime ({school_crime_radius_km}km radius): {avg_school_crime:.1f}"
            )

        for school in schools:
            if "crime_count" in school and "crime_index" in school:
                school_crimes = school["crime_count"]
                school_idx = school["crime_index"]
                print(
                    f"- {school['SCHNAME']} ({school['TOWN']}) - {school_crimes} crimes (Index: {school_idx:.2f})"
                )
            else:
                print(
                    f"- {school['SCHNAME']} ({school['TOWN']}) - Crime data unavailable"
                )

    # Print overall crime statistics summary
    if "crime_count" in df_selected:
        max_school_crime = df_selected["crime_count"].max()
        min_school_crime = df_selected["crime_count"].min()
        avg_school_crime = df_selected["crime_count"].mean()

        print(
            f"\nOverall Crime Statistics (per school, {school_crime_radius_km}km radius):"
        )
        print(f"Maximum crimes around any school: {max_school_crime}")
        print(f"Minimum crimes around any school: {min_school_crime}")
        print(f"Average crimes per school area: {avg_school_crime:.1f}")

        # Find schools with highest and lowest crime
        max_crime_school = df_selected.loc[
            df_selected["crime_count"] == max_school_crime
        ].iloc[0]
        min_crime_school = df_selected.loc[
            df_selected["crime_count"] == min_school_crime
        ].iloc[0]

        print(
            f"Highest crime area: {max_crime_school['SCHNAME']} ({max_crime_school['TOWN']})"
        )
        print(
            f"Lowest crime area: {min_crime_school['SCHNAME']} ({min_crime_school['TOWN']})"
        )

    # Save map
    m.save(map_filename)
    print(f"Map saved as {map_filename}")


if __name__ == "__main__":
    import argparse

    default_radius = config["clustering"]["default_cluster_radius_km"]
    default_min_schools = config["clustering"]["default_min_schools"]

    parser = argparse.ArgumentParser(description="Generate school clustering map")
    parser.add_argument(
        "--radius",
        type=float,
        default=default_radius,
        help=f"Cluster radius in km (default: {default_radius})",
    )
    parser.add_argument(
        "--min-schools",
        type=int,
        default=default_min_schools,
        help=f"Minimum schools per cluster (default: {default_min_schools})",
    )

    args = parser.parse_args()
    main(args.radius, args.min_schools)
