import os
import re
import json
import math
from collections import defaultdict

import pandas as pd
import numpy as np
import folium
from folium.features import DivIcon
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
from geopy.distance import geodesic

# Constants for clustering
DEFAULT_CLUSTER_RADIUS_KM = 5
DEFAULT_MIN_SCHOOLS = 2
PERCENTILE = 0.90

# Crime analysis constants
SCHOOL_CRIME_RADIUS_KM = 3  # Individual school crime analysis radius

# Crime data file
CRIME_DATA_FILE = "combined_crimes.csv.gz"

# Geocoding cache
geocoding_cache = {}
CACHE_FILE = "geocoding_cache.json"

def load_geocoding_cache():
    """Load geocoding cache from file if it exists"""
    global geocoding_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                geocoding_cache = json.load(f)
            print(f"Loaded {len(geocoding_cache)} cached geocoded addresses")
        except Exception as e:
            print(f"Error loading geocoding cache: {e}")
            geocoding_cache = {}
    else:
        geocoding_cache = {}

def save_geocoding_cache():
    """Save geocoding cache to file"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(geocoding_cache, f, indent=2)
        print(f"Saved {len(geocoding_cache)} unique addresses to cache")
    except Exception as e:
        print(f"Error saving geocoding cache: {e}")

def point_score_to_grade(point_score):
    """Convert point score to grade"""
    if pd.isna(point_score) or point_score == 0:
        return 'N/A'
    elif point_score >= 50:
        return 'A*'
    elif point_score >= 40:
        return 'A'
    else:
        return '≤B'

def reverse_geocode_location(lat, lon, index_name="your-place-index-name", region_name="eu-west-2"):
    """Reverse geocode coordinates to get postcode using Amazon Location Services"""
    try:
        location_client = boto3.client('location', region_name=region_name)
        response = location_client.search_place_index_for_position(
            IndexName=index_name,
            Position=[lon, lat],  # Amazon Location uses [lon, lat] order
            MaxResults=1
        )

        if response['Results']:
            place = response['Results'][0]['Place']
            if 'PostalCode' in place:
                return place['PostalCode']
            elif 'Label' in place:
                # Extract postcode from label using regex
                postcode_match = re.search(r'[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}', place['Label'])
                if postcode_match:
                    return postcode_match.group()

        return None

    except Exception as e:
        print(f"Error reverse geocoding ({lat:.4f}, {lon:.4f}): {e}")
        return None

def geocode_address(address, index_name="your-place-index-name", region_name="eu-west-2",
                   pbar=None, hits_counter=None, misses_counter=None):
    """Geocode address using Amazon Location Services with caching"""
    if not address or pd.isna(address):
        if pbar:
            pbar.update(1)
        return pd.Series({'Latitude': None, 'Longitude': None})

    # Check cache first
    if address in geocoding_cache:
        lat, lon = geocoding_cache[address]
        if pbar:
            pbar.set_postfix(cache_hits=hits_counter[0], cache_misses=misses_counter[0])
            hits_counter[0] += 1
            pbar.update(1)
        return pd.Series({'Latitude': lat, 'Longitude': lon})

    try:
        if pbar:
            pbar.set_postfix(cache_hits=hits_counter[0], cache_misses=misses_counter[0])
            misses_counter[0] += 1

        location_client = boto3.client('location', region_name=region_name)
        response = location_client.search_place_index_for_text(
            IndexName=index_name,
            Text=address,
            MaxResults=1
        )

        if response['Results']:
            coordinates = response['Results'][0]['Place']['Geometry']['Point']
            # Store in cache
            geocoding_cache[address] = (coordinates[1], coordinates[0])
            if pbar:
                pbar.update(1)
            return pd.Series({'Latitude': coordinates[1], 'Longitude': coordinates[0]})

        # Cache negative results
        geocoding_cache[address] = (None, None)
        if pbar:
            pbar.update(1)
        return pd.Series({'Latitude': None, 'Longitude': None})

    except ClientError as e:
        print(f"AWS Error geocoding {address}: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
        if pbar:
            pbar.update(1)
        return pd.Series({'Latitude': None, 'Longitude': None})
    except Exception as e:
        print(f"Error geocoding {address}: {e}")
        if pbar:
            pbar.update(1)
        return pd.Series({'Latitude': None, 'Longitude': None})

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in kilometers"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
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
        center_lon + lon_change   # max_lon
    )

def get_crime_stats_for_location(center_lat, center_lon, radius_km, crime_df=None):
    """Get crime statistics for a location within specified radius"""
    if crime_df is None:
        try:
            crime_df = pd.read_csv(CRIME_DATA_FILE)
        except FileNotFoundError:
            return {'total_crimes': 0, 'crime_types': {}, 'error': 'No crime data available'}

    # Filter out low-impact crimes
    excluded_crimes = {
        'Shoplifting', 'Bicycle theft', 'Other theft', 'Other crime',
        'Drugs', 'Anti-social behaviour', 'Criminal damage and arson'
    }
    crime_df_filtered = crime_df[~crime_df['Crime type'].isin(excluded_crimes)]

    # Use bounding box for initial filtering
    min_lat, max_lat, min_lon, max_lon = calculate_circle_bounding_box(
        center_lat, center_lon, radius_km
    )

    bbox_filtered = crime_df_filtered[
        (crime_df_filtered['Latitude'].between(min_lat, max_lat)) &
        (crime_df_filtered['Longitude'].between(min_lon, max_lon))
    ]

    if len(bbox_filtered) == 0:
        return {'total_crimes': 0, 'crime_types': {}, 'radius_km': radius_km}

    # Calculate actual distances for remaining points
    bbox_filtered = bbox_filtered.copy()
    bbox_filtered['distance'] = bbox_filtered.apply(
        lambda row: haversine_distance(
            center_lat, center_lon, row['Latitude'], row['Longitude']
        ), axis=1
    )

    # Final filter by actual radius
    final_filtered = bbox_filtered[bbox_filtered['distance'] <= radius_km]

    total_crimes = len(final_filtered)
    crime_types = final_filtered['Crime type'].value_counts().to_dict() if total_crimes > 0 else {}

    return {
        'total_crimes': total_crimes,
        'crime_types': crime_types,
        'radius_km': radius_km
    }

def format_crime_stats(crime_stats, crime_index=None):
    """Format crime statistics for display in popup"""
    if crime_stats.get('error'):
        return crime_stats['error']

    total = crime_stats['total_crimes']
    radius = crime_stats['radius_km']

    if total == 0:
        crime_index_text = " (Crime Index: 0.00)" if crime_index is not None else ""
        return f"No serious crimes in {radius}km radius{crime_index_text}"

    # Get top 3 crime types
    top_crimes = sorted(crime_stats['crime_types'].items(), key=lambda x: x[1], reverse=True)[:3]

    crime_index_text = f" (Crime Index: {crime_index:.2f})" if crime_index is not None else ""
    lines = [f"<b>{total} serious crimes</b> in {radius}km radius{crime_index_text}:"]
    lines.extend([f"• {crime_type}: {count}" for crime_type, count in top_crimes])

    return "<br>".join(lines)

# This function is no longer needed as we calculate crime indices directly in the dataframe

def filter_by_age_range(df, min_age_threshold=7):
    """Filter schools based on minimum age threshold"""
    def extract_min_age(age_range):
        if pd.isna(age_range):
            return None
        numbers = re.findall(r'\d+', str(age_range).strip())
        return int(numbers[0]) if numbers else None

    min_ages = df['AGERANGE'].apply(extract_min_age)
    valid_mask = min_ages.notna() & (min_ages <= min_age_threshold)
    return df[valid_mask]

def get_grade_color(score, is_independent):
    """Get color for school marker based on score and type"""
    if is_independent:
        if score >= 50:
            return '#FF0000'  # Red for A*
        elif score >= 40:
            return '#FFA500'  # Orange for A
        else:
            return '#FFD700'  # Gold for ≤B
    else:
        if score >= 50:
            return '#000080'  # Navy for A*
        elif score >= 40:
            return '#0000FF'  # Blue for A
        else:
            return '#4169E1'  # Royal Blue for ≤B

def get_text_color(background_color):
    """Determine appropriate text color based on background color brightness"""
    bg_color = background_color.lstrip('#')
    rgb = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
    brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
    return '#000000' if brightness > 128 else '#FFFFFF'

def cluster_schools(df, max_distance_km, min_schools=2):
    """
    Efficient school clustering algorithm using school locations as potential centers

    Returns:
        tuple: (labels array, cluster_centers_data list)
    """
    if len(df) == 0:
        return np.array([]), []

    print(f"Evaluating {len(df)} schools as potential cluster centers...")

    # Reset index to ensure consistent positioning
    df_reset = df.reset_index(drop=True)

    # Find potential cluster centers
    cluster_centers = []
    for center_idx, center_row in df_reset.iterrows():
        center_lat, center_lon = center_row['Latitude'], center_row['Longitude']

        # Find nearby schools
        nearby_schools = []
        for school_idx, school_row in df_reset.iterrows():
            distance = geodesic(
                (center_lat, center_lon),
                (school_row['Latitude'], school_row['Longitude'])
            ).km
            if distance <= max_distance_km:
                nearby_schools.append(school_idx)

        # Check if this forms a valid cluster
        if len(nearby_schools) >= min_schools:
            cluster_centers.append({
                'lat': center_lat,
                'lon': center_lon,
                'schools': nearby_schools,
                'count': len(nearby_schools),
                'center_school_idx': center_idx
            })

    print(f"Found {len(cluster_centers)} potential cluster centers")

    if not cluster_centers:
        return np.full(len(df_reset), -1), []

    # Remove redundant clusters (keep those with most unique schools)
    final_centers = []
    used_schools = set()

    # Sort by school count (descending)
    cluster_centers.sort(key=lambda x: x['count'], reverse=True)

    for center in cluster_centers:
        new_schools = set(center['schools']) - used_schools
        if len(new_schools) >= min_schools:
            final_centers.append(center)
            used_schools.update(center['schools'])

    print(f"After removing redundancy: {len(final_centers)} cluster centers")

    # Assign schools to clusters
    labels = np.full(len(df_reset), -1)

    for school_idx, school_row in df_reset.iterrows():
        min_distance = float('inf')
        best_cluster = -1

        for cluster_id, center in enumerate(final_centers):
            distance = geodesic(
                (center['lat'], center['lon']),
                (school_row['Latitude'], school_row['Longitude'])
            ).km
            if distance <= max_distance_km and distance < min_distance:
                min_distance = distance
                best_cluster = cluster_id

        labels[school_idx] = best_cluster

    # Calculate geographic centroids for final cluster centers
    print("Calculating geographic centroids for each cluster...")
    refined_centers = []

    for cluster_id, center in enumerate(final_centers):
        cluster_schools = [pos for pos, label in enumerate(labels) if label == cluster_id]

        if not cluster_schools:
            continue

        # Calculate centroid using Cartesian coordinates for accuracy
        x_coords, y_coords, z_coords = [], [], []

        for pos in cluster_schools:
            lat_rad = math.radians(df_reset.iloc[pos]['Latitude'])
            lon_rad = math.radians(df_reset.iloc[pos]['Longitude'])

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

        refined_centers.append({
            'lat': centroid_lat_deg,
            'lon': centroid_lon_deg,
            'schools': cluster_schools,
            'count': len(cluster_schools),
            'cluster_id': cluster_id,
            'postcode': postcode
        })

    clustered_count = sum(1 for label in labels if label != -1)
    print(f"Clustered {clustered_count} of {len(df_reset)} schools into {len(refined_centers)} clusters")

    return labels, refined_centers

def create_school_marker(row, label=-1, school_crime_stats=None, school_crime_index=None):
    """Create a folium marker for a school"""
    is_independent = pd.isna(row['ADMPOL_PT']) or row['ADMPOL_PT'].strip() == ''
    colour = get_grade_color(row['TB3PTSE'], is_independent)
    fontcolour = get_text_color(colour)
    admpol = 'independent' if is_independent else row['ADMPOL_PT']

    # Create address string
    address_parts = [
        str(row[field]) for field in ['ADDRESS1', 'TOWN', 'PCODE']
        if pd.notna(row[field]) and str(row[field]).strip()
    ]
    address_string = ', '.join(address_parts)

    # Create popup HTML
    popup_html = f"""
    <b>{row['SCHNAME']}</b><br>
    {address_string}<br>
    Phone: {row['TELNUM']}<br>
    Admissions Policy: {admpol}<br>
    Age Range: {row['AGERANGE']}<br>
    Gender: {row['GEND1618']}<br>
    Students (16-18): {row['TPUP1618']}<br>
    <b>Average across all years:</b><br>
    Avg best 3 A-levels (TB3PTSE): {row['TB3PTSE']:.2f} (<b>{point_score_to_grade(row['TB3PTSE'])}</b>)<br>
    Avg per A-level (TALLPPE_ALEV_1618): {row['TALLPPE_ALEV_1618']:.2f} (<b>{point_score_to_grade(row['TALLPPE_ALEV_1618'])}</b>)<br>
    <b>Year-by-year TB3PTSE scores:</b><br>
    {row.get('YEAR_SCORES', 'No year data available')}
    """
    if label != -1:
        popup_html += f"<br>Cluster: {label}"

    # Add school-specific crime information
    if school_crime_stats:
        popup_html += "<br><br><b>Local Area Safety (School-specific):</b><br>"
        popup_html += format_crime_stats(school_crime_stats, school_crime_index)

    # Create marker icon
    icon = DivIcon(
        icon_size=(30, 30),
        icon_anchor=(15, 30),
        html=f'''
            <div style="width: 30px; height: 30px;">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 0C7.58 0 4 3.58 4 8c0 5.76 8 16 8 16s8-10.24 8-16c0-4.42-3.58-8-8-8z"
                        fill="{colour}"
                        stroke="black"
                        stroke-width="1"/>
                    <text x="12" y="9" font-family="Arial" font-size="8" font-weight="bold"
                          fill="{fontcolour}" text-anchor="middle" dy=".3em">{round(row['TB3PTSE'])}</text>
                </svg>
            </div>
        '''
    )

    return folium.Marker(
        [row['Latitude'], row['Longitude']],
        popup=folium.Popup(popup_html, max_width=350),
        icon=icon
    )

def create_legend():
    """Create HTML legend for the map"""
    return '''
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
            <span style="display:inline-block; width: 10px; height: 10px; background-color: #FF0000; margin-right: 4px;"></span>A* (≥50)
        </div>
        <div style="margin-bottom: 3px;">
            <span style="display:inline-block; width: 10px; height: 10px; background-color: #FFA500; margin-right: 4px;"></span>A (40-49)
        </div>
        <div style="margin-bottom: 8px;">
            <span style="display:inline-block; width: 10px; height: 10px; background-color: #FFD700; margin-right: 4px;"></span>≤B (<40)
        </div>
        <div style="font-weight: bold; margin-bottom: 6px; font-size:10px;">State</div>
        <div style="margin-bottom: 3px;">
            <span style="display:inline-block; width: 10px; height: 10px; background-color: #000080; margin-right: 4px;"></span>A* (≥50)
        </div>
        <div style="margin-bottom: 3px;">
            <span style="display:inline-block; width: 10px; height: 10px; background-color: #0000FF; margin-right: 4px;"></span>A (40-49)
        </div>
        <div>
            <span style="display:inline-block; width: 10px; height: 10px; background-color: #4169E1; margin-right: 4px;"></span>≤B (<40)
        </div>
    </div>
    '''

def main(cluster_radius_km=DEFAULT_CLUSTER_RADIUS_KM, min_schools=DEFAULT_MIN_SCHOOLS):
    """Main function to process schools and create map"""
    # Load geocoding cache
    load_geocoding_cache()

    # Load crime data
    print("Loading crime data...")
    try:
        crime_df = pd.read_csv(CRIME_DATA_FILE)
        print(f"Loaded {len(crime_df)} crime records")
    except FileNotFoundError:
        print(f"Warning: Crime data file {CRIME_DATA_FILE} not found. Crime statistics will not be available.")
        crime_df = None

    # Find and load all school data files
    import glob
    csv_files = glob.glob("20*ks5final*.csv.gz")
    if not csv_files:
        print("Error: No school data files found matching pattern '20*ks5final*.csv.gz'")
        return

    print(f"Found {len(csv_files)} data files: {csv_files}")

    # Load and combine all years
    all_dfs = []
    for csv_file in sorted(csv_files):
        year = csv_file.split('_')[0]  # Extract year from filename
        try:
            df_year = pd.read_csv(csv_file, low_memory=False, dtype={'TELNUM': str})
            df_year['YEAR'] = year
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
    columns_to_keep = [
        'SCHNAME', 'ADDRESS1', 'TOWN', 'PCODE', 'TELNUM', 'ADMPOL_PT', 'GEND1618',
        'AGERANGE', 'TPUP1618', 'TALLPPE_ALEV_1618', 'TB3PTSE', 'YEAR'
    ]

    try:
        df_selected = df[columns_to_keep].copy()
    except KeyError as e:
        print(f"Error: Missing columns in data file: {e}")
        return

    # Convert numeric columns
    numeric_columns = ['TB3PTSE', 'TALLPPE_ALEV_1618']
    for col in numeric_columns:
        df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce').fillna(0)

    # Calculate average scores per school across all years
    school_groups = df_selected.groupby('SCHNAME')

    # Create consolidated dataframe with averages and year-specific data
    consolidated_data = []
    for school_name, group_df in school_groups:
        # Calculate averages (only for non-zero scores)
        valid_scores = group_df[group_df['TB3PTSE'] > 0]
        if len(valid_scores) == 0:
            continue
            
        avg_tb3ptse = valid_scores['TB3PTSE'].mean()
        avg_tallppe = valid_scores['TALLPPE_ALEV_1618'].mean()

        # Get the most recent row for other fields
        latest_row = group_df.iloc[-1].copy()
        latest_row['TB3PTSE'] = avg_tb3ptse
        latest_row['TALLPPE_ALEV_1618'] = avg_tallppe

        # Create year-specific scores string for popup
        year_scores = []
        for _, year_row in valid_scores.iterrows():
            year_scores.append(f"{year_row['YEAR']}: {year_row['TB3PTSE']:.2f}")
        latest_row['YEAR_SCORES'] = '<br>'.join(year_scores)

        consolidated_data.append(latest_row)

    df_selected = pd.DataFrame(consolidated_data)
    print(f"Consolidated to {len(df_selected)} unique schools")

    # Filter by percentile and age range
    df_selected = df_selected[df_selected['TB3PTSE'] >= df_selected['TB3PTSE'].quantile(PERCENTILE)]
    print(f"Number of schools in P{PERCENTILE}: {len(df_selected)}")

#    df_selected = filter_by_age_range(df_selected, min_age_threshold=7)
#    print(f"Number of schools after age filtering: {len(df_selected)}")

    # Create full address for geocoding
    df_selected['full_address'] = df_selected.apply(
        lambda row: f"{row['ADDRESS1']}, {row['TOWN']}, {row['PCODE']}", axis=1
    )

    # Geocode addresses with progress tracking
    total_addresses = len(df_selected)
    cache_hits = [0]
    cache_misses = [0]

    print(f"Geocoding {total_addresses} addresses...")
    with tqdm(total=total_addresses, desc="Geocoding", unit="address") as pbar:
        geocoded_results = df_selected['full_address'].apply(
            lambda addr: geocode_address(addr, pbar=pbar, hits_counter=cache_hits, misses_counter=cache_misses)
        )

    df_selected['Latitude'] = geocoded_results['Latitude']
    df_selected['Longitude'] = geocoded_results['Longitude']

    print(f"Geocoding complete - Cache hits: {cache_hits[0]} ({cache_hits[0]/total_addresses:.1%}), "
          f"Cache misses: {cache_misses[0]} ({cache_misses[0]/total_addresses:.1%})")

    # Save cache and remove schools without coordinates
    save_geocoding_cache()
    df_selected = df_selected.dropna(subset=['Latitude', 'Longitude'])
    print(f"Schools with valid coordinates: {len(df_selected)}")

    # Save processed data
    output_csv = "processed_school_data.csv"
    df_selected.sort_values(by='TB3PTSE', ascending=False).to_csv(output_csv, index=False)
    print(f"Saved processed data to {output_csv}")

    # Create map
    center_lat = df_selected['Latitude'].mean()
    center_lon = df_selected['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

    # Add title
    title_html = f'''
        <h3 align="center" style="font-size:20px">
        <a href="https://www.compare-school-performance.service.gov.uk/download-data">
        Schools by A-Level Performance, Multi-Year Average ({PERCENTILE} Percentile)
        </a>
        <br>
        <span style="font-size:16px">Clustered with {cluster_radius_km}km radius (min {min_schools} schools per cluster)</span>
        </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Perform clustering
    labels, cluster_centers_data = cluster_schools(df_selected, cluster_radius_km, min_schools)

    # Calculate crime statistics for individual schools and add to dataframe
    print("Calculating crime statistics for individual schools...")
    df_selected['crime_count'] = 0
    school_crime_stats = []

    with tqdm(total=len(df_selected), desc="School crime analysis", unit="school") as pbar:
        for idx, row in df_selected.iterrows():
            crime_stats = get_crime_stats_for_location(
                row['Latitude'], row['Longitude'], SCHOOL_CRIME_RADIUS_KM, crime_df
            )
            school_crime_stats.append(crime_stats)
            df_selected.at[idx, 'crime_count'] = crime_stats['total_crimes']
            pbar.update(1)

    # Calculate crime indices using pandas native methods
    df_selected['crime_index'] = pd.qcut(
        df_selected['crime_count'],
        q=10,
        labels=False,
        duplicates='drop'
    ) / 10.0

    # Handle edge cases where we might have fewer than 10 unique values
    if df_selected['crime_index'].max() < 0.9:
        max_val = df_selected['crime_index'].max()
        if max_val > 0:  # Avoid division by zero
            df_selected['crime_index'] = df_selected['crime_index'] * (0.9 / max_val)

    # Add cluster circles and center markers
    for cluster_id, center_data in enumerate(cluster_centers_data):
        center_lat, center_lon = center_data['lat'], center_data['lon']
        cluster_size = center_data['count']
        postcode = center_data.get('postcode', 'Unknown')

        # Add cluster circle
        folium.Circle(
            location=[center_lat, center_lon],
            radius=cluster_radius_km * 1000,  # Convert to meters
            color="#1E90FF",
            fill=True,
            fill_opacity=0.2,
            weight=2,
            popup=f"Cluster {cluster_id}: {cluster_size} schools"
        ).add_to(m)

        # Add center marker
        cluster_icon = DivIcon(
            icon_size=(20, 20),
            icon_anchor=(10, 10),
            html=f'''
                <div style="width: 20px; height: 20px;">
                    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="12" r="10" fill="#FF4500" stroke="#000000" stroke-width="2"/>
                        <text x="12" y="12" font-family="Arial" font-size="10" font-weight="bold"
                              fill="#FFFFFF" text-anchor="middle" dy=".3em">{cluster_id}</text>
                    </svg>
                </div>
            '''
        )

        # Get postcode from cluster center data
        postcode = center_data.get('postcode', 'Unknown')
        postcode_text = f" ({postcode})" if postcode and postcode != 'Unknown' else ""

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
            icon=cluster_icon
        ).add_to(m)

    # Add school markers with individual crime statistics
    for idx, row in df_selected.iterrows():
        # Find the position in the reset dataframe
        position = df_selected.index.get_loc(idx)
        label = labels[position] if position < len(labels) else -1

        # Get school-specific crime data
        school_crime_stat = school_crime_stats[position] if position < len(school_crime_stats) else None
        school_crime_idx = row['crime_index'] if 'crime_index' in row else None

        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            marker = create_school_marker(row, label, school_crime_stat, school_crime_idx)
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
    print(f"Unclustered schools: {unclustered_count} of {len(df_selected)} ({unclustered_count/len(df_selected):.1%})")

    # Print detailed cluster information
    for cluster_id, schools in clusters.items():
        center_data = cluster_centers_data[cluster_id] if cluster_id < len(cluster_centers_data) else {}
        postcode = center_data.get('postcode', 'Unknown')
        postcode_text = f" ({postcode})" if postcode and postcode != 'Unknown' else ""

        print(f"\nCluster {cluster_id}{postcode_text}:")

        # Calculate average school-level crime for this cluster
        cluster_school_positions = [df_selected.index.get_loc(school.name) for school in schools
                                  if school.name in df_selected.index]
        cluster_school_crimes = [school_crime_stats[pos]['total_crimes']
                               for pos in cluster_school_positions
                               if pos < len(school_crime_stats)]

        if cluster_school_crimes:
            avg_school_crime = sum(cluster_school_crimes) / len(cluster_school_crimes)
            print(f"  Average school-level crime ({SCHOOL_CRIME_RADIUS_KM}km radius): {avg_school_crime:.1f}")

        for school in schools:
            if 'crime_count' in school and 'crime_index' in school:
                school_crimes = school['crime_count']
                school_idx = school['crime_index']
                print(f"- {school['SCHNAME']} ({school['TOWN']}) - {school_crimes} crimes (Index: {school_idx:.2f})")
            else:
                print(f"- {school['SCHNAME']} ({school['TOWN']}) - Crime data unavailable")

    # Print overall crime statistics summary
    if 'crime_count' in df_selected:
        max_school_crime = df_selected['crime_count'].max()
        min_school_crime = df_selected['crime_count'].min()
        avg_school_crime = df_selected['crime_count'].mean()

        print(f"\nOverall Crime Statistics (per school, {SCHOOL_CRIME_RADIUS_KM}km radius):")
        print(f"Maximum crimes around any school: {max_school_crime}")
        print(f"Minimum crimes around any school: {min_school_crime}")
        print(f"Average crimes per school area: {avg_school_crime:.1f}")

        # Find schools with highest and lowest crime
        max_crime_school = df_selected.loc[df_selected['crime_count'] == max_school_crime].iloc[0]
        min_crime_school = df_selected.loc[df_selected['crime_count'] == min_school_crime].iloc[0]

        print(f"Highest crime area: {max_crime_school['SCHNAME']} ({max_crime_school['TOWN']})")
        print(f"Lowest crime area: {min_crime_school['SCHNAME']} ({min_crime_school['TOWN']})")

    # Save map
    map_filename = "schools_map.html"
    m.save(map_filename)
    print(f"Map saved as {map_filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate school clustering map')
    parser.add_argument('--radius', type=float, default=DEFAULT_CLUSTER_RADIUS_KM,
                       help=f'Cluster radius in km (default: {DEFAULT_CLUSTER_RADIUS_KM})')
    parser.add_argument('--min-schools', type=int, default=DEFAULT_MIN_SCHOOLS,
                       help=f'Minimum schools per cluster (default: {DEFAULT_MIN_SCHOOLS})')

    args = parser.parse_args()
    main(args.radius, args.min_schools)
