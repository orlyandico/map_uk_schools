import os
import math
import logging
from collections import defaultdict

import pandas as pd
import numpy as np
import folium
from folium.features import DivIcon
import boto3
from sklearn.neighbors import BallTree

# Import shared library
import school_data_lib


# ITU-R BT.601 luma coefficients for perceived brightness calculation
LUMA_RED = 299
LUMA_GREEN = 587
LUMA_BLUE = 114
LUMA_DIVISOR = 1000


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('plot_schools.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Load configuration at module level
config = school_data_lib.load_config()


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
                import re
                postcode_match = re.search(
                    r"[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}", place["Label"]
                )
                if postcode_match:
                    return postcode_match.group()

        return None

    except Exception as e:
        logger.error(f"Error reverse geocoding ({lat:.4f}, {lon:.4f}): {e}")
        return None


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
    brightness = (rgb[0] * LUMA_RED + rgb[1] * LUMA_GREEN + rgb[2] * LUMA_BLUE) / LUMA_DIVISOR
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

    logger.info(f"Evaluating {len(df)} schools as potential cluster centers...")

    # Reset index to ensure consistent positioning
    df_reset = df.reset_index(drop=True)

    # Prepare coordinates for BallTree (needs radians for haversine metric)
    coords = np.radians(df_reset[["Latitude", "Longitude"]].values)

    # Build BallTree with haversine metric for geographic distances
    # Haversine accounts for Earth's curvature
    tree = BallTree(coords, metric='haversine')

    # Convert km to radians for BallTree query
    radius_radians = max_distance_km / school_data_lib.EARTH_RADIUS_KM

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

    logger.info(f"Found {len(cluster_centers)} potential cluster centers")

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

    logger.info(f"After removing redundancy: {len(final_centers)} cluster centers")

    # Assign schools to clusters
    labels = np.full(len(df_reset), -1)

    for cluster_id, center in enumerate(final_centers):
        for school_idx in center["schools"]:
            labels[school_idx] = cluster_id

    # Calculate geographic centroids for final cluster centers
    logger.info("Calculating geographic centroids for each cluster...")
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
    logger.info(
        f"Clustered {clustered_count} of {len(df_reset)} schools into {len(refined_centers)} clusters"
    )

    return labels, refined_centers


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
    Avg best 3 A-levels (TB3PTSE): {row["TB3PTSE"]:.2f} (<b>{school_data_lib.point_score_to_grade(row["TB3PTSE"], config)}</b>)<br>
    Avg per A-level (TALLPPE_ALEV_1618): {row["TALLPPE_ALEV_1618"]:.2f} (<b>{school_data_lib.point_score_to_grade(row["TALLPPE_ALEV_1618"], config)}</b>)<br>
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


def create_and_save_map(df_selected, labels, cluster_centers_data, school_crime_stats,
                        cluster_radius_km, min_schools, school_crime_radius_km,
                        percentile, map_filename):
    """
    Create folium map with schools, clusters, and statistics

    Args:
        df_selected: Schools dataframe with all data
        labels: Cluster labels array
        cluster_centers_data: List of cluster center dictionaries
        school_crime_stats: List of crime statistics per school
        cluster_radius_km: Cluster radius in km
        min_schools: Minimum schools per cluster
        school_crime_radius_km: Crime radius around schools in km
        percentile: Percentile filter used
        map_filename: Output filename for map
    """
    # Validate we have data to display
    if len(df_selected) == 0:
        logger.error("No schools with valid coordinates to display on map")
        return

    # Create map
    center_lat = df_selected["Latitude"].mean()
    center_lon = df_selected["Longitude"].mean()

    # Validate map center coordinates
    if pd.isna(center_lat) or pd.isna(center_lon):
        logger.error("Unable to calculate map center - no valid coordinates")
        return

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

    # Save map
    m.save(map_filename)
    logger.info(f"Map saved as {map_filename}")


def print_cluster_statistics(df_selected, labels, cluster_centers_data,
                            school_crime_stats, cluster_radius_km,
                            min_schools, school_crime_radius_km):
    """
    Print detailed cluster and crime statistics

    Args:
        df_selected: Schools dataframe
        labels: Cluster labels
        cluster_centers_data: Cluster center information
        school_crime_stats: Crime statistics per school
        cluster_radius_km: Cluster radius
        min_schools: Minimum schools per cluster
        school_crime_radius_km: Crime radius around schools
    """
    # Print cluster summary
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            clusters[label].append(df_selected.iloc[idx])

    logger.info(f"\nClusters within {cluster_radius_km} km (minimum {min_schools} schools):")
    logger.info(f"Number of clusters: {len(clusters)}")
    unclustered_count = sum(1 for l in labels if l == -1)
    logger.info(
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

        logger.info(f"\nCluster {cluster_id}{postcode_text}:")

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
            logger.info(
                f"  Average school-level crime ({school_crime_radius_km}km radius): {avg_school_crime:.1f}"
            )

        for school in schools:
            if "crime_count" in school and "crime_index" in school:
                school_crimes = school["crime_count"]
                school_idx = school["crime_index"]
                logger.info(
                    f"- {school['SCHNAME']} ({school['TOWN']}) - {school_crimes} crimes (Index: {school_idx:.2f})"
                )
            else:
                logger.info(
                    f"- {school['SCHNAME']} ({school['TOWN']}) - Crime data unavailable"
                )

    # Print overall crime statistics summary
    if "crime_count" in df_selected:
        max_school_crime = df_selected["crime_count"].max()
        min_school_crime = df_selected["crime_count"].min()
        avg_school_crime = df_selected["crime_count"].mean()

        logger.info(
            f"\nOverall Crime Statistics (per school, {school_crime_radius_km}km radius):"
        )
        logger.info(f"Maximum crimes around any school: {max_school_crime}")
        logger.info(f"Minimum crimes around any school: {min_school_crime}")
        logger.info(f"Average crimes per school area: {avg_school_crime:.1f}")

        # Find schools with highest and lowest crime (with bounds checking)
        max_crime_matches = df_selected[df_selected["crime_count"] == max_school_crime]
        min_crime_matches = df_selected[df_selected["crime_count"] == min_school_crime]

        if not max_crime_matches.empty and not min_crime_matches.empty:
            max_crime_school = max_crime_matches.iloc[0]
            min_crime_school = min_crime_matches.iloc[0]

            logger.info(
                f"Highest crime area: {max_crime_school['SCHNAME']} ({max_crime_school['TOWN']})"
            )
            logger.info(
                f"Lowest crime area: {min_crime_school['SCHNAME']} ({min_crime_school['TOWN']})"
            )
        else:
            logger.info("Crime area details: Insufficient data to determine specific schools")


def main(cluster_radius_km=None, min_schools=None):
    """
    Main orchestration function to process schools and create map

    This function coordinates the entire pipeline:
    1. Validate configuration
    2. Load data (crime data, school data)
    3. Process and consolidate schools
    4. Geocode and enrich with crime stats
    5. Perform clustering
    6. Create visualization
    7. Print statistics

    Args:
        cluster_radius_km: Cluster radius in km (uses config default if None)
        min_schools: Minimum schools per cluster (uses config default if None)
    """
    # Validate configuration first
    try:
        school_data_lib.validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        return

    # Use config defaults if not specified
    if cluster_radius_km is None:
        cluster_radius_km = config["clustering"]["default_cluster_radius_km"]
    if min_schools is None:
        min_schools = config["clustering"]["default_min_schools"]

    # Extract configuration values
    crime_data_file = config["crime"]["crime_data_file"]
    school_crime_radius_km = config["crime"]["school_crime_radius_km"]
    percentile = config["filtering"]["percentile"]
    school_data_pattern = config["data"]["school_data_pattern"]
    output_csv = config["output"]["processed_csv"]
    map_filename = config["output"]["map_filename"]

    # Step 1: Load caches
    school_data_lib.load_geocoding_cache(config)
    school_data_lib.load_crime_cache(config)

    # Step 2: Load crime data
    crime_df = school_data_lib.load_crime_data(crime_data_file)

    # Step 3: Load and combine school data files
    df = school_data_lib.load_school_data_files(school_data_pattern)
    if df is None:
        return

    # Step 4: Process and consolidate schools
    df_selected = school_data_lib.process_and_consolidate_schools(df, config)
    if df_selected is None:
        return

    # Step 5: Filter by percentile
    df_selected = school_data_lib.filter_schools_by_percentile(df_selected, percentile, config)

    # Step 6: Geocode schools and enrich with crime data
    df_selected = school_data_lib.geocode_and_enrich_schools(df_selected, crime_df, config)

    # Step 7: Validate we have schools to process
    if len(df_selected) == 0:
        logger.error("No schools with valid coordinates to display on map")
        return

    # Step 8: Save processed data
    df_selected.sort_values(by="TB3PTSE", ascending=False).to_csv(
        output_csv, index=False
    )
    logger.info(f"Saved processed data to {output_csv}")

    # Step 9: Perform clustering
    labels, cluster_centers_data = cluster_schools(
        df_selected, cluster_radius_km, min_schools
    )

    # Step 10: Extract and index crime data
    df_selected, school_crime_stats = school_data_lib.extract_and_index_crime_data(
        df_selected, school_crime_radius_km
    )

    # Step 11: Create and save map
    create_and_save_map(
        df_selected, labels, cluster_centers_data, school_crime_stats,
        cluster_radius_km, min_schools, school_crime_radius_km,
        percentile, map_filename
    )

    # Step 12: Print cluster and crime statistics
    print_cluster_statistics(
        df_selected, labels, cluster_centers_data,
        school_crime_stats, cluster_radius_km,
        min_schools, school_crime_radius_km
    )


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
