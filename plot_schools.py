"""
Generate a static cluster-overview map (schools_map.html).

Runs BallTree clustering over high-performing schools in
processed_school_data.csv and emits a self-contained Leaflet HTML showing
every school plus translucent cluster-area circles and cluster-centroid
markers. Cluster postcodes are reverse-geocoded via AWS Location Services.

Prerequisite: run generate_school_data.py first to produce
processed_school_data.csv and the cache files.
"""

import os
import math
import json
import logging
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
import boto3

import school_data_lib


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("plot_schools.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

config = school_data_lib.load_config()


def reverse_geocode_location(lat, lon, index_name=None, region_name=None):
    """Reverse-geocode (lat, lon) to a postcode via Amazon Location Services."""
    import re

    if index_name is None:
        index_name = config["geocoding"]["index_name"]
    if region_name is None:
        region_name = config["geocoding"]["region_name"]

    try:
        client = boto3.client("location", region_name=region_name)
        response = client.search_place_index_for_position(
            IndexName=index_name,
            Position=[lon, lat],
            MaxResults=1,
        )
        if response["Results"]:
            place = response["Results"][0]["Place"]
            if "PostalCode" in place:
                return place["PostalCode"]
            if "Label" in place:
                m = re.search(r"[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}", place["Label"])
                if m:
                    return m.group()
        return None
    except Exception as e:
        logger.warning(f"Reverse geocode failed for ({lat:.4f}, {lon:.4f}): {e}")
        return None


def cluster_schools(df, max_distance_km, min_schools=2):
    """
    Identify geographic clusters of schools via a vectorised pairwise
    haversine distance matrix, then a greedy cover over dense candidates.

    Returns:
        (labels, cluster_centers) where labels[i] is the cluster id for the
        i-th row of df (-1 if unclustered), and cluster_centers is a list of
        dicts describing each cluster's centroid and member indices.
    """
    if len(df) == 0:
        return np.array([]), []

    df_reset = df.reset_index(drop=True)
    lat = np.radians(df_reset["Latitude"].values)
    lon = np.radians(df_reset["Longitude"].values)

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    distances_km = 2.0 * school_data_lib.EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    within = distances_km <= max_distance_km

    candidates = []
    for idx in range(len(df_reset)):
        members = np.nonzero(within[idx])[0].tolist()
        if len(members) >= min_schools:
            candidates.append({"schools": members, "count": len(members)})

    logger.info(f"Found {len(candidates)} candidate centres from {len(df_reset)} schools")

    if not candidates:
        return np.full(len(df_reset), -1), []

    # Greedy cover: keep dense candidates first, drop candidates that would
    # just re-cover the same schools.
    candidates.sort(key=lambda c: c["count"], reverse=True)
    final = []
    used = set()
    for c in candidates:
        new_members = set(c["schools"]) - used
        if len(new_members) >= min_schools:
            final.append(c)
            used.update(c["schools"])

    logger.info(f"After deduplication: {len(final)} clusters")

    labels = np.full(len(df_reset), -1)
    for cluster_id, c in enumerate(final):
        for i in c["schools"]:
            labels[i] = cluster_id

    # Recompute centroids on the Cartesian unit sphere for accuracy.
    refined = []
    for cluster_id, c in enumerate(final):
        members = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        if not members:
            continue

        xs, ys, zs = [], [], []
        for pos in members:
            lat_r = math.radians(df_reset.iloc[pos]["Latitude"])
            lon_r = math.radians(df_reset.iloc[pos]["Longitude"])
            xs.append(math.cos(lat_r) * math.cos(lon_r))
            ys.append(math.cos(lat_r) * math.sin(lon_r))
            zs.append(math.sin(lat_r))

        mx, my, mz = np.mean(xs), np.mean(ys), np.mean(zs)
        lon_deg = math.degrees(math.atan2(my, mx))
        lat_deg = math.degrees(math.atan2(mz, math.sqrt(mx ** 2 + my ** 2)))

        refined.append({
            "cluster_id": cluster_id,
            "lat": lat_deg,
            "lon": lon_deg,
            "count": len(members),
            "schools": members,
            "postcode": reverse_geocode_location(lat_deg, lon_deg),
        })

    clustered = int(np.sum(labels != -1))
    logger.info(f"Clustered {clustered}/{len(df_reset)} schools into {len(refined)} clusters")
    return labels, refined


def _clean(value, default=""):
    if pd.isna(value):
        return default
    s = str(value)
    return default if s.lower() == "nan" else s


def build_schools_payload(df, geocoding_cache, crime_cache, labels):
    """
    Build the JSON-serialisable list of school objects for the HTML payload.

    Mirrors the structure used by create_standalone_app.py so popup logic on
    the client can be near-identical.
    """
    schools = []
    crime_counts = []
    per_school_crime = []

    for pos, (_, row) in enumerate(df.iterrows()):
        address_parts = [
            _clean(row[f]) for f in ("ADDRESS1", "TOWN", "PCODE")
            if _clean(row[f])
        ]
        full_address = ", ".join(address_parts) if address_parts else None

        lat, lon = None, None
        if full_address and full_address in geocoding_cache:
            coords = geocoding_cache[full_address]
            if coords and len(coords) == 2:
                lat, lon = coords[0], coords[1]
        if lat is None and "Latitude" in row and pd.notna(row["Latitude"]):
            lat, lon = float(row["Latitude"]), float(row["Longitude"])
        if lat is None:
            logger.warning(f"No coordinates for {row['SCHNAME']}; skipping")
            continue

        crime_key = f"{lat:.6f},{lon:.6f},3"
        crime_stats = crime_cache.get(crime_key)
        crime_count = crime_stats.get("total_crimes", 0) if crime_stats else 0
        crime_counts.append(crime_count)
        per_school_crime.append((crime_stats, crime_count))

        label = int(labels[pos]) if pos < len(labels) else -1
        schools.append({
            "SCHNAME": row["SCHNAME"],
            "ADDRESS1": _clean(row["ADDRESS1"]),
            "TOWN": _clean(row["TOWN"]),
            "PCODE": _clean(row["PCODE"]),
            "TELNUM": _clean(row["TELNUM"]),
            "ADMPOL_PT": _clean(row["ADMPOL_PT"]),
            "GEND1618": _clean(row["GEND1618"]),
            "AGERANGE": _clean(row["AGERANGE"]),
            "TPUP1618": float(row["TPUP1618"]) if pd.notna(row["TPUP1618"]) else 0,
            "TALLPPE_ALEV_1618": float(row["TALLPPE_ALEV_1618"]) if pd.notna(row["TALLPPE_ALEV_1618"]) else 0,
            "TB3PTSE": float(row["TB3PTSE"]) if pd.notna(row["TB3PTSE"]) else 0,
            "YEAR_SCORES": _clean(row.get("YEAR_SCORES")),
            "Latitude": lat,
            "Longitude": lon,
            "cluster_id": label if label != -1 else None,
        })

    crime_indices = pd.Series(crime_counts).rank(pct=True).tolist() if crime_counts else []
    for i, school in enumerate(schools):
        stats, count = per_school_crime[i]
        if stats:
            school["crime_stats"] = stats
            school["crime_count"] = count
            school["crime_index"] = crime_indices[i]

    return schools


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UK Schools Cluster Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; height: 100vh; overflow: hidden; }
#map { width: 100%; height: 100%; }

.title-bar {
    position: absolute; top: 10px; left: 50px; right: 10px;
    background: white; padding: 10px 15px; border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3); z-index: 1000;
    font-size: 13px; text-align: center;
}
.title-bar h1 { font-size: 16px; margin-bottom: 4px; font-weight: 600; }
.title-bar a { color: #4169E1; text-decoration: none; }
.title-bar a:hover { text-decoration: underline; }
.title-bar .sub { font-size: 11px; color: #666; }

.legend {
    position: absolute; bottom: 20px; right: 10px;
    background: white; padding: 10px 12px; border-radius: 6px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3); z-index: 1000;
    font-size: 11px; max-height: 60vh; overflow-y: auto;
}
.legend-section { font-weight: 600; margin: 6px 0 4px; font-size: 10px; text-transform: uppercase; color: #444; }
.legend-section:first-child { margin-top: 0; }
.legend-row { display: flex; align-items: center; margin-bottom: 3px; }
.legend-swatch { display: inline-block; width: 10px; height: 10px; margin-right: 6px; border: 1px solid rgba(0,0,0,0.3); }

.leaflet-popup-content { margin: 10px; min-width: 300px; max-width: 400px; font-size: 9pt; }
.popup-header { font-weight: bold; font-size: 9pt; margin-bottom: 8px; color: #333; grid-column: 1 / -1; }
.popup-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; }
.popup-section { display: flex; flex-direction: column; }
.popup-section.full-width { grid-column: 1 / -1; }
.popup-label { font-weight: 600; color: #666; font-size: 9pt; margin-bottom: 2px; }
.popup-value { color: #333; font-size: 9pt; line-height: 1.3; }
.grade-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-weight: bold; font-size: 9pt; margin-left: 4px; }
.grade-a-star { background: #FFD700; color: #000; }
.grade-a { background: #FFA500; color: #000; }
.grade-b { background: #87CEEB; color: #000; }

@media (max-width: 768px) {
    .title-bar { left: 5px; right: 5px; top: 5px; padding: 8px 10px; font-size: 11px; }
    .title-bar h1 { font-size: 13px; }
    .legend { bottom: 10px; right: 5px; font-size: 10px; max-width: 45vw; }
}
</style>
</head>
<body>
<div id="map"></div>
<div class="title-bar" id="titleBar"></div>
<div class="legend" id="legend"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const SCHOOLS = __SCHOOLS__;
const CLUSTERS = __CLUSTERS__;
const PARAMS = __PARAMS__;

const map = L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19
}).addTo(map);
L.control.scale({ position: 'bottomleft', metric: true, imperial: false, maxWidth: 150 }).addTo(map);

if (SCHOOLS.length > 0) {
    const bounds = L.latLngBounds(SCHOOLS.map(s => [s.Latitude, s.Longitude]));
    map.fitBounds(bounds, { padding: [30, 30] });
} else {
    map.setView([52.4862, -1.8904], 7);
}

function markerColor(score, isIndependent) {
    const c = PARAMS.colors[isIndependent ? 'independent' : 'state'];
    if (score >= PARAMS.a_star_threshold) return c.a_star;
    if (score >= PARAMS.a_threshold) return c.a;
    return c.b_or_below;
}
function textColor(bg) {
    const h = bg.replace('#', '');
    const r = parseInt(h.substr(0,2), 16);
    const g = parseInt(h.substr(2,2), 16);
    const b = parseInt(h.substr(4,2), 16);
    return (r*299 + g*587 + b*114)/1000 > 128 ? '#000' : '#fff';
}
function grade(score) {
    if (!score) return 'N/A';
    if (score >= PARAMS.a_star_threshold) return 'A*';
    if (score >= PARAMS.a_threshold) return 'A';
    return '\u2264B';
}

function schoolIcon(school) {
    const isIndependent = !school.ADMPOL_PT || !school.ADMPOL_PT.trim();
    const color = markerColor(school.TB3PTSE, isIndependent);
    const txt = textColor(color);
    const score = Math.round(school.TB3PTSE);
    return L.divIcon({
        className: '',
        iconSize: [30, 40],
        iconAnchor: [15, 40],
        popupAnchor: [0, -38],
        html: `<svg width="30" height="40" viewBox="0 0 24 32" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 0C7.58 0 4 3.58 4 8c0 5.76 8 16 8 16s8-10.24 8-16c0-4.42-3.58-8-8-8z"
                  fill="${color}" stroke="black" stroke-width="1"/>
            <text x="12" y="10" font-family="Arial" font-size="7" font-weight="bold"
                  fill="${txt}" text-anchor="middle">${score}</text>
        </svg>`
    });
}

function clusterIcon(c) {
    return L.divIcon({
        className: '',
        iconSize: [24, 24],
        iconAnchor: [12, 12],
        popupAnchor: [0, -12],
        html: `<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" fill="${PARAMS.colors.cluster.center_marker}" stroke="#000" stroke-width="2"/>
            <text x="12" y="12" font-family="Arial" font-size="10" font-weight="bold"
                  fill="#fff" text-anchor="middle" dy=".3em">${c.cluster_id}</text>
        </svg>`
    });
}

function gradeClass(g) {
    return g === 'A*' ? 'a-star' : (g === 'A' ? 'a' : 'b');
}

function schoolPopup(s) {
    const isIndependent = !s.ADMPOL_PT || !s.ADMPOL_PT.trim();
    const type = isIndependent ? 'Independent' : s.ADMPOL_PT;
    const addrParts = [s.ADDRESS1, s.TOWN, s.PCODE].filter(p => p && p.trim());
    const fullAddress = addrParts.length ? addrParts.join(', ') : 'Address not available';
    const tb3Grade = grade(s.TB3PTSE);
    const tallppeGrade = grade(s.TALLPPE_ALEV_1618);

    let html = `
        <div class="popup-header">${s.SCHNAME}</div>
        <div class="popup-grid">
            <div class="popup-section full-width">
                <div class="popup-label">Address</div>
                <div class="popup-value">${fullAddress}</div>
            </div>
            <div class="popup-section">
                <div class="popup-label">Type</div>
                <div class="popup-value">${type}</div>
            </div>
            <div class="popup-section">
                <div class="popup-label">Gender</div>
                <div class="popup-value">${s.GEND1618}</div>
            </div>
            <div class="popup-section">
                <div class="popup-label">Age Range</div>
                <div class="popup-value">${s.AGERANGE}</div>
            </div>
            <div class="popup-section">
                <div class="popup-label">Students (16-18)</div>
                <div class="popup-value">${s.TPUP1618}</div>
            </div>
            <div class="popup-section">
                <div class="popup-label">Contact</div>
                <div class="popup-value">\u260E ${s.TELNUM}</div>
            </div>
            <div class="popup-section">
                <div class="popup-label">Cluster</div>
                <div class="popup-value">${(s.cluster_id === null || s.cluster_id === undefined) ? '\u2014' : s.cluster_id}</div>
            </div>
            <div class="popup-section">
                <div class="popup-label">TB3PTSE (Best 3 A-levels)</div>
                <div class="popup-value">${s.TB3PTSE.toFixed(2)} <span class="grade-badge grade-${gradeClass(tb3Grade)}">${tb3Grade}</span></div>
            </div>
            <div class="popup-section">
                <div class="popup-label">TALLPPE (Per A-level)</div>
                <div class="popup-value">${s.TALLPPE_ALEV_1618.toFixed(2)} <span class="grade-badge grade-${gradeClass(tallppeGrade)}">${tallppeGrade}</span></div>
            </div>`;

    if (s.YEAR_SCORES) {
        html += `
            <div class="popup-section full-width">
                <div class="popup-label">Year-by-Year TB3PTSE</div>
                <div class="popup-value">${s.YEAR_SCORES.replace(/<br>/g, ', ')}</div>
            </div>`;
    }

    if (s.crime_stats) {
        const idx = (s.crime_index || 0).toFixed(2);
        let safety = `${s.crime_stats.total_crimes} serious crimes (Index: ${idx})`;
        if (s.crime_stats.crime_types && Object.keys(s.crime_stats.crime_types).length > 0) {
            const top3 = Object.entries(s.crime_stats.crime_types)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3)
                .map(([t, n]) => `${t}: ${n}`)
                .join(', ');
            safety += ' \u2014 ' + top3;
        }
        html += `
            <div class="popup-section full-width">
                <div class="popup-label">Local Safety (${PARAMS.school_crime_radius_km}km)</div>
                <div class="popup-value">${safety}</div>
            </div>`;
    }

    html += '</div>';
    return html;
}

function clusterPopup(c) {
    const pc = c.postcode ? ` (${c.postcode})` : '';
    return `
        <div class="popup-header">Cluster ${c.cluster_id}${pc}</div>
        <div class="popup-grid">
            <div class="popup-section">
                <div class="popup-label">Schools</div>
                <div class="popup-value">${c.count}</div>
            </div>
            <div class="popup-section">
                <div class="popup-label">Radius</div>
                <div class="popup-value">${PARAMS.cluster_radius_km} km</div>
            </div>
            <div class="popup-section full-width">
                <div class="popup-label">Centre</div>
                <div class="popup-value">${c.lat.toFixed(4)}, ${c.lon.toFixed(4)}</div>
            </div>
        </div>`;
}

// Draw cluster circles first, then cluster centre markers, then school markers on top.
CLUSTERS.forEach(c => {
    L.circle([c.lat, c.lon], {
        radius: PARAMS.cluster_radius_km * 1000,
        color: PARAMS.colors.cluster.circle,
        fillColor: PARAMS.colors.cluster.circle,
        fillOpacity: 0.15,
        weight: 2
    }).bindPopup(clusterPopup(c)).addTo(map);
});
CLUSTERS.forEach(c => {
    L.marker([c.lat, c.lon], { icon: clusterIcon(c), zIndexOffset: 500 })
        .bindPopup(clusterPopup(c)).addTo(map);
});
SCHOOLS.forEach(s => {
    L.marker([s.Latitude, s.Longitude], { icon: schoolIcon(s), zIndexOffset: 1000 })
        .bindPopup(schoolPopup(s)).addTo(map);
});

document.getElementById('titleBar').innerHTML = `
    <h1><a href="https://www.compare-school-performance.service.gov.uk/download-data" target="_blank">
        Schools by A-Level Performance &mdash; Multi-Year Average (${(PARAMS.percentile*100).toFixed(0)}th percentile)
    </a></h1>
    <div class="sub">Clustered within ${PARAMS.cluster_radius_km}km radius (min ${PARAMS.min_schools} schools). Crime index is percentile over cohort, ${PARAMS.school_crime_radius_km}km radius.
    ${SCHOOLS.length} schools, ${CLUSTERS.length} clusters.</div>
`;

const c = PARAMS.colors;
document.getElementById('legend').innerHTML = `
    <div class="legend-section">Independent</div>
    <div class="legend-row"><span class="legend-swatch" style="background:${c.independent.a_star}"></span>A* (\u2265${PARAMS.a_star_threshold})</div>
    <div class="legend-row"><span class="legend-swatch" style="background:${c.independent.a}"></span>A (${PARAMS.a_threshold}\u2013${PARAMS.a_star_threshold-1})</div>
    <div class="legend-row"><span class="legend-swatch" style="background:${c.independent.b_or_below}"></span>\u2264B (<${PARAMS.a_threshold})</div>
    <div class="legend-section">State</div>
    <div class="legend-row"><span class="legend-swatch" style="background:${c.state.a_star}"></span>A* (\u2265${PARAMS.a_star_threshold})</div>
    <div class="legend-row"><span class="legend-swatch" style="background:${c.state.a}"></span>A (${PARAMS.a_threshold}\u2013${PARAMS.a_star_threshold-1})</div>
    <div class="legend-row"><span class="legend-swatch" style="background:${c.state.b_or_below}"></span>\u2264B (<${PARAMS.a_threshold})</div>
    <div class="legend-section">Clusters</div>
    <div class="legend-row"><span class="legend-swatch" style="background:${c.cluster.circle}; opacity:0.5"></span>Cluster area</div>
    <div class="legend-row"><span class="legend-swatch" style="background:${c.cluster.center_marker}; border-radius:50%"></span>Cluster centre</div>
`;
</script>
</body>
</html>
"""


def render_map(schools, clusters, cluster_radius_km, min_schools,
               school_crime_radius_km, percentile, output_path):
    """Render the Leaflet HTML and write it to disk."""
    params = {
        "cluster_radius_km": cluster_radius_km,
        "min_schools": min_schools,
        "school_crime_radius_km": school_crime_radius_km,
        "percentile": percentile,
        "a_star_threshold": config["grading"]["a_star_threshold"],
        "a_threshold": config["grading"]["a_threshold"],
        "colors": config["colors"],
    }

    html = (
        HTML_TEMPLATE
        .replace("__SCHOOLS__", json.dumps(schools, ensure_ascii=False))
        .replace("__CLUSTERS__", json.dumps(clusters, ensure_ascii=False))
        .replace("__PARAMS__", json.dumps(params, ensure_ascii=False))
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Saved {output_path} ({size_mb:.2f} MB)")


def print_cluster_statistics(df, labels, cluster_centers,
                             cluster_radius_km, min_schools,
                             school_crime_radius_km):
    """Log per-cluster and overall crime statistics."""
    clusters = defaultdict(list)
    for pos, label in enumerate(labels):
        if label != -1:
            clusters[int(label)].append(df.iloc[pos])

    unclustered = int((labels == -1).sum())
    logger.info(
        f"\nClusters within {cluster_radius_km}km (min {min_schools} schools): {len(clusters)}"
    )
    logger.info(
        f"Unclustered schools: {unclustered} of {len(df)} "
        f"({unclustered / max(len(df), 1):.1%})"
    )

    for cluster_id in sorted(clusters):
        members = clusters[cluster_id]
        centre = next(
            (c for c in cluster_centers if c["cluster_id"] == cluster_id), {}
        )
        pc = centre.get("postcode")
        pc_txt = f" ({pc})" if pc else ""
        logger.info(f"\nCluster {cluster_id}{pc_txt}:")

        crime_counts = [s["crime_count"] for s in members if "crime_count" in s]
        if crime_counts:
            logger.info(
                f"  Mean school-level crime ({school_crime_radius_km}km): "
                f"{np.mean(crime_counts):.1f}"
            )

        for s in members:
            if "crime_count" in s and "crime_index" in s:
                logger.info(
                    f"- {s['SCHNAME']} ({s['TOWN']}) - "
                    f"{s['crime_count']} crimes (Index: {s['crime_index']:.2f})"
                )
            else:
                logger.info(f"- {s['SCHNAME']} ({s['TOWN']}) - Crime data unavailable")

    if "crime_count" in df.columns:
        cc = df["crime_count"]
        logger.info(
            f"\nOverall crime (per school, {school_crime_radius_km}km radius): "
            f"max {cc.max()}, min {cc.min()}, mean {cc.mean():.1f}"
        )


def load_cache_files(crime_cache_path, geocoding_cache_path):
    """Load cache JSON files, stripping metadata entries."""
    crime_cache, geocoding_cache = {}, {}

    if os.path.exists(crime_cache_path):
        with open(crime_cache_path, "r") as f:
            crime_cache = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        logger.info(f"Loaded {len(crime_cache)} crime cache entries")
    else:
        logger.warning(f"Crime cache not found at {crime_cache_path}")

    if os.path.exists(geocoding_cache_path):
        with open(geocoding_cache_path, "r") as f:
            geocoding_cache = json.load(f)
        logger.info(f"Loaded {len(geocoding_cache)} geocoding cache entries")
    else:
        logger.warning(f"Geocoding cache not found at {geocoding_cache_path}")

    return crime_cache, geocoding_cache


def main(cluster_radius_km=None, min_schools=None,
         input_csv=None, output_html=None):
    """Load processed data, cluster, render cluster map, print stats."""
    try:
        school_data_lib.validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

    if cluster_radius_km is None:
        cluster_radius_km = config["clustering"]["default_cluster_radius_km"]
    if min_schools is None:
        min_schools = config["clustering"]["default_min_schools"]

    input_csv = input_csv or config["output"]["processed_csv"]
    output_html = output_html or config["output"]["map_filename"]
    school_crime_radius_km = config["crime"]["school_crime_radius_km"]
    percentile = config["filtering"]["percentile"]
    crime_cache_path = config["caching"]["crime_cache_file"]
    geocoding_cache_path = config["caching"]["geocoding_cache_file"]

    if not os.path.exists(input_csv):
        logger.error(
            f"Processed data not found at {input_csv}. "
            f"Run generate_school_data.py first."
        )
        return

    crime_cache, geocoding_cache = load_cache_files(crime_cache_path, geocoding_cache_path)
    df = pd.read_csv(input_csv, dtype={"TELNUM": str})
    logger.info(f"Loaded {len(df)} schools from {input_csv}")

    if "Latitude" not in df.columns or df["Latitude"].isna().all():
        logger.error("Processed CSV has no Latitude column; re-run generate_school_data.py")
        return

    df = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    labels, cluster_centers = cluster_schools(df, cluster_radius_km, min_schools)

    schools = build_schools_payload(df, geocoding_cache, crime_cache, labels)

    # Attach crime stats/counts back onto df rows for the stats printer.
    by_name = {s["SCHNAME"]: s for s in schools}
    for col in ("crime_count", "crime_index"):
        df[col] = [by_name.get(n, {}).get(col) for n in df["SCHNAME"]]

    clusters_payload = [
        {
            "cluster_id": c["cluster_id"],
            "lat": c["lat"],
            "lon": c["lon"],
            "count": c["count"],
            "postcode": c.get("postcode"),
        }
        for c in cluster_centers
    ]

    render_map(
        schools, clusters_payload,
        cluster_radius_km, min_schools,
        school_crime_radius_km, percentile,
        output_html,
    )
    print_cluster_statistics(
        df, labels, cluster_centers,
        cluster_radius_km, min_schools, school_crime_radius_km,
    )


if __name__ == "__main__":
    default_radius = config["clustering"]["default_cluster_radius_km"]
    default_min_schools = config["clustering"]["default_min_schools"]

    parser = argparse.ArgumentParser(
        description="Generate the static cluster-overview map (schools_map.html)."
    )
    parser.add_argument("--radius", type=float, default=default_radius,
                        help=f"Cluster radius in km (default: {default_radius})")
    parser.add_argument("--min-schools", type=int, default=default_min_schools,
                        help=f"Minimum schools per cluster (default: {default_min_schools})")
    parser.add_argument("--input", default=None,
                        help="Processed school data CSV (default: from config)")
    parser.add_argument("--output", default=None,
                        help="Output HTML file (default: from config)")
    args = parser.parse_args()

    main(args.radius, args.min_schools, args.input, args.output)
