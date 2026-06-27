"""
Create a standalone HTML with both KS5 (secondary) and KS2 (primary) school data
embedded in one file.

Reads:
  - processed_school_data.csv + crime_cache.json + geocoding_cache.json  (KS5)
  - ks2_processed_school_data.csv + ks2_crime_cache.json + ks2_geocoding_cache.json (KS2)
  - config.json + ks2_config.json

Emits:
  - school_finder.html

Both pipelines' data prep logic is adapted from their single-pipeline counterparts.
The HTML template uses marker shape to distinguish school types:
  - KS5 (secondary): teardrop pin with TB3PTSE score
  - KS2 (primary):   circle with % meeting expected standard
"""

import os
import json
import logging
import argparse

import pandas as pd

import ks2_school_data_lib
import times_ranking


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_json(path, label):
    if not os.path.exists(path):
        logger.warning(f"{label} not found: {path}")
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Loaded {label}: {len(data)} entries from {path}")
        return data
    except Exception as e:
        logger.warning(f"Could not load {label} ({path}): {e}")
        return {}


def _clean(val, default=""):
    if pd.isna(val):
        return default
    s = str(val)
    return default if s.lower() == "nan" else s


def _clean_int(val, default=""):
    """Return integer string (drops .0). Upstream pipelines coerce integer
    columns to floats when NaNs are present; this restores the display form."""
    if pd.isna(val):
        return default
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        s = str(val).strip()
        return s if s and s.lower() != "nan" else default


def _clean_phone(val, default=""):
    """Return UK phone as a string, restoring leading 0 if the CSV stored it
    as a number (common: 2076603288 → 02076603288)."""
    if pd.isna(val):
        return default
    try:
        s = str(int(float(val)))
    except (ValueError, TypeError):
        s = str(val).strip()
        if s.endswith(".0"):
            s = s[:-2]
    if not s or s.lower() == "nan":
        return default
    if s.isdigit() and not s.startswith("0") and len(s) == 10:
        s = "0" + s
    return s


# ---------------------------------------------------------------------------
# KS5 data loading
# ---------------------------------------------------------------------------

def load_ks5_schools(csv_file, crime_cache, geocoding_cache):
    try:
        df = pd.read_csv(csv_file, dtype={"TELNUM": str})
    except FileNotFoundError:
        logger.warning(f"KS5 CSV not found: {csv_file}")
        return []

    logger.info(f"Loaded {len(df)} KS5 schools from {csv_file}")

    times = times_ranking.load_secondary()
    if not times.loaded:
        logger.warning(
            f"Times secondary CSV ({times_ranking.SECONDARY_CSV}) not found; "
            "skipping Times ranks for secondaries"
        )

    schools = []

    for _, row in df.iterrows():
        addr_parts = []
        for field in ["ADDRESS1", "TOWN", "PCODE"]:
            v = row.get(field)
            if pd.notna(v) and str(v).strip() and str(v).lower() != "nan":
                addr_parts.append(str(v))
        full_addr = ", ".join(addr_parts) if addr_parts else None

        lat = lon = None
        if full_addr and full_addr in geocoding_cache:
            coords = geocoding_cache[full_addr]
            if coords and len(coords) == 2:
                lat, lon = coords[0], coords[1]
        if lat is None and "Latitude" in row and pd.notna(row["Latitude"]):
            lat, lon = float(row["Latitude"]), float(row["Longitude"])
        if lat is None:
            continue

        crime_key = f"{lat:.6f},{lon:.6f},3"
        crime_stats = crime_cache.get(crime_key)
        crime_count = crime_stats.get("total_crimes", 0) if crime_stats else 0

        school = {
            "type": "ks5",
            "SCHNAME": _clean(row.get("SCHNAME")),
            "ADDRESS1": _clean(row.get("ADDRESS1")),
            "TOWN": _clean(row.get("TOWN")),
            "PCODE": _clean(row.get("PCODE")),
            "TELNUM": _clean_phone(row.get("TELNUM")),
            "ADMPOL_PT": _clean(row.get("ADMPOL_PT")),
            "GEND1618": _clean(row.get("GEND1618")),
            "AGERANGE": _clean(row.get("AGERANGE")),
            "TPUP1618": float(row["TPUP1618"]) if pd.notna(row.get("TPUP1618")) else 0,
            "TALLPPE_ALEV_1618": float(row["TALLPPE_ALEV_1618"]) if pd.notna(row.get("TALLPPE_ALEV_1618")) else 0,
            "TB3PTSE": float(row["TB3PTSE"]) if pd.notna(row.get("TB3PTSE")) else 0,
            "YEAR_SCORES": _clean(row.get("YEAR_SCORES")),
            "Latitude": lat,
            "Longitude": lon,
        }
        if crime_stats:
            school["crime_stats"] = crime_stats
            school["crime_count"] = crime_count
        times_rank = times.lookup(row.get("SCHNAME"), row.get("TOWN", ""))
        if times_rank:
            school["times_rank"] = times_rank
        schools.append(school)

    logger.info(f"Prepared {len(schools)} KS5 schools with coordinates")
    return schools


# ---------------------------------------------------------------------------
# KS2 data loading
# ---------------------------------------------------------------------------

def load_ks2_schools(csv_file, crime_cache, geocoding_cache, crime_radius_km):
    try:
        df = pd.read_csv(csv_file, dtype={"TelephoneNum": str})
    except FileNotFoundError:
        logger.warning(f"KS2 CSV not found: {csv_file}")
        return []

    logger.info(f"Loaded {len(df)} KS2 schools from {csv_file}")

    times = times_ranking.load_primary()
    if not times.loaded:
        logger.warning(
            f"Times primary CSV ({times_ranking.PRIMARY_CSV}) not found; "
            "skipping Times ranks for primaries"
        )

    def build_addr(row):
        parts = []
        for f in ["Street", "Locality", "Town", "County", "Postcode"]:
            v = row.get(f, "")
            if pd.notna(v) and str(v).strip() and str(v).lower() != "nan":
                parts.append(str(v).strip())
        return ", ".join(parts) if parts else None

    schools = []

    for _, row in df.iterrows():
        lat = row.get("Latitude") if "Latitude" in row and pd.notna(row.get("Latitude")) else None
        lon = row.get("Longitude") if "Longitude" in row and pd.notna(row.get("Longitude")) else None

        if lat is None:
            addr = build_addr(row)
            if addr and addr in geocoding_cache:
                coords = geocoding_cache[addr]
                if coords and len(coords) == 2:
                    lat, lon = coords[0], coords[1]

        if lat is None:
            continue

        crime_key = f"{float(lat):.6f},{float(lon):.6f},{crime_radius_km}"
        crime_stats = crime_cache.get(crime_key)
        crime_count = crime_stats.get("total_crimes", 0) if crime_stats else 0

        school = {
            "type": "ks2",
            "school_urn": _clean(row.get("school_urn")),
            "school_name": _clean(row.get("school_name")),
            "EstablishmentName": _clean(row.get("EstablishmentName")),
            "Street": _clean(row.get("Street")),
            "Locality": _clean(row.get("Locality")),
            "Town": _clean(row.get("Town")),
            "County": _clean(row.get("County")),
            "Postcode": _clean(row.get("Postcode")),
            "TelephoneNum": _clean_phone(row.get("TelephoneNum")),
            "Gender": _clean(row.get("Gender")),
            "NumberOfPupils": _clean_int(row.get("NumberOfPupils")),
            "StatutoryLowAge": _clean_int(row.get("StatutoryLowAge")),
            "StatutoryHighAge": _clean_int(row.get("StatutoryHighAge")),
            "expected_pct": float(row["expected_pct"]) if pd.notna(row.get("expected_pct")) else 0,
            "higher_pct": float(row["higher_pct"]) if pd.notna(row.get("higher_pct")) else None,
            "year_scores": _clean(row.get("year_scores")),
            "Latitude": float(lat),
            "Longitude": float(lon),
        }
        if crime_stats:
            school["crime_stats"] = crime_stats
            school["crime_count"] = crime_count
        times_rank = times.lookup(
            row.get("EstablishmentName") or row.get("school_name"),
            row.get("Town", ""),
        )
        if times_rank:
            school["times_rank"] = times_rank
        schools.append(school)

    logger.info(f"Prepared {len(schools)} KS2 schools with coordinates")
    return schools


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def get_html_template():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>UK Schools Finder (KS2 + KS5)</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; height: 100vh; overflow: hidden; }
        #map { width: 100%; height: 100%; }

        .controls {
            position: absolute; top: 10px; left: 50px; right: 10px;
            z-index: 1000; display: flex; flex-direction: column; gap: 10px;
        }
        .search-container { display: flex; gap: 10px; flex-wrap: wrap; }
        .search-input-wrapper { flex: 1; min-width: 200px; display: flex; gap: 5px; }

        #addressInput {
            flex: 1; padding: 12px 15px; border: 2px solid #ddd; border-radius: 8px;
            font-size: 16px; background: white; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
        #addressInput:focus { outline: none; border-color: #4169E1; }

        button {
            padding: 12px 20px; border: none; border-radius: 8px; font-size: 16px;
            font-weight: 600; cursor: pointer; background: white; color: #333;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3); transition: all 0.2s; white-space: nowrap;
        }
        button:hover { background: #f0f0f0; }
        button:active { transform: scale(0.98); }
        .btn-primary { background: #4169E1; color: white; }
        .btn-primary:hover { background: #3557c9; }
        .btn-location { background: #005C29; color: white; }
        .btn-location:hover { background: #004820; }

        .info-panel {
            background: white; padding: 12px 15px; border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 14px;
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 10px;
        }
        .info-panel .status { color: #666; }
        .info-panel .count { font-weight: bold; }
        .info-panel .count-ks5 { color: #4169E1; }
        .info-panel .count-ks2 { color: #228B22; }

        .loading {
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            background: white; padding: 20px 40px; border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3); z-index: 2000; font-size: 16px; display: none;
        }
        .loading.show { display: block; }

        .error {
            background: #ff4444; color: white; padding: 12px 15px; border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3); display: none;
        }
        .error.show { display: block; }

        .school-detail {
            position: fixed; bottom: 10px; right: 10px; width: 340px;
            max-width: calc(100vw - 20px); max-height: 60vh; overflow-y: auto;
            background: white; padding: 15px; border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3); z-index: 1000; display: none; font-size: 9pt;
        }
        .school-detail.show { display: block; }
        .school-detail .close-btn {
            position: absolute; top: 10px; right: 10px; background: #ff4444; color: white;
            border: none; border-radius: 50%; width: 24px; height: 24px; font-size: 16px;
            line-height: 1; cursor: pointer; padding: 0; box-shadow: none;
        }
        .school-detail .close-btn:hover { background: #cc0000; }

        .legend {
            position: fixed; bottom: 10px; left: 10px; background: white;
            padding: 10px 14px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            font-size: 11px; z-index: 1000; min-width: 180px;
        }
        .legend-section { margin-bottom: 8px; }
        .legend-section:last-child { margin-bottom: 0; }
        .legend-title { font-weight: bold; margin-bottom: 4px; font-size: 11px; color: #333; }
        .legend-item { display: flex; align-items: center; margin-bottom: 3px; font-size: 10px; }
        .legend-shape { width: 14px; height: 14px; margin-right: 6px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; }
        .legend-circle { width: 10px; height: 10px; border-radius: 50%; border: 1px solid #333; }
        .legend-teardrop { width: 10px; height: 14px; border: 1px solid #333; border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%; clip-path: polygon(50% 100%, 0 40%, 50% 0, 100% 40%); }

        @media (max-width: 768px) {
            .controls { top: 5px; left: 5px; right: 5px; }
            .search-container { flex-direction: column; }
            .search-input-wrapper { min-width: 100%; }
            button { width: 100%; }
            .info-panel { font-size: 12px; padding: 10px; }
            .school-detail { width: calc(100vw - 20px); max-height: 50vh; bottom: 5px; right: 5px; }
            .legend { display: none; }
        }

        .leaflet-popup-content { margin: 10px; min-width: 280px; max-width: 380px; font-size: 9pt; }
        .popup-header { font-weight: bold; font-size: 10pt; margin-bottom: 8px; color: #333; }
        .popup-subheader { font-size: 8pt; color: #888; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        .popup-times-rank { font-weight: bold; font-size: 8pt; color: #555; margin-top: -4px; margin-bottom: 8px; }
        .popup-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
        .popup-section { display: flex; flex-direction: column; }
        .popup-section.full-width { grid-column: 1 / -1; }
        .popup-label { font-weight: 600; color: #666; font-size: 8pt; margin-bottom: 2px; }
        .popup-value { color: #333; font-size: 9pt; line-height: 1.3; }
        .grade-badge {
            display: inline-block; padding: 1px 6px; border-radius: 3px;
            font-weight: bold; font-size: 8pt; margin-left: 4px;
        }
        .grade-a-star { background: #FFD700; color: #000; }
        .grade-a { background: #FFA500; color: #000; }
        .grade-b { background: #87CEEB; color: #000; }
        .grade-high { background: #005C29; color: #fff; }
        .grade-good { background: #228B22; color: #fff; }
        .grade-avg  { background: #8FBC8F; color: #000; }

        .leaflet-control-scale { margin-bottom: 10px !important; }
    </style>
</head>
<body>
    <div id="map"></div>

    <div class="controls">
        <div class="search-container">
            <div class="search-input-wrapper">
                <input type="text" id="addressInput" placeholder="Enter address or postcode...">
                <button id="searchBtn" class="btn-primary">Search</button>
            </div>
            <button id="locationBtn" class="btn-location">📍 My Location</button>
        </div>
        <div class="error" id="errorMsg"></div>
        <div class="info-panel">
            <span class="status" id="statusMsg">Loading school data...</span>
            <span>
                <span class="count count-ks2" id="countKs2">0</span> primaries
                &nbsp;·&nbsp;
                <span class="count count-ks5" id="countKs5">0</span> secondaries
            </span>
        </div>
    </div>

    <div class="loading show" id="loading">Loading school data...</div>

    <div class="school-detail" id="schoolDetail">
        <button class="close-btn" id="closeDetail">×</button>
        <div id="schoolDetailContent"></div>
    </div>

    <div class="legend">
        <div class="legend-section">
            <div class="legend-title">KS5 (secondary, teardrop)</div>
            <div class="legend-item">
                <div class="legend-shape"><div class="legend-circle" style="background:#000080"></div></div>
                State A* (TB3PTSE ≥__KS5_A_STAR__)
            </div>
            <div class="legend-item">
                <div class="legend-shape"><div class="legend-circle" style="background:#FF0000"></div></div>
                Indep A* (TB3PTSE ≥__KS5_A_STAR__)
            </div>
            <div class="legend-item">
                <div class="legend-shape"><div class="legend-circle" style="background:#0000FF"></div></div>
                State A / <span style="color:#FFA500">Indep A</span>
            </div>
        </div>
        <div class="legend-section">
            <div class="legend-title">KS2 (primary, circle)</div>
            <div class="legend-item">
                <div class="legend-shape"><div class="legend-circle" style="background:#005C29"></div></div>
                High (≥__KS2_HIGH__%)
            </div>
            <div class="legend-item">
                <div class="legend-shape"><div class="legend-circle" style="background:#228B22"></div></div>
                Good (≥__KS2_EXPECTED__%)
            </div>
            <div class="legend-item">
                <div class="legend-shape"><div class="legend-circle" style="background:#8FBC8F"></div></div>
                Above average
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const CONFIG = {
            RADIUS_KM: 5,
            MIN_RADIUS_KM: 0.5,
            KS5: {
                A_STAR_THRESHOLD: __KS5_A_STAR__,
                A_THRESHOLD: __KS5_A__,
                COLORS: {
                    independent: { a_star: '#FF0000', a: '#FFA500', b_or_below: '#FFD700' },
                    state:       { a_star: '#000080', a: '#0000FF', b_or_below: '#4169E1' }
                }
            },
            KS2: {
                HIGH_THRESHOLD: __KS2_HIGH__,
                EXPECTED_THRESHOLD: __KS2_EXPECTED__,
                COLORS: {
                    high: '__KS2_COLOR_HIGH__',
                    expected: '__KS2_COLOR_EXPECTED__',
                    below_expected: '__KS2_COLOR_BELOW__'
                }
            }
        };

        let map, schoolsData = [], currentMarkers = [], reticleCircle = null;

        function initMap() {
            map = L.map('map').setView([52.4862, -1.8904], 7);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors', maxZoom: 19
            }).addTo(map);
            L.control.scale({ position: 'bottomleft', metric: true, imperial: false, maxWidth: 150 }).addTo(map);

            const center = map.getCenter();
            reticleCircle = L.circle([center.lat, center.lng], {
                radius: CONFIG.RADIUS_KM * 1000,
                color: '#4169E1', fillColor: '#4169E1', fillOpacity: 0.08, weight: 1.5
            }).addTo(map);

            map.on('moveend', updateSchoolsForCurrentView);
            map.on('move', updateReticlePosition);
            map.on('zoomend', adjustReticleForZoom);
        }

        async function loadSchoolData() {
            try {
                schoolsData = __SCHOOLS_DATA__;
                const ks5 = schoolsData.filter(s => s.type === 'ks5').length;
                const ks2 = schoolsData.filter(s => s.type === 'ks2').length;
                updateStatus(`Loaded ${ks2} primaries and ${ks5} secondaries`);
                showLoading(false);
                updateSchoolsForCurrentView();
            } catch (err) {
                showError('Failed to load school data: ' + err.message);
                showLoading(false);
            }
        }

        function calculateDistance(lat1, lon1, lat2, lon2) {
            const R = 6371;
            const dLat = (lat2 - lat1) * Math.PI / 180;
            const dLon = (lon2 - lon1) * Math.PI / 180;
            const a = Math.sin(dLat/2)**2 +
                      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                      Math.sin(dLon/2)**2;
            return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        }

        function getTextColor(hex) {
            const h = hex.replace('#', '');
            const r = parseInt(h.substr(0, 2), 16);
            const g = parseInt(h.substr(2, 2), 16);
            const b = parseInt(h.substr(4, 2), 16);
            return (r * 299 + g * 587 + b * 114) / 1000 > 128 ? '#000' : '#fff';
        }

        // ---- KS5 marker + popup ----

        function ks5MarkerColor(score, isIndependent) {
            const c = isIndependent ? CONFIG.KS5.COLORS.independent : CONFIG.KS5.COLORS.state;
            if (score >= CONFIG.KS5.A_STAR_THRESHOLD) return c.a_star;
            if (score >= CONFIG.KS5.A_THRESHOLD) return c.a;
            return c.b_or_below;
        }

        function ks5Grade(score) {
            if (!score || score === 0) return 'N/A';
            if (score >= CONFIG.KS5.A_STAR_THRESHOLD) return 'A*';
            if (score >= CONFIG.KS5.A_THRESHOLD) return 'A';
            return '≤B';
        }

        function ks5GradeBadgeClass(grade) {
            return grade === 'A*' ? 'a-star' : grade === 'A' ? 'a' : 'b';
        }

        function createKs5Icon(school) {
            const isIndependent = !school.ADMPOL_PT || school.ADMPOL_PT.trim() === '';
            const color = ks5MarkerColor(school.TB3PTSE, isIndependent);
            const textColor = getTextColor(color);
            const score = Math.round(school.TB3PTSE);
            return L.divIcon({
                html: `<svg width="30" height="40" viewBox="0 0 24 32" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 0C7.58 0 4 3.58 4 8c0 5.76 8 16 8 16s8-10.24 8-16c0-4.42-3.58-8-8-8z"
                          fill="${color}" stroke="black" stroke-width="1"/>
                    <text x="12" y="10" font-family="Arial" font-size="7" font-weight="bold"
                          fill="${textColor}" text-anchor="middle">${score}</text>
                </svg>`,
                className: '',
                iconSize: [30, 40],
                iconAnchor: [15, 40],
                popupAnchor: [150, 0]
            });
        }

        function createKs5Popup(school) {
            const isIndependent = !school.ADMPOL_PT || school.ADMPOL_PT.trim() === '';
            const admpol = isIndependent ? 'Independent' : school.ADMPOL_PT;
            const tb3Grade = ks5Grade(school.TB3PTSE);
            const tallppeGrade = ks5Grade(school.TALLPPE_ALEV_1618);

            const addressParts = [school.ADDRESS1, school.TOWN, school.PCODE].filter(v => v && v.trim());
            const fullAddress = addressParts.length ? addressParts.join(', ') : 'Address not available';

            let html = `
                <div class="popup-subheader">Secondary (KS5)</div>
                <div class="popup-header">${school.SCHNAME}</div>
                ${school.times_rank ? `<div class="popup-times-rank">Times Parent Power rank #${school.times_rank}</div>` : ''}
                <div class="popup-grid">
                    <div class="popup-section full-width">
                        <div class="popup-label">Address</div>
                        <div class="popup-value">${fullAddress}</div>
                    </div>
                    <div class="popup-section">
                        <div class="popup-label">Type</div>
                        <div class="popup-value">${admpol}</div>
                    </div>
                    <div class="popup-section">
                        <div class="popup-label">Gender</div>
                        <div class="popup-value">${school.GEND1618 || '—'}</div>
                    </div>
                    <div class="popup-section">
                        <div class="popup-label">Age range</div>
                        <div class="popup-value">${school.AGERANGE || '—'}</div>
                    </div>
                    <div class="popup-section">
                        <div class="popup-label">Pupils (16-18)</div>
                        <div class="popup-value">${school.TPUP1618 || '—'}</div>
                    </div>
                    <div class="popup-section">
                        <div class="popup-label">Phone</div>
                        <div class="popup-value">${school.TELNUM || '—'}</div>
                    </div>
                    <div class="popup-section">
                        <div class="popup-label">TB3PTSE (Best 3)</div>
                        <div class="popup-value">${school.TB3PTSE.toFixed(2)} <span class="grade-badge grade-${ks5GradeBadgeClass(tb3Grade)}">${tb3Grade}</span></div>
                    </div>
                    <div class="popup-section">
                        <div class="popup-label">TALLPPE (Per)</div>
                        <div class="popup-value">${school.TALLPPE_ALEV_1618.toFixed(2)} <span class="grade-badge grade-${ks5GradeBadgeClass(tallppeGrade)}">${tallppeGrade}</span></div>
                    </div>`;

            if (school.YEAR_SCORES) {
                html += `
                    <div class="popup-section full-width">
                        <div class="popup-label">Year-by-year TB3PTSE</div>
                        <div class="popup-value">${school.YEAR_SCORES.replace(/<br>/g, ', ')}</div>
                    </div>`;
            }

            if (school.crime_stats) {
                const cs = school.crime_stats;
                const ci = school.crime_index != null ? school.crime_index.toFixed(2) : '—';
                let txt = `${cs.total_crimes} serious crimes (Index: ${ci})`;
                if (cs.crime_types && Object.keys(cs.crime_types).length > 0) {
                    const top = Object.entries(cs.crime_types).sort((a, b) => b[1] - a[1]).slice(0, 3);
                    txt += ' — ' + top.map(([t, n]) => `${t}: ${n}`).join(', ');
                }
                html += `
                    <div class="popup-section full-width">
                        <div class="popup-label">Local safety (3km)</div>
                        <div class="popup-value">${txt}</div>
                    </div>`;
            }

            html += '</div>';
            return html;
        }

        // ---- KS2 marker + popup ----

        function ks2MarkerColor(pct) {
            if (pct >= CONFIG.KS2.HIGH_THRESHOLD) return CONFIG.KS2.COLORS.high;
            if (pct >= CONFIG.KS2.EXPECTED_THRESHOLD) return CONFIG.KS2.COLORS.expected;
            return CONFIG.KS2.COLORS.below_expected;
        }

        function ks2Grade(pct) {
            if (!pct || pct === 0) return null;
            if (pct >= CONFIG.KS2.HIGH_THRESHOLD) return { label: `High (≥${CONFIG.KS2.HIGH_THRESHOLD}%)`, cls: 'grade-high' };
            if (pct >= CONFIG.KS2.EXPECTED_THRESHOLD) return { label: `Good (≥${CONFIG.KS2.EXPECTED_THRESHOLD}%)`, cls: 'grade-good' };
            return { label: 'Above avg', cls: 'grade-avg' };
        }

        function createKs2Icon(school) {
            const color = ks2MarkerColor(school.expected_pct);
            const textColor = getTextColor(color);
            const label = Math.round(school.expected_pct) + '%';
            return L.divIcon({
                html: `<svg width="22" height="22" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="11" fill="${color}" stroke="#333" stroke-width="0.8"/>
                    <text x="12" y="15" font-family="Arial" font-size="10" font-weight="bold"
                          fill="${textColor}" text-anchor="middle">${label}</text>
                </svg>`,
                className: '',
                iconSize: [22, 22],
                iconAnchor: [11, 11],
                popupAnchor: [160, -11]
            });
        }

        function createKs2Popup(school) {
            const address = [school.Street, school.Locality, school.Town, school.Postcode]
                .filter(v => v && v.trim()).join(', ') || 'Address not available';
            const grade = ks2Grade(school.expected_pct);
            const gradeBadge = grade ? `<span class="grade-badge ${grade.cls}">${grade.label}</span>` : '';

            let html = `
                <div class="popup-subheader">Primary (KS2)</div>
                <div class="popup-header">${school.EstablishmentName || school.school_name}</div>
                ${school.times_rank ? `<div class="popup-times-rank">Times Parent Power rank #${school.times_rank}</div>` : ''}
                <div class="popup-grid">
                    <div class="popup-section full-width">
                        <div class="popup-label">Address</div>
                        <div class="popup-value">${address}</div>
                    </div>
                    <div class="popup-section">
                        <div class="popup-label">Gender</div>
                        <div class="popup-value">${school.Gender || '—'}</div>
                    </div>
                    <div class="popup-section">
                        <div class="popup-label">Age range</div>
                        <div class="popup-value">${school.StatutoryLowAge || '?'}–${school.StatutoryHighAge || '?'}</div>
                    </div>
                    <div class="popup-section">
                        <div class="popup-label">Pupils</div>
                        <div class="popup-value">${school.NumberOfPupils || '—'}</div>
                    </div>
                    <div class="popup-section">
                        <div class="popup-label">Phone</div>
                        <div class="popup-value">${school.TelephoneNum || '—'}</div>
                    </div>
                    <div class="popup-section full-width">
                        <div class="popup-label">KS2 avg % meeting expected (R+W+M)</div>
                        <div class="popup-value">
                            <strong>${school.expected_pct.toFixed(1)}%</strong>${gradeBadge}
                            ${school.higher_pct ? ` &nbsp;|&nbsp; Higher standard: ${school.higher_pct.toFixed(1)}%` : ''}
                        </div>
                    </div>`;

            if (school.year_scores) {
                html += `
                    <div class="popup-section full-width">
                        <div class="popup-label">Year-by-year</div>
                        <div class="popup-value">${school.year_scores.replace(/<br>/g, ' &nbsp;·&nbsp; ')}</div>
                    </div>`;
            }

            if (school.crime_stats) {
                const cs = school.crime_stats;
                const ci = school.crime_index != null ? school.crime_index.toFixed(2) : '—';
                let crimeText = `${cs.total_crimes} serious crimes in ${cs.radius_km}km (Index: ${ci})`;
                if (cs.crime_types && Object.keys(cs.crime_types).length > 0) {
                    const top = Object.entries(cs.crime_types).sort((a, b) => b[1] - a[1]).slice(0, 3);
                    crimeText += ' — ' + top.map(([t, n]) => `${t}: ${n}`).join(', ');
                }
                html += `
                    <div class="popup-section full-width">
                        <div class="popup-label">Local safety (${cs.radius_km}km)</div>
                        <div class="popup-value">${crimeText}</div>
                    </div>`;
            }

            html += '</div>';
            return html;
        }

        // ---- Dispatch ----

        function createMarkerIcon(school) {
            return school.type === 'ks5' ? createKs5Icon(school) : createKs2Icon(school);
        }

        function createPopupContent(school) {
            return school.type === 'ks5' ? createKs5Popup(school) : createKs2Popup(school);
        }

        function displaySchools(centerLat, centerLon, isSearch = false, radiusKm = CONFIG.RADIUS_KM) {
            currentMarkers.forEach(m => map.removeLayer(m));
            currentMarkers = [];

            reticleCircle.setLatLng([centerLat, centerLon]);

            const nearby = schoolsData.filter(s =>
                calculateDistance(centerLat, centerLon, s.Latitude, s.Longitude) <= radiusKm
            );

            let nKs2 = 0, nKs5 = 0;
            nearby.forEach(school => {
                const marker = L.marker([school.Latitude, school.Longitude], {
                    icon: createMarkerIcon(school)
                }).addTo(map);
                marker.on('click', () => showSchoolDetail(school));
                currentMarkers.push(marker);
                if (school.type === 'ks2') nKs2++; else nKs5++;
            });

            document.getElementById('countKs2').textContent = nKs2;
            document.getElementById('countKs5').textContent = nKs5;
            updateStatus(`${nearby.length} schools within ${radiusKm.toFixed(1)}km`);

            if (isSearch) {
                map.setView([centerLat, centerLon], 12);
            }
        }

        function updateReticlePosition() {
            if (reticleCircle && map) {
                const c = map.getCenter();
                reticleCircle.setLatLng([c.lat, c.lng]);
            }
        }

        function adjustReticleForZoom() {
            if (!map || !reticleCircle) return;
            const mapSize = map.getSize();
            const minDim = Math.min(mapSize.x, mapSize.y);
            const maxCirclePx = minDim * 0.6;
            const center = map.getCenter();
            const zoom = map.getZoom();
            const mPerPx = 40075016.686 * Math.cos(center.lat * Math.PI / 180) / Math.pow(2, zoom + 8);
            const maxRadiusM = (maxCirclePx / 2) * mPerPx;
            const constrained = Math.min(CONFIG.RADIUS_KM * 1000, maxRadiusM);
            const actual = Math.max(constrained, CONFIG.MIN_RADIUS_KM * 1000);
            reticleCircle.setRadius(actual);
        }

        function updateSchoolsForCurrentView() {
            if (!map || schoolsData.length === 0) return;
            const c = map.getCenter();
            const radiusKm = reticleCircle.getRadius() / 1000;
            displaySchools(c.lat, c.lng, false, radiusKm);
        }

        async function searchAddress() {
            const addr = document.getElementById('addressInput').value.trim();
            if (!addr) { showError('Please enter an address or postcode'); return; }
            showLoading(true);
            updateStatus('Searching...');
            try {
                const query = addr.includes('UK') ? addr : `${addr}, UK`;
                const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&countrycodes=gb&limit=1`;
                const resp = await fetch(url, { headers: { 'User-Agent': 'UK Schools Finder (Combined)' } });
                if (!resp.ok) throw new Error('Geocoding service unavailable');
                const results = await resp.json();
                showLoading(false);
                if (results.length === 0) {
                    showError('Address not found. Try a different search term.');
                    updateStatus('Ready');
                    return;
                }
                displaySchools(parseFloat(results[0].lat), parseFloat(results[0].lon), true);
            } catch (err) {
                showLoading(false);
                showError('Search failed: ' + err.message);
                updateStatus('Ready');
            }
        }

        function useCurrentLocation() {
            if (!navigator.geolocation) { showError('Geolocation not supported'); return; }
            showLoading(true);
            updateStatus('Getting location...');
            navigator.geolocation.getCurrentPosition(
                pos => { showLoading(false); displaySchools(pos.coords.latitude, pos.coords.longitude, true); },
                err => {
                    showLoading(false);
                    const msgs = { 1: 'Location permission denied.', 2: 'Location unavailable.', 3: 'Location request timed out.' };
                    showError(msgs[err.code] || 'Unable to get location');
                },
                { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
            );
        }

        function showLoading(show) { document.getElementById('loading').classList.toggle('show', show); }
        function showError(msg) {
            const el = document.getElementById('errorMsg');
            el.textContent = msg; el.classList.add('show');
            setTimeout(() => el.classList.remove('show'), 5000);
        }
        function updateStatus(msg) { document.getElementById('statusMsg').textContent = msg; }
        function showSchoolDetail(school) {
            document.getElementById('schoolDetailContent').innerHTML = createPopupContent(school);
            document.getElementById('schoolDetail').classList.add('show');
        }

        document.getElementById('searchBtn').addEventListener('click', searchAddress);
        document.getElementById('locationBtn').addEventListener('click', useCurrentLocation);
        document.getElementById('addressInput').addEventListener('keypress', e => { if (e.key === 'Enter') searchAddress(); });
        document.getElementById('closeDetail').addEventListener('click', () => {
            document.getElementById('schoolDetail').classList.remove('show');
        });

        initMap();
        loadSchoolData();
    </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def create_combined_html(
    ks5_csv="processed_school_data.csv",
    ks5_crime_cache="crime_cache.json",
    ks5_geocoding_cache="geocoding_cache.json",
    ks5_config="config.json",
    ks2_csv="ks2_processed_school_data.csv",
    ks2_crime_cache="ks2_crime_cache.json",
    ks2_geocoding_cache="ks2_geocoding_cache.json",
    ks2_config="ks2_config.json",
    output_file="school_finder.html",
):
    logger.info("Building combined KS2 + KS5 standalone HTML...")

    # Load configs
    ks2_cfg = ks2_school_data_lib.load_config(ks2_config)
    ks2_high = ks2_cfg["grading"]["high_threshold"]
    ks2_expected = ks2_cfg["grading"]["expected_threshold"]
    ks2_colors = ks2_cfg["colors"]
    ks2_crime_radius = ks2_cfg["crime"]["school_crime_radius_km"]

    ks5_a_star = 50
    ks5_a = 40
    if os.path.exists(ks5_config):
        try:
            with open(ks5_config) as f:
                cfg = json.load(f)
            ks5_a_star = cfg.get("grading", {}).get("a_star_threshold", ks5_a_star)
            ks5_a = cfg.get("grading", {}).get("a_threshold", ks5_a)
        except Exception as e:
            logger.warning(f"Could not read {ks5_config}: {e}; using defaults")

    # Load caches
    ks5_crime = _load_json(ks5_crime_cache, "KS5 crime cache")
    ks5_crime = {k: v for k, v in ks5_crime.items() if not k.startswith("_")}
    ks5_geo = _load_json(ks5_geocoding_cache, "KS5 geocoding cache")
    ks2_crime = _load_json(ks2_crime_cache, "KS2 crime cache")
    ks2_crime = {k: v for k, v in ks2_crime.items() if not k.startswith("_")}
    ks2_geo = _load_json(ks2_geocoding_cache, "KS2 geocoding cache")

    # Load schools
    ks5_schools = load_ks5_schools(ks5_csv, ks5_crime, ks5_geo)
    ks2_schools = load_ks2_schools(ks2_csv, ks2_crime, ks2_geo, ks2_crime_radius)

    schools = ks5_schools + ks2_schools
    if not schools:
        logger.error("No schools loaded — aborting.")
        return None

    # Pooled crime-index percentile: rank every school against the combined
    # cohort so the index is comparable between primaries and secondaries.
    # Only rank schools that have crime data; leave others without an index.
    ranked_idx = [i for i, s in enumerate(schools) if "crime_count" in s]
    if ranked_idx:
        counts = pd.Series([schools[i]["crime_count"] for i in ranked_idx])
        indices = counts.rank(pct=True).tolist()
        for pos, i in enumerate(ranked_idx):
            schools[i]["crime_index"] = indices[pos]
    logger.info(f"Computed pooled crime index over {len(ranked_idx)} schools")

    # Emit HTML
    schools_json = json.dumps(schools, indent=2, ensure_ascii=False)
    html = get_html_template()
    replacements = {
        "__SCHOOLS_DATA__": schools_json,
        "__KS5_A_STAR__": str(ks5_a_star),
        "__KS5_A__": str(ks5_a),
        "__KS2_HIGH__": str(ks2_high),
        "__KS2_EXPECTED__": str(ks2_expected),
        "__KS2_COLOR_HIGH__": ks2_colors["high"],
        "__KS2_COLOR_EXPECTED__": ks2_colors["expected"],
        "__KS2_COLOR_BELOW__": ks2_colors["below_expected"],
    }
    for k, v in replacements.items():
        html = html.replace(k, v)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    logger.info(f"Saved {output_file} ({size_mb:.2f} MB)")
    logger.info(f"  {len(ks2_schools)} primaries (KS2) + {len(ks5_schools)} secondaries (KS5) = {len(schools)} schools")
    logger.info(f"\nOpen {output_file} in any browser or upload to S3.")
    return schools


def main():
    parser = argparse.ArgumentParser(description="Build a combined KS2+KS5 standalone HTML")
    parser.add_argument("--ks5-csv", default="processed_school_data.csv")
    parser.add_argument("--ks5-crime-cache", default="crime_cache.json")
    parser.add_argument("--ks5-geocoding-cache", default="geocoding_cache.json")
    parser.add_argument("--ks5-config", default="config.json")
    parser.add_argument("--ks2-csv", default="ks2_processed_school_data.csv")
    parser.add_argument("--ks2-crime-cache", default="ks2_crime_cache.json")
    parser.add_argument("--ks2-geocoding-cache", default="ks2_geocoding_cache.json")
    parser.add_argument("--ks2-config", default="ks2_config.json")
    parser.add_argument("--output", default="school_finder.html")
    args = parser.parse_args()

    create_combined_html(
        ks5_csv=args.ks5_csv,
        ks5_crime_cache=args.ks5_crime_cache,
        ks5_geocoding_cache=args.ks5_geocoding_cache,
        ks5_config=args.ks5_config,
        ks2_csv=args.ks2_csv,
        ks2_crime_cache=args.ks2_crime_cache,
        ks2_geocoding_cache=args.ks2_geocoding_cache,
        ks2_config=args.ks2_config,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
