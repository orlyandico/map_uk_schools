"""
Create a standalone HTML file with embedded KS2 primary school data.

Reads ks2_processed_school_data.csv and the cache files, then generates
ks2_school_finder.html — a self-contained mobile-friendly web app with:
  - Movable reticle circle showing schools within radius
  - Address / postcode search (Nominatim)
  - GPS location button
  - School popup showing KS2 metrics, year-by-year breakdown, crime stats
  - Colour-coded markers: dark green (high), green (good), light green (above avg)
"""

import os
import json
import logging
import argparse

import pandas as pd

import ks2_school_data_lib


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML / JS template
# ---------------------------------------------------------------------------

def get_html_template():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>UK Primary Schools Finder (KS2)</title>
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
        #addressInput:focus { outline: none; border-color: #228B22; }

        button {
            padding: 12px 20px; border: none; border-radius: 8px; font-size: 16px;
            font-weight: 600; cursor: pointer; background: white; color: #333;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3); transition: all 0.2s; white-space: nowrap;
        }
        button:hover { background: #f0f0f0; }
        button:active { transform: scale(0.98); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-primary { background: #228B22; color: white; }
        .btn-primary:hover { background: #1a6e1a; }
        .btn-location { background: #005C29; color: white; }
        .btn-location:hover { background: #004820; }

        .info-panel {
            background: white; padding: 12px 15px; border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 14px;
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 10px;
        }
        .info-panel .status { color: #666; }
        .info-panel .count { font-weight: bold; color: #228B22; }

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
            position: fixed; bottom: 10px; right: 10px; width: 320px;
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

        /* Legend */
        .legend {
            position: fixed; bottom: 10px; left: 10px; background: white;
            padding: 10px 14px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            font-size: 11px; z-index: 1000; min-width: 150px;
        }
        .legend-title { font-weight: bold; margin-bottom: 6px; font-size: 12px; }
        .legend-item { display: flex; align-items: center; margin-bottom: 4px; }
        .legend-dot { width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; flex-shrink: 0; }

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
        .popup-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
        .popup-section { display: flex; flex-direction: column; }
        .popup-section.full-width { grid-column: 1 / -1; }
        .popup-label { font-weight: 600; color: #666; font-size: 8pt; margin-bottom: 2px; }
        .popup-value { color: #333; font-size: 9pt; line-height: 1.3; }
        .grade-badge {
            display: inline-block; padding: 1px 6px; border-radius: 3px;
            font-weight: bold; font-size: 8pt; margin-left: 4px;
        }
        .grade-high { background: #005C29; color: #fff; }
        .grade-good { background: #228B22; color: #fff; }
        .grade-avg  { background: #8FBC8F; color: #000; }
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
            <span class="count" id="schoolCount">0 schools</span>
        </div>
    </div>

    <div class="loading show" id="loading">Loading school data...</div>

    <div class="school-detail" id="schoolDetail">
        <button class="close-btn" id="closeDetail">×</button>
        <div id="schoolDetailContent"></div>
    </div>

    <div class="legend">
        <div class="legend-title">KS2 Expected Standard</div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#005C29"></div>
            High (≥__HIGH_THRESHOLD__%)</div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#228B22"></div>
            Good (≥__EXPECTED_THRESHOLD__%)</div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#8FBC8F"></div>
            Above average</div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const CONFIG = {
            RADIUS_KM: 5,
            MIN_RADIUS_KM: 0.5,
            HIGH_THRESHOLD: __HIGH_THRESHOLD__,
            EXPECTED_THRESHOLD: __EXPECTED_THRESHOLD__,
            COLORS: {
                high: '__COLOR_HIGH__',
                expected: '__COLOR_EXPECTED__',
                below_expected: '__COLOR_BELOW__'
            }
        };

        let map, schoolsData = [], currentMarkers = [], currentCircle = null, reticleCircle = null;

        function initMap() {
            map = L.map('map').setView([52.4862, -1.8904], 7);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors', maxZoom: 19
            }).addTo(map);
            L.control.scale({ position: 'bottomleft', metric: true, imperial: false, maxWidth: 150 }).addTo(map);

            const center = map.getCenter();
            reticleCircle = L.circle([center.lat, center.lng], {
                radius: CONFIG.RADIUS_KM * 1000,
                color: '#228B22', fillColor: '#228B22', fillOpacity: 0.08, weight: 2
            }).addTo(map);

            map.on('moveend', updateSchoolsForCurrentView);
            map.on('move', updateReticlePosition);
            map.on('zoomend', adjustReticleForZoom);
        }

        async function loadSchoolData() {
            try {
                schoolsData = __SCHOOLS_DATA__;
                showLoading(false);
                updateStatus(`Loaded ${schoolsData.length} schools`);
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

        function getMarkerColor(pct) {
            if (pct >= CONFIG.HIGH_THRESHOLD) return CONFIG.COLORS.high;
            if (pct >= CONFIG.EXPECTED_THRESHOLD) return CONFIG.COLORS.expected;
            return CONFIG.COLORS.below_expected;
        }

        function getTextColor(hex) {
            const h = hex.replace('#', '');
            const r = parseInt(h.substr(0, 2), 16);
            const g = parseInt(h.substr(2, 2), 16);
            const b = parseInt(h.substr(4, 2), 16);
            return (r * 299 + g * 587 + b * 114) / 1000 > 128 ? '#000' : '#fff';
        }

        function pctToGrade(pct) {
            if (!pct || pct === 0) return null;
            if (pct >= CONFIG.HIGH_THRESHOLD) return { label: `High (≥${CONFIG.HIGH_THRESHOLD}%)`, cls: 'grade-high' };
            if (pct >= CONFIG.EXPECTED_THRESHOLD) return { label: `Good (≥${CONFIG.EXPECTED_THRESHOLD}%)`, cls: 'grade-good' };
            return { label: 'Above avg', cls: 'grade-avg' };
        }

        function createMarkerIcon(school) {
            const color = getMarkerColor(school.expected_pct);
            const textColor = getTextColor(color);
            const label = Math.round(school.expected_pct) + '%';
            return L.divIcon({
                html: `<svg width="36" height="46" viewBox="0 0 24 32" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 0C7.58 0 4 3.58 4 8c0 5.76 8 16 8 16s8-10.24 8-16c0-4.42-3.58-8-8-8z"
                          fill="${color}" stroke="#333" stroke-width="0.8"/>
                    <text x="12" y="9.5" font-family="Arial" font-size="5.5" font-weight="bold"
                          fill="${textColor}" text-anchor="middle">${label}</text>
                </svg>`,
                className: '',
                iconSize: [36, 46],
                iconAnchor: [18, 46],
                popupAnchor: [160, 0]
            });
        }

        function createPopupContent(school) {
            const address = [school.Street, school.Locality, school.Town, school.Postcode]
                .filter(v => v && v.trim()).join(', ') || 'Address not available';

            const grade = pctToGrade(school.expected_pct);
            const gradeBadge = grade
                ? `<span class="grade-badge ${grade.cls}">${grade.label}</span>`
                : '';

            let html = `
                <div class="popup-header">${school.EstablishmentName || school.school_name}</div>
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

        function displaySchools(centerLat, centerLon, isSearch = false, radiusKm = CONFIG.RADIUS_KM) {
            currentMarkers.forEach(m => map.removeLayer(m));
            currentMarkers = [];

            reticleCircle.setLatLng([centerLat, centerLon]);

            const nearby = schoolsData.filter(s =>
                calculateDistance(centerLat, centerLon, s.Latitude, s.Longitude) <= radiusKm
            );

            nearby.forEach(school => {
                const marker = L.marker([school.Latitude, school.Longitude], {
                    icon: createMarkerIcon(school)
                }).addTo(map);
                marker.on('click', () => showSchoolDetail(school));
                marker.bindPopup(createPopupContent(school), { maxWidth: 400 });
                currentMarkers.push(marker);
            });

            updateSchoolCount(nearby.length);
            updateStatus(`${nearby.length} school${nearby.length !== 1 ? 's' : ''} within ${radiusKm.toFixed(1)}km`);

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
                const resp = await fetch(url, { headers: { 'User-Agent': 'UK Primary Schools Finder' } });
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
        function updateSchoolCount(n) {
            document.getElementById('schoolCount').textContent = `${n} school${n !== 1 ? 's' : ''}`;
        }
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
# Data preparation
# ---------------------------------------------------------------------------

def create_standalone_html(
    csv_file="ks2_processed_school_data.csv",
    output_file="ks2_school_finder.html",
    crime_cache_file="ks2_crime_cache.json",
    geocoding_cache_file="ks2_geocoding_cache.json",
    config_file="ks2_config.json",
):
    logger.info("Creating standalone KS2 HTML application...")

    config = ks2_school_data_lib.load_config(config_file)
    high_threshold = config["grading"]["high_threshold"]
    expected_threshold = config["grading"]["expected_threshold"]
    colors = config["colors"]

    # Load caches
    crime_cache = {}
    geocoding_cache = {}
    for path, cache in [(crime_cache_file, "crime"), (geocoding_cache_file, "geocoding")]:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                if cache == "crime":
                    crime_cache = {k: v for k, v in data.items() if not k.startswith("_")}
                    logger.info(f"Loaded {len(crime_cache)} crime cache entries")
                else:
                    geocoding_cache = data
                    logger.info(f"Loaded {len(geocoding_cache)} geocoding cache entries")
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")

    # Load CSV
    try:
        df = pd.read_csv(csv_file, dtype={"TelephoneNum": str})
    except FileNotFoundError:
        logger.error(f"CSV not found: {csv_file}. Run ks2_generate_school_data.py first.")
        return None
    logger.info(f"Loaded {len(df)} schools from {csv_file}")

    schools = []
    crime_counts = []
    school_crime_data = []

    # Build address for cache lookup
    def build_addr(row):
        parts = []
        for f in ["Street", "Locality", "Town", "County", "Postcode"]:
            v = row.get(f, "")
            if pd.notna(v) and str(v).strip() and str(v).lower() != "nan":
                parts.append(str(v).strip())
        return ", ".join(parts) if parts else None

    def clean(val, default=""):
        if pd.isna(val):
            return default
        s = str(val)
        return default if s.lower() == "nan" else s

    for _, row in df.iterrows():
        # Coordinates: from CSV first, then geocoding cache
        lat = row.get("Latitude") if "Latitude" in row and pd.notna(row.get("Latitude")) else None
        lon = row.get("Longitude") if "Longitude" in row and pd.notna(row.get("Longitude")) else None

        if lat is None:
            addr = build_addr(row)
            if addr and addr in geocoding_cache:
                coords = geocoding_cache[addr]
                if coords and len(coords) == 2:
                    lat, lon = coords[0], coords[1]

        if lat is None:
            logger.warning(f"No coordinates for {row.get('EstablishmentName', row.get('school_name', '?'))}")
            continue

        # Crime data
        crime_cache_key = f"{float(lat):.6f},{float(lon):.6f},{config['crime']['school_crime_radius_km']}"
        crime_stats = crime_cache.get(crime_cache_key)
        crime_count = crime_stats["total_crimes"] if crime_stats and "total_crimes" in crime_stats else 0

        crime_counts.append(crime_count)
        school_crime_data.append((crime_stats, crime_count))

        school = {
            "school_urn": clean(row.get("school_urn")),
            "school_name": clean(row.get("school_name")),
            "EstablishmentName": clean(row.get("EstablishmentName")),
            "Street": clean(row.get("Street")),
            "Locality": clean(row.get("Locality")),
            "Town": clean(row.get("Town")),
            "County": clean(row.get("County")),
            "Postcode": clean(row.get("Postcode")),
            "TelephoneNum": clean(row.get("TelephoneNum")),
            "Gender": clean(row.get("Gender")),
            "NumberOfPupils": float(row["NumberOfPupils"]) if pd.notna(row.get("NumberOfPupils")) else None,
            "StatutoryLowAge": clean(row.get("StatutoryLowAge")),
            "StatutoryHighAge": clean(row.get("StatutoryHighAge")),
            "expected_pct": float(row["expected_pct"]) if pd.notna(row.get("expected_pct")) else 0,
            "higher_pct": float(row["higher_pct"]) if pd.notna(row.get("higher_pct")) else None,
            "year_scores": clean(row.get("year_scores")),
            "Latitude": float(lat),
            "Longitude": float(lon),
        }
        schools.append(school)

    # Crime indices
    logger.info(f"Calculating crime indices for {len(schools)} schools...")
    crime_indices = pd.Series(crime_counts).rank(pct=True).tolist()
    for i, school in enumerate(schools):
        cs, cc = school_crime_data[i]
        if cs:
            school["crime_stats"] = cs
            school["crime_count"] = cc
            school["crime_index"] = crime_indices[i]

    logger.info(f"Prepared {len(schools)} schools")

    # Build HTML
    schools_json = json.dumps(schools, indent=2, ensure_ascii=False)
    html = get_html_template()
    html = html.replace("__SCHOOLS_DATA__", schools_json)
    html = html.replace("__HIGH_THRESHOLD__", str(high_threshold))
    html = html.replace("__EXPECTED_THRESHOLD__", str(expected_threshold))
    html = html.replace("__COLOR_HIGH__", colors["high"])
    html = html.replace("__COLOR_EXPECTED__", colors["expected"])
    html = html.replace("__COLOR_BELOW__", colors["below_expected"])

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    schools_with_crime = sum(1 for s in schools if "crime_stats" in s)
    logger.info(f"Saved {output_file} ({size_mb:.2f} MB)")
    logger.info(f"Schools with crime data: {schools_with_crime} ({schools_with_crime/len(schools)*100:.0f}%)")
    logger.info(f"\nOpen {output_file} in any browser or upload to S3.")
    return schools


def main():
    parser = argparse.ArgumentParser(description="Create standalone KS2 school finder HTML")
    parser.add_argument("--input", default="ks2_processed_school_data.csv")
    parser.add_argument("--output", default="ks2_school_finder.html")
    parser.add_argument("--crime-cache", default="ks2_crime_cache.json")
    parser.add_argument("--geocoding-cache", default="ks2_geocoding_cache.json")
    parser.add_argument("--config", default="ks2_config.json")
    args = parser.parse_args()

    create_standalone_html(
        csv_file=args.input,
        output_file=args.output,
        crime_cache_file=args.crime_cache,
        geocoding_cache_file=args.geocoding_cache,
        config_file=args.config,
    )


if __name__ == "__main__":
    main()
