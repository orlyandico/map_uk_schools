#!/usr/bin/env python3
"""
Create a standalone HTML file with embedded school data

This script generates a self-contained HTML file that includes all school data
embedded directly in the HTML, eliminating the need for external JSON files
or a web server.

Output: school_finder.html
"""

import os
import json
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_cache_files(
    crime_cache_file="crime_cache.json", geocoding_cache_file="geocoding_cache.json"
):
    """
    Load crime and geocoding cache files

    Args:
        crime_cache_file: Path to crime cache JSON
        geocoding_cache_file: Path to geocoding cache JSON

    Returns:
        tuple: (crime_cache dict, geocoding_cache dict)
    """
    crime_cache = {}
    geocoding_cache = {}

    # Load crime cache
    if os.path.exists(crime_cache_file):
        try:
            with open(crime_cache_file, "r") as f:
                crime_cache = json.load(f)
            # Remove metadata entries
            crime_cache = {
                k: v for k, v in crime_cache.items() if not k.startswith("_")
            }
            logger.info(f"Loaded {len(crime_cache)} crime cache entries")
        except Exception as e:
            logger.warning(f"Failed to load crime cache: {e}")

    # Load geocoding cache
    if os.path.exists(geocoding_cache_file):
        try:
            with open(geocoding_cache_file, "r") as f:
                geocoding_cache = json.load(f)
            logger.info(f"Loaded {len(geocoding_cache)} geocoding cache entries")
        except Exception as e:
            logger.warning(f"Failed to load geocoding cache: {e}")

    return crime_cache, geocoding_cache


def get_html_template():
    """Return the HTML template as a string"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>UK Schools Finder</title>

    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            height: 100vh;
            overflow: hidden;
        }

        #map {
            width: 100%;
            height: 100%;
        }

        .controls {
            position: absolute;
            top: 10px;
            left: 50px;
            right: 10px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .search-container {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .search-input-wrapper {
            flex: 1;
            min-width: 200px;
            display: flex;
            gap: 5px;
        }

        #addressInput {
            flex: 1;
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            background: white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }

        #addressInput:focus {
            outline: none;
            border-color: #4169E1;
        }

        button {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            background: white;
            color: #333;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            transition: all 0.2s;
            white-space: nowrap;
        }

        button:hover {
            background: #f0f0f0;
        }

        button:active {
            transform: scale(0.98);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .btn-primary {
            background: #4169E1;
            color: white;
        }

        .btn-primary:hover {
            background: #3557c9;
        }

        .btn-location {
            background: #FF4500;
            color: white;
        }

        .btn-location:hover {
            background: #e03d00;
        }

        .info-panel {
            background: white;
            padding: 12px 15px;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            font-size: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .info-panel .status {
            color: #666;
        }

        .info-panel .count {
            font-weight: bold;
            color: #4169E1;
        }

        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px 40px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 2000;
            font-size: 16px;
            display: none;
        }

        .loading.show {
            display: block;
        }

        .error {
            background: #ff4444;
            color: white;
            padding: 12px 15px;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            display: none;
        }

        .error.show {
            display: block;
        }

        /* School detail panel */
        .school-detail {
            position: fixed;
            bottom: 10px;
            right: 10px;
            width: 320px;
            max-width: calc(100vw - 20px);
            max-height: 60vh;
            overflow-y: auto;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            display: none;
            font-size: 9pt;
        }

        .school-detail.show {
            display: block;
        }

        .school-detail .close-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            background: #ff4444;
            color: white;
            border: none;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            font-size: 16px;
            line-height: 1;
            cursor: pointer;
            padding: 0;
            box-shadow: none;
        }

        .school-detail .close-btn:hover {
            background: #cc0000;
        }

        /* Mobile adjustments */
        @media (max-width: 768px) {
            .controls {
                top: 5px;
                left: 5px;
                right: 5px;
            }

            .search-container {
                flex-direction: column;
            }

            .search-input-wrapper {
                min-width: 100%;
            }

            button {
                width: 100%;
            }

            .info-panel {
                font-size: 12px;
                padding: 10px;
            }

            .school-detail {
                width: calc(100vw - 20px);
                max-height: 50vh;
                bottom: 5px;
                right: 5px;
            }
        }

        /* Scale control positioning */
        .leaflet-control-scale {
            margin-bottom: 10px !important;
        }

        /* Custom popup styles */
        .leaflet-popup-content {
            margin: 10px;
            min-width: 300px;
            max-width: 400px;
            font-size: 9pt;
        }

        .popup-header {
            font-weight: bold;
            font-size: 9pt;
            margin-bottom: 8px;
            color: #333;
            grid-column: 1 / -1;
        }

        .popup-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px 12px;
        }

        .popup-section {
            display: flex;
            flex-direction: column;
        }

        .popup-section.full-width {
            grid-column: 1 / -1;
        }

        .popup-label {
            font-weight: 600;
            color: #666;
            font-size: 9pt;
            margin-bottom: 2px;
        }

        .popup-value {
            color: #333;
            font-size: 9pt;
            line-height: 1.3;
        }

        .popup-value strong {
            font-size: 9pt;
            font-weight: 600;
        }

        .popup-value small {
            font-size: 9pt;
        }

        .grade-badge {
            display: inline-block;
            padding: 1px 6px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 9pt;
            margin-left: 4px;
        }

        .grade-a-star {
            background: #FFD700;
            color: #000;
        }

        .grade-a {
            background: #FFA500;
            color: #000;
        }

        .grade-b {
            background: #87CEEB;
            color: #000;
        }
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
            <button id="locationBtn" class="btn-location">📍 Use My Location</button>
        </div>
        <div class="error" id="errorMsg"></div>
        <div class="info-panel">
            <span class="status" id="statusMsg">Load data to begin</span>
            <span class="count" id="schoolCount">0 schools</span>
        </div>
    </div>

    <div class="loading" id="loading">Loading school data...</div>

    <div class="school-detail" id="schoolDetail">
        <button class="close-btn" id="closeDetail">×</button>
        <div id="schoolDetailContent"></div>
    </div>

    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <script>
        // Configuration
        const CONFIG = {
            RADIUS_KM: 10,
            A_STAR_THRESHOLD: 50,
            A_THRESHOLD: 40,
            COLORS: {
                independent: {
                    a_star: '#FF0000',
                    a: '#FFA500',
                    b_or_below: '#FFD700'
                },
                state: {
                    a_star: '#000080',
                    a: '#0000FF',
                    b_or_below: '#4169E1'
                }
            }
        };

        // Global state
        let map;
        let schoolsData = [];
        let currentMarkers = [];
        let currentCircle = null;
        let currentCenter = null;
        let reticleCircle = null;

        // Initialize map
        function initMap() {
            map = L.map('map').setView([52.4862, -1.8904], 7); // Center of England

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors',
                maxZoom: 19
            }).addTo(map);

            // Add scale control to bottom left
            L.control.scale({
                position: 'bottomleft',
                metric: true,
                imperial: false,
                maxWidth: 150
            }).addTo(map);

            // Create permanent reticle circle
            const center = map.getCenter();
            reticleCircle = L.circle([center.lat, center.lng], {
                radius: CONFIG.RADIUS_KM * 1000,
                color: '#4169E1',
                fillColor: '#4169E1',
                fillOpacity: 0.1,
                weight: 1
            }).addTo(map);

            // Add map event listeners after map is created
            map.on('moveend', updateSchoolsForCurrentView);
            map.on('move', updateReticlePosition);
        }

        // Load school data (embedded)
        async function loadSchoolData() {
            try {
                showLoading(true);
                // Data embedded directly in HTML
                schoolsData = __SCHOOLS_DATA__;
                updateStatus(`Loaded ${schoolsData.length} schools`);
                showLoading(false);
            } catch (error) {
                showError('Failed to load school data: ' + error.message);
                showLoading(false);
            }
        }

        // Calculate distance between two points (Haversine formula)
        function calculateDistance(lat1, lon1, lat2, lon2) {
            const R = 6371; // Earth's radius in km
            const dLat = (lat2 - lat1) * Math.PI / 180;
            const dLon = (lon2 - lon1) * Math.PI / 180;
            const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                     Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                     Math.sin(dLon/2) * Math.sin(dLon/2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            return R * c;
        }

        // Get schools within radius
        function getSchoolsInRadius(centerLat, centerLon, radiusKm) {
            return schoolsData.filter(school => {
                const distance = calculateDistance(centerLat, centerLon, school.Latitude, school.Longitude);
                return distance <= radiusKm;
            });
        }

        // Get marker color
        function getMarkerColor(score, isIndependent) {
            const colors = isIndependent ? CONFIG.COLORS.independent : CONFIG.COLORS.state;
            if (score >= CONFIG.A_STAR_THRESHOLD) return colors.a_star;
            if (score >= CONFIG.A_THRESHOLD) return colors.a;
            return colors.b_or_below;
        }

        // Get text color based on background brightness
        function getTextColor(bgColor) {
            const hex = bgColor.replace('#', '');
            const r = parseInt(hex.substr(0, 2), 16);
            const g = parseInt(hex.substr(2, 2), 16);
            const b = parseInt(hex.substr(4, 2), 16);
            const brightness = (r * 299 + g * 587 + b * 114) / 1000;
            return brightness > 128 ? '#000000' : '#FFFFFF';
        }

        // Point score to grade
        function pointScoreToGrade(score) {
            if (!score || score === 0) return 'N/A';
            if (score >= CONFIG.A_STAR_THRESHOLD) return 'A*';
            if (score >= CONFIG.A_THRESHOLD) return 'A';
            return '≤B';
        }

        // Create popup content
        function createPopupContent(school) {
            const isIndependent = !school.ADMPOL_PT || school.ADMPOL_PT.trim() === '';
            const admpol = isIndependent ? 'Independent' : school.ADMPOL_PT;
            const tb3Grade = pointScoreToGrade(school.TB3PTSE);
            const tallppeGrade = pointScoreToGrade(school.TALLPPE_ALEV_1618);

            // Build address (data is already cleaned by Python)
            let addressParts = [];
            if (school.ADDRESS1 && school.ADDRESS1.trim()) {
                addressParts.push(school.ADDRESS1);
            }
            if (school.TOWN && school.TOWN.trim()) {
                addressParts.push(school.TOWN);
            }
            if (school.PCODE && school.PCODE.trim()) {
                addressParts.push(school.PCODE);
            }
            const fullAddress = addressParts.length > 0 ? addressParts.join(', ') : 'Address not available';

            let html = `
                <div class="popup-header">${school.SCHNAME}</div>
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
                        <div class="popup-value">${school.GEND1618}</div>
                    </div>

                    <div class="popup-section">
                        <div class="popup-label">Age Range</div>
                        <div class="popup-value">${school.AGERANGE}</div>
                    </div>

                    <div class="popup-section">
                        <div class="popup-label">Students (16-18)</div>
                        <div class="popup-value">${school.TPUP1618}</div>
                    </div>

                    <div class="popup-section">
                        <div class="popup-label">Contact</div>
                        <div class="popup-value">☎️ ${school.TELNUM}</div>
                    </div>

                    <div class="popup-section">
                        <div class="popup-label">TB3PTSE (Best 3 A-levels)</div>
                        <div class="popup-value">${school.TB3PTSE.toFixed(2)} <span class="grade-badge grade-${tb3Grade === 'A*' ? 'a-star' : tb3Grade === 'A' ? 'a' : 'b'}">${tb3Grade}</span></div>
                    </div>

                    <div class="popup-section">
                        <div class="popup-label">TALLPPE (Per A-level)</div>
                        <div class="popup-value">${school.TALLPPE_ALEV_1618.toFixed(2)} <span class="grade-badge grade-${tallppeGrade === 'A*' ? 'a-star' : tallppeGrade === 'A' ? 'a' : 'b'}">${tallppeGrade}</span></div>
                    </div>
            `;

            if (school.YEAR_SCORES) {
                html += `
                    <div class="popup-section full-width">
                        <div class="popup-label">Year-by-Year TB3PTSE</div>
                        <div class="popup-value">${school.YEAR_SCORES.replace(/<br>/g, ', ')}</div>
                    </div>
                `;
            }

            if (school.crime_stats) {
                const crimeStats = school.crime_stats;
                const crimeIndex = school.crime_index || 0;
                html += `
                    <div class="popup-section full-width">
                        <div class="popup-label">Local Safety (3km)</div>
                        <div class="popup-value">
                            ${crimeStats.total_crimes} serious crimes (Index: ${crimeIndex.toFixed(2)})`;

                if (crimeStats.crime_types && Object.keys(crimeStats.crime_types).length > 0) {
                    const topCrimes = Object.entries(crimeStats.crime_types)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 3);
                    html += ' — ';
                    html += topCrimes.map(([type, count]) => `${type}: ${count}`).join(', ');
                }

                html += `
                        </div>
                    </div>
                `;
            }

            html += '</div>'; // Close popup-grid

            return html;
        }

        // Create custom marker icon
        function createMarkerIcon(school) {
            const isIndependent = !school.ADMPOL_PT || school.ADMPOL_PT.trim() === '';
            const color = getMarkerColor(school.TB3PTSE, isIndependent);
            const textColor = getTextColor(color);
            const score = Math.round(school.TB3PTSE);

            return L.divIcon({
                html: `
                    <svg width="30" height="40" viewBox="0 0 24 32" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 0C7.58 0 4 3.58 4 8c0 5.76 8 16 8 16s8-10.24 8-16c0-4.42-3.58-8-8-8z"
                            fill="${color}"
                            stroke="black"
                            stroke-width="1"/>
                        <text x="12" y="10" font-family="Arial" font-size="7" font-weight="bold"
                              fill="${textColor}" text-anchor="middle">${score}</text>
                    </svg>
                `,
                className: '',
                iconSize: [30, 40],
                iconAnchor: [15, 40],
                popupAnchor: [150, 0]
            });
        }

        // Display schools on map
        function displaySchools(centerLat, centerLon, isAddressSearch = false) {
            // Clear existing markers only (keep reticle circle)
            currentMarkers.forEach(marker => map.removeLayer(marker));
            currentMarkers = [];
            if (currentCircle) {
                map.removeLayer(currentCircle);
                currentCircle = null;
            }

            // Store current center
            currentCenter = { lat: centerLat, lon: centerLon };

            // Update reticle position
            reticleCircle.setLatLng([centerLat, centerLon]);

            // Get schools in radius
            const nearbySchools = getSchoolsInRadius(centerLat, centerLon, CONFIG.RADIUS_KM);

            // Add markers for each school
            nearbySchools.forEach(school => {
                const marker = L.marker([school.Latitude, school.Longitude], {
                    icon: createMarkerIcon(school)
                }).addTo(map);

                marker.on('click', () => {
                    showSchoolDetail(school);
                });

                currentMarkers.push(marker);
            });

            // Update UI
            updateSchoolCount(nearbySchools.length);
            updateStatus(`Showing ${nearbySchools.length} schools within ${CONFIG.RADIUS_KM}km`);

            // Zoom appropriately for address searches
            if (isAddressSearch) {
                // Set zoom level so circle fills most of vertical height
                // Approximate zoom level for 10km radius to fill ~80% of screen height
                const targetZoom = 11;
                map.setView([centerLat, centerLon], targetZoom);
            }
        }

        // Use current location
        function useCurrentLocation() {
            if (!navigator.geolocation) {
                showError('Geolocation is not supported by your browser');
                return;
            }

            showLoading(true);
            updateStatus('Getting your location...');

            navigator.geolocation.getCurrentPosition(
                (position) => {
                    showLoading(false);
                    displaySchools(position.coords.latitude, position.coords.longitude, true);
                },
                (error) => {
                    showLoading(false);
                    let errorMsg = 'Unable to get your location';
                    if (error.code === error.PERMISSION_DENIED) {
                        errorMsg = 'Location permission denied. Please enable location access.';
                    } else if (error.code === error.POSITION_UNAVAILABLE) {
                        errorMsg = 'Location information unavailable';
                    } else if (error.code === error.TIMEOUT) {
                        errorMsg = 'Location request timed out';
                    }
                    showError(errorMsg);
                },
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 0
                }
            );
        }

        // Search by address using Nominatim (OpenStreetMap geocoding)
        async function searchAddress() {
            const address = document.getElementById('addressInput').value.trim();
            if (!address) {
                showError('Please enter an address or postcode');
                return;
            }

            showLoading(true);
            updateStatus('Searching for address...');

            try {
                // Add UK bias to search
                const searchQuery = address.includes('UK') ? address : `${address}, UK`;
                const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&countrycodes=gb&limit=1`;

                const response = await fetch(url, {
                    headers: {
                        'User-Agent': 'UK Schools Finder Web App'
                    }
                });

                if (!response.ok) {
                    throw new Error('Geocoding service unavailable');
                }

                const results = await response.json();
                showLoading(false);

                if (results.length === 0) {
                    showError('Address not found. Please try a different search term.');
                    updateStatus('Ready');
                    return;
                }

                const location = results[0];
                displaySchools(parseFloat(location.lat), parseFloat(location.lon), true);

            } catch (error) {
                showLoading(false);
                showError('Failed to search address: ' + error.message);
                updateStatus('Ready');
            }
        }

        // UI Helper functions
        function showLoading(show) {
            document.getElementById('loading').classList.toggle('show', show);
        }

        function showError(message) {
            const errorEl = document.getElementById('errorMsg');
            errorEl.textContent = message;
            errorEl.classList.add('show');
            setTimeout(() => {
                errorEl.classList.remove('show');
            }, 5000);
        }

        function updateStatus(message) {
            document.getElementById('statusMsg').textContent = message;
        }

        function updateSchoolCount(count) {
            document.getElementById('schoolCount').textContent = `${count} school${count !== 1 ? 's' : ''}`;
        }

        function showSchoolDetail(school) {
            const content = createPopupContent(school);
            document.getElementById('schoolDetailContent').innerHTML = content;
            document.getElementById('schoolDetail').classList.add('show');
        }

        function hideSchoolDetail() {
            document.getElementById('schoolDetail').classList.remove('show');
        }

        // Update reticle position during map movement
        function updateReticlePosition() {
            if (reticleCircle && map) {
                const center = map.getCenter();
                reticleCircle.setLatLng([center.lat, center.lng]);
            }
        }

        // Update schools based on map center
        function updateSchoolsForCurrentView() {
            if (!map || schoolsData.length === 0) return;
            
            const center = map.getCenter();
            displaySchools(center.lat, center.lng);
        }

        // Event listeners
        document.getElementById('locationBtn').addEventListener('click', useCurrentLocation);
        document.getElementById('searchBtn').addEventListener('click', searchAddress);
        document.getElementById('addressInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                searchAddress();
            }
        });
        document.getElementById('closeDetail').addEventListener('click', hideSchoolDetail);

        // Initialize
        initMap();
        loadSchoolData();
    </script>
</body>
</html>"""


def create_standalone_html(
    csv_file="processed_school_data.csv",
    output_file="school_finder.html",
    crime_cache_file="crime_cache.json",
    geocoding_cache_file="geocoding_cache.json",
):
    """
    Create standalone HTML file with embedded school data

    Args:
        csv_file: Path to processed school data CSV
        output_file: Path to output HTML file (default: school_finder.html)
        crime_cache_file: Path to crime cache JSON
        geocoding_cache_file: Path to geocoding cache JSON
    """
    logger.info("Creating standalone HTML application...")

    try:
        # Load cache files
        crime_cache, geocoding_cache = load_cache_files(
            crime_cache_file, geocoding_cache_file
        )

        # Read CSV
        df = pd.read_csv(csv_file, dtype={"TELNUM": str})
        logger.info(f"Loaded {len(df)} schools from CSV")

        # Prepare school data
        schools = []
        crime_counts = []
        school_crime_data = []

        for idx, row in df.iterrows():
            # Build full address for geocoding cache lookup (same logic as library)
            address_parts = []
            for field in ['ADDRESS1', 'TOWN', 'PCODE']:
                value = row[field]
                if pd.notna(value) and str(value).strip() and str(value).lower() != 'nan':
                    address_parts.append(str(value))
            full_address = ", ".join(address_parts) if address_parts else None

            # Get coordinates
            lat, lon = None, None
            if full_address and full_address in geocoding_cache:
                coords = geocoding_cache[full_address]
                if coords and len(coords) == 2:
                    lat, lon = coords[0], coords[1]

            # Fallback to CSV
            if lat is None and "Latitude" in row and pd.notna(row["Latitude"]):
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])

            if lat is None:
                logger.warning(f"No coordinates for {row['SCHNAME']}")
                continue

            # Get crime stats
            crime_cache_key = f"{lat:.6f},{lon:.6f},3"
            crime_stats = crime_cache.get(crime_cache_key)

            crime_count = 0
            if crime_stats and "total_crimes" in crime_stats:
                crime_count = crime_stats["total_crimes"]

            crime_counts.append(crime_count)
            school_crime_data.append((crime_stats, crime_count))

            # Build school object (convert NaN to empty string for cleaner output)
            def clean_value(value, default=""):
                """Convert NaN/None to default value"""
                if pd.isna(value):
                    return default
                return str(value) if str(value).lower() != 'nan' else default

            school = {
                "SCHNAME": row["SCHNAME"],
                "ADDRESS1": clean_value(row["ADDRESS1"]),
                "TOWN": clean_value(row["TOWN"]),
                "PCODE": clean_value(row["PCODE"]),
                "TELNUM": clean_value(row["TELNUM"]),
                "ADMPOL_PT": clean_value(row["ADMPOL_PT"]),
                "GEND1618": clean_value(row["GEND1618"]),
                "AGERANGE": clean_value(row["AGERANGE"]),
                "TPUP1618": float(row["TPUP1618"]) if pd.notna(row["TPUP1618"]) else 0,
                "TALLPPE_ALEV_1618": float(row["TALLPPE_ALEV_1618"]) if pd.notna(row["TALLPPE_ALEV_1618"]) else 0,
                "TB3PTSE": float(row["TB3PTSE"]) if pd.notna(row["TB3PTSE"]) else 0,
                "YEAR_SCORES": clean_value(row.get("YEAR_SCORES")),
                "Latitude": lat,
                "Longitude": lon,
            }

            schools.append(school)

        # Calculate crime indices
        logger.info("Calculating crime indices...")
        crime_counts_series = pd.Series(crime_counts)
        crime_indices = crime_counts_series.rank(pct=True).tolist()

        # Add crime data
        for i, school in enumerate(schools):
            crime_stats, crime_count = school_crime_data[i]
            if crime_stats:
                school["crime_stats"] = crime_stats
                school["crime_count"] = crime_count
                school["crime_index"] = crime_indices[i]

        logger.info(f"Prepared {len(schools)} schools with data")

        # Convert schools data to JSON string
        schools_json = json.dumps(schools, indent=2, ensure_ascii=False)

        # Get HTML template and embed data
        html_content = get_html_template()
        html_content = html_content.replace("__SCHOOLS_DATA__", schools_json)

        # Write standalone HTML
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Successfully created {output_file}")

        # File size info
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")

        # Statistics
        schools_with_crime = sum(1 for s in schools if "crime_stats" in s)
        logger.info(
            f"Schools with crime data: {schools_with_crime} ({schools_with_crime / len(schools) * 100:.1f}%)"
        )

        logger.info(f"\n✓ Standalone app created successfully!")
        logger.info(f"✓ You can now:")
        logger.info(f"  - Open {output_file} directly in any browser")
        logger.info(f"  - Upload to S3/CDN for hosting")
        logger.info(f"  - Share as a single file")

        return schools

    except Exception as e:
        logger.error(f"Error creating standalone HTML: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create standalone HTML with embedded school data",
        epilog="""
Examples:
  # Create school_finder.html
  python create_standalone_app.py

  # Custom output filename
  python create_standalone_app.py --output my_schools.html

Notes:
  - Resulting file is self-contained (no external dependencies except CDN)
  - Can be opened directly in browser (file://)
  - Perfect for S3, GitHub Pages, or sharing via email
  - File size typically 0.2-0.5 MB depending on number of schools
        """,
    )

    parser.add_argument(
        "--input",
        default="processed_school_data.csv",
        help="Input CSV file (default: processed_school_data.csv)",
    )
    parser.add_argument(
        "--output",
        default="school_finder.html",
        help="Output HTML file (default: school_finder.html)",
    )
    parser.add_argument(
        "--crime-cache",
        default="crime_cache.json",
        help="Crime cache file (default: crime_cache.json)",
    )
    parser.add_argument(
        "--geocoding-cache",
        default="geocoding_cache.json",
        help="Geocoding cache file (default: geocoding_cache.json)",
    )

    args = parser.parse_args()

    create_standalone_html(
        csv_file=args.input,
        output_file=args.output,
        crime_cache_file=args.crime_cache,
        geocoding_cache_file=args.geocoding_cache,
    )


if __name__ == "__main__":
    main()
