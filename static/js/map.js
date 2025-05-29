// Constants
const DEFAULT_CENTER = [20, 0];
const DEFAULT_ZOOM = 4;
const TEMPERATURE_RANGE = {
    MIN: -40,
    MAX: 40
};

// Initialize variables
let currentYear = 2024;  // Fixed to 2024
let heatmapLayer = null;
let currentTooltip = null;

// Temperature gradient configuration
const temperatureGradient = {
    0.0: '#91319A',  // Cold Purple (-40°C)
    0.1: '#2B65EC',  // Deep Blue (-32°C)
    0.2: '#3D9EFF',  // Light Blue (-24°C)
    0.3: '#51B8FF',  // Cyan Blue (-16°C)
    0.4: '#6CCDFF',  // Light Cyan (-8°C)
    0.5: '#80FFE5',  // Cyan (0°C)
    0.6: '#8FFF75',  // Light Green (8°C)
    0.7: '#FFFF00',  // Yellow (16°C)
    0.8: '#FFB300',  // Orange (24°C)
    0.9: '#FF6B00',  // Dark Orange (32°C)
    1.0: '#FF1700'   // Red (40°C)
};

// Map configuration
const mapConfig = {
    center: DEFAULT_CENTER,
    zoom: DEFAULT_ZOOM,
    zoomControl: false,
    minZoom: 2,
    maxZoom: 8,
    attributionControl: false,
    worldCopyJump: true,  // Better handling of world wrap
    maxBounds: [[-90, -180], [90, 180]]  // Restrict map bounds
};

// Initialize the map
const map = L.map('map', mapConfig);

// Add base layer with retina support
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 9,
    attribution: '©OpenStreetMap, ©CartoDB',
    subdomains: 'abcd',  // Use all available subdomains
    detectRetina: true   // Support for retina displays
}).addTo(map);

// Initialize heatmap layer with optimized settings
function initHeatLayer(data = []) {
    if (heatmapLayer) {
        map.removeLayer(heatmapLayer);
    }

    heatmapLayer = L.heatLayer(data, {
        radius: getAdaptiveRadius(),
        blur: getAdaptiveBlur(),
        maxZoom: 8,
        max: 1.0,
        gradient: temperatureGradient,
        minOpacity: 0.5,
        maxOpacity: 0.8,
        scaleRadius: true,
        useLocalExtrema: false
    }).addTo(map);
}

// Adaptive radius based on zoom level and device pixel ratio
function getAdaptiveRadius() {
    const zoom = map.getZoom();
    const pixelRatio = window.devicePixelRatio || 1;
    return Math.max(15, Math.min(25, 20 * pixelRatio)) * (zoom < 4 ? 0.75 : 1);
}

// Adaptive blur based on zoom level
function getAdaptiveBlur() {
    const zoom = map.getZoom();
    return zoom < 4 ? 10 : 15;
}

// Loading indicator functions
function showLoading() {
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

// Error handling with timeout
function showError(message, duration = 5000) {
    const errorDiv = document.getElementById('error-message');
    if (!errorDiv) return;

    errorDiv.textContent = message;
    errorDiv.style.display = 'block';

    if (errorDiv._hideTimeout) {
        clearTimeout(errorDiv._hideTimeout);
    }

    errorDiv._hideTimeout = setTimeout(() => {
        errorDiv.style.display = 'none';
    }, duration);
}

// Update heatmap data with error handling and retry
async function updateHeatmap(retryCount = 0) {
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 1000; // 1 second

    try {
        showLoading();

        const bounds = map.getBounds();
        const response = await fetch('/api/heatmap-data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                bounds: {
                    _southWest: bounds.getSouthWest(),
                    _northEast: bounds.getNorthEast()
                },
                zoom: map.getZoom()
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        const points = data.data.map(point => {
            const normalizedTemp = (point.temperature - TEMPERATURE_RANGE.MIN) / 
                                 (TEMPERATURE_RANGE.MAX - TEMPERATURE_RANGE.MIN);
            return [
                point.lat,
                point.lon,
                Math.max(0, Math.min(1, normalizedTemp)) // Ensure value is between 0 and 1
            ];
        });

        initHeatLayer(points);
        hideLoading();

    } catch (error) {
        console.error('Error updating heatmap:', error);
        
        if (retryCount < MAX_RETRIES) {
            setTimeout(() => {
                updateHeatmap(retryCount + 1);
            }, RETRY_DELAY * (retryCount + 1));
        } else {
            showError('Failed to update temperature data. Please try again later.');
            hideLoading();
        }
    }
}

// Debounce function to limit update frequency
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Map event listeners with debouncing
const debouncedUpdate = debounce(updateHeatmap, 250);
map.on('moveend', debouncedUpdate);
map.on('zoomend', debouncedUpdate);

// Add controls
L.control.scale({
    position: 'bottomleft',
    imperial: false,
    maxWidth: 200
}).addTo(map);

L.control.zoom({
    position: 'bottomleft'
}).addTo(map);

// Attribution control
L.control.attribution({
    position: 'bottomright',
    prefix: '© Temperature Data'
}).addTo(map);

// Layer control
const layerControl = L.control({position: 'topright'});
layerControl.onAdd = function() {
    const div = L.DomUtil.create('div', 'layer-control');
    div.innerHTML = `
        <div class="layer-toggle">
            <label>
                <input type="checkbox" checked> Temperature Layer
            </label>
        </div>
    `;
    
    const checkbox = div.querySelector('input');
    checkbox.addEventListener('change', (e) => {
        if (e.target.checked) {
            if (heatmapLayer) map.addLayer(heatmapLayer);
        } else {
            if (heatmapLayer) map.removeLayer(heatmapLayer);
        }
    });
    
    return div;
};
layerControl.addTo(map);

// Initial update
updateHeatmap();