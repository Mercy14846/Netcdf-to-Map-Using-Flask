// Constants
const AFRICA_CENTER = [0, 20];  // Centered on Africa
const DEFAULT_ZOOM = 4;
const TEMPERATURE_RANGE = {
    MIN: -40,
    MAX: 40
};

// Memory management
let currentData = null;
let dataCache = new Map();
const MAX_CACHE_SIZE = 50;

// Debounce configuration
const DEBOUNCE_DELAY = 250;
const CHUNK_SIZE = 1000;

// Initialize variables
let currentYear = 2024;
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
    center: AFRICA_CENTER,
    zoom: DEFAULT_ZOOM,
    zoomControl: false,
    minZoom: 2,
    maxZoom: 8,
    attributionControl: false,
    worldCopyJump: true,
    maxBounds: [[-90, -180], [90, 180]]
};

// Initialize the map
const map = L.map('map', mapConfig);

// Base layers
const baseLayers = {
    'Vector': L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 9,
        attribution: '©OpenStreetMap, ©CartoDB',
        subdomains: 'abcd',
        detectRetina: true
    }).addTo(map),  // Add default layer immediately
    'Satellite': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 9,
        attribution: '©Esri',
        detectRetina: true
    })
};

// Initialize overlays object
const overlays = {};

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
    });
    
    // Add heatmap layer to map by default
    heatmapLayer.addTo(map);
    
    // Update overlays object
    overlays['Temperature Layer'] = heatmapLayer;
    
    // Refresh layer control
    if (layerControl) {
        layerControl.remove();
    }
    addLayerControl();
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

// Cache management
function manageCache() {
    if (dataCache.size > MAX_CACHE_SIZE) {
        const oldestKey = dataCache.keys().next().value;
        dataCache.delete(oldestKey);
    }
}

// Chunk processing
function processDataChunks(data, chunkSize = CHUNK_SIZE) {
    const chunks = [];
    for (let i = 0; i < data.length; i += chunkSize) {
        chunks.push(data.slice(i, i + chunkSize));
    }
    return chunks;
}

// Optimized data processing
async function processChunkAsync(chunk) {
    return chunk.map(point => {
        const normalizedTemp = (point.temperature - TEMPERATURE_RANGE.MIN) / 
                             (TEMPERATURE_RANGE.MAX - TEMPERATURE_RANGE.MIN);
        return [
            point.lat,
            point.lon,
            Math.max(0, Math.min(1, normalizedTemp))
        ];
    });
}

// Update heatmap with chunked processing
async function updateHeatmapOptimized(data) {
    const chunks = processDataChunks(data);
    const processedChunks = [];
    
    for (const chunk of chunks) {
        const processed = await processChunkAsync(chunk);
        processedChunks.push(...processed);
        
        // Update visualization incrementally
        if (heatmapLayer) {
            heatmapLayer.setLatLngs(processedChunks);
        }
        
        // Allow UI updates
        await new Promise(resolve => setTimeout(resolve, 0));
    }
    
    return processedChunks;
}

// Optimized heatmap update with caching and retry
async function updateHeatmap(retryCount = 0) {
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 1000;

    try {
        showLoading();

        const bounds = map.getBounds();
        const cacheKey = `${bounds.getNorth()}_${bounds.getSouth()}_${bounds.getEast()}_${bounds.getWest()}_${map.getZoom()}`;
        
        // Check cache first
        if (dataCache.has(cacheKey)) {
            const cachedData = dataCache.get(cacheKey);
            initHeatLayer(cachedData);
            hideLoading();
            return;
        }

        const response = await fetch('/api/heatmap-data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Cache-Control': 'max-age=3600'
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

        // Process data in chunks
        const processedData = await updateHeatmapOptimized(data.data);
        
        // Cache the processed data
        dataCache.set(cacheKey, processedData);
        manageCache();

        initHeatLayer(processedData);
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

// Layer control setup
let layerControl = null;

function addLayerControl() {
    // Remove existing layer control if it exists
    if (layerControl) {
        layerControl.remove();
    }

    // Only create layer control if we have layers to show
    if (Object.keys(overlays).length > 0 || Object.keys(baseLayers).length > 0) {
        // Create new layer control
        layerControl = L.control.layers(baseLayers, overlays, {
            position: 'topright',
            collapsed: false,
            sortLayers: true,
            hideSingleBase: false
        }).addTo(map);

        // Force the layer control to be visible
        setTimeout(() => {
            const layerControlElement = document.querySelector('.leaflet-control-layers');
            if (layerControlElement) {
                layerControlElement.style.display = 'block';
                layerControlElement.classList.add('leaflet-control-layers-expanded');
            }
        }, 100);
    }
}

// Initial layer control setup (only for base layers)
addLayerControl();

// Attribution control
L.control.attribution({
    position: 'bottomright',
    prefix: '© Temperature Data'
}).addTo(map);

// Initial update
updateHeatmap();

// Ensure layer control stays visible after map updates
map.on('overlayremove overlayadd baselayerchange', () => {
    const layerControlElement = document.querySelector('.leaflet-control-layers');
    if (layerControlElement) {
        layerControlElement.style.display = 'block';
        layerControlElement.classList.add('leaflet-control-layers-expanded');
    }
});