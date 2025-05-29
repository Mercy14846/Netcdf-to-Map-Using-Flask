// Initialize variables
let currentYear = 2024;  // Fixed to 2024
let heatmapLayer = null;
let currentTooltip = null;

// Custom heat layer with willReadFrequently set to true
L.HeatLayer = L.HeatLayer.extend({
    _initCanvas: function () {
        let canvas = L.DomUtil.create('canvas', 'leaflet-heatmap-layer');
        
        // Initialize with default size
        canvas.width = 100;
        canvas.height = 100;
        
        let animated = this._map.options.zoomAnimation && L.Browser.any3d;
        L.DomUtil.addClass(canvas, 'leaflet-zoom-' + (animated ? 'animated' : 'hide'));

        // Set willReadFrequently to true for better performance
        this._canvas = canvas;
        this._ctx = canvas.getContext('2d', { willReadFrequently: true });
        
        // Initial size setup
        this._updateCanvasSize();
        
        return canvas;
    },

    _updateCanvasSize: function () {
        if (this._map && this._canvas) {
            let size = this._map.getSize();
            this._canvas.width = size.x;
            this._canvas.height = size.y;
            this._width = this._canvas.width;
            this._height = this._canvas.height;
            this._draw();
        }
    },

    onAdd: function (map) {
        this._map = map;
        if (!this._canvas) {
            this._initCanvas();
        }

        map._panes.overlayPane.appendChild(this._canvas);

        map.on('moveend', this._reset, this);
        map.on('resize', this._updateCanvasSize, this);

        if (map.options.zoomAnimation && L.Browser.any3d) {
            map.on('zoomanim', this._animateZoom, this);
        }

        this._reset();
    }
});

L.heatLayer = function (latlngs, options) {
    return new L.HeatLayer(latlngs, options);
};

// Initialize the map with better default view
const map = L.map('map', {
    center: [20, 0],
    zoom: 4,
    zoomControl: false,
    minZoom: 2,
    maxZoom: 8,
    attributionControl: false
});

// Add a clean base layer (Carto's Positron for a cleaner look)
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 9,
    attribution: '©OpenStreetMap, ©CartoDB'
}).addTo(map);

// Define enhanced temperature gradient with OpenWeatherMap-like colors
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

// Initialize heatmap layer with enhanced settings
heatmapLayer = L.heatLayer([], {
    radius: 20,      // Smaller radius for more precise temperature regions
    blur: 10,        // Less blur for sharper boundaries
    maxZoom: 8,
    max: 1.0,
    gradient: temperatureGradient,
    minOpacity: 0.5, // Increased minimum opacity
    maxOpacity: 0.8
}).addTo(map);

// Add attribution control in bottom right
L.control.attribution({
    position: 'bottomright'
}).addTo(map);

// Add custom control for map layers
const layerControl = L.control({position: 'topright'});
layerControl.onAdd = function(map) {
    const div = L.DomUtil.create('div', 'layer-control');
    div.innerHTML = `
        <div class="layer-toggle">
            <label>
                <input type="checkbox" checked> Temperature
            </label>
        </div>
    `;
    return div;
};
layerControl.addTo(map);

// Function to convert temperature to color value
function getTemperatureColor(temp) {
    const normalizedTemp = (temp + 40) / 80;
    const stops = Object.entries(temperatureGradient);
    for (let i = 0; i < stops.length - 1; i++) {
        const [pos1, color1] = stops[i];
        const [pos2, color2] = stops[i + 1];
        if (normalizedTemp >= parseFloat(pos1) && normalizedTemp <= parseFloat(pos2)) {
            return color1;
        }
    }
    return temperatureGradient[1.0];
}

// Loading indicator functions
function showLoading() {
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

// Error handling
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

// Update heatmap data
function updateHeatmap() {
    showLoading();
    
    fetch('/api/heatmap-data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify({
            bounds: {
                _southWest: map.getBounds().getSouthWest(),
                _northEast: map.getBounds().getNorthEast()
            },
            zoom: map.getZoom()
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        
        const points = data.data.map(point => {
            const normalizedTemp = (point.temperature + 40) / 80;
            return [
                point.lat,
                point.lon,
                Math.pow(normalizedTemp, 1.1) // Slight emphasis on higher temperatures
            ];
        });
        
        if (heatmapLayer) {
            map.removeLayer(heatmapLayer);
        }
        
        heatmapLayer = L.heatLayer(points, {
            radius: map.getZoom() < 4 ? 15 : 20,
            blur: map.getZoom() < 4 ? 10 : 15,
            maxZoom: 8,
            max: 1.0,
            gradient: temperatureGradient,
            minOpacity: 0.5,
            maxOpacity: 0.8
        }).addTo(map);
        
        hideLoading();
    })
    .catch(error => {
        console.error('Error updating heatmap:', error);
        showError('Failed to update temperature data');
        hideLoading();
    });
}

// Map event listeners
map.on('moveend', updateHeatmap);
map.on('zoomend', updateHeatmap);

// Add scale control
L.control.scale({
    position: 'bottomleft',
    imperial: false // Use metric only
}).addTo(map);

// Add zoom control
L.control.zoom({
    position: 'bottomleft'
}).addTo(map);

// Initial update
updateHeatmap();