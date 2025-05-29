// Initialize variables
let currentYear = 2024;  // Fixed to 2024
let heatmapLayer = null;
let currentTooltip = null;

// Initialize the map with better default view
const map = L.map('map', {
    center: [20, 0],  // Center map at equator
    zoom: 4,         // Default zoom level
    zoomControl: false,
    minZoom: 2,      // Restrict minimum zoom
    maxZoom: 8     // Maximum zoom level
});

// Add multiple base layers
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 9
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Define enhanced temperature gradient with more color stops
const temperatureGradient = {
    0.0: '#000083',  // Deep Blue (-40°C)
    0.1: '#0000FF',  // Blue (-32°C)
    0.2: '#0080FF',  // Light Blue (-24°C)
    0.3: '#00FFFF',  // Cyan (-16°C)
    0.4: '#00FF80',  // Blue-Green (-8°C)
    0.5: '#00FF00',  // Green (0°C)
    0.6: '#80FF00',  // Yellow-Green (8°C)
    0.7: '#FFFF00',  // Yellow (16°C)
    0.8: '#FF8000',  // Orange (24°C)
    0.9: '#FF0000',  // Red (32°C)
    1.0: '#800000'   // Dark Red (40°C)
};

// Initialize heatmap layer with enhanced settings
heatmapLayer = L.heatLayer([], {
    radius: 25,
    blur: 15,
    maxZoom: 10,
    max: 1.0,
    gradient: temperatureGradient,
    minOpacity: 0.4,
    maxOpacity: 0.8
}).addTo(map);

// Function to convert temperature to color value (for tooltip or other UI elements)
function getTemperatureColor(temp) {
    // Normalize temperature from -40 to +40 range to 0-1
    const normalizedTemp = (temp + 40) / 80;
    // Find the appropriate color stop
    const stops = Object.entries(temperatureGradient);
    for (let i = 0; i < stops.length - 1; i++) {
        const [pos1, color1] = stops[i];
        const [pos2, color2] = stops[i + 1];
        if (normalizedTemp >= parseFloat(pos1) && normalizedTemp <= parseFloat(pos2)) {
            return color1;
        }
    }
    return temperatureGradient[1.0]; // Return max temperature color if above range
}

// Add loading indicator functions
function showLoading() {
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

// Error handling function
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

function clearError() {
    const errorDiv = document.getElementById('error-message');
    errorDiv.style.display = 'none';
}

// Update heatmap data with enhanced visualization
function updateHeatmap() {
    showLoading();
    console.log('Fetching heatmap data for year:', currentYear);
    
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
    .then(response => {
        console.log('Response status:', response.status);
        return response.json();
    })
    .then(data => {
        if (data.error) throw new Error(data.error);
        console.log('Received data points:', data.data.length);
        
        // Transform data for heatmap with enhanced intensity calculation
        const points = data.data.map(point => {
            const normalizedTemp = (point.temperature + 40) / 80;
            // Add intensity weighting based on temperature extremes
            const intensity = Math.pow(normalizedTemp, 1.2); // Slight emphasis on higher temperatures
            return [
                point.lat,
                point.lon,
                intensity
            ];
        });
        
        console.log('Processed points:', points.length);
        if (points.length > 0) {
            console.log('Sample point:', points[0]);
        }
        
        // Remove existing layer and create new one with enhanced settings
        if (heatmapLayer) {
            map.removeLayer(heatmapLayer);
        }
        
        heatmapLayer = L.heatLayer(points, {
            radius: map.getZoom() < 4 ? 15 : 25, // Adaptive radius based on zoom
            blur: map.getZoom() < 4 ? 10 : 15,   // Adaptive blur based on zoom
            maxZoom: 10,
            max: 1.0,
            gradient: temperatureGradient,
            minOpacity: 0.4,
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
L.control.scale({position: 'bottomleft'}).addTo(map);

// Add zoom control
L.control.zoom({position: 'bottomleft'}).addTo(map);

// Initial update
updateHeatmap();