// Initialize the map with better default view
const map = L.map('map', {
    center: [6.1511677, 31.1710389],
    zoom: 4,
    zoomControl: false
});

// Add multiple base layers
const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors'
});

const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
});

// Set default base layer
osmLayer.addTo(map);

// Create temperature layer
const temperatureLayer = L.tileLayer('/tiles/{z}/{x}/{y}.png', {
    maxZoom: 19,
    opacity: 0.7
});

// Add layers to map
temperatureLayer.addTo(map);

// Create layer control panel
const controlPanel = L.control({position: 'topright'});

controlPanel.onAdd = function (map) {
    const div = L.DomUtil.create('div', 'control-panel');
    div.innerHTML = `
        <h3>Layers</h3>
        <div class="layer-control">
            <div class="layer-item">
                <input type="radio" id="osm" name="base-layer" value="osm" checked>
                <label for="osm">OpenStreetMap</label>
            </div>
            <div class="layer-item">
                <input type="radio" id="satellite" name="base-layer" value="satellite">
                <label for="satellite">Satellite</label>
            </div>
            <div class="layer-item">
                <input type="checkbox" id="temp-layer" checked>
                <label for="temp-layer">Temperature</label>
            </div>
        </div>
        <div class="opacity-control">
            <strong>Layer Opacity</strong><br>
            <input type="range" class="opacity-slider" min="0" max="1" step="0.1" value="0.7">
        </div>
    `;

    // Add event listeners
    setTimeout(() => {
        const osmRadio = div.querySelector('#osm');
        const satelliteRadio = div.querySelector('#satellite');
        const tempCheckbox = div.querySelector('#temp-layer');
        const opacitySlider = div.querySelector('.opacity-slider');

        osmRadio.addEventListener('change', () => {
            map.removeLayer(satelliteLayer);
            map.addLayer(osmLayer);
        });

        satelliteRadio.addEventListener('change', () => {
            map.removeLayer(osmLayer);
            map.addLayer(satelliteLayer);
        });

        tempCheckbox.addEventListener('change', () => {
            if (tempCheckbox.checked) {
                temperatureLayer.addTo(map);
            } else {
                map.removeLayer(temperatureLayer);
            }
        });

        opacitySlider.addEventListener('input', (e) => {
            temperatureLayer.setOpacity(e.target.value);
        });
    }, 0);

    return div;
};

controlPanel.addTo(map);

// Add legend with improved styling
const legend = L.control({position: 'bottomright'});

legend.onAdd = function (map) {
    const div = L.DomUtil.create('div', 'info legend');
    
    // Create a gradient background for the legend
    const gradientHeight = 200;
    const tempRange = [-40, 50];  // min to max temperature
    
    div.innerHTML = `
        <div class="legend-title">Temperature (°C)</div>
        <div class="gradient-box" style="
            height: ${gradientHeight}px;
            background: linear-gradient(
                to bottom,
                #cc1414,  /* Very hot */
                #ff4333,  /* Hot */
                #ff9d35,  /* Warm */
                #ffed3d,  /* Mild */
                #7dfa80,  /* Cool */
                #48f5f7,  /* Cold */
                #2681f2,  /* Very cold */
                #91009b   /* Extremely cold */
            );
            width: 30px;
            margin-right: 10px;
            float: left;
        "></div>
        <div class="gradient-labels" style="margin-left: 40px;">
            <div style="height: ${gradientHeight}px; position: relative;">
                <span style="position: absolute; top: 0;">${tempRange[1]}°C</span>
                <span style="position: absolute; top: 50%;">5°C</span>
                <span style="position: absolute; bottom: 0;">${tempRange[0]}°C</span>
            </div>
        </div>
    `;

    return div;
};

legend.addTo(map);

// Add scale control in bottom left
L.control.scale({position: 'bottomleft'}).addTo(map);

// Add zoom control in bottom left
L.control.zoom({position: 'bottomleft'}).addTo(map);

// Improved loading indicator
const loadingControl = L.control({position: 'topright'});

loadingControl.onAdd = function (map) {
    const div = L.DomUtil.create('div', 'loading-control');
    div.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
    div.style.display = 'none';
    return div;
};

loadingControl.addTo(map);

// Enhanced click handling for temperature data
map.on('click', async function(e) {
    const lat = e.latlng.lat.toFixed(4);
    const lng = e.latlng.lng.toFixed(4);
    
    try {
        const response = await fetch('/time-series', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                latitude: lat,
                longitude: lng
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            console.error('Error:', data.error);
            return;
        }
        
        // Create popup content with temperature info
        const temp = data.data[data.data.length - 1].temperature.toFixed(1);
        const popupContent = `
            <div class="temperature-popup">
                <img src="/static/img/temp-icon.svg" class="weather-icon" alt="Temperature">
                <span class="temperature-value">${temp}°C</span>
                <br>
                Lat: ${lat}° N<br>
                Lon: ${lng}° E
            </div>
        `;
        
        L.popup()
            .setLatLng(e.latlng)
            .setContent(popupContent)
            .openOn(map);
            
    } catch (error) {
        console.error('Error fetching temperature data:', error);
    }
});

// Show/hide loading indicator
function showLoading() {
    document.querySelector('.loading-control').style.display = 'block';
}

function hideLoading() {
    document.querySelector('.loading-control').style.display = 'none';
}

// Add loading events
temperatureLayer.on('loading', showLoading);
temperatureLayer.on('load', hideLoading);
temperatureLayer.on('tileerror', hideLoading);

// Function to get color based on temperature using gradient interpolation
function getColor(temp) {
    // Define color stops for the gradient
    const colorStops = [
        { temp: -40, color: '#91009b' },  // Extremely cold
        { temp: -30, color: '#2681f2' },  // Very cold
        { temp: -20, color: '#48f5f7' },  // Cold
        { temp: -10, color: '#7dfa80' },  // Cool
        { temp: 0, color: '#ffed3d' },    // Mild
        { temp: 20, color: '#ff9d35' },   // Warm
        { temp: 35, color: '#ff4333' },   // Hot
        { temp: 50, color: '#cc1414' }    // Very hot
    ];

    // Find the color stops between which the temperature falls
    for (let i = 0; i < colorStops.length - 1; i++) {
        if (temp <= colorStops[i + 1].temp) {
            const t = (temp - colorStops[i].temp) / (colorStops[i + 1].temp - colorStops[i].temp);
            return interpolateColor(colorStops[i].color, colorStops[i + 1].color, t);
        }
    }
    return colorStops[colorStops.length - 1].color;
}

// Helper function to interpolate between two colors
function interpolateColor(color1, color2, t) {
    // Convert hex to RGB
    const rgb1 = hexToRgb(color1);
    const rgb2 = hexToRgb(color2);
    
    // Interpolate each component
    const r = Math.round(rgb1.r + (rgb2.r - rgb1.r) * t);
    const g = Math.round(rgb1.g + (rgb2.g - rgb1.g) * t);
    const b = Math.round(rgb1.b + (rgb2.b - rgb1.b) * t);
    
    // Convert back to hex
    return rgbToHex(r, g, b);
}

// Helper function to convert hex to RGB
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

// Helper function to convert RGB to hex
function rgbToHex(r, g, b) {
    return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}