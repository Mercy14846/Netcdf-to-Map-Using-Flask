// Initialize the map with better default view
const map = L.map('map', {
    center: [20, 0],
    zoom: 3,
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
    const temps = [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50];
    
    div.innerHTML = '<div class="legend-title">Temperature (°C)</div>';
    
    for (let i = 0; i < temps.length; i++) {
        const from = temps[i];
        const to = temps[i + 1];
        
        div.innerHTML +=
            '<i style="background:' + getColor(from + 1) + '"></i> ' +
            from + (to ? '&ndash;' + to : '+') + '<br>';
    }

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

// Function to get color based on temperature
function getColor(temp) {
    // Enhanced temperature color scale
    if (temp <= -40) return '#68001a';
    if (temp <= -35) return '#7a0024';
    if (temp <= -30) return '#8c002e';
    if (temp <= -25) return '#9e0038';
    if (temp <= -20) return '#b00042';
    if (temp <= -15) return '#c2004c';
    if (temp <= -10) return '#d40056';
    if (temp <= -5) return '#e60060';
    if (temp <= 0) return '#ff206e';
    if (temp <= 5) return '#ff4081';
    if (temp <= 10) return '#ff6094';
    if (temp <= 15) return '#ff80a7';
    if (temp <= 20) return '#ffa0ba';
    if (temp <= 25) return '#ffc0cd';
    if (temp <= 30) return '#ffe0e0';
    if (temp <= 35) return '#ffffff';
    if (temp <= 40) return '#e6ffff';
    if (temp <= 45) return '#ccffff';
    if (temp <= 50) return '#99ffff';
    return '#80ffff';
}