// Initialize the map
const map = L.map('map').setView([0, 0], 2);

// Add OpenStreetMap tiles as the base layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

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

// Create a custom temperature layer
const temperatureLayer = L.tileLayer('/tiles/{z}/{x}/{y}.png', {
    maxZoom: 19,
    opacity: 0.7,
    attribution: 'Temperature Data'
}).addTo(map);

// Add legend
const legend = L.control({position: 'bottomright'});

legend.onAdd = function (map) {
    const div = L.DomUtil.create('div', 'info legend');
    const temps = [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50];
    
    div.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
    div.style.padding = '10px';
    div.style.borderRadius = '5px';
    div.style.lineHeight = '18px';
    div.style.color = '#555';

    // Add legend title
    div.innerHTML = '<strong>Temperature (°C)</strong><br>';

    // Add colored boxes for each interval
    for (let i = 0; i < temps.length; i++) {
        const from = temps[i];
        const to = temps[i + 1];
        
        div.innerHTML +=
            '<i style="background:' + getColor(from + 1) + '; width: 18px; height: 18px; float: left; margin-right: 8px; opacity: 0.7"></i> ' +
            from + (to ? '&ndash;' + to : '+') + '<br>';
    }

    return div;
};

legend.addTo(map);

// Add scale control
L.control.scale().addTo(map);

// Add zoom control
map.zoomControl.setPosition('bottomleft');

// Function to update temperature layer opacity
function updateOpacity(value) {
    temperatureLayer.setOpacity(value);
}

// Add opacity control
const opacityControl = L.control({position: 'bottomright'});

opacityControl.onAdd = function (map) {
    const div = L.DomUtil.create('div', 'opacity-control');
    div.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
    div.style.padding = '10px';
    div.style.marginBottom = '10px';
    div.style.borderRadius = '5px';
    
    div.innerHTML = `
        <strong>Layer Opacity</strong><br>
        <input type="range" min="0" max="1" step="0.1" value="0.7" 
               onchange="updateOpacity(this.value)" 
               style="width: 100px;">
    `;
    
    return div;
};

opacityControl.addTo(map);

// Add loading indicator
const loadingControl = L.control({position: 'topright'});

loadingControl.onAdd = function (map) {
    const div = L.DomUtil.create('div', 'loading-control');
    div.style.display = 'none';
    div.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
    div.style.padding = '10px';
    div.style.borderRadius = '5px';
    div.innerHTML = 'Loading...';
    return div;
};

loadingControl.addTo(map);

// Show/hide loading indicator
function showLoading() {
    document.querySelector('.loading-control').style.display = 'block';
}

function hideLoading() {
    document.querySelector('.loading-control').style.display = 'none';
}

// Add event listeners for tile loading
temperatureLayer.on('loading', showLoading);
temperatureLayer.on('load', hideLoading);
temperatureLayer.on('tileerror', hideLoading);

// Note: Click event handling for time series data is now managed in index.html