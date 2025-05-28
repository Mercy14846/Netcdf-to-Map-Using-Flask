// Initialize the map with better default view
const map = L.map('map', {
    center: [20, 0],  // Center map at equator
    zoom: 3,         // Default zoom level similar to OpenWeatherMap
    zoomControl: false,
    minZoom: 2,      // Restrict minimum zoom
    maxZoom: 18      // Maximum zoom level
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

// Create temperature layer with improved options
let heatmapLayer = null;
let legendControl = null;

// Create layer controls in OpenWeatherMap style
const layerControl = L.control({position: 'topleft'});
layerControl.onAdd = function (map) {
    const div = L.DomUtil.create('div', 'layer-control');
    div.innerHTML = `
        <div class="layer-button">
            <button id="toggleTemp" class="control-button">
                <i class="fas fa-temperature-high"></i> Temperature
            </button>
        </div>
    `;
    return div;
};
layerControl.addTo(map);

// Extend L.Canvas to set willReadFrequently
const CanvasLayer = L.Canvas.extend({
    _initContainer: function() {
        L.Canvas.prototype._initContainer.call(this);
        this._ctx.canvas.setAttribute('willReadFrequently', 'true');
    }
});

// Define the color gradient stops
const colorGradient = {
    0.0: '#91003f',  // Deep purple (Very cold)
    0.1: '#7f1f7f',  // Purple
    0.2: '#4c2c9b',  // Blue-purple
    0.3: '#1f3b9b',  // Dark blue
    0.4: '#2f7eb6',  // Medium blue
    0.5: '#40b6e5',  // Light blue
    0.6: '#6be5bf',  // Turquoise
    0.7: '#8fef73',  // Light green
    0.8: '#efef45',  // Yellow
    0.9: '#ef4524',  // Red
    1.0: '#cc0000'   // Deep red (Very hot)
};

// Create legend control
function createLegend(min, max, segments) {
    if (legendControl) {
        map.removeControl(legendControl);
    }

    legendControl = L.control({position: 'bottomright'});
    legendControl.onAdd = function (map) {
        const div = L.DomUtil.create('div', 'info legend');
        
        // Create legend title
        div.innerHTML = '<div class="legend-title">Temperature (°C)</div>';
        div.innerHTML += '<div class="legend-container">';
        
        // Create continuous gradient bar
        const gradientBar = document.createElement('div');
        gradientBar.className = 'gradient-bar';
        
        // Create gradient background
        const stops = Object.entries(colorGradient)
            .map(([pos, color]) => `${color} ${pos * 100}%`)
            .join(', ');
        gradientBar.style.background = `linear-gradient(to right, ${stops})`;
        
        div.appendChild(gradientBar);

        // Add temperature labels
        const labelContainer = document.createElement('div');
        labelContainer.className = 'temp-labels';
        
        // Add labels for each segment
        segments.forEach((segment, index) => {
            const label = document.createElement('span');
            label.className = 'temp-label';
            label.style.left = `${(index / (segments.length - 1)) * 100}%`;
            label.textContent = `${Math.round(segment.start)}°`;
            labelContainer.appendChild(label);
        });
        
        // Add the final label
        const finalLabel = document.createElement('span');
        finalLabel.className = 'temp-label';
        finalLabel.style.left = '100%';
        finalLabel.textContent = `${Math.round(max)}°`;
        labelContainer.appendChild(finalLabel);
        
        div.appendChild(labelContainer);
        div.innerHTML += '</div>';
        
        return div;
    };
    legendControl.addTo(map);
}

// Fetch heatmap data and initialize the layer
fetch('/api/heatmap-data')
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error:', data.error);
            return;
        }

        // Calculate optimal radius based on data resolution
        const latRes = data.resolution.lat;
        const lonRes = data.resolution.lon;
        const avgRes = (latRes + lonRes) / 2;
        const baseRadius = Math.max(5, Math.ceil(1 / avgRes));

        // Create heatmap layer with custom configuration
        heatmapLayer = new L.HeatLayer(data.data, {
            radius: baseRadius,
            minOpacity: 0.35,
            gradient: colorGradient,
        }).addTo(map);

        // Set map bounds based on data extent with some padding
        const latPad = (data.bounds.lat[1] - data.bounds.lat[0]) * 0.1;
        const lonPad = (data.bounds.lon[1] - data.bounds.lon[0]) * 0.1;
        map.fitBounds([
            [data.bounds.lat[0] - latPad, data.bounds.lon[0] - lonPad],
            [data.bounds.lat[1] + latPad, data.bounds.lon[1] + lonPad]
        ]);

        // Create legend with segments
        createLegend(data.min, data.max, data.segments);

        // Set up toggle button
        const toggleButton = document.getElementById('toggleTemp');
        let isVisible = true;

        toggleButton.addEventListener('click', () => {
            if (isVisible) {
                map.removeLayer(heatmapLayer);
                toggleButton.classList.remove('active');
            } else {
                heatmapLayer.addTo(map);
                toggleButton.classList.add('active');
            }
            isVisible = !isVisible;
        });

        // Initially activate the button
        toggleButton.classList.add('active');
    })
    .catch(error => {
        console.error('Error loading heatmap data:', error);
    });

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
            if (tempCheckbox.checked && heatmapLayer) {
                heatmapLayer.addTo(map);
            } else if (heatmapLayer) {
                map.removeLayer(heatmapLayer);
            }
        });

        opacitySlider.addEventListener('input', (e) => {
            if (heatmapLayer) {
                heatmapLayer.setOptions({ opacity: e.target.value });
            }
        });
    }, 0);

    return div;
};

controlPanel.addTo(map);

// Add scale control in bottom left
L.control.scale({position: 'bottomleft'}).addTo(map);

// Add zoom control in bottom left
L.control.zoom({position: 'bottomleft'}).addTo(map);

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

// Function to get color based on temperature using gradient interpolation
function getColor(temp) {
    // Define color stops for the gradient with brighter colors
    const colorStops = [
        { temp: -40, color: '#0000FF' },  // Bright Blue (Very cold)
        { temp: -30, color: '#00FFFF' },  // Cyan
        { temp: -20, color: '#00FF90' },  // Bright Turquoise
        { temp: -10, color: '#00FF00' },  // Bright Green
        { temp: -5, color: '#80FF00' },   // Lime Green
        { temp: 0, color: '#FFFF00' },    // Bright Yellow
        { temp: 5, color: '#FFC000' },    // Bright Orange
        { temp: 10, color: '#FF8000' },   // Dark Orange
        { temp: 15, color: '#FF4000' },   // Light Red
        { temp: 20, color: '#FF0000' },   // Pure Red
        { temp: 25, color: '#FF0040' },   // Red-Pink
        { temp: 30, color: '#FF0080' },   // Bright Pink
        { temp: 35, color: '#FF00FF' },   // Magenta
        { temp: 40, color: '#800080' }    // Purple
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