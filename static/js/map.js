// Initialize the map with better default view
const map = L.map('map', {
    center: [20, 0],  // Center map at equator
    zoom: 3,         // Default zoom level
    zoomControl: false,
    minZoom: 2,      // Restrict minimum zoom
    maxZoom: 18      // Maximum zoom level
});

// Add multiple base layers
const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Initialize variables
let currentYear = 1840;
let isPlaying = false;
let playInterval = null;
let heatmapLayer = null;
let currentTooltip = null;

// Initialize heatmap layer
heatmapLayer = L.heatLayer([], {
    radius: 25,
    blur: 15,
    maxZoom: 10,
    max: 1.0,
    gradient: {
        0.0: '#0000FF',  // Very Cold (-40°C)
        0.2: '#00FFFF',  // Cold
        0.4: '#00FF00',  // Cool
        0.6: '#FFFF00',  // Mild
        0.8: '#FFA500',  // Warm
        1.0: '#FF0000'   // Hot (40°C)
    }
}).addTo(map);

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

// Update heatmap data
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
            year: currentYear,
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
        
        // Transform data for heatmap
        const points = data.data.map(point => [
            point.lat,
            point.lon,
            (point.temperature + 40) / 80  // Normalize temperature to 0-1
        ]);
        
        console.log('Processed points:', points.length);
        if (points.length > 0) {
            console.log('Sample point:', points[0]);
        }
        
        // Remove existing layer and create new one
        if (heatmapLayer) {
            map.removeLayer(heatmapLayer);
        }
        
        heatmapLayer = L.heatLayer(points, {
            radius: 25,
            blur: 15,
            maxZoom: 10,
            max: 1.0,
            gradient: {
                0.0: '#0000FF',  // Very Cold (-40°C)
                0.2: '#00FFFF',  // Cold
                0.4: '#00FF00',  // Cool
                0.6: '#FFFF00',  // Mild
                0.8: '#FFA500',  // Warm
                1.0: '#FF0000'   // Hot (40°C)
            }
        }).addTo(map);
        
        hideLoading();
    })
    .catch(error => {
        console.error('Error updating heatmap:', error);
        showError('Failed to update temperature data');
        hideLoading();
    });
}

// Time controls
const yearSlider = document.getElementById('year-slider');
const currentYearDisplay = document.getElementById('current-year');
const playPauseBtn = document.getElementById('play-pause');
const prevYearBtn = document.getElementById('prev-year');
const nextYearBtn = document.getElementById('next-year');

function updateYear(year) {
    currentYear = year;
    currentYearDisplay.textContent = year;
    yearSlider.value = year;
    updateHeatmap();
}

function togglePlayPause() {
    isPlaying = !isPlaying;
    playPauseBtn.textContent = isPlaying ? '⏸' : '▶';

    if (isPlaying) {
        playInterval = setInterval(() => {
            let nextYear = parseInt(currentYear) + 1;
            if (nextYear > 2024) {
                nextYear = 1840;
            }
            updateYear(nextYear);
        }, 1000);
    } else {
        clearInterval(playInterval);
    }
}

// Event listeners
yearSlider.addEventListener('input', (e) => updateYear(parseInt(e.target.value)));
playPauseBtn.addEventListener('click', togglePlayPause);
prevYearBtn.addEventListener('click', () => updateYear(Math.max(1840, currentYear - 1)));
nextYearBtn.addEventListener('click', () => updateYear(Math.min(2024, currentYear + 1)));

// Map event listeners
map.on('moveend', updateHeatmap);
map.on('zoomend', updateHeatmap);

// Temperature tooltip
let followTooltip = L.tooltip({
    permanent: false,
    direction: 'top',
    offset: [0, -20],
    className: 'temp-tooltip'
});

// Throttle function
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Handle mousemove on map
map.on('mousemove', throttle(function(e) {
    if (map.getZoom() < 4) return; // Only show tooltip at closer zoom levels

    fetch('/time-series', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify({
            latitude: parseFloat(e.latlng.lat.toFixed(6)),
            longitude: parseFloat(e.latlng.lng.toFixed(6)),
            year: parseInt(currentYear)
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) throw new Error(data.error);
        
        const timeSeriesData = data.data[0];
        const temp = timeSeriesData.temperature;
        
        // Update tooltip content and position
        followTooltip
            .setContent(`${temp.toFixed(1)}°C`)
            .setLatLng(e.latlng);
        
        if (!map.hasLayer(followTooltip)) {
            followTooltip.addTo(map);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        if (map.hasLayer(followTooltip)) {
            map.removeLayer(followTooltip);
        }
    });
}, 100));

// Remove tooltip when mouse leaves map
map.on('mouseout', function() {
    if (map.hasLayer(followTooltip)) {
        map.removeLayer(followTooltip);
    }
});

// Add scale control
L.control.scale({position: 'bottomleft'}).addTo(map);

// Add zoom control
L.control.zoom({position: 'bottomleft'}).addTo(map);

// Initial update
updateHeatmap();