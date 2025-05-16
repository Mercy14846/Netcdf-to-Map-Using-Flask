// Initialize the map
let map = L.map('map', {
    center: [12.04, 11.07],
    minZoom: 1,
    zoom: 4,
    zoomSnap: 1,
    maxBounds: [[-84, -Infinity], [84, Infinity]],
    maxBoundsViscosity: 1.0
});

// Add base map layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
}).addTo(map);

// Add data layer
let dataLayer = L.tileLayer('/tiles/{z}/{x}/{y}.png', {
    attribution: 'Temperature Data',
    minZoom: 1,
    maxZoom: 10,
    opacity: 0.75,
    tileSize: 256,
    updateWhenIdle: false,
    updateWhenZooming: true,
    keepBuffer: 2
}).addTo(map);

// Add debug info for tile loading
dataLayer.on('loading', function() {
    console.log('Loading tiles...');
});

dataLayer.on('load', function() {
    console.log('All tiles loaded');
});

dataLayer.on('tileloadstart', function(e) {
    console.log('Loading tile:', e.url);
});

dataLayer.on('tileerror', function(e) {
    console.error('Error loading tile:', e.url);
});

// Note: Click event handling for time series data is now managed in index.html