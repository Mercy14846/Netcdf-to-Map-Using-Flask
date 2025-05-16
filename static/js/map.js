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
    attribution: 'Map Data',
    maxZoom: 7,
    opacity: 0.75
}).addTo(map);

// Add click event handler for time series data
map.on('click', function(e) {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;
    
    // Update coordinates display
    document.getElementById('lat-update').textContent = lat.toFixed(4);
    document.getElementById('lon-update').textContent = lon.toFixed(4);
    
    // Fetch time series data
    fetch('/time-series', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            latitude: lat,
            longitude: lon
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error:', data.error);
            return;
        }
        
        // Parse the time series data
        const timeSeriesData = JSON.parse(data.data);
        
        // Update temperature display
        if (timeSeriesData.length > 0) {
            document.getElementById('temp-update').textContent = 
                timeSeriesData[timeSeriesData.length - 1].temperature.toFixed(2) + '°C';
        }
        
        // Plot time series
        plotTimeSeries(timeSeriesData);
    })
    .catch(error => console.error('Error:', error));
});

// Function to plot time series using D3.js
function plotTimeSeries(data) {
    // Clear previous chart
    d3.select('#time-series-chart').selectAll('*').remove();
    
    // Set chart dimensions
    const margin = {top: 20, right: 20, bottom: 30, left: 50};
    const width = 350 - margin.left - margin.right;
    const height = 350 - margin.top - margin.bottom;
    
    // Create SVG container
    const svg = d3.select('#time-series-chart')
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Create scales
    const x = d3.scaleTime()
        .domain(d3.extent(data, d => new Date(d.year)))
        .range([0, width]);
        
    const y = d3.scaleLinear()
        .domain([
            d3.min(data, d => d.temperature),
            d3.max(data, d => d.temperature)
        ])
        .range([height, 0]);
    
    // Add X axis
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x));
    
    // Add Y axis
    svg.append('g')
        .call(d3.axisLeft(y));
    
    // Add line
    const line = d3.line()
        .x(d => x(new Date(d.year)))
        .y(d => y(d.temperature));
    
    svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', 'steelblue')
        .attr('stroke-width', 1.5)
        .attr('d', line);
}