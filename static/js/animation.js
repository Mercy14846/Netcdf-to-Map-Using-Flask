// Animation control variables
let animationData = null;
let currentFrameIndex = 0;
let animationInterval = null;
let isPlaying = false;

// Animation configuration
const ANIMATION_SPEED = 2000;  // 2 seconds per frame
const RETRY_DELAY = 1000;
const MAX_RETRIES = 3;
const CACHE_SIZE = 50;
const CHUNK_SIZE = 1000;

// Initialize animation controls
function initializeAnimation() {
    const controlPanel = L.control({ position: 'bottomright' });
    
    controlPanel.onAdd = function() {
        const div = L.DomUtil.create('div', 'animation-control-panel');
        div.innerHTML = `
            <div class="animation-controls">
                <button id="playPauseBtn" title="Play/Pause">
                    <i class="fas fa-play"></i>
                </button>
                <input type="range" id="timeSlider" min="0" max="23" value="0">
                <span id="currentTime">00:00</span>
            </div>
        `;
        return div;
    };
    
    controlPanel.addTo(map);
    
    // Add event listeners
    document.getElementById('playPauseBtn').addEventListener('click', toggleAnimation);
    document.getElementById('timeSlider').addEventListener('input', handleSliderChange);
    
    // Load initial data
    loadAnimationData();
}

// Load animation data with optimizations
async function loadAnimationData(retryCount = 0) {
    try {
        showLoading();
        
        // Check cache first
        const cacheKey = 'animation_data';
        if (dataCache.has(cacheKey)) {
            animationData = dataCache.get(cacheKey);
            initializeAnimationControls();
            hideLoading();
            return;
        }
        
        const response = await fetch('/api/animation-data', {
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'max-age=3600'
            }
        });
        
        if (!response.ok) throw new Error('Failed to load animation data');
        
        const rawData = await response.json();
        
        if (!rawData || !rawData.data || rawData.data.length === 0) {
            throw new Error('Invalid data format received');
        }
        
        // Process data in chunks
        animationData = await processAnimationData(rawData);
        
        // Cache the processed data
        dataCache.set(cacheKey, animationData);
        manageCache();
        
        // Initialize controls and start animation
        initializeAnimationControls();
        startAnimation();
        hideLoading();
        
    } catch (error) {
        console.error('Error loading animation data:', error);
        
        if (retryCount < MAX_RETRIES) {
            console.log(`Retrying data load (${retryCount + 1}/${MAX_RETRIES})...`);
            setTimeout(() => {
                loadAnimationData(retryCount + 1);
            }, RETRY_DELAY * (retryCount + 1));
        } else {
            showError('Failed to load temperature animation data. Please refresh the page.');
            hideLoading();
        }
    }
}

// Process animation data in chunks
async function processAnimationData(rawData) {
    const processedData = {
        timestamps: rawData.timestamps,
        temperature_range: rawData.temperature_range,
        colors: rawData.colors,
        data: []
    };
    
    // Process each frame's data in chunks
    for (const frame of rawData.data) {
        const chunks = chunkArray(frame.points, CHUNK_SIZE);
        const processedPoints = [];
        
        for (const chunk of chunks) {
            const processed = await processPointsChunk(chunk);
            processedPoints.push(...processed);
            
            // Allow UI updates
            await new Promise(resolve => setTimeout(resolve, 0));
        }
        
        processedData.data.push({
            hour: frame.hour,
            points: processedPoints
        });
    }
    
    return processedData;
}

// Process a chunk of points
function processPointsChunk(points) {
    return points.map(point => ({
        lat: point.lat,
        lon: point.lon,
        temperature: normalizeTemperature(point.temperature)
    }));
}

// Split array into chunks
function chunkArray(array, size) {
    const chunks = [];
    for (let i = 0; i < array.length; i += size) {
        chunks.push(array.slice(i, i + size));
    }
    return chunks;
}

// Manage cache size
function manageCache() {
    if (dataCache.size > CACHE_SIZE) {
        const oldestKey = dataCache.keys().next().value;
        dataCache.delete(oldestKey);
    }
}

// Initialize animation controls
function initializeAnimationControls() {
    const slider = document.getElementById('timeSlider');
    slider.max = animationData.timestamps.length - 1;
    slider.value = currentFrameIndex;
    updateFrame(currentFrameIndex);
}

// Update frame with optimized rendering
function updateFrame(frameIndex) {
    if (!animationData || !animationData.data[frameIndex]) return;
    
    const frameData = animationData.data[frameIndex];
    document.getElementById('currentTime').textContent = frameData.hour;
    document.getElementById('timeSlider').value = frameIndex;
    
    // Update heatmap layer efficiently
    const points = frameData.points.map(point => [
        point.lat,
        point.lon,
        point.temperature
    ]);
    
    // Create or update heatmap layer
    if (!heatmapLayer) {
        initHeatLayer(points);
    } else {
        requestAnimationFrame(() => {
            heatmapLayer.setLatLngs(points);
        });
    }
}

// Toggle animation playback
function toggleAnimation() {
    if (!animationData) return;
    
    isPlaying = !isPlaying;
    const btn = document.getElementById('playPauseBtn');
    btn.innerHTML = isPlaying ? '<i class="fas fa-pause"></i>' : '<i class="fas fa-play"></i>';
    
    if (isPlaying) {
        startAnimation();
    } else {
        stopAnimation();
    }
}

// Start animation with performance optimization
function startAnimation() {
    if (!isPlaying || !animationData) return;
    
    stopAnimation(); // Clear any existing interval
    
    animationInterval = setInterval(() => {
        currentFrameIndex = (currentFrameIndex + 1) % animationData.data.length;
        requestAnimationFrame(() => {
            updateFrame(currentFrameIndex);
        });
    }, ANIMATION_SPEED);
}

// Stop animation
function stopAnimation() {
    if (animationInterval) {
        clearInterval(animationInterval);
        animationInterval = null;
    }
}

// Handle slider changes
function handleSliderChange(e) {
    currentFrameIndex = parseInt(e.target.value);
    updateFrame(currentFrameIndex);
}

// Normalize temperature value
function normalizeTemperature(temp) {
    if (!animationData || !animationData.temperature_range) return 0;
    
    const min = animationData.temperature_range.min;
    const max = animationData.temperature_range.max;
    return Math.max(0, Math.min(1, (temp - min) / (max - min)));
}

// Initialize animation when the page loads
document.addEventListener('DOMContentLoaded', initializeAnimation); 