// Animation control variables
let animationData = null;
let currentFrameIndex = 0;
let animationInterval = null;
let isPlaying = false;

// Animation speed in milliseconds (2 seconds per hour)
const ANIMATION_SPEED = 2000;
const RETRY_DELAY = 1000;
const MAX_RETRIES = 3;

// Initialize animation controls
function initializeAnimation() {
    // Create animation control panel
    const controlPanel = L.control({ position: 'bottomright' });
    
    controlPanel.onAdd = function() {
        const div = L.DomUtil.create('div', 'animation-control-panel');
        div.innerHTML = `
            <div class="animation-controls">
                <button id="playPauseBtn" title="Play/Pause">
                    <i class="fas fa-pause"></i>
                </button>
                <input type="range" id="timeSlider" min="0" max="23" value="0">
                <span id="currentTime"></span>
            </div>
        `;
        return div;
    };
    
    controlPanel.addTo(map);
    
    // Add event listeners
    document.getElementById('playPauseBtn').addEventListener('click', toggleAnimation);
    document.getElementById('timeSlider').addEventListener('input', handleSliderChange);
    
    // Load animation data
    loadAnimationData();
}

// Load temperature data for animation with retry mechanism
async function loadAnimationData(retryCount = 0) {
    try {
        showLoading();
        const response = await fetch('/api/animation-data', {
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache'
            }
        });
        
        if (!response.ok) throw new Error('Failed to load animation data');
        
        animationData = await response.json();
        
        if (!animationData || !animationData.data || animationData.data.length === 0) {
            throw new Error('Invalid data format received');
        }
        
        // Initialize slider
        const slider = document.getElementById('timeSlider');
        slider.max = animationData.timestamps.length - 1;
        slider.value = 0;
        
        // Show initial frame and start animation
        updateFrame(0);
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

// Start animation automatically
function startAnimation() {
    if (!isPlaying) {
        isPlaying = true;
        document.getElementById('playPauseBtn').innerHTML = '<i class="fas fa-pause"></i>';
        animationInterval = setInterval(() => {
            currentFrameIndex = (currentFrameIndex + 1) % animationData.timestamps.length;
            updateFrame(currentFrameIndex);
        }, ANIMATION_SPEED);
    }
}

// Update the map with current frame data
function updateFrame(frameIndex) {
    if (!animationData) return;
    
    const frameData = animationData.data[frameIndex];
    document.getElementById('currentTime').textContent = frameData.hour;
    document.getElementById('timeSlider').value = frameIndex;
    
    // Update heatmap layer
    const points = frameData.points.map(point => [
        point.lat,
        point.lon,
        normalizeTemperature(point.temperature)
    ]);
    
    initHeatLayer(points);
}

// Normalize temperature value to 0-1 range for heatmap
function normalizeTemperature(temp) {
    const { min, max } = animationData.temperature_range;
    return (temp - min) / (max - min);
}

// Toggle animation play/pause
function toggleAnimation() {
    const btn = document.getElementById('playPauseBtn');
    
    if (isPlaying) {
        clearInterval(animationInterval);
        btn.innerHTML = '<i class="fas fa-play"></i>';
    } else {
        animationInterval = setInterval(() => {
            currentFrameIndex = (currentFrameIndex + 1) % animationData.timestamps.length;
            updateFrame(currentFrameIndex);
        }, ANIMATION_SPEED);
        btn.innerHTML = '<i class="fas fa-pause"></i>';
    }
    
    isPlaying = !isPlaying;
}

// Handle manual slider change
function handleSliderChange(event) {
    currentFrameIndex = parseInt(event.target.value);
    updateFrame(currentFrameIndex);
    
    // Pause animation when manually changing time
    if (isPlaying) {
        toggleAnimation();
    }
}

// Initialize animation when the map is ready
document.addEventListener('DOMContentLoaded', initializeAnimation); 