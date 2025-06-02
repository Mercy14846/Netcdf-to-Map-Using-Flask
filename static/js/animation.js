// Animation control variables
let animationData = null;
let currentFrameIndex = 0;
let animationInterval = null;
let isPlaying = false;

// Animation speed in milliseconds
const ANIMATION_SPEED = 1000;

// Initialize animation controls
function initializeAnimation() {
    // Create animation control panel
    const controlPanel = L.control({ position: 'bottomright' });
    
    controlPanel.onAdd = function() {
        const div = L.DomUtil.create('div', 'animation-control-panel');
        div.innerHTML = `
            <div class="animation-controls">
                <button id="playPauseBtn" title="Play/Pause">
                    <i class="fas fa-play"></i>
                </button>
                <input type="range" id="timeSlider" min="0" max="100" value="0">
                <span id="currentDate"></span>
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

// Load temperature data for animation
async function loadAnimationData() {
    try {
        showLoading();
        const response = await fetch('/api/animation-data');
        if (!response.ok) throw new Error('Failed to load animation data');
        
        animationData = await response.json();
        
        // Initialize slider
        const slider = document.getElementById('timeSlider');
        slider.max = animationData.timestamps.length - 1;
        slider.value = 0;
        
        // Show initial frame
        updateFrame(0);
        hideLoading();
        
    } catch (error) {
        console.error('Error loading animation data:', error);
        showError('Failed to load temperature animation data');
        hideLoading();
    }
}

// Update the map with current frame data
function updateFrame(frameIndex) {
    if (!animationData) return;
    
    const timestamp = animationData.timestamps[frameIndex];
    document.getElementById('currentDate').textContent = timestamp;
    document.getElementById('timeSlider').value = frameIndex;
    
    // Update heatmap layer
    const frameData = animationData.data.filter(d => d.time === timestamp);
    const points = frameData.map(point => [
        point.latitude,
        point.longitude,
        normalizeTemperature(point[temp_var])
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
}

// Initialize animation when the map is ready
document.addEventListener('DOMContentLoaded', initializeAnimation); 