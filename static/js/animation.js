// Animation control variables
let animationData = null;
let currentFrameIndex = 0;
let animationInterval = null;
let isPlaying = false;
let lastFrameTime = 0;

// Animation configuration
const ANIMATION_SPEED = 2000;  // 2 seconds per frame
const TRANSITION_DURATION = 500;  // 500ms transition duration
const RETRY_DELAY = 1000;
const MAX_RETRIES = 3;
const CACHE_SIZE = 50;
const MIN_FRAME_TIME = 16;  // ~60fps cap

// Animation state
let currentFrame = null;
let nextFrame = null;
let transitionProgress = 0;
let dataCache = new Map();

// Debug logging
const DEBUG = true;
function log(...args) {
    if (DEBUG) {
        console.log('[Animation]', ...args);
    }
}

// Error handling
function handleError(error, context) {
    console.error(`[Animation Error] ${context}:`, error);
    showError(`Animation error: ${context}. Please refresh the page.`);
}

// Cache management
function manageCache() {
    try {
        if (dataCache.size > CACHE_SIZE) {
            const keysIterator = dataCache.keys();
            const oldestKey = keysIterator.next().value;
            dataCache.delete(oldestKey);
            log('Cache cleaned, removed:', oldestKey);
        }
    } catch (error) {
        handleError(error, 'Cache management');
    }
}

// Initialize animation controls
function initializeAnimation() {
    try {
        log('Initializing animation controls...');
        
        const controlPanel = L.control({ position: 'bottomright' });
        
        controlPanel.onAdd = function() {
            log('Creating control panel...');
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
        log('Control panel added to map');
        
        // Add event listeners with error handling
        const playPauseBtn = document.getElementById('playPauseBtn');
        const timeSlider = document.getElementById('timeSlider');
        
        if (!playPauseBtn || !timeSlider) {
            throw new Error('Required control elements not found');
        }
        
        playPauseBtn.addEventListener('click', toggleAnimation);
        timeSlider.addEventListener('input', handleSliderChange);
        
        // Load initial data
        log('Loading initial animation data...');
        loadAnimationData();
    } catch (error) {
        handleError(error, 'Initialization');
    }
}

// Load animation data with optimizations and error handling
async function loadAnimationData(retryCount = 0) {
    try {
        log('Loading animation data, attempt:', retryCount + 1);
        showLoading();
        
        // Check cache first
        const cacheKey = 'animation_data';
        if (dataCache.has(cacheKey)) {
            log('Using cached animation data');
            animationData = dataCache.get(cacheKey);
            initializeAnimationControls();
            hideLoading();
            return;
        }
        
        log('Fetching fresh animation data...');
        const response = await fetch('/api/animation-data', {
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'max-age=3600'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const rawData = await response.json();
        log('Received raw data:', rawData);
        
        if (!rawData || !rawData.data || rawData.data.length === 0) {
            throw new Error('Invalid data format received');
        }
        
        // Process data in chunks
        log('Processing animation data...');
        animationData = await processAnimationData(rawData);
        log('Animation data processed:', animationData);
        
        // Cache the processed data
        dataCache.set(cacheKey, animationData);
        manageCache();
        
        // Initialize controls and start animation
        log('Initializing controls and starting animation...');
        initializeAnimationControls();
        startAnimation();
        hideLoading();
        
    } catch (error) {
        console.error('Error loading animation data:', error);
        
        if (retryCount < MAX_RETRIES) {
            log(`Retrying data load (${retryCount + 1}/${MAX_RETRIES})...`);
            setTimeout(() => {
                loadAnimationData(retryCount + 1);
            }, RETRY_DELAY * (retryCount + 1));
        } else {
            handleError(error, 'Data loading failed after max retries');
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

// Initialize animation controls
function initializeAnimationControls() {
    const slider = document.getElementById('timeSlider');
    slider.max = animationData.timestamps.length - 1;
    slider.value = currentFrameIndex;
    updateFrame(currentFrameIndex);
    
    // Start playing automatically
    isPlaying = true;
    const playPauseBtn = document.getElementById('playPauseBtn');
    playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
    startAnimation();
}

// Update frame with optimized rendering and flow effect
function updateFrame(frameIndex) {
    if (!animationData || !animationData.data[frameIndex]) return;
    
    const frameData = animationData.data[frameIndex];
    document.getElementById('currentTime').textContent = frameData.hour;
    document.getElementById('timeSlider').value = frameIndex;
    
    // Prepare current and next frame data
    currentFrame = frameData.points;
    const nextIndex = (frameIndex + 1) % animationData.data.length;
    nextFrame = animationData.data[nextIndex].points;
    
    // Interpolate between frames for smooth transition
    const points = interpolateFrames(currentFrame, nextFrame, transitionProgress);
    
    // Update heatmap with flow effect
    if (typeof window.updateHeatmapLayer === 'function') {
        window.updateHeatmapLayer(points, TRANSITION_DURATION);
    } else {
        // Fallback if updateHeatmapLayer is not available
        if (!heatmapLayer) {
            initHeatLayer(points);
        } else {
            heatmapLayer.setLatLngs(points);
        }
    }
}

// Interpolate between two frames for smooth transitions
function interpolateFrames(frame1, frame2, progress) {
    if (!frame1 || !frame2) return frame1 || frame2;
    
    return frame1.map((point, i) => {
        const next = frame2[i] || point;
        return [
            point.lat + (next.lat - point.lat) * progress,
            point.lon + (next.lon - point.lon) * progress,
            point.temperature + (next.temperature - point.temperature) * progress
        ];
    });
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

// Start animation with improved timing
function startAnimation() {
    if (!isPlaying || !animationData) return;
    
    stopAnimation(); // Clear any existing animation
    
    lastFrameTime = performance.now();
    
    function animate(currentTime) {
        if (!isPlaying) return;
        
        const deltaTime = currentTime - lastFrameTime;
        
        // Ensure minimum frame time for performance
        if (deltaTime < MIN_FRAME_TIME) {
            animationInterval = requestAnimationFrame(animate);
            return;
        }
        
        transitionProgress = (deltaTime % ANIMATION_SPEED) / ANIMATION_SPEED;
        
        if (deltaTime >= ANIMATION_SPEED) {
            currentFrameIndex = (currentFrameIndex + 1) % animationData.data.length;
            lastFrameTime = currentTime;
        }
        
        try {
            updateFrame(currentFrameIndex);
        } catch (error) {
            handleError(error, 'Frame update');
            stopAnimation();
            return;
        }
        
        animationInterval = requestAnimationFrame(animate);
    }
    
    animationInterval = requestAnimationFrame(animate);
}

// Stop animation with cleanup
function stopAnimation() {
    if (animationInterval) {
        cancelAnimationFrame(animationInterval);
        animationInterval = null;
    }
}

// Handle slider changes with validation
function handleSliderChange(e) {
    try {
        const newIndex = parseInt(e.target.value);
        if (isNaN(newIndex) || newIndex < 0 || !animationData || newIndex >= animationData.data.length) {
            throw new Error('Invalid slider value');
        }
        currentFrameIndex = newIndex;
        updateFrame(currentFrameIndex);
    } catch (error) {
        handleError(error, 'Slider change');
    }
}

// Normalize temperature value
function normalizeTemperature(temp) {
    if (!animationData || !animationData.temperature_range) return 0;
    
    const min = animationData.temperature_range.min;
    const max = animationData.temperature_range.max;
    return Math.max(0, Math.min(1, (temp - min) / (max - min)));
}

// Initialize animation when the page loads
document.addEventListener('DOMContentLoaded', () => {
    log('DOM loaded, checking map initialization...');
    if (typeof map !== 'undefined') {
        log('Map is ready, initializing animation...');
        initializeAnimation();
    } else {
        console.error('Map not initialized');
        setTimeout(() => {
            log('Retrying animation initialization...');
            if (typeof map !== 'undefined') {
                initializeAnimation();
            } else {
                showError('Failed to initialize map. Please refresh the page.');
            }
        }, 1000);
    }
}); 