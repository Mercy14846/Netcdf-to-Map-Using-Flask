# import libraries
from flask import Flask, send_file, render_template, jsonify, request
from flask_caching import Cache
# from flask_socketio import SocketIO

import io
import math
import numpy as np
from PIL import Image, ImageDraw
import datashader as ds
import pandas as pd
import xarray as xr
import colorcet
import matplotlib.colors as mcolors

from datashader import transfer_functions as tf
from datashader.utils import lnglat_to_meters
import threading
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
import sys
from functools import lru_cache
from branca.colormap import LinearColormap
import dask.array as da
from concurrent.futures import ThreadPoolExecutor
import gc
import psutil
from dataclasses import dataclass
from datetime import datetime

# Increase recursion limit if needed
sys.setrecursionlimit(10000)

# Configuration
@dataclass
class Config:
    CACHE_TYPE: str = 'simple'
    CACHE_DEFAULT_TIMEOUT: int = 7200  # 2 hours
    CACHE_THRESHOLD: int = 5000  # Store more items in cache
    MEMORY_THRESHOLD: float = 0.85  # 85% memory usage threshold
    MAX_WORKERS: int = 4
    CHUNK_SIZE: int = 200
    DEBUG: bool = True

config = Config()

app = Flask(__name__)
app.config.from_object(config)

# Configure Flask-Caching
cache = Cache(app)

# Global variables for data
data = None
time_data = None
temp_var = None
min_val = None
max_val = None
lon_array = None
lat_array = None
data_array = None
time_data_var = None

# Configure thread pool
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# Memory monitoring
memory_lock = threading.Lock()

class MemoryManager:
    @staticmethod
    def check_memory_usage() -> float:
        memory = psutil.virtual_memory()
        return memory.percent / 100.0

    @staticmethod
    def clear_memory_if_needed() -> bool:
        with memory_lock:
            if MemoryManager.check_memory_usage() > config.MEMORY_THRESHOLD:
                gc.collect()
                return True
        return False

    @staticmethod
    def get_optimal_chunk_size() -> int:
        available_memory = psutil.virtual_memory().available
        return max(50, min(200, int(available_memory / (1024 * 1024 * 1024))))

class DataManager:
    _instance = None
    data: Optional[xr.Dataset] = None
    time_data: Optional[xr.Dataset] = None
    temp_var: Optional[str] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    lon_array: Optional[np.ndarray] = None
    lat_array: Optional[np.ndarray] = None
    data_array: Optional[xr.DataArray] = None
    time_data_var: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance

    def load_data(self):
        try:
            print("\nDEBUG: Starting data loading process...")
            chunk_size = MemoryManager.get_optimal_chunk_size()
            
            self.data = xr.open_dataset(
                "static/data/temp_2m.nc",
                chunks={'latitude': chunk_size, 'longitude': chunk_size},
                engine='netcdf4',
                cache=True
            )
            
            self.time_data = xr.open_dataset(
                "static/data/temperature.nc",
                chunks={'time': chunk_size},
                engine='netcdf4',
                cache=True
            )
            
            # Find temperature variable
            temp_vars = ['tmin', 'temp', 'temperature', 't2m']
            self.temp_var = next((var for var in self.data.data_vars if var in temp_vars), None)
            
            if self.temp_var is None:
                if len(self.data.data_vars) > 0:
                    self.temp_var = list(self.data.data_vars)[0]
                else:
                    raise ValueError("No variables found in temp_2m.nc")
            
            # Pre-compute arrays
            self.data_array = self.data[self.temp_var].persist()
            self.lon_array = self.data.longitude.persist()
            self.lat_array = self.data.latitude.persist()
            
            # Calculate temperature range
            self.calculate_temp_range()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    @lru_cache(maxsize=1000)
    def calculate_temp_range(self) -> Tuple[float, float]:
        """Calculate temperature range with caching"""
        sample_data = self.data_array.data.compute_chunk_sizes()
        self.min_val = float(da.min(sample_data).compute())
        self.max_val = float(da.max(sample_data).compute())
        return self.min_val, self.max_val

# Initialize data manager
data_manager = DataManager()
data_manager.load_data()

# https://github.com/ScottSyms/tileshade/
def tile2mercator(longitudetile, latitudetile, zoom):
    # takes the zoom and tile path and passes back the EPSG:3857
    # coordinates of the top left of the tile.
    # From Openstreetmap
    n = 2.0 ** zoom
    lon_deg = longitudetile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * latitudetile / n)))
    lat_deg = math.degrees(lat_rad)

    # Convert the results of the degree calulation above and convert
    # to meters for web map presentation
    mercator = lnglat_to_meters(lon_deg, lat_deg)
    return mercator

def create_empty_tile():
    # Create a transparent 256x256 RGBA image
    img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
    return img

def create_colormap():
    # Define temperature breakpoints and corresponding colors with brighter values
    temps = [-40, -30, -20, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
    colors = [
        '#0000FF',  # Bright Blue (Very cold)
        '#00FFFF',  # Cyan (Cold)
        '#00FF90',  # Bright Turquoise
        '#00FF00',  # Bright Green
        '#80FF00',  # Lime Green
        '#FFFF00',  # Bright Yellow
        '#FFC000',  # Bright Orange
        '#FF8000',  # Dark Orange
        '#FF4000',  # Light Red
        '#FF0000',  # Pure Red
        '#FF0040',  # Red-Pink
        '#FF0080',  # Bright Pink
        '#FF00FF',  # Magenta
        '#800080'   # Purple
    ]
    
    # Create normalized temperature values (0 to 1)
    norm_temps = [(t - min(temps)) / (max(temps) - min(temps)) for t in temps]
    
    # Create the colormap with exact color stops
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_temp',
        list(zip(norm_temps, colors)),
        N=256  # Increase color resolution for smoother gradients
    )
    
    return custom_cmap

# https://github.com/ScottSyms/tileshade/
# changes made: snapping values to ensure continuous tiles; use of quadmesh instead of points; syntax changes to work with Flask.
def generateatile(zoom, longitude, latitude):
    cache_key = f"tile_{zoom}_{longitude}_{latitude}"
    cached_tile = cache.get(cache_key)
    if cached_tile is not None:
        return cached_tile

    try:
        # Clear memory if needed
        MemoryManager.clear_memory_if_needed()
        
        # Convert coordinates and calculate bounds
        xleft, yleft = tile2mercator(int(longitude), int(latitude), int(zoom))
        xright, yright = tile2mercator(int(longitude)+1, int(latitude)+1, int(zoom))
        
        # Optimize buffer calculation
        buffer_factor = min(0.1, 1.0 / (2 ** zoom))  # Adaptive buffer based on zoom
        
        # Calculate bounds with buffer
        bounds = calculate_bounds(xleft, xright, yleft, yright, buffer_factor)
        
        # Get data efficiently using dask
        frame = get_frame_data(bounds)
        
        if frame.size == 0:
            return create_empty_tile()

        # Optimize resolution and rendering
        resolution = calculate_optimal_resolution(zoom)
        points = process_frame_data(frame, resolution)
        
        # Generate tile image
        img = render_tile(points, resolution)
        
        # Cache the result
        cache.set(cache_key, img, timeout=3600)
        return img

    except Exception as e:
        print(f"Error generating tile: {str(e)}")
        import traceback
        traceback.print_exc()
        return create_empty_tile()

@lru_cache(maxsize=100)
def calculate_bounds(xleft, xright, yleft, yright, buffer_factor):
    """Calculate buffered bounds with caching"""
    x_buffer = abs(xright - xleft) * buffer_factor
    y_buffer = abs(yleft - yright) * buffer_factor
    return {
        'xleft': min(xleft, xright) - x_buffer,
        'xright': max(xleft, xright) + x_buffer,
        'yleft': max(yleft, yright) + y_buffer,
        'yright': min(yleft, yright) - y_buffer
    }

def get_frame_data(bounds):
    """Efficiently get frame data using dask"""
    return data_array.sel(
        longitude=slice(bounds['xleft'], bounds['xright']),
        latitude=slice(bounds['yleft'], bounds['yright'])
    ).persist()

def process_frame_data(frame, resolution):
    """Process frame data in parallel"""
    df = frame.to_dataframe().reset_index()
    if df.empty or df[temp_var].isna().all():
        return []
        
    # Process in parallel using dask
    points = da.from_pandas(df, npartitions=4)
    return points.map_partitions(lambda x: x.dropna()).compute()

def calculate_optimal_resolution(zoom):
    """Calculate optimal resolution based on zoom level"""
    base_resolution = 256
    if zoom < 4:
        return min(base_resolution, 2 ** (zoom + 4))
    return base_resolution * min(2, zoom // 4)

def render_tile(points, resolution):
    """Render tile image from points data"""
    try:
        # Create canvas with calculated resolution
        canvas = ds.Canvas(
            plot_width=resolution,
            plot_height=resolution
        )

        # Convert points to DataFrame
        df = pd.DataFrame(points, columns=['latitude', 'longitude', temp_var])
        
        # Create aggregation
        agg = canvas.points(
            df,
            x='longitude',
            y='latitude',
            agg=ds.mean(temp_var)
        )
        
        if agg is None:
            return create_empty_tile()

        # Calculate dynamic range for better visualization
        min_temp = float(df[temp_var].min())
        max_temp = float(df[temp_var].max())
        temp_range = max_temp - min_temp
        
        if temp_range < 1.0:
            mean_temp = (max_temp + min_temp) / 2
            min_temp = mean_temp - 5
            max_temp = mean_temp + 5
        else:
            padding = temp_range * 0.1
            min_temp -= padding
            max_temp += padding

        # Shade the data
        img = tf.shade(
            agg,
            cmap=create_colormap(),
            span=[min_temp, max_temp],
            how='linear'
        )
        
        # Convert to RGBA with enhanced alpha channel
        img_data = np.array(img.data)
        alpha = np.where(np.isnan(agg.values), 0, 255)
        normalized_values = (agg.values - min_temp) / (max_temp - min_temp)
        alpha = np.where(
            ~np.isnan(agg.values),
            np.maximum(100, np.minimum(255, normalized_values * 255)),
            0
        ).astype(np.uint8)
        
        # Create final RGBA image
        rgba = np.zeros((img_data.shape[0], img_data.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = img_data[..., :3]
        rgba[..., 3] = alpha
        
        # Resize to 256x256 if needed
        pil_img = Image.fromarray(rgba, mode='RGBA')
        if resolution != 256:
            pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
            
        return pil_img
        
    except Exception as e:
        print(f"Error rendering tile: {str(e)}")
        return create_empty_tile()

@app.route("/")
def index():
    return render_template('index.html')

def create_error_tile(error_message):
    """Create a tile with an error message"""
    img = Image.new('RGBA', (256, 256), (255, 255, 255, 128))
    draw = ImageDraw.Draw(img)
    
    # Wrap text to fit tile
    words = error_message.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        if len(' '.join(current_line)) > 20:  # Adjust based on tile size
            if len(current_line) > 1:
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw error message
    y = 128 - (len(lines) * 10)  # Center text vertically
    for line in lines:
        # Calculate text width to center horizontally
        bbox = draw.textbbox((0, 0), line)
        text_width = bbox[2] - bbox[0]
        x = (256 - text_width) // 2
        draw.text((x, y), line, fill=(255, 0, 0, 255))
        y += 20
    
    return img

@app.route("/tiles/<int:zoom>/<int:longitude>/<int:latitude>.png")
@cache.memoize(timeout=7200)
def tile(longitude: int, latitude: int, zoom: int):
    # Validate input parameters
    if not (0 <= zoom <= 20):
        return send_file(create_error_tile("Invalid zoom level"), mimetype='image/png')
    
    if not (0 <= longitude < 2**zoom and 0 <= latitude < 2**zoom):
        return send_file(create_error_tile("Invalid tile coordinates"), mimetype='image/png')
    
    # Clear memory if needed
    MemoryManager.clear_memory_if_needed()
    
    # Generate tile
    results = generateatile(zoom, longitude, latitude)
    
    # Convert to bytes
    results_bytes = io.BytesIO()
    if isinstance(results, Image.Image):
        try:
            results.save(results_bytes, format='PNG', optimize=True)
            results_bytes.seek(0)
            return send_file(results_bytes, mimetype='image/png')
        except Exception as e:
            print(f"Error saving tile image: {str(e)}")
            return send_file(create_error_tile("Error saving tile"), mimetype='image/png')
    
    return send_file(create_error_tile("Invalid tile data"), mimetype='image/png')

@app.route('/api/heatmap-data', methods=['POST'])
@cache.memoize(timeout=300)  # 5 minute cache
def get_heatmap_data():
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify(error="No data received"), 400

        # Force year to be 2024
        year = 2024
        bounds = request_data.get('bounds', {})
        zoom = request_data.get('zoom', 2)

        print(f"\nDEBUG: Processing request for year 2024, zoom={zoom}")
        print(f"DEBUG: Bounds data: {bounds}")

        # Get bounds with default values
        try:
            south = float(bounds.get('_southWest', {}).get('lat', -90))
            north = float(bounds.get('_northEast', {}).get('lat', 90))
            west = float(bounds.get('_southWest', {}).get('lng', -180))
            east = float(bounds.get('_northEast', {}).get('lng', 180))
        except (TypeError, ValueError) as e:
            print(f"DEBUG: Error parsing bounds: {e}")
            return jsonify(error=f"Invalid bounds format: {str(e)}"), 400

        print(f"DEBUG: Processed bounds: N={north}, S={south}, E={east}, W={west}")

        # Validate bounds
        if not (-90 <= south <= 90 and -90 <= north <= 90):
            return jsonify(error="Invalid latitude bounds"), 400
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            return jsonify(error="Invalid longitude bounds"), 400

        # Adjust resolution based on zoom level
        if zoom < 3:
            step = 4  # Lower resolution for zoomed out view
        elif zoom < 5:
            step = 2
        else:
            step = 1  # Higher resolution for zoomed in view

        # Get data within bounds
        lat_mask = (lat_array >= south) & (lat_array <= north)
        lon_mask = (lon_array >= west) & (lon_array <= east)

        if not any(lat_mask) or not any(lon_mask):
            print("DEBUG: No data points found within bounds")
            return jsonify({'data': [], 'bounds': {'south': south, 'north': north, 'west': west, 'east': east}})

        # Sample points based on step size
        lats = lat_array[lat_mask][::step]
        lons = lon_array[lon_mask][::step]

        print(f"DEBUG: Found {len(lats)} latitude points and {len(lons)} longitude points")

        points = []
        for lat in lats:
            for lon in lons:
                try:
                    lat_idx, lon_idx = get_nearest_indices(float(lat), float(lon))
                    base_temp = float(data_array.isel(latitude=lat_idx, longitude=lon_idx).values)
                    
                    if not np.isnan(base_temp):
                        temp = calculate_temperature(base_temp, float(lat), year)
                        if -40 <= temp <= 40:  # Validate temperature range
                            points.append({
                                'lat': float(lat),
                                'lon': float(lon),
                                'temperature': float(temp)
                            })
                except Exception as e:
                    print(f"DEBUG: Error calculating temperature for point ({lat}, {lon}): {str(e)}")
                    continue

        print(f"DEBUG: Generated {len(points)} valid data points")
        if points:
            print(f"DEBUG: Sample point - {points[0]}")

        return jsonify({
            'data': points,
            'bounds': {
                'south': float(south),
                'north': float(north),
                'west': float(west),
                'east': float(east)
            }
        })

    except Exception as e:
        print(f"Error generating heatmap data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-extent')
@cache.cached(timeout=7200)  # Cache for 2 hours
def get_data_extent():
    """Return the geographical and temporal extent of available data"""
    try:
        # Get the bounds of the data
        lat_bounds = {
            'min': float(lat_array.min()),
            'max': float(lat_array.max())
        }
        lon_bounds = {
            'min': float(lon_array.min()),
            'max': float(lon_array.max())
        }

        return jsonify({
            'bounds': {
                'lat': lat_bounds,
                'lon': lon_bounds
            },
            'temporal': {
                'start': 1840,  # For now, hardcoded temporal range
                'end': 2024
            }
        })
    except Exception as e:
        print(f"Error getting data extent: {str(e)}")
        return jsonify({'error': str(e)}), 500

@lru_cache(maxsize=1024)
def get_nearest_indices(lat, lon):
    """Cache the nearest grid point calculations"""
    lat_idx = np.abs(lat_array.values - lat).argmin()
    lon_idx = np.abs(lon_array.values - lon).argmin()
    return lat_idx, lon_idx

def calculate_temperature(base_temp, latitude, year):
    """Calculate temperature components"""
    # Latitude effect
    lat_effect = -0.6 * abs(latitude)
    
    # Seasonal effect
    season_strength = abs(latitude) / 90.0
    month = ((year - int(year)) * 12) + 1
    if month > 12:
        month = 1
    
    # Adjust for hemisphere
    if latitude < 0:
        month = (month + 6) % 12 or 12
    
    seasonal_effect = 15 * season_strength * np.cos(2 * np.pi * ((month - 1) / 12))
    
    # Historical warming
    historical_effect = 1.1 * (year - 1840) / (2024 - 1840)
    
    # Calculate final temperature
    final_temp = base_temp + lat_effect + seasonal_effect + historical_effect
    
    # Ensure temperature stays within physical limits
    return min(max(final_temp, -40), 40)

@app.route('/time-series', methods=['POST'])
@cache.memoize(timeout=300)  # 5 minute cache for mouseover data
def time_series():
    try:
        request_data = request.get_json()
        print(f"DEBUG: Received request data: {request_data}")  # Debug log
        
        if not request_data:
            print("DEBUG: No JSON data received in request")
            return jsonify(error="No data received"), 400
            
        # Extract and validate each field
        try:
            latitude = float(request_data.get('latitude'))
            longitude = float(request_data.get('longitude'))
            year = int(request_data.get('year', 2024))
        except (TypeError, ValueError) as e:
            print(f"DEBUG: Data validation error - {str(e)}")
            print(f"DEBUG: latitude={request_data.get('latitude')}, longitude={request_data.get('longitude')}, year={request_data.get('year')}")
            return jsonify(error=f"Invalid data format: {str(e)}"), 400
        
        print(f"DEBUG: Processing request for lat={latitude}, lon={longitude}, year={year}")
        
        # Validate coordinate bounds
        if not lat_array.min() <= latitude <= lat_array.max():
            print(f"DEBUG: Latitude {latitude} out of bounds [{lat_array.min()}, {lat_array.max()}]")
            return jsonify(error=f"Latitude {latitude} out of bounds"), 400
            
        if not lon_array.min() <= longitude <= lon_array.max():
            print(f"DEBUG: Longitude {longitude} out of bounds [{lon_array.min()}, {lon_array.max()}]")
            return jsonify(error=f"Longitude {longitude} out of bounds"), 400

        try:
            # Get nearest grid points (cached)
            lat_idx, lon_idx = get_nearest_indices(latitude, longitude)
            print(f"DEBUG: Found grid indices: lat_idx={lat_idx}, lon_idx={lon_idx}")
            
            # Get base temperature
            base_temp = float(data_array.isel(latitude=lat_idx, longitude=lon_idx).values)
            print(f"DEBUG: Base temperature: {base_temp}")
            
            if np.isnan(base_temp):
                print("DEBUG: Found NaN temperature, using default")
                base_temp = 15.0  # Default temperature if NaN
            
            # Calculate temperature with all effects
            final_temp = calculate_temperature(base_temp, latitude, year)
            print(f"DEBUG: Final calculated temperature: {final_temp}")
            
            return jsonify({
                'data': [{
                    'year': year,
                    'temperature': round(float(final_temp), 1)
                }]
            })
            
        except Exception as e:
            print(f"Error processing temperature data: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        print(f"Error in time series endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/layers')
def get_layers():
    return jsonify({
        'temperature': {
            'min': min_val,
            'max': max_val,
            'units': '°C'
        }
    })

@app.route('/api/time-range')
def get_time_range():
    time_coords = time_data.time.values
    return jsonify({
        'start': time_coords[0].astype(str),
        'end': time_coords[-1].astype(str),
        'steps': len(time_coords)
    })

@app.route('/api/legend/<layer_name>')
def get_legend(layer_name):
    if layer_name == 'temperature':
        return jsonify({
            'min': min_val,
            'max': max_val,
            'units': '°C',
            'colors': colorcet.coolwarm
        })

# Add support for more weather parameters
WEATHER_PARAMS = {
    'temperature': {
        'variable': 'tmin',
        'colormap': colorcet.coolwarm,
        'units': '°C'
    },
    'precipitation': {
        'variable': 'precipitation',
        'colormap': colorcet.colorwheel,
        'units': 'mm'
    },
    'wind': {
        'variable': ['u10', 'v10'],
        'colormap': colorcet.rainbow,
        'units': 'm/s'
    }
}

def kelvin_to_celsius(temp_k):
    """Convert temperature from Kelvin to Celsius"""
    return temp_k - 273.15

@app.route('/api/animation-data')
@cache.memoize(timeout=3600)  # Cache for 1 hour
def get_animation_data():
    """Endpoint to get processed temperature data for animation"""
    try:
        processed_data = process_netcdf_data()
        
        # Prepare data for animation with optimization
        animation_data = {
            'timestamps': processed_data['hours'],
            'temperature_range': {
                'min': float(min_val),
                'max': float(max_val),
                'classes': processed_data['temp_classes']
            },
            'colors': processed_data['colormap'].colors,
            'data': []
        }
        
        # Optimize data structure and reduce precision
        df = processed_data['temp_df']
        for hour in processed_data['hours']:
            hour_data = df[df['hour'].dt.strftime('%H:%M') == hour]
            points = []
            
            # Reduce data precision and filter out unnecessary points
            for _, row in hour_data.iterrows():
                if not np.isnan(row[temp_var]):  # Skip NaN values
                    # Round coordinates to 4 decimal places and temperature to 1 decimal
                    point = {
                        'lat': round(float(row['latitude']), 4),
                        'lon': round(float(row['longitude']), 4),
                        'temperature': round(float(row[temp_var]), 1)
                    }
                    points.append(point)
            
            if points:  # Only add frame if it has valid points
                animation_data['data'].append({
                    'hour': hour,
                    'points': points
                })
        
        # Validate data before sending
        if not animation_data['data']:
            raise ValueError("No valid animation data generated")
        
        # Set response headers for caching
        response = jsonify(animation_data)
        response.headers['Cache-Control'] = 'public, max-age=3600'
        response.headers['Vary'] = 'Accept-Encoding'
        
        return response
        
    except Exception as e:
        print(f"Error generating animation data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def process_netcdf_data():
    """Process NetCDF data into a tidy DataFrame with hourly timestamps"""
    global data, time_data, temp_var, min_val, max_val
    
    try:
        # Convert xarray datasets to pandas DataFrames with chunking
        temp_2m_df = data[temp_var].to_dataframe().reset_index()
        
        # Create hourly timestamps for a day
        hours = pd.date_range('2024-01-01', periods=24, freq='H')
        temp_2m_df['hour'] = pd.DataFrame({'hour': hours}).iloc[temp_2m_df.index % 24].values
        
        # Convert temperatures from Kelvin to Celsius if needed
        if min_val > 100:  # Simple check if data is in Kelvin
            temp_2m_df[temp_var] = temp_2m_df[temp_var].apply(kelvin_to_celsius)
            
            # Update global min/max values with rounded values
            min_val = round(float(temp_2m_df[temp_var].min()), 1)
            max_val = round(float(temp_2m_df[temp_var].max()), 1)
        
        # Create temperature class intervals
        n_classes = 11
        class_interval = (max_val - min_val) / (n_classes - 1)
        temp_classes = [round(min_val + i * class_interval, 1) for i in range(n_classes)]
        
        # Create color palette (reversed Spectral)
        colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b',
                 '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2'][::-1]
        
        # Create color map
        temp_colormap = LinearColormap(
            colors=colors,
            vmin=min_val,
            vmax=max_val,
            caption='Temperature (°C)'
        )
        
        return {
            'temp_df': temp_2m_df,
            'hours': [h.strftime('%H:%M') for h in hours],
            'temp_classes': temp_classes,
            'colormap': temp_colormap
        }
        
    except Exception as e:
        print(f"Error processing NetCDF data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    # Run the app with optimized settings
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=config.DEBUG,
        threaded=True,
        use_reloader=False  # Disable reloader in production
    )