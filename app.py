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

# Increase recursion limit if needed
sys.setrecursionlimit(10000)

app = Flask(__name__)

# Configure Flask-Caching with increased timeout and larger threshold
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 7200,  # 2 hours
    'CACHE_THRESHOLD': 5000  # Store more items in cache
})

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

def load_data():
    global data, time_data, temp_var, min_val, max_val, lon_array, lat_array, data_array, time_data_var
    
    try:
        print("\nDEBUG: Starting data loading process...")
        
        # Load the datasets with chunking for better memory management
        try:
            data = xr.open_dataset("static/data/temp_2m.nc", chunks={'latitude': 100, 'longitude': 100})
            print("DEBUG: Successfully loaded temp_2m.nc")
            print("DEBUG: temp_2m.nc structure:", data)
        except Exception as e:
            print(f"ERROR loading temp_2m.nc: {str(e)}")
            raise
            
        try:
            time_data = xr.open_dataset("static/data/temperature.nc", chunks={'time': 100})
            print("DEBUG: Successfully loaded temperature.nc")
            print("DEBUG: temperature.nc structure:", time_data)
        except Exception as e:
            print(f"ERROR loading temperature.nc: {str(e)}")
            raise
        
        print("DEBUG: Available variables in temp_2m.nc:", list(data.data_vars))
        print("DEBUG: Available variables in temperature.nc:", list(time_data.data_vars))
        
        # Find temperature variable
        temp_vars = ['tmin', 'temp', 'temperature', 't2m']
        temp_var = next((var for var in data.data_vars if var in temp_vars), None)
        
        if temp_var is None:
            print("DEBUG: Could not find standard temperature variable, using first available")
            if len(data.data_vars) > 0:
                temp_var = list(data.data_vars)[0]
                print(f"DEBUG: Using variable: {temp_var}")
            else:
                raise ValueError("No variables found in temp_2m.nc")
        
        print(f"DEBUG: Selected temperature variable: {temp_var}")
        
        try:
            # Calculate min/max values
            data_values = data[temp_var].values
            print("DEBUG: Shape of temperature data:", data_values.shape)
            print("DEBUG: Sample of temperature values:", data_values.flatten()[:10])
            
            min_val = float(np.nanmin(data_values))
            max_val = float(np.nanmax(data_values))
            print(f"DEBUG: Temperature range: {min_val} to {max_val}")
            
            # Extract dimensions
            lon_array = data['longitude']
            lat_array = data['latitude']
            print("DEBUG: Longitude range:", float(lon_array.min()), "to", float(lon_array.max()))
            print("DEBUG: Latitude range:", float(lat_array.min()), "to", float(lat_array.max()))
            
            data_array = data[temp_var]
        except Exception as e:
            print(f"ERROR processing temperature data: {str(e)}")
            raise
        
        # Extract time series data
        try:
            time_vars = ['tmin', 'temperature']
            time_data_var = next((time_data[var] for var in time_vars if var in time_data.data_vars), None)
            
            if time_data_var is None:
                if len(time_data.data_vars) > 0:
                    time_data_var = time_data[list(time_data.data_vars)[0]]
                    print(f"DEBUG: Using time variable: {list(time_data.data_vars)[0]}")
                else:
                    raise ValueError("No variables found in temperature.nc")
        except Exception as e:
            print(f"ERROR processing time data: {str(e)}")
            raise
            
        print("DEBUG: Data loading completed successfully")
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Initialize data at startup
print("\nDEBUG: Starting application, loading initial data...")
load_data()

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
    # Add caching
    cache_key = f"tile_{zoom}_{longitude}_{latitude}"
    cached_tile = cache.get(cache_key)
    if cached_tile is not None:
        return cached_tile

    try:
        # Convert tile coordinates to mercator
        xleft, yleft = tile2mercator(int(longitude), int(latitude), int(zoom))
        xright, yright = tile2mercator(int(longitude)+1, int(latitude)+1, int(zoom))

        # Convert mercator coordinates back to lat/lon with added buffer
        buffer_factor = 0.1  # 10% buffer
        xleft = xleft / 20037508.34 * 180
        xright = xright / 20037508.34 * 180
        yleft = math.degrees(2 * math.atan(math.exp(yleft / 20037508.34 * math.pi)) - math.pi/2)
        yright = math.degrees(2 * math.atan(math.exp(yright / 20037508.34 * math.pi)) - math.pi/2)

        # Calculate buffer size based on tile dimensions
        x_buffer = abs(xright - xleft) * buffer_factor
        y_buffer = abs(yleft - yright) * buffer_factor

        print(f"DEBUG: Processing tile: zoom={zoom}, lon={longitude}, lat={latitude}")
        print(f"DEBUG: Original bounds: ({xleft}, {yleft}) to ({xright}, {yright})")
        print(f"DEBUG: Buffer sizes: x={x_buffer:.6f}, y={y_buffer:.6f}")

        # Get data for the tile with dynamic buffer
        frame = data_array.sel(
            longitude=slice(min(xleft, xright) - x_buffer, max(xleft, xright) + x_buffer),
            latitude=slice(max(yleft, yright) + y_buffer, min(yleft, yright) - y_buffer)
        )

        if frame.size == 0:
            print("DEBUG: No data in selected region")
            return create_empty_tile()

        # Calculate optimal resolution based on zoom level
        base_resolution = 256  # Base tile size
        if zoom < 4:
            resolution = min(base_resolution, 2 ** (zoom + 4))
        else:
            resolution = base_resolution * min(2, zoom // 4)  # Increase resolution at higher zooms
        
        print(f"DEBUG: Using resolution {resolution} for zoom level {zoom}")

        # Create canvas with calculated resolution
        canvas = ds.Canvas(
            plot_width=resolution,
            plot_height=resolution,
            x_range=(xleft - x_buffer, xright + x_buffer),
            y_range=(yright - y_buffer, yleft + y_buffer)
        )

        # Convert data to DataFrame for datashader
        df = frame.to_dataframe().reset_index()
        
        if df.empty or df[temp_var].isna().all():
            print("DEBUG: No valid temperature data in region")
            return create_empty_tile()

        # Calculate spread value based on zoom level and data density
        points_per_pixel = df.shape[0] / (resolution * resolution)
        
        # Adaptive spread calculation
        if zoom < 4:
            # More spread at low zoom levels
            base_spread = 4
        elif zoom < 8:
            # Moderate spread at medium zoom levels
            base_spread = 3
        else:
            # Minimal spread at high zoom levels
            base_spread = 2

        # Adjust spread based on data density
        if points_per_pixel < 0.1:
            # Increase spread for sparse data
            spread = base_spread + 1
        elif points_per_pixel > 1.0:
            # Reduce spread for dense data
            spread = max(1, base_spread - 1)
        else:
            spread = base_spread

        print(f"DEBUG: Points per pixel: {points_per_pixel:.3f}, Using spread: {spread}")

        # Create aggregation using points
        agg = canvas.points(
            df,
            x='longitude',
            y='latitude',
            agg=ds.mean(temp_var)
        )

        # Only apply spread if we have valid data and need it
        if agg is not None and spread > 1:
            print(f"DEBUG: Applying spread of {spread} pixels")
            agg = ds.tf.spread(agg, px=spread)
        
        if agg is None:
            print("DEBUG: Aggregation failed")
            return create_empty_tile()

        # Enhanced shading with dynamic range adjustment
        min_temp = float(df[temp_var].min())
        max_temp = float(df[temp_var].max())
        temp_range = max_temp - min_temp

        # Adjust color range based on data distribution
        if temp_range < 1.0:
            # Very small temperature range, center around the mean
            mean_temp = (max_temp + min_temp) / 2
            min_temp = mean_temp - 5
            max_temp = mean_temp + 5
        else:
            # Add padding to the range to smooth color transitions
            padding = temp_range * 0.1
            min_temp -= padding
            max_temp += padding

        print(f"DEBUG: Temperature range: {min_temp:.1f}°C to {max_temp:.1f}°C")

        # Shade the data with calculated range
        img = tf.shade(
            agg,
            cmap=create_colormap(),
            span=[min_temp, max_temp],
            how='linear'
        )
        
        # Convert to RGBA with enhanced alpha channel
        img_data = np.array(img.data)
        
        # Create alpha channel based on data validity and intensity
        alpha = np.where(np.isnan(agg.values), 0, 255)
        # Adjust alpha based on value intensity
        normalized_values = (agg.values - min_temp) / (max_temp - min_temp)
        alpha = np.where(
            ~np.isnan(agg.values),
            np.maximum(100, np.minimum(255, normalized_values * 255)),
            0
        ).astype(np.uint8)
        
        # Create final RGBA image with transparency
        rgba = np.zeros((img_data.shape[0], img_data.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = img_data[..., :3]
        rgba[..., 3] = alpha

        # Resize to 256x256 if needed
        if resolution != 256:
            pil_img = Image.fromarray(rgba, mode='RGBA')
            pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
        else:
            pil_img = Image.fromarray(rgba, mode='RGBA')
        
        # Cache the result
        cache.set(cache_key, pil_img, timeout=3600)
        return pil_img

    except Exception as e:
        print(f"Error generating tile: {str(e)}")
        import traceback
        traceback.print_exc()
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
def tile(longitude, latitude, zoom):
    try:
        # Validate zoom level
        if zoom < 0 or zoom > 20:
            return send_file(create_error_tile("Invalid zoom level").save(io.BytesIO(), format='PNG'), mimetype='image/png')
        
        # Check if coordinates are within valid range
        if longitude < 0 or longitude >= 2**zoom or latitude < 0 or latitude >= 2**zoom:
            return send_file(create_error_tile("Invalid tile coordinates").save(io.BytesIO(), format='PNG'), mimetype='image/png')
        
        results = generateatile(zoom, longitude, latitude)
        
        # Convert results to bytes
        results_bytes = io.BytesIO()
        if isinstance(results, Image.Image):
            try:
                results.save(results_bytes, format='PNG')
            except Exception as e:
                print(f"Error saving tile image: {str(e)}")
                return send_file(create_error_tile("Error saving tile").save(io.BytesIO(), format='PNG'), mimetype='image/png')
        else:
            print("Invalid tile result type")
            return send_file(create_error_tile("Invalid tile data").save(io.BytesIO(), format='PNG'), mimetype='image/png')
        
        results_bytes.seek(0)
        return send_file(results_bytes, mimetype='image/png')
        
    except Exception as e:
        print(f"Error generating tile: {str(e)}")
        import traceback
        traceback.print_exc()
        return send_file(create_error_tile("Tile generation error").save(io.BytesIO(), format='PNG'), mimetype='image/png')

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

if __name__ == '__main__':
    # Run the app on all network interfaces (0.0.0.0)
    # This makes it accessible from other devices on the network
    # Default port is 5000, but you can change it if needed
    app.run(
        host='0.0.0.0',  # Listen on all network interfaces
        port=8080,       # You can change this port if needed
        debug=True       # Keep debug mode for development
    )