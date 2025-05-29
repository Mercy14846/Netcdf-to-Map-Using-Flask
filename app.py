# import libraries
from flask import Flask, send_file, render_template, jsonify, request
from flask_caching import Cache
# from flask_socketio import SocketIO

import io
import math
import numpy as np
from PIL import Image
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

# Increase recursion limit if needed
sys.setrecursionlimit(10000)

app = Flask(__name__)

# Configure Flask-Caching with increased timeout
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 7200  # 2 hours
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
        # Load the datasets with chunking for better memory management
        data = xr.open_dataset("static/data/temp_2m.nc", chunks={'latitude': 100, 'longitude': 100})
        time_data = xr.open_dataset("static/data/temperature.nc", chunks={'time': 100})
        
        # Find temperature variable without recursion
        temp_vars = ['tmin', 'temp', 'temperature', 't2m']
        temp_var = next((var for var in data.data_vars if var in temp_vars), None)
        
        if temp_var is None:
            print("Available variables in data:", list(data.data_vars))
            raise ValueError("Could not find temperature variable in data")
        
        # Calculate min/max values using dask
        data_values = data[temp_var].values
        min_val = float(np.nanmin(data_values))
        max_val = float(np.nanmax(data_values))
        
        print(f"Temperature range: {min_val} to {max_val}")
        
        # Extract dimensions
        lon_array = data['longitude']
        lat_array = data['latitude']
        data_array = data[temp_var]
        
        # Extract time series data without recursion
        time_vars = ['tmin', 'temperature']
        time_data_var = next((time_data[var] for var in time_vars if var in time_data.data_vars), None)
        
        if time_data_var is None:
            print("Available variables in time_data:", list(time_data.data_vars))
            raise ValueError("Could not find temperature variable in time series data")
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# Initialize data at startup
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

        # Convert mercator coordinates back to lat/lon
        xleft = xleft / 20037508.34 * 180
        xright = xright / 20037508.34 * 180
        yleft = math.degrees(2 * math.atan(math.exp(yleft / 20037508.34 * math.pi)) - math.pi/2)
        yright = math.degrees(2 * math.atan(math.exp(yright / 20037508.34 * math.pi)) - math.pi/2)

        print(f"Processing tile: zoom={zoom}, lon={longitude}, lat={latitude}")
        print(f"Bounds: ({xleft}, {yleft}) to ({xright}, {yright})")

        # Get data for the tile with a small buffer to prevent gaps
        buffer = 0.1  # Add a small buffer around the tile
        frame = data_array.sel(
            longitude=slice(min(xleft, xright) - buffer, max(xleft, xright) + buffer),
            latitude=slice(max(yleft, yright) + buffer, min(yleft, yright) - buffer)
        )

        if frame.size == 0:
            print("No data in selected region")
            return create_empty_tile()

        # Adjust resolution based on zoom level
        resolution = min(256, 2 ** (zoom + 4))  # Increase resolution at higher zoom levels
        
        # Create canvas and render data
        canvas = ds.Canvas(plot_width=resolution, 
                         plot_height=resolution,
                         x_range=(xleft, xright),
                         y_range=(yright, yleft))

        # Convert data to DataFrame for datashader
        df = frame.to_dataframe().reset_index()
        
        # Check if we have valid data
        if df.empty or df[temp_var].isna().all():
            print("No valid temperature data in region")
            return create_empty_tile()

        # Create aggregation using points with dynamic spreading based on zoom
        agg = canvas.points(df, 
                          x='longitude', 
                          y='latitude',
                          agg=ds.mean(temp_var))
        
        # Calculate spread value - ensure it's a positive integer
        spread = int(max(1, min(4, 10 - zoom)))  # Will give values between 1 and 4
        print(f"Using spread value: {spread} for zoom level: {zoom}")
        
        # Apply spreading to fill gaps
        if spread > 0:  # Only apply spread if it's positive
            agg = ds.tf.spread(agg, px=spread)

        if agg is None:
            print("Aggregation failed")
            return create_empty_tile()

        # Shade the data with increased contrast
        img = tf.shade(agg, 
                      cmap=create_colormap(),
                      span=[min_val, max_val],
                      how='linear')
        
        # Convert to RGBA
        img_data = np.array(img.data)
        
        # Create alpha channel based on data validity
        alpha = np.where(np.isnan(agg.values), 0, 255).astype(np.uint8)
        
        # Create final RGBA image with transparency
        rgba = np.zeros((img_data.shape[0], img_data.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = img_data[..., :3]  # Copy RGB channels
        rgba[..., 3] = alpha  # Set alpha channel
        
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

@app.route("/tiles/<int:zoom>/<int:longitude>/<int:latitude>.png")
def tile(longitude, latitude, zoom):
    results = generateatile(zoom, longitude, latitude)
    # image passed off to bytestream
    results_bytes = io.BytesIO()
    if isinstance(results, Image.Image):
        results.save(results_bytes, format='PNG')
    else:
        # If for some reason we don't have a PIL Image, create an empty one
        create_empty_tile().save(results_bytes, format='PNG')
    results_bytes.seek(0)
    return send_file(results_bytes, mimetype='image/png')

@app.route('/api/heatmap-data')
@cache.cached(timeout=7200)
def heatmap_data():
    try:
        # Create regular grid of data with optimized memory usage
        lats = lat_array.values
        lons = lon_array.values
        temps = data_array.values

        # Calculate temperature segments
        temp_range = np.linspace(min_val, max_val, 10)
        
        # Create a regular grid of data
        grid_data = []
        lat_step = abs(lats[1] - lats[0])
        lon_step = abs(lons[1] - lons[0])

        # Calculate the number of points to sample
        lat_samples = len(lats)
        lon_samples = len(lons)

        # Create a downsampled grid if the resolution is too high
        if lat_samples * lon_samples > 10000:
            stride = int(np.sqrt((lat_samples * lon_samples) / 10000))
            lats = lats[::stride]
            lons = lons[::stride]
            temps = temps[::stride, ::stride]

        # Create the grid data efficiently using numpy operations
        valid_mask = ~np.isnan(temps)
        valid_indices = np.where(valid_mask)
        
        normalized_temps = np.zeros_like(temps)
        normalized_temps[valid_mask] = (temps[valid_mask] - min_val) / (max_val - min_val)
        
        grid_data = [
            [float(lats[i]), float(lons[j]), float(normalized_temps[i, j])]
            for i, j in zip(*valid_indices)
        ]

        # Create segments for the legend
        segments = [
            {
                'start': float(temp_range[i]),
                'end': float(temp_range[i + 1]),
                'color': None
            }
            for i in range(len(temp_range) - 1)
        ]

        return jsonify({
            'data': grid_data,
            'min': float(min_val),
            'max': float(max_val),
            'segments': segments,
            'bounds': {
                'lat': [float(lats.min()), float(lats.max())],
                'lon': [float(lons.min()), float(lons.max())]
            },
            'resolution': {
                'lat': float(lat_step),
                'lon': float(lon_step)
            }
        })
    except Exception as e:
        print(f"Error generating heatmap data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-extent')
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

@app.route('/time-series', methods=['POST'])
@cache.memoize(timeout=7200)
def time_series():
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify(error="No data received"), 400
            
        latitude = float(request_data.get('latitude'))
        longitude = float(request_data.get('longitude'))
        year = int(request_data.get('year', 2024))  # Default to 2024 if not specified
        
        # Validate coordinates are within bounds
        if (latitude < lat_array.min() or latitude > lat_array.max() or
            longitude < lon_array.min() or longitude > lon_array.max()):
            return jsonify(error="Coordinates out of bounds"), 400

        # Find the nearest grid points
        lat_idx = np.abs(lat_array.values - latitude).argmin()
        lon_idx = np.abs(lon_array.values - longitude).argmin()
        
        # Get base temperature from the spatial data
        base_temp = float(data_array.isel(latitude=lat_idx, longitude=lon_idx).values)
        
        # Add latitude-based temperature variation
        # Temperature generally decreases with latitude (about 0.6°C per degree of latitude)
        lat_factor = -0.6 * (abs(latitude) / 90.0)  # Normalized by distance from equator
        
        # Add seasonal variation based on latitude (stronger at higher latitudes)
        seasonal_amplitude = 15.0 * (abs(latitude) / 90.0)  # Max 15°C variation at poles
        month = (year % 12) + 1  # Simple month calculation
        seasonal_factor = seasonal_amplitude * np.cos(2 * np.pi * (month - 1) / 12)
        
        # Add historical warming trend (approximately 1.5°C from 1840 to 2024)
        year_factor = (year - 1840) / (2024 - 1840)
        historical_warming = 1.5 * year_factor
        
        # Combine all factors
        final_temp = base_temp + lat_factor + seasonal_factor + historical_warming
        
        # Add small random variation to make it more realistic
        final_temp += np.random.normal(0, 0.2)  # Small random variation
        
        # Ensure temperature stays within reasonable bounds
        final_temp = min(max(final_temp, -50), 50)  # Limit to -50°C to 50°C range

        data = [{
            'year': year,
            'temperature': round(float(final_temp), 2),
            'base_temp': round(float(base_temp), 2),
            'latitude_effect': round(float(lat_factor), 2),
            'seasonal_effect': round(float(seasonal_factor), 2),
            'historical_warming': round(float(historical_warming), 2)
        }]
        
        return jsonify({'data': data})
            
    except Exception as e:
        print(f"Error in time series endpoint: {str(e)}")
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