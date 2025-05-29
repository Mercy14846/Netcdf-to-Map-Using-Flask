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

# Increase recursion limit if needed
sys.setrecursionlimit(10000)

app = Flask(__name__)

# Configure Flask-Caching with increased timeout
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 7200,  # 2 hours
    'CACHE_THRESHOLD': 1000  # Maximum number of items to store in cache
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
        data = xr.open_dataset("static/data/temp_2m.nc", chunks={'latitude': 100, 'longitude': 100})
        time_data = xr.open_dataset("static/data/temperature.nc", chunks={'time': 100})
        
        print("DEBUG: Available variables in temp_2m.nc:", list(data.data_vars))
        print("DEBUG: Available variables in temperature.nc:", list(time_data.data_vars))
        
        # Find temperature variable without recursion
        temp_vars = ['tmin', 'temp', 'temperature', 't2m']
        temp_var = next((var for var in data.data_vars if var in temp_vars), None)
        
        if temp_var is None:
            print("DEBUG: Could not find standard temperature variable, checking all variables:", list(data.data_vars))
            # If standard names not found, take the first variable as temperature
            temp_var = list(data.data_vars)[0]
            print(f"DEBUG: Using first available variable as temperature: {temp_var}")
        
        print(f"DEBUG: Selected temperature variable: {temp_var}")
        print(f"DEBUG: Temperature variable shape: {data[temp_var].shape}")
        
        # Calculate min/max values using dask
        data_values = data[temp_var].values
        min_val = float(np.nanmin(data_values))
        max_val = float(np.nanmax(data_values))
        
        print(f"DEBUG: Temperature range: {min_val} to {max_val}")
        
        # Extract dimensions
        lon_array = data['longitude']
        lat_array = data['latitude']
        data_array = data[temp_var]
        
        print("DEBUG: Coordinate ranges:")
        print(f"Latitude: {float(lat_array.min())} to {float(lat_array.max())}")
        print(f"Longitude: {float(lon_array.min())} to {float(lon_array.max())}")
        
        # Extract time series data without recursion
        time_vars = ['tmin', 'temperature']
        time_data_var = next((time_data[var] for var in time_vars if var in time_data.data_vars), None)
        
        if time_data_var is None:
            print("DEBUG: Could not find standard time series variable, checking all variables:", list(time_data.data_vars))
            # If standard names not found, take the first variable as temperature
            time_data_var = time_data[list(time_data.data_vars)[0]]
            print(f"DEBUG: Using first available variable for time series: {list(time_data.data_vars)[0]}")
        
        print("DEBUG: Data loading completed successfully")
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
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
        
        print(f"\nDEBUG: Processing request for lat={latitude}, lon={longitude}, year={year}")
        
        # Validate coordinates are within bounds
        if (latitude < lat_array.min() or latitude > lat_array.max() or
            longitude < lon_array.min() or longitude > lon_array.max()):
            print(f"DEBUG: Coordinates out of bounds - Requested: ({latitude}, {longitude}), Bounds: ({lat_array.min()}, {lat_array.max()}), ({lon_array.min()}, {lon_array.max()})")
            return jsonify(error="Coordinates out of bounds"), 400

        try:
            # Find nearest grid points
            lat_idx = np.abs(lat_array.values - latitude).argmin()
            lon_idx = np.abs(lon_array.values - longitude).argmin()
            
            print(f"DEBUG: Selected grid points - lat_idx={lat_idx}, lon_idx={lon_idx}")
            print(f"DEBUG: Actual coordinates - lat={lat_array.values[lat_idx]}, lon={lon_array.values[lon_idx]}")
            
            # Get base temperature from the data
            base_temp = float(data_array.isel(latitude=lat_idx, longitude=lon_idx).values)
            if np.isnan(base_temp):
                print("DEBUG: Warning - NaN value found in base temperature")
                base_temp = 15.0  # Default temperature if NaN
            
            # Temperature adjustments based on real-world patterns:
            
            # 1. Latitude effect: Temperature decreases with distance from equator
            # Approximately -0.6°C per degree of latitude from equator
            lat_effect = -0.6 * abs(latitude)  # Direct degree effect
            
            # 2. Elevation effect (estimated from location)
            # Rough estimate: -6.5°C per 1000m elevation
            elevation_effect = 0  # Would need elevation data for more accuracy
            
            # 3. Seasonal effect (varies with latitude)
            # Stronger seasonal variation at higher latitudes
            season_strength = abs(latitude) / 90.0  # 0 at equator, 1 at poles
            # Calculate month (1-12) based on fractional part of the year
            month = ((year - int(year)) * 12) + 1
            if month > 12:  # Ensure month is between 1 and 12
                month = 1
            
            # Adjust seasonal effect based on hemisphere
            if latitude < 0:  # Southern hemisphere
                month = (month + 6) % 12  # Shift seasons by 6 months
                if month == 0:
                    month = 12
            
            seasonal_effect = 15 * season_strength * np.cos(2 * np.pi * ((month - 1) / 12))
            
            # 4. Historical warming trend
            # Global average temperature has increased by about 1.1°C since 1880
            historical_effect = 1.1 * (year - 1840) / (2024 - 1840)
            
            # 5. Ocean moderating effect
            # Simplified ocean effect based on longitude (rough approximation)
            ocean_effect = 0
            
            # Combine all effects
            final_temp = base_temp + lat_effect + elevation_effect + seasonal_effect + historical_effect + ocean_effect
            
            # Ensure temperature stays within physical limits
            final_temp = min(max(final_temp, -40), 40)
            
            print(f"DEBUG: Temperature calculation components:")
            print(f"Base temp: {base_temp:.2f}°C")
            print(f"Latitude effect: {lat_effect:.2f}°C")
            print(f"Seasonal effect: {seasonal_effect:.2f}°C (Month: {month}, Strength: {season_strength:.2f})")
            print(f"Historical warming: {historical_effect:.2f}°C")
            print(f"Final temperature: {final_temp:.2f}°C")

            data = [{
                'year': year,
                'temperature': round(float(final_temp), 2),
                'components': {
                    'base_temp': round(float(base_temp), 2),
                    'latitude_effect': round(float(lat_effect), 2),
                    'seasonal_effect': round(float(seasonal_effect), 2),
                    'historical_warming': round(float(historical_effect), 2),
                    'month': int(month)
                }
            }]
            
            return jsonify({'data': data})
            
        except Exception as e:
            print(f"DEBUG: Error processing temperature data: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
            
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