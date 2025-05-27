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

app = Flask(__name__)

# Configure Flask-Caching
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 3600
})

# import dataset and handle missing values properly
try:
    data = xr.open_dataset("static/data/temp_2m.nc")
    time_data = xr.open_dataset("static/data/temperature.nc")
    
    # Print dataset info for debugging
    print("Main data structure:", data)
    print("Time data structure:", time_data)
    
    # Ensure the temperature variable exists and get its name
    temp_var = None
    for var in data.data_vars:
        if var in ['tmin', 'temp', 'temperature', 't2m']:
            temp_var = var
            break
    
    if temp_var is None:
        print("Available variables in data:", list(data.data_vars))
        raise ValueError("Could not find temperature variable in data")
    
    # find min/max data values to set global colorbar
    min_val = float(data[temp_var].min())
    max_val = float(data[temp_var].max())
    
    print(f"Temperature range: {min_val} to {max_val}")
    
    # extract dimensions
    lon_array = data['longitude']
    lat_array = data['latitude']
    data_array = data[temp_var]
    
except Exception as e:
    print(f"Error loading data: {str(e)}")
    raise

# Extract time series data array - assuming 'tmin' or 'temperature' is the variable name
time_data_var = None
for var in time_data.data_vars:
    if var in ['tmin', 'temperature']:
        time_data_var = time_data[var]
        break
if time_data_var is None:
    print("Available variables in time_data:", list(time_data.data_vars))
    raise ValueError("Could not find temperature variable in time series data")

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

        # Get data for the tile
        frame = data_array.sel(
            longitude=slice(min(xleft, xright), max(xleft, xright)),
            latitude=slice(max(yleft, yright), min(yleft, yright))
        )

        if frame.size == 0:
            print("No data in selected region")
            return create_empty_tile()

        # Create canvas and render data
        canvas = ds.Canvas(plot_width=256, plot_height=256,
                         x_range=(xleft, xright),
                         y_range=(yright, yleft))

        # Convert data to DataFrame for datashader
        df = frame.to_dataframe().reset_index()
        
        # Check if we have valid data
        if df.empty or df[temp_var].isna().all():
            print("No valid temperature data in region")
            return create_empty_tile()

        # Create aggregation using points instead of quadmesh
        agg = canvas.points(df, 
                          x='longitude', 
                          y='latitude',
                          agg=ds.mean(temp_var))

        if agg is None:
            print("Aggregation failed")
            return create_empty_tile()

        # Shade the data
        img = tf.shade(agg, 
                      cmap=create_colormap(),
                      span=[min_val, max_val],
                      how='linear')

        # Convert to RGBA
        img_data = np.array(img.data)
        
        # Create alpha channel
        alpha = np.where(np.isnan(agg.values), 0, 255).astype(np.uint8)
        
        # Create final RGBA image
        rgba = np.zeros((img_data.shape[0], img_data.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = img_data[..., :3]
        rgba[..., 3] = alpha

        # Convert to PIL image
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

@app.route('/time-series', methods=['POST'])
def get_time_series():
    try:
        request_data = request.get_json()
        if not request_data:
            print("No JSON data received in request")
            return jsonify(error="No data received"), 400
            
        latitude = float(request_data.get('latitude'))
        longitude = float(request_data.get('longitude'))
        
        print(f"Processing request for lat={latitude}, lon={longitude}")
        
        # Get the time series data using the correct coordinate names
        ts_slice = time_data_var.sel(
            longitude=longitude,
            latitude=latitude,
            method="nearest"
        )
        
        print("Time series slice:", ts_slice)
        print("Time series slice type:", type(ts_slice))
        
        # Get time values from the dataset
        time_values = pd.to_datetime(time_data.time.values)
        print("Time values:", time_values)
        print("Time values type:", type(time_values))
        print("Time values year type:", type(time_values.year))
        
        # Handle both array and scalar year values
        if isinstance(time_values.year, (int, np.integer)):
            years = [int(time_values.year)]
            print("Single year value:", years[0])
        else:
            years = time_values.year.tolist()  # Convert to list to ensure it's iterable
            print("Year values:", years[:5], "...")
        
        try:
            # If ts_slice is a DataArray with time dimension
            temp_values = ts_slice.values
            if not isinstance(temp_values, np.ndarray):
                temp_values = np.array([temp_values])
            print("Temperature values shape:", temp_values.shape)
            print("First few temperature values:", temp_values[:5])
        except Exception as e:
            print("Error processing temperature values:", str(e))
            # If ts_slice is a scalar value
            temp_values = np.full(len(years), float(ts_slice))
            print("Created constant temperature array:", temp_values[:5])
        
        # Create DataFrame ensuring both arrays are the same length
        df_slice = pd.DataFrame({
            'year': years[:len(temp_values)],
            'temperature': temp_values[:len(years)]
        })
        
        print("Final DataFrame info:")
        print(df_slice.info())
        print("First few rows:")
        print(df_slice.head())
        
        # Convert to JSON-serializable format
        df_slice['year'] = df_slice['year'].astype(int)
        df_slice['temperature'] = df_slice['temperature'].astype(float)
        
        # Convert to list of dictionaries for JSON serialization
        data_list = [
            {'year': int(row['year']), 'temperature': float(row['temperature'])}
            for _, row in df_slice.iterrows()
        ]
        
        return jsonify({'data': data_list})
            
    except Exception as e:
        print(f"Error processing time series: {str(e)}")
        print("Dataset structure:", time_data)
        print("Available coordinates:", time_data.coords)
        print("Available variables:", time_data.variables)
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
   app.run(debug=True)