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

from datashader import transfer_functions as tf
from datashader.utils import lnglat_to_meters

app = Flask(__name__)

# Configure Flask-Caching
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 3600
})

# import dataset
data = xr.open_dataset("static/data/temp_2m.nc")
time_data = xr.open_dataset("static/data/temperature.nc")

# Print dataset info for debugging
print("Time data structure:", time_data)

# find min/max data values to set global colorbar
min_val = float(data['tmin'].min())
max_val = float(data['tmax'].max())

# extract dimensions
lon_array = data['longitude']
lat_array = data['latitude']
data_array = data['tmin']

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

# https://github.com/ScottSyms/tileshade/
# changes made: snapping values to ensure continuous tiles; use of quadmesh instead of points; syntax changes to work with Flask.
def generateatile(zoom, longitude, latitude):
    # Add caching
    cache_key = f"tile_{zoom}_{longitude}_{latitude}"
    cached_tile = cache.get(cache_key)
    if cached_tile is not None:
        return cached_tile

    # The function takes the zoom and tile path from the web request,
    # and determines the top left and bottom right coordinates of the tile.
    # This information is used to query against the dataframe.
    xleft, yleft = tile2mercator(int(longitude), int(latitude), int(zoom))
    xright, yright = tile2mercator(int(longitude)+1, int(latitude)+1, int(zoom))

    # Convert mercator coordinates back to lat/lon
    xleft = xleft / 20037508.34 * 180
    xright = xright / 20037508.34 * 180
    yleft = math.degrees(2 * math.atan(math.exp(yleft / 20037508.34 * math.pi)) - math.pi/2)
    yright = math.degrees(2 * math.atan(math.exp(yright / 20037508.34 * math.pi)) - math.pi/2)

    print(f"Tile coordinates: zoom={zoom}, lon={longitude}, lat={latitude}")
    print(f"Mercator bounds: ({xleft}, {yleft}) to ({xright}, {yright})")

    # Get all longitude and latitude values
    lon_values = lon_array.values
    lat_values = lat_array.values

    # Find nearest coordinates
    xleft_idx = abs(lon_values - xleft).argmin()
    xright_idx = abs(lon_values - xright).argmin()
    yleft_idx = abs(lat_values - yleft).argmin()
    yright_idx = abs(lat_values - yright).argmin()

    # Ensure we have at least 2 points in each dimension
    if xleft_idx == xright_idx and xleft_idx < len(lon_values) - 1:
        xright_idx = xleft_idx + 1
    elif xleft_idx == xright_idx:
        xleft_idx = xright_idx - 1

    if yleft_idx == yright_idx and yleft_idx < len(lat_values) - 1:
        yright_idx = yleft_idx + 1
    elif yleft_idx == yright_idx:
        yleft_idx = yright_idx - 1

    # Get the actual coordinate values
    xleft_snapped = lon_values[xleft_idx]
    xright_snapped = lon_values[xright_idx]
    yleft_snapped = lat_values[yleft_idx]
    yright_snapped = lat_values[yright_idx]

    # Ensure correct ordering
    if xleft_snapped > xright_snapped:
        xleft_snapped, xright_snapped = xright_snapped, xleft_snapped
    if yleft_snapped < yright_snapped:
        yleft_snapped, yright_snapped = yright_snapped, yleft_snapped

    print(f"Data bounds: ({xleft_snapped}, {yleft_snapped}) to ({xright_snapped}, {yright_snapped})")

    # Add error handling
    try:
        frame = data.sel(
            longitude=slice(xleft_snapped, xright_snapped),
            latitude=slice(yleft_snapped, yright_snapped)
        )
        
        print(f"Frame shape: {frame.sizes}")
        
        # Check if frame is empty using sizes
        if any(size == 0 for size in frame.sizes.values()):
            print("Empty frame, returning empty tile")
            return create_empty_tile()

        csv = ds.Canvas(plot_width=256, plot_height=256, 
                       x_range=(xleft, xright), 
                       y_range=(yright, yleft))
        agg = csv.quadmesh(frame, x='longitude', y='latitude', 
                          agg=ds.mean('tmin'))
        img = tf.shade(agg, cmap=colorcet.coolwarm, 
                      span=[min_val, max_val], 
                      how="linear")
        
        # Convert datashader image to PIL Image
        pil_img = Image.fromarray(np.array(img.data))
        
        # Cache the result
        cache.set(cache_key, pil_img, timeout=3600)
        return pil_img
    except Exception as e:
        print(f"Error generating tile: {str(e)}")
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
        
        # Get time values from the dataset
        time_values = pd.to_datetime(time_data.time.values)
        
        # Handle both array and scalar year values
        if isinstance(time_values.year, (int, np.integer)):
            years = [int(time_values.year)]
        else:
            years = time_values.year.tolist()  # Convert to list to ensure it's iterable
        
        try:
            # If ts_slice is a DataArray with time dimension
            temp_values = ts_slice.values
            if not isinstance(temp_values, np.ndarray):
                temp_values = np.array([temp_values])
        except:
            # If ts_slice is a scalar value
            temp_values = np.full(len(years), float(ts_slice))
        
        # Create DataFrame ensuring both arrays are the same length
        df_slice = pd.DataFrame({
            'year': years[:len(temp_values)],
            'temperature': temp_values[:len(years)]
        })
        
        print("DataFrame shape:", df_slice.shape)
        print("DataFrame head:", df_slice.head())
        
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