# import libraries
from flask import Flask, send_file, render_template, jsonify, request
# from flask_socketio import SocketIO

import io
import math

import datashader as ds
import pandas as pd
import xarray as xr
import colorcet

from datashader import transfer_functions as tf
from datashader.utils import lnglat_to_meters

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


# https://github.com/ScottSyms/tileshade/
# changes made: snapping values to ensure continuous tiles; use of quadmesh instead of points; syntax changes to work with Flask.
def generateatile(zoom, longitude, latitude):
    # The function takes the zoom and tile path from the web request,
    # and determines the top left and bottom right coordinates of the tile.
    # This information is used to query against the dataframe.
    xleft, yleft = tile2mercator(int(longitude), int(latitude), int(zoom))
    xright, yright = tile2mercator(int(longitude)+1, int(latitude)+1, int(zoom))

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

    # The dataframe query gets passed to Datashader to construct the graphic.
    frame = data.sel(
        longitude=slice(xleft_snapped, xright_snapped),
        latitude=slice(yleft_snapped, yright_snapped)
    )

    # First the graphic is created, then the dataframe is passed to the Datashader aggregator.
    csv = ds.Canvas(plot_width=256, plot_height=256, x_range=(xleft, xright), y_range=(yright, yleft))
    agg = csv.quadmesh(frame, x='longitude', y='latitude', agg=ds.mean('tmin'))

    # The image is created from the aggregate object, a color map and aggregation function.
    img = tf.shade(agg, cmap=colorcet.coolwarm, span=[min_val, max_val], how="linear")
    return img.to_pil()

app = Flask(__name__)

# socketio = SocketIO(app)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/tiles/<int:zoom>/<int:longitude>/<int:latitude>.png")
def tile(longitude, latitude, zoom):
    results = generateatile(zoom, longitude, latitude)
    # image passed off to bytestream
    results_bytes = io.BytesIO()
    results.save(results_bytes, 'PNG')
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
        print(f"Available variables in time_data: {list(time_data.data_vars)}")
        
        # Get the time series data using the correct coordinate names
        ts_slice = time_data_var.sel(
            longitude=longitude,
            latitude=latitude,
            method="nearest"
        )
        
        # Create a DataFrame with the time index and temperature values
        if isinstance(ts_slice, xr.DataArray):
            # If we have a DataArray with a time dimension
            df_slice = pd.DataFrame({
                'year': ts_slice.time.values if hasattr(ts_slice, 'time') else range(len(ts_slice)),
                'temperature': ts_slice.values
            })
        else:
            # If we have a scalar value, create a single row DataFrame
            df_slice = pd.DataFrame({
                'year': [2023],  # or whatever year is appropriate
                'temperature': [float(ts_slice)]
            })
        
        print("DataFrame shape:", df_slice.shape)
        print("DataFrame head:", df_slice.head())
        
        # Convert to JSON-serializable format
        df_slice['year'] = df_slice['year'].astype(str)
        df_slice['temperature'] = df_slice['temperature'].astype(float)
        
        json_data = df_slice.to_json(orient='records')
        return jsonify({'data': json_data})
            
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

if __name__ == '__main__':
   app.run(debug=True)