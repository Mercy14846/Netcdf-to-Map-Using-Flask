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
        
        # Convert to dataframe with time index
        df_slice = ts_slice.to_dataframe()
        
        # Reset index and ensure we have a proper time column
        if isinstance(df_slice.index, pd.MultiIndex):
            df_slice = df_slice.reset_index()
        else:
            df_slice = df_slice.reset_index().rename(columns={'index': 'time'})
        
        print("DataFrame columns:", df_slice.columns.tolist())
        print("DataFrame head:", df_slice.head().to_dict('records'))
        
        # Get the variable name that was actually found in time_data_var
        temp_col = time_data_var.name
        time_col = 'time' if 'time' in df_slice.columns else 'year'
        
        if time_col in df_slice.columns:
            result_df = df_slice[[time_col, temp_col]].copy()
            print("Final data shape:", result_df.shape)
            print("Final columns:", result_df.columns.tolist())
            print("Sample data:", result_df.head().to_dict('records'))
            
            # Ensure the data is serializable
            result_df[time_col] = result_df[time_col].astype(str)
            json_data = result_df.to_dict('records')
            return jsonify({'data': json_data})
        else:
            print(f"Missing time column. Available columns: {df_slice.columns}")
            return jsonify({'error': 'Time column not found in data'}), 500
            
    except Exception as e:
        print(f"Error processing time series: {str(e)}")
        print("Dataset structure:", time_data)
        print("Available coordinates:", time_data.coords)
        print("Available variables:", time_data.variables)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
   app.run(debug=True)