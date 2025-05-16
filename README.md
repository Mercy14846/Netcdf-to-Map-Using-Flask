# Netcdf-to-Map-Using-Flask

A Flask web application that visualizes NetCDF data on an interactive map using Leaflet.js and datashader.

## Features
- Interactive map visualization of NetCDF data
- Dynamic tile generation
- Time series data display on click
- Responsive design
- Caching for improved performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Netcdf-to-Map-Using-Flask.git
cd Netcdf-to-Map-Using-Flask
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your NetCDF file in the appropriate directory
2. Run the Flask application:
```bash
python app.py
```
3. Open your browser and navigate to `http://localhost:5000`

## Configuration

You can modify the following settings in `config.py`:
- Map center coordinates
- Zoom levels
- Cache settings
- Data file paths