import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry
import csv

# LOCATION VARIABLES

# SD78
LONGITUDE = 121.79
LATITUDE = 49.30


# DATA VARIABLES
# keep dates in "yyyy-mm-dd" format, open meteo only provides data in full day increments
START_DATE = "2023-03-01"
END_DATE = "2023-10-04"

# LOAD PROFILE
sd78 = "NimbaDataConversion/SD78_hourly_load.csv"

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
	"latitude": LATITUDE,
	"longitude": LONGITUDE,
	"start_date": START_DATE,
	"end_date": END_DATE,
	"hourly": ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()
hourly_shortwave_radiation = hourly.Variables(2).ValuesAsNumpy()

hourly_data = {"time": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

file = pd.read_csv(sd78, usecols = ["load_gross_real_power_kW"])
        
hourly_data["load"] = file["load_gross_real_power_kW"]
hourly_data["tmp1"] = hourly_temperature_2m
hourly_data["wind1"] = hourly_wind_speed_10m
hourly_data["swrad1"] = hourly_shortwave_radiation


hourly_dataframe = pd.DataFrame(data = hourly_data)
print(hourly_dataframe)


# # Write CSV file
hourly_dataframe.to_csv('historic.csv', encoding='utf-8', index=False)
