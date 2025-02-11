import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 49.186772977390234, #location coords (use harrison hot springs elementary school)
	"longitude": -122.84903502213014, #location coords
	"hourly": ["temperature_2m", "shortwave_radiation"],
	"forecast_days": 3
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
hourly_shortwave_radiation = hourly.Variables(1).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
#pull wind speed

hourly_dataframe = pd.DataFrame(data = hourly_data)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 150)
print(hourly_dataframe)

# Define the solar generation calculation function
def solar_gen(sr, a, df):
    return sr*a*df

# Define the area (in square meters) and derate factor
a = 100  # Example area in square meters (pull sd78 data??)
df = 0.75  # Example derate factor (75%)

# Calculate solar generation for each timestamp and append to the dataframe
hourly_dataframe['solar_generation'] = hourly_dataframe['shortwave_radiation'].apply(
    lambda x: solar_gen(x, a, df)
)

# Print the updated dataframe with solar generation
print(hourly_dataframe)