from io import StringIO
import openmeteo_requests
import pandas as pd
import requests
import requests_cache
from retry_requests import retry


# Function to check the status of requests- If an error is found in the request the error code is printed
def status_check(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response
    else:
        print(response.status_code)
        return None


# Sets up the TfL open data to be ingested in a seamless way.
def trip_data(url, file_list):
    url_list = [url + x for x in file_list]
    dfs = []
    for url in url_list:
        data = status_check(url)
        print('New Download Starting...')
        df = pd.read_csv(StringIO(data.text), low_memory=False)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df, dfs


# This code applies all chosen parameters and is non-exhaustive. It includes caching and the conversion to
# Pandas DataFrames.
def weather_data(url):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below

    params = {
        "latitude": 51.5085,
        "longitude": -0.1257,
        "start_date": "2023-01-02",
        "end_date": "2023-05-28",
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "weather_code", "wind_speed_10m",
                   "wind_direction_10m", "is_day"],
        "timezone": "GMT"}
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°E {response.Longitude()}°N")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(2).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(3).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(5).ValuesAsNumpy()
    hourly_is_day = hourly.Variables(6).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s"),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ), "temperature_2m": hourly_temperature_2m, "relative_humidity_2m": hourly_relative_humidity_2m,
        "apparent_temperature": hourly_apparent_temperature, "weather_code": hourly_weather_code,
        "wind_speed_10m": hourly_wind_speed_10m, "wind_direction_10m": hourly_wind_direction_10m,
        "is_day": hourly_is_day}

    hourly_dataframe = pd.DataFrame(data=hourly_data)

    return hourly_dataframe


# Bike Trip Data
BIKE_STATS_URL = 'https://cycling.data.tfl.gov.uk/usage-stats/'
bike_files = ["351JourneyDataExtract02Jan2023-08Jan2023.csv",
              "352JourneyDataExtract09Jan2023-15Jan2023.csv", "353JourneyDataExtract16Jan2023-22Jan2023.csv",
              "354JourneyDataExtract23Jan2023-29Jan2023.csv", "355JourneyDataExtract30Jan2023-05Feb2023.csv",
              "356JourneyDataExtract06Feb2023-12Feb2023.csv", "357JourneyDataExtract13Feb2023-19Feb2023.csv",
              "358JourneyDataExtract20Feb2023-26Feb2023.csv", "359JourneyDataExtract27Feb2023-05Mar2023.csv",
              "360JourneyDataExtract06Mar2023-12Mar2023.csv", "361JourneyDataExtract13Mar2023-19Mar2023.csv",
              "362JourneyDataExtract20Mar2023-26Mar2023.csv", "363JourneyDataExtract27Mar2023-02Apr2023.csv",
              "364JourneyDataExtract03Apr2023-09Apr2023.csv", "365JourneyDataExtract10Apr2023-16Apr2023.csv",
              "366JourneyDataExtract17Apr2023-23Apr2023.csv", "367JourneyDataExtract24Apr2023-30Apr2023.csv",
              "368JourneyDataExtract01May2023-07May2023.csv", "369JourneyDataExtract08May2023-14May2023.csv",
              "370JourneyDataExtract15May2023-21May2023.csv", "371JourneyDataExtract22May2023-28May2023.csv",]
trip_data, df_list = trip_data(BIKE_STATS_URL, bike_files)
trip_data.columns = map(str.lower, trip_data.columns)
trip_data['start date'] = pd.to_datetime(trip_data['start date'])
trip_data['trips'] = 1
trips = trip_data[['start date', 'trips']]
trips.set_index('start date', inplace=True)
trips = trips.resample('h').sum()

# London Weather Data
WEATHER_DATA_API = "https://archive-api.open-meteo.com/v1/archive"
weather = weather_data(WEATHER_DATA_API)
weather['date'] = pd.to_datetime(weather['date'])
weather.set_index('date', inplace=True)

# Merged dataset - Trips and Weather
merged_df = pd.merge(trips, weather, left_index=True, right_index=True)


"https://www.kaggle.com/code/zhikchen/xgb-feature-engineering-bike-share-forecast"

