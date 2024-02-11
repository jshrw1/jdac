import requests
import matplotlib
import numpy as np
import pandas as pd
import requests_cache
import xgboost as xgb
from io import StringIO
import openmeteo_requests
import matplotlib.pyplot as plt
from retry_requests import retry
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

matplotlib.use('TkAgg')

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
data = pd.merge(trips, weather, left_index=True, right_index=True)

# Get the earliest and latest date in the dataset
print("Earliest date in dataset: {}".format(min(data.index)))
print("Latest date in dataset: {}".format(max(data.index)))

# check for missing values and check for inconsistencies
print(data.isnull().sum())
summary = data.describe()

#Plot som histograms to visualise
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
axs = axs.ravel()
for i, column in enumerate(data.columns):
    axs[i].hist(data[column], bins=20)
    axs[i].set_title('{} Distribution'.format(column))
plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()

# Explore weather data in more detail group data by weather_code and get the average cnt
data_weather = data.groupby('weather_code')['trips'].mean()
fig = plt.figure(figsize = (12,3))
plt.bar(data_weather.index, data_weather.values)
plt.xlabel('Weather Code')
plt.ylabel('Average Number of Bikes Shared')
plt.title('Number of Bikes Shared vs. Weather Code')
plt.show()

# Explore temperature in more detail
t_bins = np.arange(-10.0, 40.0, 5.0)
data["temp_bins"] = pd.cut(data["temperature_2m"], bins=t_bins)
data["temp_bins"] = data["temp_bins"].apply(lambda x: x.right)
temp_data = data.groupby('temp_bins')['trips'].mean()

t_bins = np.arange(-10.0, 40.0, 5.0)
data["a_temp_bins"] = pd.cut(data["apparent_temperature"], bins=t_bins)
data["a_temp_bins"] = data["a_temp_bins"].apply(lambda x: x.right)
a_temp_data = data.groupby('a_temp_bins')['trips'].mean()

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(temp_data.index, temp_data.values, width=0.7, label='Actual')
ax.bar(a_temp_data.index.astype(float) + 0.7, a_temp_data.values, width=0.7, label='Feels Like')
ax.set_xlabel('Temperature Bins (C)')
ax.set_ylabel('Average Number of Bikes Shared')
ax.set_title('Number of Bikes Shared vs. Acutal and Feels Like temperature Bins')
ax.legend()
plt.show()

# Explore humidity
hum_bins = np.arange(30.0, 110.0, 10.0)
data["hum_bins"] = pd.cut(data["relative_humidity_2m"].astype(float), bins = hum_bins)
data["hum_bins"] = data["hum_bins"].apply(lambda x: x.right)
data_hum = data.groupby('hum_bins')['trips'].mean()

fig = plt.figure(figsize = (12,3))
plt.bar(data_hum.index, data_hum.values)
plt.xlabel('Humidity %')
plt.ylabel('Average Number of Bikes Shared')
plt.title('Number of Bikes Shared vs. Humidity %')
plt.xticks(hum_bins)
plt.show()

# Explore Wind Speed
wind_speed_bins = np.arange(-5.0, 65.0, 5.0)
data["wind_speed_bins"] = pd.cut(data["wind_speed_10m"], bins = wind_speed_bins)
data["wind_speed_bins"] = data["wind_speed_bins"].apply(lambda x: x.right)
data_wind_speed = data.groupby('wind_speed_bins')['trips'].mean()


fig = plt.figure(figsize = (12,3))
plt.bar(data_wind_speed.index, data_wind_speed.values)
plt.xlabel('Wind Speed km/h')
plt.ylabel('Average Number of Bikes Shared')
plt.title('Number of Bikes Shared vs. Wind Speed km/h')
plt.xticks(wind_speed_bins[1:])
plt.show()

# Explore daily trips
fig = plt.figure(figsize = (12,3))
plt.plot(data.index, data['trips'])
plt.xlabel('Date')
plt.ylabel('Number of Bikes Shared')
plt.title('Number of Bikes Shared over time')
plt.show()

# Could do more data exploration - as time series exlement could look to decompose into seasonal adjustment
# Existing EDA should suffice for demonstration - Build model
# Remove binned samples for EDA
data = data.drop(['temp_bins','a_temp_bins','hum_bins','wind_speed_bins'], axis=1)
data['hour'] = data.index.hour
data['dayofweek'] = data.index.dayofweek
data['weekofyear'] = data.index.isocalendar().week
data['month'] = data.index.month

# Split the dataset into training and testing sets (80% train, 20% test)
X = data.drop(['trips'], axis=1)
y = data['trips']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define xgboost model
model = xgb.XGBRegressor()
# Define the hyperparameters for tuning
params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.001]
}

# perform hyperparameter tuning using gridsearch
grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
model.fit(X_train, y_train)

# Model test
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)

print("MSE: {}".format(mse))
print("RMSE: {}".format(rmse))
print("R-Squared: {}".format(r_squared))

# Plot estimator vs actual
fig = plt.figure(figsize=(12,3))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label="Predictions", color="r")
plt.ylabel('Number of Bikes Shared')
plt.title('Number of Bikes Shared Predicted vs Actual')
plt.legend()
plt.show()
