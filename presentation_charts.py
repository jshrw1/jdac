##############
# Step 1 draw in data and process so that it can be used at a later stage
##############
import shutil
import zipfile
import requests
import pandas as pd
from io import BytesIO
import geopandas as gp
from io import StringIO
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# URLs to make requests from
BIKE_STATS_URL = 'https://cycling.data.tfl.gov.uk/usage-stats/'
BIKE_POINTS_URL = "https://api.tfl.gov.uk/bikepoint"
POSTCODE_LOOKUP_URLS = ["https://api.postcodes.io/postcodes", "https://findthatpostcode.uk"]
HEALTH_URL = "https://fingertipsws.phe.org.uk/api/all_data/csv/by_group_id?v=/0-6cf7ae9c/&parent_area_code=E12000007" \
             "&parent_area_type_id=6&child_area_type_id=501&group_id=1938132701&category_area_code=null "


# Function to check the status of requests- If an error is found in the request the error code is printed
def status_check(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response
    else:
        print(response.status_code)
        return None


# Creates a function to extract json data from Bike Point URL
def bike_points(url):
    # Checks API status and reads data as JSON
    response = status_check(url)
    parsed = response.json()
    data = []

    # Iterates through the JSON data to extract information relevant to us
    for p in parsed:
        bp_id = int(p["id"].replace("BikePoints_", ""))
        name = p["commonName"]
        latitude = p["lat"]
        longitude = p["lon"]
        num_docks = 0
        num_bikes = 0
        num_empty = 0

        for x in p["additionalProperties"]:
            if x["key"] == "NbDocks":
                num_docks = int(x["value"])
            if x["key"] == "NbBikes":
                num_bikes = int(x["value"])
            if x["key"] == "NbEmptyDocks":
                num_empty = int(x["value"])

        num_broken = num_docks - num_bikes - num_empty
        data.append([bp_id, name, latitude, longitude, num_docks, num_bikes, num_empty, num_broken])

    cols = ['id', 'name', 'latitude', 'longitude', 'num_docks', 'num_bikes', 'num_empty', 'num_broken']
    return pd.DataFrame(data, columns=cols)


# A function used to convert lists of latitude and longitudes to Local Authority names and Codes. Uses postcodes.io
# as default as its faster and then uses findthatpostcode to fill any gaps
def lat_lon_to_la(lat_lon_list):
    la_codes = []
    la_names = []

    for lat, lon in lat_lon_list:
        try:
            api_url = f"{POSTCODE_LOOKUP_URLS[0]}?lat={lat}&lon={lon}"
            response = status_check(api_url)
            result = response.json()
            la_code = result['result'][0]['codes']['admin_district']
            la_codes.append(la_code)
            la_name = result['result'][0]['admin_district']
            la_names.append(la_name)
        except TypeError as e:
            print(lat, lon, e)
            api_url = f"{POSTCODE_LOOKUP_URLS[1]}/points/{lat},{lon}"
            response = status_check(api_url)
            result = response.json()
            la_code = result['included'][0]['attributes']['laua']
            la_codes.append(la_code)
            la_name = result['included'][0]['attributes']['laua_name']
            la_names.append(la_name)

    return la_codes, la_names


# Sets up the TfL open data to be ingested in a seamless way.
def trip_data(url, file_list):
    url_list = [url + x for x in file_list]
    dfs = []
    for url in url_list:
        data = status_check(url)
        df = pd.read_csv(StringIO(data.text))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df


# Download bike point data
bike_point = bike_points(BIKE_POINTS_URL)
# Convert latitude and longitude into more useful location data
coordinates = list(zip(bike_point['latitude'], bike_point['longitude']))
bike_point['la_code'], bike_point['la_name'] = lat_lon_to_la(coordinates)

# Download trip data
bike_files = ['376JourneyDataExtract01Jul2023-14Jul2023.csv', '377JourneyDataExtract15Jul2023-31Jul2023.csv',
              '378JourneyDataExtract01Aug2023-14Aug2023.csv', '378JourneyDataExtract15Aug2023-31Aug2023.csv',
              '379JourneyDataExtract01Sep2023-14Sep2023.csv', '380JourneyDataExtract15Sep2023-30Sep2023.csv']
trip_data = trip_data(BIKE_STATS_URL, bike_files)
trip_data.columns = map(str.lower, trip_data.columns)
# Trip data additional variables
trip_data['start date'] = pd.to_datetime(trip_data['start date'])
trip_data['day'] = trip_data['start date'].dt.day_name()
trip_data['count'] = 1
trip_data['hour'] = trip_data['start date'].dt.hour
trip_data['weekday'] = (trip_data['start date'].dt.weekday < 5).astype(int)

# Download health data
health = pd.read_csv(HEALTH_URL, low_memory=False)
health.columns = map(str.lower, health.columns)

# Filter on relevant indcators
indicators = list(health['indicator name'].unique())
indicators_of_interest = selected_items = [indicators[i] for i in [0, 2, 4, 9, 13, 14, 15, 21]]
health = health[health['indicator name'].isin(indicators_of_interest)]

# Filter on time frequency of data, london only data and data for all persons
health = health[health['time period range'] == '1y']
health = health[health['parent name'] == 'London region']
health = health[health['sex'] == 'Persons']

##############
# Step 2 Create some maps using data collated at local authority level.
##############

MAP_FILE_URL = "https://data.london.gov.uk/download/statistical-gis-boundary-files-london/9ba8c833-6370-4b11-abdc" \
               "-314aa020d5e0/statistical-gis-boundaries-london.zip "


def extract_map_df(url, file_name):
    response = requests.get(url)
    zip_file = zipfile.ZipFile(BytesIO(response.content))
    file_prefix = file_name

    # Extract all files with the specified prefix to a temporary location
    extracted_files = [file for file in zip_file.namelist() if file.startswith(file_prefix)]
    for file in extracted_files:
        zip_file.extract(file, 'temp_directory')

    # Read the extracted shapefile into a GeoPandas DataFrame
    _df = gp.read_file('temp_directory/' + f'{file_prefix}.shx')
    shutil.rmtree('temp_directory')

    return _df


def map_variable(data_frame, variable, title, annotation, save_name):
    # Configure plot
    fig, ax = plt.subplots(1, figsize=(10, 6))
    # Plot Data
    data_frame.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    # Chart Settings
    ax.axis('off')
    ax.set_title(f'{title}', fontdict={'fontsize': '14', 'fontweight': '3'})
    ax.annotate(f'{annotation}', xy=(0.1, .08), xycoords='figure fraction', horizontalalignment='left',
                verticalalignment='top', fontsize=10, color='#555555')
    fig.savefig(f'pngs\{save_name}.png', dpi=300)
    plt.close()

# Configure dataframe with london map data
file_path = 'statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW'
map_df = extract_map_df(MAP_FILE_URL, file_path)
# Create definition of Inner London
inner_ldn = ['Camden', 'Greenwich', 'Hackney', 'Hammersmith and Fulham', 'Islington', 'Kensington and Chelsea',
             'Lambeth', 'Lewisham', 'Southwark', 'Tower Hamlets', 'Wandsworth', 'Westminster', 'City of London',
             'Newham']
map_df['inner_ldn'] = map_df['NAME'].apply(lambda x: 'T' if x in inner_ldn else 'F')
map_df = map_df[map_df['inner_ldn'] == 'T']
map_df.columns = map(str.lower, map_df.columns)
map_df = map_df[["name", "gss_code", "geometry"]]

# Map bike points
la = bike_point.groupby(['la_code', 'la_name']).size().reset_index(name='Bike Docks')
bikes_la = pd.merge(map_df, la, left_on='gss_code', right_on='la_code', how='left')
bikes_la['Bike Docks'] = bikes_la['Bike Docks'].fillna(0)
map_variable(bikes_la, 'Bike Docks', 'Santander Cycles Bike Points in Inner London', 'Source: TfL unified API', 'Bike '
                                                                                                                'docks')
# Configure health data ready for merging
# Refreshed list of indicators after filters
indicators = list(health['indicator name'].unique())

# Create separate DataFrames for each individual
individual_dfs = {}
for value in indicators:
    individual_dfs[value] = health[health['indicator name'] == value]

for key, dataframe in individual_dfs.items():
    max_time = dataframe['time period'].max()
    latest_rows = dataframe[dataframe['time period'] == max_time]
    latest_dataframe = pd.DataFrame(latest_rows, columns=dataframe.columns)
    individual_dfs[key] = latest_dataframe

# Extract information from the first dataframe
first_dataframe_key = list(individual_dfs.keys())[2]
first_dataframe = individual_dfs[first_dataframe_key]
first_dataframe['category'] = first_dataframe['category'].fillna('All')
first_dataframe = first_dataframe[first_dataframe['category'] == 'All']

unique_value = first_dataframe['indicator name'].unique()[0]
health = first_dataframe[['area code', 'area name', 'value']].copy()
health.rename(columns={'value': unique_value}, inplace=True)

for key, dataframe in individual_dfs.items():
    if key != first_dataframe_key:  # Skip the first dataframe since it's already in new_dataframe
        unique_value = dataframe['indicator name'].unique()[0]

        temp_df = dataframe[['area code', 'value']].copy()
        temp_df.rename(columns={'value': unique_value}, inplace=True)

        health = pd.merge(health, temp_df, how='outer', on=['area code'])

health_la = pd.merge(map_df, health, left_on='gss_code', right_on='area code', how='left')
for x in indicators:
    map_variable(health_la, f'{x}', f'{x}', 'Source: Public Health Data API', f'{x}')

##############
# Step 3 Create some charts using bike trip data
##############
# Trips by day of week
trips_by_day = trip_data.groupby(['day'])['count'].sum().reset_index()
trips_by_day.set_index('day', inplace=True)
trips_by_day['count'] = trips_by_day['count'] / 13
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
trips_by_day = trips_by_day.reindex(days_order)

# Trips by hour
trips_by_hour_weekday = trip_data[trip_data['weekday'] == 1].groupby(['hour'])['count'].sum().reset_index()
trips_by_hour_weekday.rename(columns={'count': 'Weekday'}, inplace=True)
trips_by_hour_weekend = trip_data[trip_data['weekday'] == 0].groupby(['hour'])['count'].sum().reset_index()
trips_by_hour_weekend.rename(columns={'count': 'Weekend'}, inplace=True)
trips_by_hour = pd.merge(trips_by_hour_weekday, trips_by_hour_weekend, on='hour')
trips_by_hour['Weekday'] = trips_by_hour['Weekday'] / 65
trips_by_hour['Weekend'] = trips_by_hour['Weekend'] / 27
trips_by_hour.rename(columns={'hour': 'Hour of the Day'}, inplace=True)
trips_by_hour.set_index('Hour of the Day', inplace=True)

# Trips by duration
weekday_trip_time = trip_data[trip_data['weekday'] == 1]['total duration (ms)'] / 1000 / 60
weekend_trip_time = trip_data[trip_data['weekday'] == 0]['total duration (ms)'] / 1000 / 60

# Create charts using the datasets above

# Trips by day
fig, ax = plt.subplots(1, figsize=(10, 6))
trips_by_day.plot(kind='bar', ax=ax, legend=False)
# Set the title
ax.set_title('Average Number of Bike Trips Taken by Day - 2023 Q3', fontdict={'fontsize': 14, 'fontweight': '3'})
# Increase the bottom margin to leave space for the annotation
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(top=0.85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Add annotation just outside the plot
fig.text(0.1, 0.01, f'Source: public TfL data', ha='left', fontsize=10, color='#555555')
fig.savefig('pngs\Trips by day.png', dpi=300)
plt.close()

# Trips by hour
fig, ax = plt.subplots(1, figsize=(10, 6))
trips_by_hour.plot(kind='bar', ax=ax, legend=True)
# Set the title
ax.set_title('Average Number of Bike Trips Taken by Hour - 2023 Q3', fontdict={'fontsize': 14, 'fontweight': '3'})
# Increase the bottom margin to leave space for the annotation
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(top=0.85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Add annotation just outside the plot
fig.text(0.1, 0.01, f'Source: public TfL data', ha='left', fontsize=10, color='#555555')
fig.savefig('pngs\Trips by hour.png', dpi=300)
plt.close()

# Trip Duration Histogram
fig, axes = plt.subplots(1, figsize=(10, 6))
axes.hist([weekday_trip_time, weekend_trip_time], 40, range=[0, 42], rwidth=1, label=['Weekdays', 'Weekends'])
axes.set_xlabel('Bike ride duration (minutes)')
axes.set_ylabel('Counts')
axes.set_title('Histogram of average bike ride durations - 2023 Q3', fontdict={'fontsize': 14, 'fontweight': '3'})
axes.legend()
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(top=0.85)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
# Add annotation just outside the plot
fig.text(0.1, 0.05, f'Source: public TfL data', ha='left', fontsize=10, color='#555555')
fig.savefig('pngs\Trip duration.png', dpi=300)
plt.close()