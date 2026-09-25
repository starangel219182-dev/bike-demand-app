import pandas as pd

# Load the full dataset
df = pd.read_csv("data/bike_data.csv")

# Keep calendar, weather, and the 'casual' target column
casual_df = df[[
    'season', 'yr', 'mnth', 'hr', 'holiday', 
    'weekday', 'workingday', 'weathersit', 
    'temp', 'atemp', 'hum', 'windspeed', 'casual'
]]

# Save it to data/
casual_df.to_csv("data/casual_bike_data.csv", index=False)
print("Saved data/casual_bike_data.csv with shape:", casual_df.shape)