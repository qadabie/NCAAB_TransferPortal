import pandas as pd
import numpy as np

# Read the parquet file
df = pd.read_parquet('opponent_tracker_data.parquet')

# Convert date to datetime if it's not already
df['date'] = pd.to_datetime(df['date'])

# Sort by team, year, and date
df = df.sort_values(['team', 'year', 'date'])

# Number games within each team-year combination
df['game_number'] = df.groupby(['team', 'year']).cumcount() + 1

# Create a date for the second week of March for each year
# Assuming second week of March starts around March 8th
df['year'] = df['date'].dt.year
df['march_second_week'] = pd.to_datetime(df['year'].astype(str) + '-03-08')

# Filter games before the second week of March
before_march = df[df['date'] < df['march_second_week']]

# Get the last game number before second week of March for each team-year
last_game_before_march = before_march.groupby(['team', 'year'])['game_number'].max().reset_index()
last_game_before_march.columns = ['team', 'year', 'last_game_before_march_2nd_week']
last_game_before_march['last_game_before_march_2nd_week'] = last_game_before_march['last_game_before_march_2nd_week'].astype(int)

#Save the result to a new csv file
last_game_before_march.to_csv('last_regular_season_game.csv', index=False)