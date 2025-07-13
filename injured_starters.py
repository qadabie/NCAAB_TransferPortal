import pandas as pd
import numpy as np

# Read player_box_scores.parquet
box = pd.read_parquet('player_box_scores.parquet')

# Convert game_date to datetime
box['game_date'] = pd.to_datetime(box['game_date'])

# Sort values by athlete_id and game_date
box = box.sort_values(by = ['athlete_id', 'game_date'])

# Create next_game_id and next_minutes columns
box['next_minutes'] = box.groupby('athlete_id')['minutes'].shift(-1)
box['next_game_id'] = box.groupby('athlete_id')['game_id'].shift(-1)

# Assuming injured starters are those who started a game but did not play in the next game
injured_starters = box[(box['starter'] == True) & (box['next_minutes'].isna()) & box['next_game_id'].notna()]

# Select relevant columns
injured_starters = injured_starters[['game_id',
                                    'game_date',
                                    'athlete_id',
                                    'athlete_display_name',
                                    'starter']]

# Rename athlete_display_name to player and athlete_id to player_id
injured_starters = injured_starters.rename(columns={'athlete_display_name': 'player', 
                                                    'athlete_id': 'player_id'})

# Save the DataFrame to a CSV file
injured_starters.to_csv('injured_starters.csv', index=False)

