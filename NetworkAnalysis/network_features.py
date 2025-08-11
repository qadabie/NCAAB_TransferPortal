import pandas as pd
from db_connect import get_connection
import psycopg2

conn = get_connection()
# Read the head of player_box_scores.parquet (input player_box_scores
# and the off_network_features)
player_box_scores = pd.read_parquet('2025_player_box_scores.parquet')
player_box_scores.drop_duplicates(inplace=True)
off_network_features = pd.read_csv('24_25_off_network_features.csv')

# Rename player_id to athlete_id for merging
off_network_features = off_network_features.rename(columns={'player_id': 'athlete_id'})
off_network_features['athlete_id'] = off_network_features['athlete_id'].astype(float)
player_box_scores['athlete_id'] = player_box_scores['athlete_id'].astype(float)
player_box_scores['season'] = player_box_scores['season'].astype(str)
player_box_scores['active'] = (player_box_scores['minutes'] > 0).astype(int)

# Merge dataframes
merged_df = player_box_scores.merge(
    off_network_features,
    on=['game_id', 'athlete_id'],
    how='left',
    suffixes=('', '_off')
)

# Convert boolean columns to integers for aggregation
merged_df['starter'] = merged_df['starter'].astype(int)
#merged_df['active'] = merged_df['active'].astype(int)

# Group by player_id, player_name and season and calculate aggregates
result = merged_df.groupby(['athlete_id', 'player_name', 'season','team_id','team_display_name']).agg(
    avg_in_deg=('in_deg', 'mean'),
    avg_out_deg=('out_deg', 'mean'),
    avg_total_deg=('total_deg', 'mean'),
    avg_possessions=('possessions', 'mean'),
    total_possessions=('possessions', 'sum'),
    avg_minutes=('minutes', 'mean'),
    total_minutes=('minutes', 'sum'),
    total_starts=('starter', 'sum'),
    total_games=('active', 'sum')
).reset_index()

# Rename athlete_id back to player_id
result = result.rename(columns={'athlete_id': 'player_id'})
print(result.head(10))

transfer_df = pd.read_sql_query("SELECT * FROM player_table", conn)
transfer_test = transfer_df[transfer_df['player'] == 'Adam Sanago']
# print("\nJosh Proctor in transfer_df:")
print(transfer_test)
result['season'] = result['season'].astype(int)

# Standardize columns for comparison
result['year'] = result['season']
result['player'] = result['player_name']
# Remove periods/dots from player names
result['player'] = result['player'].str.replace('.', '', regex=False)
result['player'] = result['player'].str.replace(' Jr', '', regex=False)
result['player'] = result['player'].str.replace("'", '', regex=False)
result['player'] = result['player'].str.replace("-", '', regex=False)
result['player'] = result['player'].str.replace(' II', '', regex=False)
result = result[result['total_games'] > 3]
transfer_df['player'] = transfer_df['player'].str.replace('.', '', regex=False)
transfer_df['player'] = transfer_df['player'].str.replace(' Jr', '', regex=False)
transfer_df['player'] = transfer_df['player'].str.replace("'", '', regex=False)
transfer_df['player'] = transfer_df['player'].str.replace("-", '', regex=False)
# Remove numbers at the end of names in player column
#result['player'] = result['player'].str.replace(r' \d+$', '', regex=True)
transfer_df['player'] = transfer_df['player'].str.replace(r' \d+$', '', regex=True)
# Get unique player-year combinations from both dataframes
result_players = set(zip(result['player'].str.lower(), result['year']))
transfer_players = set(zip(transfer_df['player'].str.lower(), transfer_df['year'].astype(int)))

# Convert player names to lowercase for case-insensitive matching
result['player_lower'] = result['player'].str.lower()
transfer_df['player_lower'] = transfer_df['player'].str.lower()

# Merge the dataframes on player name and year
merged_data = transfer_df.merge(
    result[['player_lower', 'year', 'avg_in_deg', 'avg_out_deg', 'avg_total_deg', 
            'avg_possessions', 'total_possessions', 'avg_minutes', 'total_minutes', 
            'total_starts', 'total_games', 'team_id', 'team_display_name']],
    on=['player_lower', 'year'],
    how='left'
)

# Drop the temporary lowercase columns
merged_data = merged_data.drop(columns=['player_lower'])
merged_data = merged_data[merged_data['year'] ==2025]  # Filter for seasons 2022 and later
# Print the head with all columns
print("Merged data head:")
print(merged_data.head())
print(f"Total rows in merged data: {len(merged_data)}")

# Save to parquet file
merged_data.to_parquet('24_25_player_stats_with_network.parquet', index=False)
print("Data saved to 24 _25_player_stats_with_network.parquet")

