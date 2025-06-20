import pandas as pd
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
# Try to locate and read the file
file_path = "22_23_mbb_pbp.parquet"  # Updated file extension
df = pd.read_parquet(file_path)
# Check if the column exists before dropping it
columns_to_drop = ['period_display_value','clock_display_value','wallclock','home_team_mascot','home_team_name_alt',
                  'away_team_mascot','away_team_name_alt','game_spread_available','period_number']
# Filter out columns that don't exist in the dataframe
columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df.drop(columns = columns_to_drop, inplace=True)
#print(df['sequence_number'].head(25))
print(df.shape)
#Filter only relevant plays
type_ids = [558, #Jump Shot
            572, #Layup Shot
            574, #Dunk Shot
            437, #Tip Shot
            586, #Offensive Rebound
            540, #Made Free Throw
            584, #Subsitution
            598, #Lost Ball Turnover
            ]
df = df[df['type_id'].isin(type_ids)]
# Create poss_end column based on team_id changes
df['poss_end'] = False  # Initialize all to False

# For each game, check if team_id changes between consecutive plays
for game_id in df['game_id'].unique():
    game_df = df[df['game_id'] == game_id].copy()
    game_df = game_df.sort_values(['game_id', 'sequence_number'])
    
    # Compare current team_id with next team_id (shift(-1))
    # If they're different, mark as possession end
    game_df['poss_end'] = (game_df['team_id'] != game_df['team_id'].shift(-1))
    
    # Last play of the game is always a possession end
    if not game_df.empty:
        game_df.iloc[-1, game_df.columns.get_loc('poss_end')] = True
    
    # Update the main dataframe
    df.loc[game_df.index, 'poss_end'] = game_df['poss_end']
# Handle free throw sequences
# Group by game_id and sequence_number, then for each group of free throws (type_id=540)
# only mark the last one as possession end
# Group free throws by game_id, team_id, and consecutive sequence_numbers
df['ft_group'] = (
    (df['type_id'] == 540) & 
    ((df['team_id'] != df['team_id'].shift(1)) | 
     (df['sequence_number'] != df['sequence_number'].shift(1) + 1))
).cumsum()

# Find the last free throw in each group
last_in_ft_group = df[df['type_id'] == 540].groupby(['game_id', 'ft_group']).tail(1).index

# Set all free throws to False initially
df.loc[df['type_id'] == 540, 'poss_end'] = False

# Drop the temporary column
df.drop('ft_group', axis=1, inplace=True)
# Then set only the last free throw in each sequence to True if it's a scoring play
df.loc[last_in_ft_group, 'poss_end'] = df.loc[last_in_ft_group, 'scoring_play']
# Initialize home_pos and away_pos columns with zeros
df['home_pos'] = -99
df['away_pos'] = -99
# Process each game separately

for game_id in tqdm(df['game_id'].unique()):
    game_mask = df['game_id'] == game_id
    game_df = df[game_mask]
    
    # Get all plays where possessions end
    poss_end_plays = game_df[game_df['poss_end'] == True].copy()
    
    # Count possessions for home team
    home_poss_count = poss_end_plays[poss_end_plays['team_id'] == poss_end_plays['home_team_id']].shape[0]
    
    # Count possessions for away team
    away_poss_count = poss_end_plays[poss_end_plays['team_id'] == poss_end_plays['away_team_id']].shape[0]
    
    # Set initial possession count for first plays of each team in the game
    first_home_play_idx = game_df[game_df['team_id'] == game_df['home_team_id']].index.min()
    first_away_play_idx = game_df[game_df['team_id'] == game_df['away_team_id']].index.min()
    
    if pd.notna(first_home_play_idx):
        df.loc[first_home_play_idx, 'home_pos'] = 1
    if pd.notna(first_away_play_idx):
        df.loc[first_away_play_idx, 'away_pos'] = 1
    
    # Create possession sequences
    home_poss_idx = poss_end_plays[poss_end_plays['team_id'] == poss_end_plays['home_team_id']].index
    away_poss_idx = poss_end_plays[poss_end_plays['team_id'] == poss_end_plays['away_team_id']].index
    
    # Assign cumulative possession counts
    for i, idx in enumerate(sorted(home_poss_idx)):
        df.loc[idx, 'home_pos'] = i + 1
    
    for i, idx in enumerate(sorted(away_poss_idx)):
        df.loc[idx, 'away_pos'] = i + 1
    # Set possession number to 0 for first row of each game for the non-active team
    first_play = game_df.iloc[0]
    first_idx = game_df.index[0]
    # If first play is by home team, set away_pos to 0
    if first_play['team_id'] == first_play['home_team_id']:
        df.loc[first_idx, 'away_pos'] = 0
    # If first play is by away team, set home_pos to 0
    else:
        df.loc[first_idx, 'home_pos'] = 0
    # Forward fill possession counts within each game
    df.loc[game_mask, 'home_pos'] = df.loc[game_mask, 'home_pos'].replace(-99, method='ffill')
    df.loc[game_mask, 'away_pos'] = df.loc[game_mask, 'away_pos'].replace(-99, method='ffill')
    

# Save the modified DataFrame to a new Parquet file
output_file_path = "22_23_mbb_pbp_with_possessions.parquet"
df.to_parquet(output_file_path, index=False)
print(f"Data with possessions saved to {output_file_path}")