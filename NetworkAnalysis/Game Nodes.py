import pandas as pd
from tqdm import tqdm
#Change the path to the parquet file as needed
df = pd.read_parquet(r'data\pbp_cleaned\24_25_mbb_pbp_with_possessions.parquet')
def create_game_nodes(game_id,df):
    home_game_df = df[(df['game_id'] == game_id) & (df['team_id'] == df['home_team_id'])]
    away_game_df = df[(df['game_id'] == game_id) & (df['team_id'] == df['away_team_id'])]
    poss_summary = []
    game_node_df = pd.DataFrame(columns=['game_id','team_id','season','poss_number','poss_summary'])
    previous_score = 0
    for row in range(len(home_game_df)):
        row_data = home_game_df.iloc[row]
        if pd.notna(row_data['player2_id']):
            poss_summary.append(str(row_data['player2_id']))
        poss_summary.append(str(row_data['player1_id']))
        if row_data['shooting_play'] == True:
            score_shot = str(row_data['score_value'])+"_shot"
            poss_summary.append(score_shot)
        if row_data['poss_end'] == True:
            if row_data['shooting_play'] == False:
                poss_summary.append("turnover")
            score_delta = row_data['home_score'] - previous_score
            previous_score = row_data['home_score']
            score_change = str(score_delta) + "_points"
            poss_summary.append(score_change)
                # Add current possession to DataFrame when possession ends
            game_node_df = pd.concat([
                game_node_df,
                pd.DataFrame({
                    'game_id': [game_id],
                    'team_id': [home_game_df['home_team_id'].iloc[0]],
                    'season': [home_game_df['season'].iloc[0]],
                    'poss_number': [len(game_node_df) + 1],
                    'poss_summary': [poss_summary],
                    'home': [1]
                })
            ], ignore_index=True)
            poss_summary = []
    # Process away team possessions
    previous_score = 0
    for row in range(len(away_game_df)):
        row_data = away_game_df.iloc[row]
        if pd.notna(row_data['player2_id']):
            poss_summary.append(str(row_data['player2_id']))
        poss_summary.append(str(row_data['player1_id']))
        if row_data['shooting_play'] == True:
            score_shot = str(row_data['score_value'])+ "_shot"
            poss_summary.append(score_shot)
        if row_data['poss_end'] == True:
            if row_data['shooting_play'] == False:
                poss_summary.append("turnover")
            score_delta = row_data['away_score'] - previous_score
            previous_score = row_data['away_score']
            score_change = str(score_delta) + "_points"
            poss_summary.append(score_change)
            # Add current possession to DataFrame when possession ends
            game_node_df = pd.concat([
                game_node_df,
                pd.DataFrame({
                    'game_id': [game_id],
                    'team_id': [away_game_df['away_team_id'].iloc[0]],
                    'season': [away_game_df['season'].iloc[0]],
                    'poss_number': [len(game_node_df) + 1],
                    'poss_summary': [poss_summary],
                    'home': [0]  
                })
            ], ignore_index=True)
            poss_summary = []   
    return game_node_df

all_game_nodes = pd.DataFrame()

for game_id in tqdm(df['game_id'].unique(), desc="Processing games"):
    try:
        game_nodes = create_game_nodes(game_id, df)
        all_game_nodes = pd.concat([all_game_nodes, game_nodes], ignore_index=True)
    except Exception as e:
        print(f"Error processing game_id {game_id}: {e}")
# Save the combined game nodes DataFrame to a parquet file
all_game_nodes.to_parquet('data/24_25_game_nodes.parquet', index=False)
# seasons = ['21_22', '22_23', '23_24', '24_25']
# all_seasons_nodes = pd.DataFrame()

# for season in seasons:
#     print(f"Processing season {season}...")
#     try:
#         # Read season data
#         season_df = pd.read_parquet(f'data/pbp_cleaned/{season}_mbb_pbp_with_possessions.parquet')
        
#         # Get unique game IDs for this season
#         season_game_ids = season_df['game_id'].unique()
        
#         # Process each game in this season
#         season_nodes = pd.DataFrame()
#         for game_id in tqdm(season_game_ids, desc=f"Games in {season}"):
#             try:
#                 game_node_df = create_game_nodes(game_id, season_df)
#                 season_nodes = pd.concat([season_nodes, game_node_df], ignore_index=True)
#             except Exception as e:
#                 print(f"Error processing game_id {game_id} in season {season}: {e}")
        
#         # Save individual season results
#         season_nodes.to_parquet(f'{season}_game_nodes.parquet', index=False)
#         print(f"Saved {len(season_nodes)} rows for season {season}")
        
#         # Add to combined results
#         all_seasons_nodes = pd.concat([all_seasons_nodes, season_nodes], ignore_index=True)
        
#     except Exception as e:
#         print(f"Error processing season {season}: {e}")

# # Save combined results
# all_seasons_nodes.to_parquet('21_25_game_nodes.parquet', index=False)
# print(f"Saved {len(all_seasons_nodes)} total rows of game nodes data across all seasons")