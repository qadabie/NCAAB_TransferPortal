import pandas as pd
from tqdm import tqdm
df = pd.read_parquet('22_23_mbb_pbp_with_possessions.parquet')
def create_game_nodes(game_id):
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
                    'poss_summary': [poss_summary]
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
    if poss_summary:
        # Add current possession to DataFrame when possession ends
        game_node_df = pd.concat([
            game_node_df,
            pd.DataFrame({
                'game_id': [game_id],
                'team_id': [away_game_df['away_team_id'].iloc[0]],
                'season': [away_game_df['season'].iloc[0]],
                'poss_number': [len(game_node_df) + 1],
                'poss_summary': [poss_summary]
            })
        ], ignore_index=True)
    return game_node_df
game_ids = df['game_id'].unique()
# game_id_1 = game_ids[0]
# game_node_df = create_game_nodes(game_id_1)
# game_node_df.to_csv('game_nodes_sample.csv', index=False)
all_game_nodes = pd.DataFrame()

for game_id in tqdm(game_ids):
    try:
        game_node_df = create_game_nodes(game_id)
        all_game_nodes = pd.concat([all_game_nodes, game_node_df], ignore_index=True)
    except Exception as e:
        print(f"Error processing game_id {game_id}: {e}")

# Save all game nodes to a single file
all_game_nodes.to_parquet('22_23_game_nodes.parquet', index=False)
print(f"Saved {len(all_game_nodes)} rows of game nodes data")