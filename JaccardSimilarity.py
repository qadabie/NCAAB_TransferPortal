import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

def calculate_weighted_jaccard_network_similarity(df):
    """
    Calculate weighted Jaccard similarity for networks based on sliding windows of games.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe containing columns: game_id, team_id, game_date, and edge weights
        Format should be one row per edge with weight value
    
    Returns:
    --------
    pandas DataFrame
        DataFrame with team_id, season, and list of jaccard_similarity values
    """
    # Ensure date is in datetime format
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Add season column (assuming season spans from July to June of next year)
    #df['season'] = df['game_date'].apply(lambda x: x.year if x.month > 6 else x.year - 1)
    
    # Sort by team_id, season, and game_date
    df = df.sort_values(['team_id', 'season', 'game_date'])
    
    results = []
    
    # Group by team and season
    for (team_id, season), group in tqdm(df.groupby(['team_id', 'season'])):
        # Get unique game_ids for this team in chronological order
        games = group['game_id'].unique()
        
        if len(games) < 7:  # Need at least 7 games to calculate first similarity
            continue
        
        jaccard_similarities = []
        def_rating_diffs = []
        # Create sliding windows
        for i in range(len(games) - 6):
            # First window: games i, i+1, i+2
            # Second window: games i+3, i+4, i+5
            first_window = group[group['game_id'].isin(games[i:i+3])]
            second_window = group[group['game_id'].isin(games[i+3:i+6])]
            
            # Create weighted edge dictionaries for both windows
            first_def = first_window['adj_de'].unique().sum()  # Assuming 'adj_de' is the defensive rating
            sec_def = second_window['adj_de'].unique().sum()
            first_edges = defaultdict(float)
            
            for _, row in first_window.iterrows():
                # Assuming edges are identified by some columns in your dataframe
                # Modify this according to your actual data structure
                edge_id = (row.get('source_node', 0), row.get('target_node', 0))
                first_edges[edge_id] += row.get('weight', 1.0)

            
            second_edges = defaultdict(float)
            for _, row in second_window.iterrows():
                edge_id = (row.get('source_node', 0), row.get('target_node', 0))
                second_edges[edge_id] += row.get('weight', 1.0)

            
            # Calculate weighted Jaccard similarity
            intersection_sum = 0
            union_sum = 0
            
            # All unique edges from both windows
            all_edges = set(first_edges.keys()).union(set(second_edges.keys()))
            
            # Calculate Defensive Rating Difference
            def_rating_diff = first_def - sec_def

            for edge in all_edges:
                weight1 = first_edges.get(edge, 0)
                weight2 = second_edges.get(edge, 0)
                
                intersection_sum += min(weight1, weight2)
                union_sum += max(weight1, weight2)
            
            jaccard_similarity = intersection_sum / union_sum if union_sum > 0 else 0
            jaccard_similarities.append(jaccard_similarity)
            def_rating_diffs.append(def_rating_diff)

        results.append({
            'team_id': team_id,
            'season': season,
            'jaccard_similarities': jaccard_similarities,
            'def_rating_diffs': def_rating_diffs
        })
    
    return pd.DataFrame(results)

# # Example usage:
# df = pd.read_parquet('all_game_network_edges.parquet')
# df2 = pd.read_parquet('player_box_scores.parquet')
# # Get unique pairs of game_id and game_date from df2
# game_date_map = df2[['game_id', 'game_date']].drop_duplicates().set_index('game_id')['game_date']

# # Create a 'game_date' column in df by mapping from game_id
# df['game_date'] = df['game_id'].map(game_date_map)
# df.loc[df['game_id'] == 401598912, 'game_date'] = pd.to_datetime('2024-02-26')
df= pd.read_parquet('merged_edges.parquet')
df.dropna(subset=['adj_de'], inplace=True)  # Drop rows where 'adj_de' is NaN
# Fill NA values in adj_de with 10th percentile
# Calculate 10th percentile of adj_de for filling NA values
adj_de_10th_percentile = df['adj_de'].quantile(0.1)
df['adj_de'].fillna(adj_de_10th_percentile, inplace=True)

# Print the count of team-season combinations with at least 100 rows
df = df.groupby(['team_id', 'season']).filter(lambda x: len(x) >= 100)
# Calculate weighted Jaccard similarity
result_df = calculate_weighted_jaccard_network_similarity(df)
print(result_df.head())
result_df.to_parquet('merged_jaccard_similarity.parquet', index=False)