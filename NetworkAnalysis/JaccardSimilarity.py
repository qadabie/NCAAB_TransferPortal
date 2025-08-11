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
        def_to_pct_diffs = []
        def_apl_diffs = []
        def_ft_rate_diffs = []
        home_diffs = []
        # Create sliding windows
        for i in range(len(games) - 6):
            # First window: games i, i+1, i+2
            # Second window: games i+3, i+4, i+5
            first_window = group[group['game_id'].isin(games[i:i+3])]
            second_window = group[group['game_id'].isin(games[i+3:i+6])]
            
            # Create weighted edge dictionaries for both windows
            first_def = first_window['adj_de'].unique().sum()  # Assuming 'adj_de' is the defensive rating
            sec_def = second_window['adj_de'].unique().sum()
            first_def_to_pct = first_window['def_to_pct'].unique().sum()  # Assuming 'def_to_pct' is the defensive percentage
            sec_def_to_pct = second_window['def_to_pct'].unique().sum()
            first_def_apl = first_window['def_apl'].unique().sum()  # Assuming 'def_apl' is the defensive adjusted points allowed
            sec_def_apl = second_window['def_apl'].unique().sum()
            first_def_ft_rate = first_window['def_ft_rate'].unique().sum()  # Assuming 'def_ft_rate' is the defensive free throw rate
            sec_def_ft_rate = second_window['def_ft_rate'].unique().sum()
            first_edges = defaultdict(float)
            # Sum the 'home' count for the first row of each game_id in the first window
            first_home_sum = first_window.groupby('game_id').head(1)['home'].sum()
            # Sum the 'home' count for the first row of each game_id in the second window
            second_home_sum = second_window.groupby('game_id').head(1)['home'].sum()
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
            def_to_pct_diff = first_def_to_pct - sec_def_to_pct
            def_apl_diff = first_def_apl - sec_def_apl
            def_ft_rate_diff = first_def_ft_rate - sec_def_ft_rate

            for edge in all_edges:
                weight1 = first_edges.get(edge, 0)
                weight2 = second_edges.get(edge, 0)
                
                intersection_sum += min(weight1, weight2)
                union_sum += max(weight1, weight2)
            
            jaccard_similarity = intersection_sum / union_sum if union_sum > 0 else 0
            jaccard_similarities.append(jaccard_similarity)
            def_rating_diffs.append(def_rating_diff)
            def_to_pct_diffs.append(def_to_pct_diff)
            def_apl_diffs.append(def_apl_diff)
            def_ft_rate_diffs.append(def_ft_rate_diff)
            home_diffs.append(first_home_sum - second_home_sum)
        results.append({
            'team_id': team_id,
            'season': season,
            'jaccard_similarities': jaccard_similarities,
            'def_rating_diffs': def_rating_diffs,
            'def_to_diffs': def_to_pct_diffs,
            'def_apl_diffs': def_apl_diffs,
            'def_ft_rate_diffs': def_ft_rate_diffs,
            'home_diffs': home_diffs,
        })
    
    return pd.DataFrame(results)

# # Example usage:
df = pd.read_parquet('all_game_network_edges.parquet')
df2 = pd.read_parquet('player_box_scores.parquet', columns=['game_id', 'game_date'])
df3 = pd.read_parquet('opponent_tracker_data.parquet')
# Get unique pairs of game_id and game_date from df2
game_date_map = df2[['game_id', 'game_date']].drop_duplicates().set_index('game_id')['game_date']

# # Create a 'game_date' column in df by mapping from game_id
df['game_date'] = df['game_id'].map(game_date_map)
df['game_date'] = pd.to_datetime(df['game_date'])
df.loc[df['game_id'] == 401598912, 'game_date'] = pd.to_datetime('2024-02-26')
df3['game_date'] = pd.to_datetime(df3['date'])
merged_df = df.merge(df3[['game_date', 'adj_de', 'def_to_pct', 'def_apl', 'def_ft_rate']], on='game_date', how='left')
merged_df.dropna(subset=['adj_de'], inplace=True)  # Drop rows where 'adj_de' is NaN
# Fill NA values in adj_de with 10th percentile
# Calculate 10th percentile of adj_de for filling NA values
adj_de_10th_percentile = merged_df['adj_de'].quantile(0.1)
merged_df['adj_de'].fillna(adj_de_10th_percentile, inplace=True)
def_to_pct_10th_percentile = merged_df['def_to_pct'].quantile(0.1)
merged_df['def_to_pct'].fillna(def_to_pct_10th_percentile, inplace=True)
def_apl_10th_percentile = merged_df['def_apl'].quantile(0.1)
merged_df['def_apl'].fillna(def_apl_10th_percentile, inplace=True)
def_ft_rate_10th_percentile = merged_df['def_ft_rate'].quantile(0.1)
merged_df['def_ft_rate'].fillna(def_ft_rate_10th_percentile, inplace=True)
merged_df.drop_duplicates(inplace=True)
merged_df= merged_df[merged_df['season'] == 2025]
# Calculate weighted Jaccard similarity
result_df = calculate_weighted_jaccard_network_similarity(merged_df)
print(result_df.head())
result_df.to_parquet('merged_jaccard_similarity_2025.parquet', index=False)