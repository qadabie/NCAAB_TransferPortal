import pandas as pd
import numpy as np

# List of years and columns to compute differences for
years = [2021, 2022, 2023, 2024]
diff_columns = ['min_pct', 'o_rtg', 'poss_pct', 'shots_pct', 'e_fg_pct', 'ts_pct', 
                'or_pct', 'dr_pct', 'a_rate', 'to_rate', 'blk_pct', 'stl_pct', 
                'f_cper40', 'f_dper40', 'ft_rate', 'ft_pct', 'fg_2_pct', 'fg_3_pct']

# Load all dataframes first
dfs = {}
for year in years:
    file_name = f"data/ncaab_players_{year}.parquet"
    try:
        dfs[year] = pd.read_parquet(file_name)
        print(f"Loaded data for {year}, shape: {dfs[year].shape}")
    except Exception as e:
        print(f"Error loading {file_name}: {e}")

# For each year from 2022 to 2024, compute differences with previous year
result_dfs = []

for year in range(2022, 2025):
    if year not in dfs or year-1 not in dfs:
        print(f"Skipping {year} due to missing data")
        continue
        
    current_df = dfs[year].copy()
    prev_df = dfs[year-1]
    
    # Initialize difference columns with NA
    for col in diff_columns:
        current_df[f"diff_{col}"] = np.nan
    
    current_df["transfer"] = False
    current_df["prev_team"] = np.nan

    # Process each player in current year
    for idx, row in current_df.iterrows():
        player_id = row["player_id"]
        # Check if player exists in previous year
        prev_player_data = prev_df[prev_df["player_id"] == player_id]
        
        if len(prev_player_data) > 0:
            # Player found in previous year
            prev_row = prev_player_data.iloc[0]
            
            # Check if player transferred
            current_df.at[idx, "transfer"] = row["team"] != prev_row["team"]
            current_df.at[idx, "prev_team"] = prev_row["team"]
            # Calculate differences for specified columns
            for col in diff_columns:
                if col in row and col in prev_row:
                    current_df.at[idx, f"diff_{col}"] = row[col] - prev_row[col]
            
    
    # Add to result dataframes
    result_dfs.append(current_df)
    print(f"Processed {year} data, shape: {current_df.shape}")

# Combine results if any exist
if result_dfs:
    final_df = pd.concat(result_dfs)
    
    # Print summary
    print("\nSummary of modified data:")
    print(f"Total number of players: {len(final_df)}")
    print(f"Number of transfers: {final_df['transfer'].sum()}")
    
    # Summary statistics for difference columns
    print("\nStatistics for difference columns:")
    diff_stats = final_df[[f"diff_{col}" for col in diff_columns]].describe()
    print(diff_stats)
    
    # Save to parquet
    output_file = "22_24_ncaab_players_with_diffs.parquet"
    final_df.to_parquet(output_file)
    print(f"\nSaved processed data to {output_file}")
else:
    print("No data was processed. Check if the parquet files exist.")