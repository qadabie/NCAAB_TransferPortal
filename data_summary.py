import pandas as pd
import os

# Try to locate and read the file
file_path = "24_25_mbb_pbp.parquet"  # Updated file extension

try:
    # Attempt to read the file based on extension
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        # Try parquet as default for this case
        df = pd.read_parquet(file_path)
    
    # Filter out "team_X" columns except for "team_Florida" and "team_Houston"
    cols_to_keep = [col for col in df.columns if not col.startswith('team_') or 
                    col in ['team_Florida', 'team_Houston']]
    df = df[cols_to_keep]
    
    # The rest of your code remains the same
    print(f"File successfully loaded: {file_path}")
    print(f"Shape: {df.shape} (rows, columns)")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    # Data types summary
    print("\nData types:")
    print(df.dtypes)
    
    # Basic statistics
    print("\nSummary statistics:")
    print(df.describe())
    
    # Sample data
    print("\nSample data (first 5 rows):")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values by column:")
    print(df.isnull().sum())
    
    # If this is basketball play-by-play data, show some specific summaries
    if 'score' in df.columns or 'points' in df.columns or 'team' in df.columns:
        print("\nBasketball-specific summary:")
        
        # Summarize by team if possible
        if 'team' in df.columns:
            print("\nEvents by team:")
            print(df['team'].value_counts())
        
        # Summarize play types if available
        if 'event_type' in df.columns:
            print("\nEvent types:")
            print(df['event_type'].value_counts())
            
except FileNotFoundError:
    print(f"File not found: {file_path}")
    print("Please ensure the file exists in the current directory or provide the full path.")
except Exception as e:
    print(f"Error reading file: {e}")
    print("This could be due to an incorrect file format or path.")