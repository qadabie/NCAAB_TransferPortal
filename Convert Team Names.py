import pandas as pd
import pyarrow
def convert_team_name(team_names, source='kenpom', target='espn'):
    """
    Convert team names between KenPom and ESPN naming conventions.
    
    Args:
        team_names (list): The team names to convert
        source (str): The source format ('kenpom' or 'espn')
        target (str): The target format ('kenpom' or 'espn')
    
    Returns:
        str: The converted team name
    """
    espn_to_kenpom_dict = {"Michigan State": "Michigan St.",
    "Iowa State": "Iowa St.",
    "Mississippi State": "Mississippi St.",
    "Ole Miss": "Mississippi",
    "UConn": "Connecticut",
    "American University": "American",
    "San José State": "San Jose St.",
    "San Diego State": "San Diego St.",
    "Alabama State": "Alabama St.",
    "Arkansas State": "Arkansas St.",
    "Oklahoma State": "Oklahoma St.",
    "Norfolk State": "Norfolk St.",
    "Colorado State": "Colorado St.",
    "Jackson State": "Jackson St.",
    "Jacksonville State": "Jacksonville St.",
    "Miami (OH)": "Miami OH",
    "South Carolina State": "South Carolina St.",
    "Wichita State": "Wichita St.",
    "Kent State": "Kent St.",
    "Kennesaw State": "Kennesaw St.",
    "Delaware State": "Delaware St.",
    "Bethune-Cookman": "Bethune Cookman",
    "California Baptist": "Cal Baptist",
    "Utah State": "Utah St.",
    "McNeese": "McNeese St.",
    "Ohio State": "Ohio St.",
    "Grambling": "Grambling St.",
    "Kansas State": "Kansas St.",
    "Florida State": "Florida St.",
    "Northwestern State": "Northwestern St.",
    "Idaho State": "Idaho St.",
    "Nicholls": "Nicholls St.",
    "Cleveland State": "Cleveland St.",
    "Portland State": "Portland St.",
    "Texas A&M-Corpus Christi": "Texas A&M Corpus Chris",
    "SE Louisiana": "Southeastern Louisiana",
    "Omaha": "Nebraska Omaha",
    "Arizona State": "Arizona St.",
    "Long Island University": "Long Island",
    "Miami": "Miami FL",
    "Mississippi Valley State": "Mississippi Valley St.",
    "Sam Houston": "Sam Houston St.",
    "East Tennessee State": "East Tennessee St.",
    "Arkansas-Pine Bluff": "Arkansas Pine Bluff",
    "Albany NY": "Albany",
    "Long Beach State": "Long Beach St.",
    "Weber State": "Weber St.",
    "Tarleton State": "Tarleton St.",
    "Oregon State": "Oregon St.",
    "Southeast Missouri State": "Southeast Missouri St.",
    "Cal State Fullerton": "Cal St. Fullerton",
    "Utah Tech": "Utah Tech",
    "St. Thomas-Minnesota": "St. Thomas",
    "Washington State": "Washington St.",
    "North Dakota State": "North Dakota St.",
    "App State": "Appalachian St.",
    "South Dakota State": "South Dakota St.",
    "Boise State": "Boise St.",
    "Illinois State": "Illinois St.",
    "Georgia State": "Georgia St.",
    "Murray State": "Murray St.",
    "Youngstown State": "Youngstown St.",
    "Coppin State": "Coppin St.",
    "Texas State": "Texas St.",
    "Tennessee State": "Tennessee St.",
    "UIC": "Illinois Chicago",
    "Cal State Bakersfield": "Cal St. Bakersfield",
    "St. Francis (PA)": "Saint Francis",
    "North Carolina State": "N.C. State",
    "Gardner-Webb": "Gardner Webb",
    "Kansas City": "Kansas City",
    "Loyola Maryland": "Loyola MD",
    "Ball State": "Ball St.",
    "Wright State": "Wright St.",
    "Fresno State": "Fresno St.",
    "Alcorn State": "Alcorn St.",
    "Morgan State": "Morgan St.",
    "East Texas A&M": "Texas A&M Commerce",
    "Montana State": "Montana St.",
    "Indiana State": "Indiana St.",
    "Penn State": "Penn St.",
    "South Carolina Upstate": "South Carolina Upstate",
    "Florida International": "Florida International",
    "IU Indy": "IUPUI",
    "Sacramento State": "Sacramento St.",
    "UT Martin": "Tennessee Martin",
    "Stonehill": "Stonehill",
    "New Mexico State": "New Mexico St.",
    "Seattle U": "Seattle",
    "Hawai'i": "Hawaii",
    "UL Monroe": "Louisiana Monroe",
    "Mercyhurst": "Mercyhurst University",
    "West Georgia": "West Georgia",
    "Missouri State": "Missouri St.",
    "Chicago State": "Chicago St.",
    "Pennsylvania": "Penn",
    "Aquinas": "Aquinas College",
    "Puerto Rico-Rio Piedras": "Puerto Rico Rio Piedras",
    "Life Pacific": "Life Pacific University",
    "Morehead State": "Morehead St.",
    "Queens University": "Queens",
    "Florida Tech": "Florida Tech",
    "Puerto Rico-Bayamon": "Puerto Rico Bayamon"
    }
    # Create the reverse mapping dictionary
    kenpom_to_espn_dict = {v: k for k, v in espn_to_kenpom_dict.items()}
    
    # Initialize result list
    converted_names = []
    
    # Process each team name in the input list
    for team_name in team_names:
        if source == "kenpom" and target == "espn":
            new_name = kenpom_to_espn_dict.get(team_name, team_name)
            converted_names.append(new_name)
        elif source == "espn" and target == "kenpom":
            new_name = espn_to_kenpom_dict.get(team_name, team_name)
            converted_names.append(new_name)
        else:
            converted_names.append(team_name)  # Keep original if no mapping exists
    
    return converted_names

if __name__ == "__main__":
    df1 = pd.read_parquet('all_game_network_edges.parquet')
    df1 = df1[df1['season'] != 2025]
    df2 = pd.read_parquet('opponent_tracker_data.parquet')
    df3 = pd.read_csv('team_table.csv')
    df4 = pd.read_parquet('player_box_scores.parquet')
    # Get unique pairs of game_id and game_date from df2
    game_date_map = df4[['game_id', 'game_date']].drop_duplicates().set_index('game_id')['game_date']

    # Create a 'game_date' column in df by mapping from game_id
    df1['game_date'] = df1['game_id'].map(game_date_map)
    df1.loc[df1['game_id'] == 401598912, 'game_date'] = pd.to_datetime('2024-02-26')
    df3['kenpom_name'] = convert_team_name(df3['team'], source='espn', target='kenpom')
    df2['espn_name'] = convert_team_name(df2['team'], source='kenpom', target='espn')
    # Create a dictionary mapping team_id to team_name from df3
    team_id_to_name = dict(zip(df3['team_id'], df3['team']))

    # Map team_id in df1 to team_name
    df1['team_name'] = df1['team_id'].map(team_id_to_name)
    # Ensure date columns are in same format before merging
    df1['game_date'] = pd.to_datetime(df1['game_date'])
    df2['game_date'] = pd.to_datetime(df2['date'])

    # Merge df1 and df2 on team_name/espn_name and game_date
    merged_df = pd.merge(
        df1,
        df2,
        left_on=['team_name', 'game_date'],
        right_on=['espn_name', 'game_date'],
        how='left'
    )
    # # Count and print number of present and missing values in 'adj_de' column
    # present = merged_df['espn_name'].notna().sum()
    # missing = merged_df['espn_name'].isna().sum()
    # total = len(merged_df)

    # print(f"adj_de column statistics:")
    # print(f"Present values: {present} ({present/total:.2%})")
    # print(f"Missing values: {missing} ({missing/total:.2%})")
    # print(f"Total rows: {total}")
    # pd.set_option('display.max_columns', None)  # Show all columns in the output
    # print(merged_df.sample(5))
    # Save the merged DataFrame to a new Parquet file
    merged_df.to_parquet('merged_edges.parquet', engine='pyarrow', index=False)