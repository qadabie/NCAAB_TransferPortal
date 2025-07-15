import pandas as pd
import numpy as np
from push_db_data import push_data_to_db

# Include name mapping to convert team names between ESPN & KenPom data
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
    "IU Indianapolis": "IUPUI",
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
    "Puerto Rico-Bayamon": "Puerto Rico Bayamon",
    "Cal State Northridge": "Cal St. Northridge",
    "UAlbany": "Albany",
    "NC State": "N.C. State",
    "South Carolina Upstate": "USC Upstate",
    "Florida International": "FIU"
    }

# reversal dictionary
kenpom_to_espn_dict = {v: k for k, v in espn_to_kenpom_dict.items()}

# Read parquet files for each season & concatenate into one dataframe
_2021 = pd.read_parquet('data/ncaab_players_2021.parquet')
_2022 = pd.read_parquet('data/ncaab_players_2022.parquet')
_2023 = pd.read_parquet('data/ncaab_players_2023.parquet')
_2024 = pd.read_parquet('data/ncaab_players_2024.parquet')

ncaa_players = pd.concat([_2021, _2022, _2023, _2024], ignore_index=True)

# Group by 'year' and 'team' and get value_counts() of the 'yr' column (indicating player classification)
ncaa_ = ncaa_players.groupby(['year','team'])['yr'].value_counts().reset_index()

# Extract only players with freshman classification
ncaa_freshmen = ncaa_.loc[ncaa_['yr'] == 'Fr']

# Create pivot table to break out rows containing # of freshmen by year to columns
ncaa_freshmen = ncaa_freshmen.pivot(index='team', columns='year', values='count').fillna(0).reset_index()

# Rename columns and convert to int
ncaa_freshmen.rename(columns={2021: '2021_fr', 2022: '2022_fr', 2023: '2023_fr', 2024: '2024_fr'}, inplace=True)
ncaa_freshmen[['2021_fr', '2022_fr', '2023_fr', '2024_fr']] = ncaa_freshmen[['2021_fr', '2022_fr', '2023_fr', '2024_fr']].astype(int)

# Read in team_table
team_table_ = pd.read_csv('team_table_data.csv')

# Strip white space and update team naming conventions for team_table and ncaa_freshmen
team_table_['team'] = team_table_['team'].str.strip()
team_table_['team'] = team_table_['team'].replace(kenpom_to_espn_dict)

ncaa_freshmen['team'] = ncaa_freshmen['team'].str.strip()
ncaa_freshmen['team'] = ncaa_freshmen['team'].replace(kenpom_to_espn_dict)

# Merge ncaa_freshmen to the team_table
team_w_freshmen = team_table_.merge(ncaa_freshmen, on='team', how='left')

# Some teams did not have any information in the ncaab_players_{year}.parquet files
# Updated their freshman totals based on the rosters on team webpages
updated_fr = {'Long Island University': {"2021_fr": 7, "2022_fr": 3, "2023_fr": 4, "2024_fr": 6},
              'Kansas City': {"2021_fr": 5, "2022_fr": 3, "2023_fr": 10, "2024_fr": 3},
              'Mercyhurst': {"2021_fr": 2, "2022_fr": 3, "2023_fr": 4, "2024_fr": 2},
              'West Georgia': {"2021_fr": 2, "2022_fr": 2, "2023_fr": 1, "2024_fr": 2},
              'Aquinas': {"2021_fr": 6, "2022_fr": 8, "2023_fr": 4, "2024_fr": 3},
              'Florida Tech': {"2021_fr": 4, "2022_fr": 5, "2023_fr": 1, "2024_fr": 1},
              'Puerto Rico-Rio Piedras': {"2021_fr": 0, "2022_fr": 0, "2023_fr": 0, "2024_fr": 0},
              'Puerto Rico-Bayamon': {"2021_fr": 0, "2022_fr": 0, "2023_fr": 0, "2024_fr": 0},
              'Life Pacific': {"2021_fr": 4, "2022_fr": 8, "2023_fr": 5, "2024_fr": 6},
}

# Add updated values to merged team_table and convert to int 
for team, fr_counts in updated_fr.items():
    team_w_freshmen.loc[team_w_freshmen['team'] == team, list(fr_counts.keys())] = list(fr_counts.values())

team_w_freshmen.fillna(0, inplace = True)
team_w_freshmen[['2021_fr', '2022_fr', '2023_fr', '2024_fr']] = team_w_freshmen[['2021_fr', '2022_fr', '2023_fr', '2024_fr']].astype(int)

# Read in 22_24_ncaab_players_with_diffs.parquet to get transfer information
transfer_df = pd.read_parquet('22_24_ncaab_players_with_diffs.parquet')

# Group by 'team' and 'year' and extract the value_counts() of 'transfer' column
transfers = transfer_df.groupby(['team', 'year'])['transfer'].value_counts().reset_index()
transfers = transfers.loc[transfers['transfer'] == True]

# Create pivot table to break out # of transfers by year to columns
ncaa_transfers = transfers.pivot(index='team', columns='year', values='count').fillna(0).reset_index()
ncaa_transfers.rename(columns = {2022: "2022_tr", 2023: "2023_tr", 2024: "2024_tr"}, inplace=True)

# Convert transfer value columns to int
# Only 3 transfer columns because 2021 was the first year of data used
ncaa_transfers[['2022_tr','2023_tr','2024_tr']] = ncaa_transfers[['2022_tr','2023_tr','2024_tr']].astype(int)

ncaa_transfers['team'] = ncaa_transfers['team'].str.strip()
ncaa_transfers['team'] = ncaa_transfers['team'].replace(kenpom_to_espn_dict)

# merge the transfer values to the current team table and fill NaN with 0
team_w_fr_tr = team_w_freshmen.merge(ncaa_transfers, how='left', on = 'team')
team_w_fr_tr.fillna(0, inplace = True)

# convert transfer total columns to int & push to database (replaced initial team_table)
team_w_fr_tr[['2022_tr','2023_tr','2024_tr']] = team_w_fr_tr[['2022_tr', '2023_tr','2024_tr']].astype(int)

#push_data_to_db(team_w_fr_tr)
