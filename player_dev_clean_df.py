
def clean_df(df):

    train_years = [2021, 2022, 2023]
    test_year = 2024

    df_hist = df[df['year'].isin(train_years)]
    player_counts = df_hist.groupby('player_id')['year'].nunique()
    players_in_all_years = player_counts[player_counts == 3].index

    df_three_years = df_hist[df_hist['player_id'].isin(players_in_all_years)]

    games_check = (df_three_years.groupby(['player_id', 'year'])['g'].min().unstack().dropna())

    players_with_10_games_each_year = games_check[(games_check >= 10).all(axis = 1)].index
    df_three_years = df_three_years[df_three_years['player_id'].isin(players_with_10_games_each_year)]

    same_team_players = \
    (df_three_years.groupby('player_id')['team'].nunique().reset_index().query("team == 1")['player_id'])

    valid_players = same_team_players

    df_model = df[df['player_id'].isin(valid_players)]

    df_train = df_model[df_model['year'].isin(train_years)]

    df_2024 = df_model[df_model['year'] == test_year]
    df_2024_transfers = df_2024[df_2024['transfer'] == True]
    df_2024_no_transfers = df_2024[df_2024['transfer'] == False]

    return df_train, df_2024_transfers, df_2024_no_transfers, valid_players


