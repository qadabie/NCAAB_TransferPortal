import pandas as pd

def model_vars(df):
    # Outcome variable
    Y = df['o_rtg'].values

    # Covariates: prior performance
    X_cols = [
        'poss_pct', 'shots_pct', 'e_fg_pct', 'ts_pct', 'or_pct', 'dr_pct',
        'a_rate', 'to_rate', 'blk_pct', 'stl_pct', 'f_cper40', 'f_dper40',
        'ft_rate', 'ftm', 'fta', 'ft_pct', 'fgm_2', 'fga_2', 'fg_2_pct', 'fgm_3', 'fga_3',
        'fg_3_pct', 'min_pct'
    ]
    X_partial = df[X_cols].copy()

    role = pd.get_dummies(df['role'], drop_first = True)
    X = pd.concat([role, X_partial], axis = 1)

    player_names = df['player']

    return Y, X.values, player_names