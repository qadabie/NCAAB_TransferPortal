import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def split_scale(df, test_size=0.2, random_state=42):
    Y, X, player_names = model_vars(df)

    X_train, X_test, Y_train, Y_test, names_train, names_test = train_test_split(
        X, Y, player_names, test_size=test_size, random_state=random_state
    )

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "names_test": names_test.reset_index(drop=True),
        "scaler_x": scaler_x
    }