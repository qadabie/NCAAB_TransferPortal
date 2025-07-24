import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def model_vars(df):

    # Parse and Impute Height
    if df['ht'].dtype == 'object':
        def parse_height(ht_str):
            try:
                feet, inches = ht_str.split('-')
                return int(feet) * 12 + int(inches)
            except:
                return np.nan
        df['height_in'] = df['ht'].apply(parse_height)
    else:
        df['height_in'] = df['ht']

    ht_col = 'height_in'
    wt_col = 'wt'

    # Impute height using similar weight players
    known_ht = df[~df[ht_col].isna() & ~df[wt_col].isna()]
    def impute_height(row):
        if pd.isna(row[ht_col]) and not pd.isna(row[wt_col]):
            similar = known_ht[(known_ht[wt_col] - row[wt_col]).abs() <= 10]
            if not similar.empty:
                return similar[ht_col].mean()
        return row[ht_col]
    df[ht_col] = df.apply(impute_height, axis=1)

    # Base Feature Set
    base_cols = [
        'poss_pct', 'shots_pct',
        'e_fg_pct', 'ts_pct', 
        'or_pct', 'dr_pct',
        'a_rate', 'to_rate', 'blk_pct', 'stl_pct', 
        'f_cper40', 'f_dper40',
        'ftm', 'fta', 'fgm_2', 'fga_2', 'fgm_3', 'fga_3'
    ]
    X_features = df[base_cols].copy()

    # Minutes
    X_features['min_pct'] = np.sqrt(df['min_pct'])

    # Offensive Engineered Features
    X_features['scoring_efficiency_volume'] = df['shots_pct'] * df['ts_pct']

    # Defensive Engineered Features
    X_features['draw_vs_commit_ratio'] = df['f_dper40'] / (df['f_cper40'] + 1e-6)

    # Physical Feature
    X_features['height_in'] = df[ht_col]

    # Role Encoding
    role_dummies = pd.get_dummies(df['role'], drop_first=True)

    # Combine All Features
    X = pd.concat([role_dummies, X_features], axis=1)

    # Outcome Variable Construction
    offense = df['o_rtg']
    defense = df['stl_pct'] + df['blk_pct'] + df['dr_pct']
    support_play = df['or_pct'] + df['a_rate']
    minutes = df['min_pct']

    scaler_y = StandardScaler()
    norm_components = scaler_y.fit_transform(
        pd.DataFrame({
            'offense': offense,
            'defense': defense,
            'support_play': support_play,
            'minutes': minutes
        })
    )
    norm_df = pd.DataFrame(norm_components, columns=['offense', 'defense', 'support_play', 'minutes'])

    # Weighting Outcome Components
    offense_weight = 1.0
    defense_weight = 0.7
    support_weight = 0.3
    minutes_weight = 0.5

    total = offense_weight + defense_weight + support_weight + minutes_weight
    offense_w = offense_weight / total
    defense_w = defense_weight / total
    support_w = support_weight / total
    minutes_w = minutes_weight / total

    Y = (
        offense_w * norm_df['offense'] +
        defense_w * norm_df['defense'] +
        support_w * norm_df['support_play'] +
        minutes_w * norm_df['minutes']
    ).values

    player_names = df['player']

    return Y, X, player_names


def split_scale(df, test_size = 0.2, random_state = 42):
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
        "names_test": names_test.reset_index(drop = True),
        "scaler_x": scaler_x
    }