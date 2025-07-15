import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import pd_fe.eng_vars as ev  # use engineered features and composite outcome
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

def model_vs_actual(df, scaler, fit_model):
    Y, X, names = ev.model_vars(df)
    X_scaled = scaler.transform(X)
    Y_pred = fit_model.predict(X_scaled)

    summary_df = pd.DataFrame({
        "player": names.reset_index(drop=True),
        "Actual Composite Score (2024)": Y,
        "Predicted": Y_pred,
        "Estimated Difference": Y - Y_pred
    })
    return summary_df

def model_results(df):
    mse = mean_squared_error(df["Actual Composite Score (2024)"], df["Predicted"])
    r2 = r2_score(df["Actual Composite Score (2024)"], df["Predicted"])

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Predicted", y="Actual Composite Score (2024)")
    plt.plot([df["Predicted"].min(), df["Predicted"].max()],
             [df["Predicted"].min(), df["Predicted"].max()],
             color='red', linestyle='--')
    plt.title("Actual vs. Predicted Composite Score (2024)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual Composite Score (2024)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Estimated Difference"], bins=20, kde=True)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Estimated Difference on Composite Score")
    plt.xlabel("Effect Size (Actual - Predicted)")
    plt.ylabel("Number of Players")
    plt.show()

    top_effects = df.sort_values("Estimated Difference", ascending=False).head()
    bottom_effects = df.sort_values("Estimated Difference").head()

    return top_effects, bottom_effects



def plot_player_radar(player_name, X_df, model, feature_names, top_n = 10):
    """
    Creates a radar (polar) chart comparing a player's scaled top-N feature values to the average player.
    """

    importances = model.final_estimator_.feature_importances_

    if len(importances) != len(feature_names):
        raise ValueError(f"Mismatch: {len(importances)} importances vs {len(feature_names)} features.")

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    top_features = (
        importance_df[~importance_df['feature'].str.startswith("pred_")]
        .sort_values(by='importance', ascending=False)
        .head(top_n)['feature']
        .tolist()
    )

    if 'player' not in X_df.columns:
        raise ValueError("DataFrame must contain a 'player' column.")

    player_row = X_df[X_df['player'].str.lower() == player_name.lower()]
    if player_row.empty:
        raise ValueError(f"Player '{player_name}' not found in DataFrame.")
    player_row = player_row.iloc[0]

    scaled_df = X_df[top_features].copy()
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(scaled_df), columns=top_features)

    player_scaled = scaled_df.loc[player_row.name]
    avg_scaled = scaled_df.mean()

    player_vals = player_scaled.tolist()
    avg_vals = avg_scaled.tolist()

    categories = top_features + [top_features[0]]
    player_vals += [player_vals[0]]
    avg_vals += [avg_vals[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=player_vals,
        theta=categories,
        fill='toself',
        name=player_name,
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatterpolar(
        r=avg_vals,
        theta=categories,
        fill='toself',
        name='Average Player',
        line=dict(color='gray', dash='dot')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=False,
                ticks='',
                showline=False,
                showgrid=True
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
                rotation=90,
                direction="clockwise",
                showline=True,
                linecolor='lightgray',
                linewidth=1,
                gridcolor='lightgray',
                gridwidth=0.5
            )
        ),
        showlegend=True,
        width=700,
        height=600,
        margin=dict(l=45, r=45, t=45, b=45),
        title=f"Top {top_n} Feature Profile for {player_name}"
    )

    return fig

def get_full_feature_names(model, df, model_vars_func = ev.model_vars):
    _, X_df, _ = model_vars_func(df)
    original_feature_names = X_df.columns.tolist()

    base_model_names = [f'pred_{name}' for name in model.named_estimators_.keys()]

    return base_model_names + original_feature_names