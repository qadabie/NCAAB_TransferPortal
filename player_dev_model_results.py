import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from transfer_vars import *

def model_vs_actual(df, scaler, fit_model):
    Y, X, names = model_vars(df)
    X_scaled = scaler.transform(X)
    Y_pred = fit_model.predict(X_scaled)

    summary_df = pd.DataFrame({
        "player": names.reset_index(drop=True),
        "Actual Offensive Rating (2024)": Y,
        "Predicted": Y_pred,
        "Estimated Difference": Y - Y_pred
    })
    return summary_df

def model_results(df):
    mse = mean_squared_error(df["Actual Offensive Rating (2024)"], df["Predicted"])
    r2 = r2_score(df["Actual Offensive Rating (2024)"], df["Predicted"])

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Predicted", y="Actual Offensive Rating (2024)")
    plt.plot([df["Predicted"].min(), df["Predicted"].max()],
             [df["Predicted"].min(), df["Predicted"].max()],
             color='red', linestyle='--')
    plt.title("Actual vs. Predicted Offensive Rating (2024)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual Offensive Rating (2024)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Estimated Difference"], bins=20, kde=True)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Estimated Difference on Offensive Rating")
    plt.xlabel("Effect Size (Actual - Predicted)")
    plt.ylabel("Number of Players")
    plt.show()

    top_effects = df.sort_values("Estimated Difference", ascending=False).head()
    bottom_effects = df.sort_values("Estimated Difference").head()

    return top_effects, bottom_effects