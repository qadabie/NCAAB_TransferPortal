# 🏀 How to Win In College Basketball during the Transfer Portal Era

Following the rule changes in 2021 and in 2024 the sport has never seen a larger flux of transfers which leads to natural questions about how this change will shape the overall college basketball landscape.  
Question 1 - How does transferring affect a player’s development?   
Question 2 - How do roster and role continuity affect the network structure of an offense?

This project has two components designed to answers these questions.  
Component 1: Individual Player Development Analysis   
Component 2: 

## 📂 Project Structure (Component 1)

### Objectives
- Predict composite player performance scores for 2024 based on 2021 - 2023 performance statistics
- Compare actual vs. predicted performance for transfer players.
- Visualize transfer impact by player and team.

### Getting Started

#### 1. Clone the repo

```bash
git clone https://github.com/qadabie18/NCAAB_TransferPortal.git
```
#### 2. Setup Dependencies
```bash
pip install requirements.txt
```
#### 3. Open Jupyter Notebook File
```bash
pd_fe/ensemble_transfer_model.ipynb
```

### Step 1: Load Data
The data used in this section of the project is stored in an Aiven PostgreSQL database. This setup facilitates pulling 
player statistics from the HoopR library (via R) and processing them using Python.

To load the data, a connection is created using `create_engine` from the `sqlalchemy` library. For this project, the 
database credentials are hardcoded for simplicity. However, for better security during future endeavors, it's recommended to store credentials 
in a `.env` file and access them using `load_dotenv` from the `dotenv` package. A commented example of this more secure 
method is included in the `db_connect.py` file. When prompted, the table name in use is `player_table`

**Python File Imports for Step 1**
```bash
from db_connect import get_connection
```

### Step 2: Clean DataFrame
In this step the dataset is cleaned to ensure consistency and reliability. The `clean_df()` function handles this preprocessing by:
- Selecting only players **who played at one school** in 2021, 2022, and 2023
- Ensuring players have played at least **10 games in each of those years**

The function returns:
- `df_train`: the cleaned dataset used for training the model
- `df_2024_transfers`: 2024 players who transferred (same school for 2021-23 but a new school in 2024)
- `df_2024_no_transfers`: 2024 players who did not transfer (same school from 2021-24)
- `valid_players`: list of the player's names meeting all conditions

***Python File Imports for Step 2**
```bash
from player_dev_clean_df import clean_df
```

### Step 3: Feature Selection, Engineering & Scaling
To prepare the data for modeling, this step performs the following:
- **Feature selection** from base performance statistics (e.g., shooting, rebounding, fouls, etc.)
- **Feature engineering** to create new variables, such as:

  - `scoring_efficiency_volume`: combining shooting volume and efficiency
  - `draw_vs_commit_ratio`: ratio of drawn fouls to committed fouls

- **Height imputation**: if a player’s height is missing, it is estimated using players with similar weight

Our **Target variable** is a **composite score** constructed by weighting four components (offense, defense, support play, and minutes)
All of this is wrapped inside two reusable functions in `eng_vars.py`:

**Python File Imports for Step 3**
```bash
import eng_vars as ev
```
### Step 4: Train Ensemble Model (Stacking Regressor)
In this step, we build and tune a **stacked ensemble regression model** to predict a player's composite performance score.

##### Model Structure
**Base Models**:
  - `RandomForestRegressor`
  - `HistGradientBoostingRegressor`
- **Final Estimator**:
  - `GradientBoostingRegressor`
- **Passthrough**: Enabled (`passthrough = True`) so the original features are available to the final estimator
- **Cross-validation**: 5-fold (`cv = 5`)

##### Hyperparameter Tuning

`GridSearchCV` is used to optimize the following hyperparameters:

- `rf__n_estimators`, `rf__max_depth`
- `gb__max_iter`
- `final_estimator__n_estimators`

To avoid retraining every time, a trained version of the model is saved in the `pd_fe` folder and can be reloaded using joblib. 

**Python File Imports for Step 4**
```bash
from player_dev_ml_models import ml_stack_model_w_grid_search

best_stack_model = joblib.load('best_stack_model.pkl')
```
### Step 5: Model Results
After training the ensemble model, we applied it to both transfer and non-transfer players from the 2024 season to evaluate 
prediction accuracy and model performance. Two distinct prediction scenarios were analyzed:

##### Non-Transfer Players (4 years at one school):
For players who remained at the same school for all four years, the model was trained on their performance from years 1 to 3 
and used to predict their performance in year 4. The prediction was then compared to the actual year 4 performance at the same school.

Model Performance: **Mean Squared Error (MSE)**: 0.10 & **R² Score**: 0.806

##### Transfer Players (transferred before year 4):
For players who transferred after three seasons at one school, the model again used years 1 to 3 for training and predicted 
what their year 4 performance would have been had they stayed. This estimated score was compared against their actual 
performance at the new school, helping assess the developmental impact of transferring.

Model Performance: **Mean Squared Error (MSE)**: 0.05 & **R² Score**: 0.899

These performance metrics suggest that the model is highly effective in capturing player development trajectories, with 
particularly strong predictive power among players who transferred after three seasons with one school and ahead of their 
fourth season in a new place 

##### Visualizing Player Profiles: Polar Chart
To complement the numeric model results, we include radar (polar) charts to visualize individual player performance profiles. 
These charts provide intuitive, side-by-side comparisons between a selected player and the average player across the model’s most important features.

Each radar plot shows:
- The top 10 most important features used by the final ensemble model, based on feature importance scores.
- A shaded blue region representing the selected player’s scaled values for each feature.
- A dashed gray region representing the average scaled values for all players.
- Axis labels for each feature (e.g., ts_pct, dr_pct, to_rate) are scaled to highlight relative differences, not raw values.

This visualization helps answer questions such as:
- Which traits make a player stand out relative to their peers?
- How might a transfer impact a player’s role or statistical footprint?

**Python File Imports for Step 5**
```bash
from p_dev_mdl_rslts import model_vs_actual, model_results, plot_player_radar, get_full_feature_names
```

## 📂 Project Structure (Component 2)

