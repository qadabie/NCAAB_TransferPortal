from db_connect import get_connection
import pandas as pd
from scipy.stats import pearsonr
from Convert_Team_Names import convert_team_name
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
from scipy import stats
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr

engine = get_connection()
pd.set_option('display.max_columns', None)  # Show all columns in the DataFrame
# Show available tables
df = pd.read_sql("SELECT * FROM team_table", con=engine)
df3 = pd.read_sql("SELECT * FROM player_table", con=engine)

# Group by team and year, sum poss_pct for transfers
poss_sum_tr = df3[df3['transfer'] == True].groupby(['team', 'year'])['min_pct'].sum().reset_index()
poss_sum_tr = poss_sum_tr.rename(columns={'min_pct': 'min_sum_tr'})

# Group by team and year, sum min_pct for freshmen (yr == 'Fr')

poss_sum_fr = df3[df3['yr'] == 'Fr'].groupby(['team', 'year'])['min_pct'].sum().reset_index()
poss_sum_fr = poss_sum_fr.rename(columns={'min_pct': 'min_sum_fr'})

# Merge the two results
poss_summary = pd.merge(poss_sum_tr, poss_sum_fr, on=['team', 'year'], how='outer')
poss_summary['min_sum_tr'] = poss_summary['min_sum_tr'].fillna(0)
poss_summary['min_sum_fr'] = poss_summary['min_sum_fr'].fillna(0)
poss_summary['min_sum_new'] = poss_summary['min_sum_tr'] + poss_summary['min_sum_fr']

df2 = pd. read_csv('team_adjust_times_and_eos_similarity.csv')
df_results = pd.read_csv('cbb.csv')
df25_results = pd.read_csv('cbb25.csv')
df25_results['TEAM'] = df25_results['Team']
df25_results['YEAR'] = 2025
df_results = pd.concat([df_results, df25_results], ignore_index=True)
# Create a df with team_id and season for each team, with seasons 2022-2025
team_seasons = []
for team_id in df['team_id'].unique():
    team_name = df[df['team_id'] == team_id]['team'].iloc[0]
    for season in range(2022, 2026):
        team_seasons.append({'team_id': team_id, 'team': team_name, 'season': season})

df_team_seasons = pd.DataFrame(team_seasons)
merged_df = pd.merge(df_team_seasons, df2, on=['team_id', 'season'], how='inner')
merged_df['team'] = convert_team_name(merged_df['team'],source='espn', target='kenpom')
merged_df = merged_df.rename(columns={'season': 'year'})
merged_df = pd.merge(merged_df, poss_summary, on=['team', 'year'], how='left')
other_name_dict = {'NC State': 'North Carolina St.',
                   }
Power_Conf = ['ACC','B12','B10','SEC', 'BE','P12']
df_results['Pwr_Cnf'] = df_results['CONF'].apply(lambda x: 1 if x in Power_Conf else 0)
# Set P12 teams in 2025 to Pwr_Cnf = 0
df_results.loc[(df_results['CONF'] == 'P12') & (df_results['YEAR'] == 2025), 'Pwr_Cnf'] = 0
df_results['W/G'] = df_results['W'] / df_results['G']

merged_df = pd.merge(merged_df, df_results[['TEAM','YEAR','CONF','Pwr_Cnf','W/G','POSTSEASON','SEED']], left_on=['team', 'year'], right_on=['TEAM', 'YEAR'], how='inner')
postseason_mapping = {
    'R68': 0,
    'R64': 0,
    'R32': 1,
    'S16': 2,
    'E8': 3,
    'F4': 4,
    '2ND': 5,
    'Champions': 6
}
expected_wins = {
    1: 4,
    2: 3,
    3: 2,
    4: 2,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
}
merged_df['POSTSEASON'] = merged_df['POSTSEASON'].map(postseason_mapping)
merged_df['EWINS'] = merged_df['SEED'].map(expected_wins)
merged_df['post_perf'] = merged_df.apply(
    lambda row: 1 if pd.notna(row['POSTSEASON']) and pd.notna(row['EWINS']) and row['POSTSEASON'] > row['EWINS']
                else (-1 if pd.notna(row['POSTSEASON']) and pd.notna(row['EWINS']) and row['POSTSEASON'] < row['EWINS']
                     else (0 if pd.notna(row['POSTSEASON']) and pd.notna(row['EWINS']) and row['POSTSEASON'] == row['EWINS']
                          else np.nan)), axis=1
)
merged_df = merged_df.drop(columns=['TEAM'])
# merged_df['new'] = merged_df['fr'] + merged_df['tr']
merged_df = merged_df.dropna(subset=['adjust_time'])
merged_df.to_csv('merged_conf.csv', index=False)
power_conf_teams_only = merged_df[merged_df['Pwr_Cnf'] == 1]
#power_conf_teams_only = power_conf_teams_only[power_conf_teams_only['tr'] <= 5]

power_conf_teams_only.dropna(subset=['min_sum_new', 'adjust_time','EoS_similarity'], inplace=True)
power_conf_teams_only = power_conf_teams_only.drop_duplicates()
power_conf_teams_only = power_conf_teams_only[power_conf_teams_only['min_sum_new'] < 475]

# Create subgroups
transfer_heavy_teams = power_conf_teams_only[power_conf_teams_only['min_sum_tr'] > 200]
freshman_heavy_teams = power_conf_teams_only[power_conf_teams_only['min_sum_fr'] > 200]
both_heavy_teams = power_conf_teams_only[(power_conf_teams_only['min_sum_tr'] > 100) & (power_conf_teams_only['min_sum_fr'] > 100)]

print(f"Total Power Conference Teams: {len(power_conf_teams_only)}")
print(power_conf_teams_only.head(2))
# Create a table for team profiles based on postseason performance
postseason_profile = power_conf_teams_only.groupby('post_perf').agg({
    'adjust_time': 'mean',
    'segments': 'mean',
    'EoS_similarity': 'mean'
}).round(3)

postseason_profile['count'] = power_conf_teams_only.groupby('post_perf').size()

# Add post_perf labels for better readability
post_perf_labels = {
    -1: 'Underperformed',
    0: 'Expected Performance',
    1: 'Overperformed'
}

postseason_profile.index = postseason_profile.index.map(lambda x: post_perf_labels.get(x, 'Expected Performance'))
print("\nTeam Profiles by Postseason Performance:")
print(postseason_profile)
# Calculate correlations
corr, p_value = spearmanr(power_conf_teams_only['adjust_time'], power_conf_teams_only['W/G'])
corr2, p_value2 = spearmanr(power_conf_teams_only['adjust_time'], power_conf_teams_only['min_sum_tr'])
corr3, p_value3 = spearmanr(power_conf_teams_only['EoS_similarity'], power_conf_teams_only['W/G'])
corr4, p_value4 = spearmanr(power_conf_teams_only['EoS_similarity'], power_conf_teams_only['min_sum_tr'])
corr5, p_value5 = spearmanr(power_conf_teams_only['segments'], power_conf_teams_only['W/G'])
corr6, p_value6 = spearmanr(power_conf_teams_only['segments'], power_conf_teams_only['min_sum_tr'])
corr7, p_value7 = spearmanr(power_conf_teams_only['min_sum_tr'], power_conf_teams_only['W/G'])
corr8, p_value8 = spearmanr(power_conf_teams_only['min_sum_fr'], power_conf_teams_only['W/G'])
corr9, p_value9 = spearmanr(power_conf_teams_only['min_sum_new'], power_conf_teams_only['W/G'])
print(f"Spearman correlation coefficient (Adjusted Time vs W/G): {corr:.4f}, p-value: {p_value:.4g}")
print(f"Spearman correlation coefficient (Adjusted Time vs Transfer Minutes): {corr2:.4f}, p-value: {p_value2:.4g}")
print(f"Spearman correlation coefficient (EoS Similarity vs W/G): {corr3:.4f}, p-value: {p_value3:.4g}")
print(f"Spearman correlation coefficient (EoS Similarity vs Transfer Minutes): {corr4:.4f}, p-value: {p_value4:.4g}")
print(f"Spearman correlation coefficient (Segments vs W/G): {corr5:.4f}, p-value: {p_value5:.4g}")
print(f"Spearman correlation coefficient (Segments vs Transfer Minutes): {corr6:.4f}, p-value: {p_value6:.4g}")
print(f"Spearman correlation coefficient (Transfer Minutes vs W/G): {corr7:.4f}, p-value: {p_value7:.4g}")
print(f"Spearman correlation coefficient (Freshman Minutes vs W/G): {corr8:.4f}, p-value: {p_value8:.4g}")
print(f"Spearman correlation coefficient (New Minutes vs W/G): {corr9:.4f}, p-value: {p_value9:.4g}")

# Create correlation charts
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
plt.subplots_adjust(hspace=1)  # Add more vertical space between rows

# Chart 1: Adjusted Time vs W/G
axes[0, 0].scatter(power_conf_teams_only['adjust_time'], power_conf_teams_only['W/G'], alpha=0.6, color='blue')
z = np.polyfit(power_conf_teams_only['adjust_time'], power_conf_teams_only['W/G'], 1)
p = np.poly1d(z)
axes[0, 0].plot(power_conf_teams_only['adjust_time'], p(power_conf_teams_only['adjust_time']), "r--", alpha=0.8)
axes[0, 0].set_xlabel('Adjusted Time')
axes[0, 0].set_ylabel('Win Rate (W/G)')
axes[0, 0].set_title(f'Adjusted Time vs Win Rate\nSpearman r = {corr:.4f}, p = {p_value:.4g}')
axes[0, 0].grid(True, alpha=0.3)

# Chart 2: EoS Similarity vs W/G
axes[0, 1].scatter(power_conf_teams_only['EoS_similarity'], power_conf_teams_only['W/G'], alpha=0.6, color='green')
z2 = np.polyfit(power_conf_teams_only['EoS_similarity'], power_conf_teams_only['W/G'], 1)
p2 = np.poly1d(z2)
axes[0, 1].plot(power_conf_teams_only['EoS_similarity'], p2(power_conf_teams_only['EoS_similarity']), "r--", alpha=0.8)
axes[0, 1].set_xlabel('EoS Similarity')
axes[0, 1].set_ylabel('Win Rate (W/G)')
axes[0, 1].set_title(f'EoS Similarity vs Win Rate\nSpearman r = {corr3:.4f}, p = {p_value3:.4g}')
axes[0, 1].grid(True, alpha=0.3)

# Chart 3: Transfer Minutes vs Segments
axes[1, 0].scatter(power_conf_teams_only['adjust_time'], power_conf_teams_only['min_sum_tr'], alpha=0.6, color='orange')
z3 = np.polyfit(power_conf_teams_only['adjust_time'], power_conf_teams_only['min_sum_tr'], 1)
p3 = np.poly1d(z3)
axes[1, 0].plot(power_conf_teams_only['adjust_time'], p3(power_conf_teams_only['adjust_time']), "r--", alpha=0.8)
axes[1, 0].set_xlabel('Adjusted Time')
axes[1, 0].set_ylabel('min_sum_tr (Transfer Minutes)')
axes[1, 0].set_title(f'Adjusted Time vs Transfer Minutes\nSpearman r = {corr2:.4f}, p = {p_value2:.4g}')
axes[1, 0].grid(True, alpha=0.3)

# Chart 4: EoS Similarity vs Transfer Minutes
axes[1, 1].scatter(power_conf_teams_only['EoS_similarity'], power_conf_teams_only['min_sum_tr'], alpha=0.6, color='purple')
z4 = np.polyfit(power_conf_teams_only['EoS_similarity'], power_conf_teams_only['min_sum_tr'], 1)
p4 = np.poly1d(z4)
axes[1, 1].plot(power_conf_teams_only['EoS_similarity'], p4(power_conf_teams_only['EoS_similarity']), "r--", alpha=0.8)
axes[1, 1].set_xlabel('EoS Similarity')
axes[1, 1].set_ylabel('min_sum_tr (Transfer Minutes)')
axes[1, 1].set_title(f'EoS Similarity vs Transfer Minutes\nSpearman r = {corr4:.4f}, p = {p_value4:.4g}')
axes[1, 1].grid(True, alpha=0.3)

#Chart 5: Segments vs W/G
axes[0, 2].scatter(power_conf_teams_only['segments'], power_conf_teams_only['W/G'], alpha=0.6, color='brown')
z5 = np.polyfit(power_conf_teams_only['segments'], power_conf_teams_only['W/G'], 1)
p5 = np.poly1d(z5)
axes[0, 2].plot(power_conf_teams_only['segments'], p5(power_conf_teams_only['segments']), "r--", alpha=0.8)
axes[0, 2].set_xlabel('Segments')
axes[0, 2].set_ylabel('Win Rate (W/G)')
axes[0, 2].set_title(f'Segments vs Win Rate\nSpearman r = {corr5:.4f}, p = {p_value5:.4g}')
axes[0, 2].grid(True, alpha=0.3)

#Chart 6: Segments vs Transfer Minutes
axes[1, 2].scatter(power_conf_teams_only['segments'], power_conf_teams_only['min_sum_tr'], alpha=0.6, color='purple')
z6 = np.polyfit(power_conf_teams_only['segments'], power_conf_teams_only['min_sum_tr'], 1)
p6 = np.poly1d(z6)
axes[1, 2].plot(power_conf_teams_only['segments'], p6(power_conf_teams_only['segments']), "r--", alpha=0.8)
axes[1, 2].set_xlabel('Segments')
axes[1, 2].set_ylabel('min_sum_tr (Transfer Minutes)')
axes[1, 2].set_title(f'Segments vs Transfer Minutes\nSpearman r = {corr6:.4f}, p = {p_value6:.4g}')
axes[1, 2].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# Adjust the figure to accommodate a third row
plt.close('all')

# Create separate chart for player minutes vs win rate
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

# Chart 1: Transfer Minutes vs W/G
axes2[0].scatter(power_conf_teams_only['min_sum_tr'], power_conf_teams_only['W/G'], alpha=0.6, color='orange')
z7 = np.polyfit(power_conf_teams_only['min_sum_tr'], power_conf_teams_only['W/G'], 1)
p7 = np.poly1d(z7)
axes2[0].plot(power_conf_teams_only['min_sum_tr'], p7(power_conf_teams_only['min_sum_tr']), "r--", alpha=0.8)
axes2[0].set_xlabel('min_sum_tr (Transfer Minutes)')
axes2[0].set_ylabel('Win Rate (W/G)')
axes2[0].set_title(f'Transfer Minutes vs Win Rate\nSpearman r = {corr7:.4f}, p = {p_value7:.4g}')
axes2[0].grid(True, alpha=0.3)

# Chart 2: Freshman Minutes vs W/G
axes2[1].scatter(power_conf_teams_only['min_sum_fr'], power_conf_teams_only['W/G'], alpha=0.6, color='darkblue')
z8 = np.polyfit(power_conf_teams_only['min_sum_fr'], power_conf_teams_only['W/G'], 1)
p8 = np.poly1d(z8)
axes2[1].plot(power_conf_teams_only['min_sum_fr'], p8(power_conf_teams_only['min_sum_fr']), "r--", alpha=0.8)
axes2[1].set_xlabel('min_sum_fr (Freshman Minutes)')
axes2[1].set_ylabel('Win Rate (W/G)')
axes2[1].set_title(f'Freshman Minutes vs Win Rate\nSpearman r = {corr8:.4f}, p = {p_value8:.4g}')
axes2[1].grid(True, alpha=0.3)

# Chart 3: New Minutes vs W/G
axes2[2].scatter(power_conf_teams_only['min_sum_new'], power_conf_teams_only['W/G'], alpha=0.6, color='darkgreen')
z9 = np.polyfit(power_conf_teams_only['min_sum_new'], power_conf_teams_only['W/G'], 1)
p9 = np.poly1d(z9)
axes2[2].plot(power_conf_teams_only['min_sum_new'], p9(power_conf_teams_only['min_sum_new']), "r--", alpha=0.8)
axes2[2].set_xlabel('min_sum_new (New Player Minutes)')
axes2[2].set_ylabel('Win Rate (W/G)')
axes2[2].set_title(f'New Player Minutes vs Win Rate\nSpearman r = {corr9:.4f}, p = {p_value9:.4g}')
axes2[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
import plotly.graph_objects as go
import plotly.express as px
# Transform smoothed data to percentage of season
merged_df.dropna(subset=['smoothed'], inplace=True)
print(len(merged_df))
def process_smoothed_data(df):
    new_df = []
    for __, row in df.iterrows():
        team = row['team']
        year = row['year']
        smoothed_value = row['smoothed'][1:-2]
        min_sum_tr = row['min_sum_tr']
        min_sum_fr = row['min_sum_fr']
        is_transfer_heavy = min_sum_tr > 200
        is_freshman_heavy = min_sum_fr > 200
        count = 0
        for item in smoothed_value.split(' '):
            if len(item) >0:
                new_df.append({
                    'team': team,
                    'year': year,
                    'smoothed_value': float(item),
                    'index_value': count,
                    'is_transfer_heavy': is_transfer_heavy,
                    'is_freshman_heavy': is_freshman_heavy,
                    'min_sum_tr': min_sum_tr,
                    'min_sum_fr': min_sum_fr
                })
                count += 1

    return pd.DataFrame(new_df) 
#print(power_conf_teams_only['smoothed'].describe())
# Process the data
expanded_df = process_smoothed_data(merged_df)
expanded_df.drop_duplicates(inplace=True)
print(expanded_df['year'].value_counts())
# Create density line plot function
def create_scatter_plot(data):
    """Create a cumulative distribution plot for adjust_time by team groups"""
    
    # Get the original merged data with adjust_time values
    # We need to work with the power_conf_teams_only data that has adjust_time
    
    # Create the plot
    fig = go.Figure()
    
    # Filter to power conference teams only
    power_conf_data = merged_df[merged_df['Pwr_Cnf'] == 1].dropna(subset=['adjust_time'])
    
    # Create subgroups
    transfer_heavy = power_conf_data[power_conf_data['min_sum_tr'] > 150]
    freshman_heavy = power_conf_data[power_conf_data['min_sum_fr'] > 150]
    all_teams = power_conf_data
    
    # Function to calculate cumulative distribution
    def calc_cumulative_dist(group_data, column):
        sorted_values = np.sort(group_data[column].values)
        n = len(sorted_values)
        cumulative_prob = np.arange(1, n + 1) / n
        return sorted_values, cumulative_prob
    
    # Calculate cumulative distributions
    all_x, all_y = calc_cumulative_dist(all_teams, 'adjust_time')
    transfer_x, transfer_y = calc_cumulative_dist(transfer_heavy, 'adjust_time')
    freshman_x, freshman_y = calc_cumulative_dist(freshman_heavy, 'adjust_time')
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=all_x,
            y=all_y,
            mode='lines',
            name='All Power Conf Teams',
            line=dict(color='rgba(128, 128, 128, 0.8)', width=2),
            hovertemplate='Adjust Time: %{x:.2f}<br>Cumulative Prob: %{y:.3f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=transfer_x,
            y=transfer_y,
            mode='lines',
            name=f'Transfer Heavy (n={len(transfer_heavy)})',
            line=dict(color='rgba(255, 0, 0, 0.8)', width=3),
            hovertemplate='Adjust Time: %{x:.2f}<br>Cumulative Prob: %{y:.3f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=freshman_x,
            y=freshman_y,
            mode='lines',
            name=f'Freshman Heavy (n={len(freshman_heavy)})',
            line=dict(color='rgba(0, 0, 255, 0.8)', width=3),
            hovertemplate='Adjust Time: %{x:.2f}<br>Cumulative Prob: %{y:.3f}<extra></extra>'
        )
    )
    
    # Add reference lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="black", opacity=0.5,
                  annotation_text="50th percentile")
    fig.add_hline(y=0.25, line_dash="dot", line_color="gray", opacity=0.3,
                  annotation_text="25th percentile")
    fig.add_hline(y=0.75, line_dash="dot", line_color="gray", opacity=0.3,
                  annotation_text="75th percentile")
    
    # Update layout
    fig.update_layout(
        title="Cumulative Distribution of Adjust Time by Team Composition",
        xaxis=dict(
            title="Adjust Time",
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        yaxis=dict(
            title="Cumulative Probability",
            range=[0, 1],
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickformat='.0%'
        ),
        height=600,
        width=1000,
        hovermode='closest',
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        plot_bgcolor='white'
    )
    
    return fig

# Create and display the scatter plot
# fig_scatter = create_scatter_plot(expanded_df)

# Display the plot
# import plotly.offline as pyo
# pyo.plot(fig_scatter, filename='scatter_plot.html', auto_open=True)

# Print summary statistics
print(f"\nData Summary:")
print(f"Total observations: {len(expanded_df)}")
print(f"Transfer heavy observations: {len(expanded_df[expanded_df['is_transfer_heavy']])}")
print(f"Freshman heavy observations: {len(expanded_df[expanded_df['is_freshman_heavy']])}")
plt.close('all')