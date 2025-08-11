import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM

# Load the parquet file
df = pd.read_parquet('opponent_tracker_data.parquet')

# Convert date column to datetime if it's not already
df['date'] = pd.to_datetime(df['date'])

# Filter for dates between November 1st and December 15th
mask = (df['date'].dt.month == 11) | ((df['date'].dt.month == 12) & (df['date'].dt.day <= 15))
filtered_df = df[mask]

# Group by team and year, then calculate average adj_de
early_de_df = filtered_df.groupby(['team', 'year'])['adj_de'].mean().reset_index()
early_de_df.rename(columns={'adj_de': 'early_de'}, inplace=True)

df2 = pd.read_csv('merged_conf.csv')
df3 = pd.read_csv('NCSOS_pull.csv')
# Merge the early_de_df with df2 on 'team' and 'year'
merged_df = pd.merge(df2, early_de_df, on=['team', 'year'], how='left')
merged_df = pd.merge(merged_df, df3[['team', 'year', 'NCSOS']], on=['team', 'year'], how='left')
merged_df = merged_df[['team', 'year','early_de','adjust_time','EoS_similarity','segments','min_sum_tr','min_sum_fr','min_sum_new','Pwr_Cnf','CONF','NCSOS','POSTSEASON']]
print(merged_df['year'].value_counts())

import statsmodels.api as sm

merged_df['EoS_similarity_z'] = (merged_df['EoS_similarity'] - merged_df['EoS_similarity'].mean()) / merged_df['EoS_similarity'].std()
merged_df['min_sum_new_z'] = (merged_df['min_sum_new'] - merged_df['min_sum_new'].mean()) / merged_df['min_sum_new'].std()
merged_df['min_sum_tr_z'] = (merged_df['min_sum_tr'] - merged_df['min_sum_tr'].mean()) / merged_df['min_sum_tr'].std()
merged_df['min_sum_fr_z'] = (merged_df['min_sum_fr'] - merged_df['min_sum_fr'].mean()) / merged_df['min_sum_fr'].std()
merged_df['early_de_z'] = (merged_df['early_de'] - merged_df['early_de'].mean()) / merged_df['early_de'].std()
merged_df['adjust_time_z'] = (merged_df['adjust_time'] - merged_df['adjust_time'].mean()) / merged_df['adjust_time'].std()
merged_df['NCSOS_z'] = (merged_df['NCSOS'] - merged_df['NCSOS'].mean()) / merged_df['NCSOS'].std()
merged_df['Pwr_Cnf'] = merged_df['Pwr_Cnf'].astype(int)
high_transfer = merged_df#[(merged_df['min_sum_tr'] >= 200)& (merged_df['min_sum_tr'] < 475)]
# Calculate min_sum_tr_z values for specific min_sum_tr values
transfer_values = [100, 150, 200, 250, 300]
mean_tr = merged_df['min_sum_tr'].mean()
std_tr = merged_df['min_sum_tr'].std()

print("min_sum_tr_z values:")
for val in transfer_values:
    z_score = (val - mean_tr) / std_tr
    print(f"min_sum_tr = {val}: min_sum_tr_z = {z_score:.3f}")
high_transfer = high_transfer[high_transfer['Pwr_Cnf'] == 1]
high_transfer['min_sum_tr'] = high_transfer['min_sum_tr']/5
high_transfer = high_transfer.dropna(subset=['NCSOS_z', 'min_sum_tr','segments'])
high_transfer = high_transfer.drop_duplicates()
# Fit mixed effects model with simpler random effects structure
model = MixedLM.from_formula('segments ~ NCSOS_z + min_sum_tr', 
                            data=high_transfer,
                            groups=high_transfer['CONF'],
                            re_formula='1')
result = model.fit(method='lbfgs', maxiter=1000)
print("Full Dataset Results:")
print(result.summary())
print(f"\nP-values:")
print(f"NCSOS_z p-value: {result.pvalues['NCSOS_z']:.6f}")
print(f"min_sum_tr p-value: {result.pvalues['min_sum_tr']:.6f}")

import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Generate prediction curves at low vs high NCSOS
x_vals = np.linspace(0, 600, 100)
ncsos_low = -1.5
ncsos_high = 1.5

# Extract fixed effect coefficients
fe = result.params
def predict(NCSOS_z, min_sum_tr):
    return (
        fe["Intercept"]
        + fe["NCSOS_z"] * NCSOS_z
        + fe["min_sum_tr"] * min_sum_tr
    )

# Set None values for POSTSEASON to -1
high_transfer['POSTSEASON'] = high_transfer['POSTSEASON'].fillna(-1)

pred_low = [predict(ncsos_low, x) for x in x_vals]
pred_high = [predict(ncsos_high, x) for x in x_vals]

import plotly.graph_objects as go
import plotly.express as px
# Create dashboard with slider
fig = go.Figure()

# Get unique postseason values for slider
postseason_values = sorted(high_transfer['POSTSEASON'].unique())

# Add traces for each postseason threshold
for threshold in postseason_values:
    # Data meeting threshold
    meeting_threshold = high_transfer[high_transfer['POSTSEASON'] >= threshold]
    # Data below threshold
    below_threshold = high_transfer[high_transfer['POSTSEASON'] < threshold]
    
    # Add below threshold data (greyed out)
    fig.add_trace(go.Scatter(
        x=below_threshold['min_sum_tr'],
        y=below_threshold['segments'],
        mode='markers',
        marker=dict(color='lightgrey', size=6, opacity=0.3),
        name=f'Below Postseason {threshold}',
        visible=(threshold == postseason_values[0]),
        showlegend=False,  # Hide from legend to reduce clutter
        hovertemplate='<b>%{customdata[0]} (%{customdata[1]})</b><br>' +
                      'Transfer % Minutes: %{x}<br>' +
                      'Segments: %{y}<br>' +
                      'NCSOS (Z): %{customdata[2]:.2f}<br>' +
                      'Postseason: %{customdata[3]}<extra></extra>',
        customdata=list(zip(below_threshold['team'], below_threshold['year'], 
                           below_threshold['NCSOS_z'], below_threshold['POSTSEASON']))
    ))
    
    # Add meeting threshold data (colored by NCSOS)
    fig.add_trace(go.Scatter(
        x=meeting_threshold['min_sum_tr'],
        y=meeting_threshold['segments'],
        mode='markers',
        marker=dict(
            color=meeting_threshold['NCSOS_z'],
            colorscale='plasma',
            size=8,
            opacity=0.8,
            cmin=-2,
            cmax=2,
            colorbar=dict(title="NCSOS (Standardized)", x=1.02)
        ),
        name=f'Postseason ≥ {threshold}',
        visible=(threshold == postseason_values[0]),
        showlegend=False,  # Hide from legend to reduce clutter
        hovertemplate='<b>%{customdata[0]} (%{customdata[1]})</b><br>' +
                      'Transfer % Minutes: %{x}<br>' +
                      'Segments: %{y}<br>' +
                      'NCSOS (Z): %{customdata[2]:.2f}<br>' +
                      'Postseason: %{customdata[3]}<extra></extra>',
        customdata=list(zip(meeting_threshold['team'], meeting_threshold['year'], 
                           meeting_threshold['NCSOS_z'], meeting_threshold['POSTSEASON']))
    ))

# Update x_vals to go from 0 to 100
x_vals = np.linspace(0, 100, 100)

pred_low = [predict(ncsos_low, x) for x in x_vals]
pred_high = [predict(ncsos_high, x) for x in x_vals]

# Add prediction lines (always visible)
fig.add_trace(go.Scatter(
    x=x_vals,
    y=pred_low,
    mode='lines',
    line=dict(dash='dash', color='blue', width=3),
    name='Low NCSOS Prediction',
    hovertemplate='Low NCSOS Prediction<br>Transfer Minutes: %{x}<br>Segments: %{y:.2f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=x_vals,
    y=pred_high,
    mode='lines',
    line=dict(dash='dash', color='red', width=3),
    name='High NCSOS Prediction',
    hovertemplate='High NCSOS Prediction<br>Transfer Minutes: %{x}<br>Segments: %{y:.2f}<extra></extra>'
))

# Create slider steps
steps = []
slider_labels = ["All Teams", "Tournament Teams", "R32", "S16", "E8", "Final Four", "Finals", "Champions"]

for i, threshold in enumerate(postseason_values):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)}],
        label=slider_labels[i] if i < len(slider_labels) else str(int(threshold))
    )
    # Show data for current threshold
    step["args"][0]["visible"][i*2] = True  # Below threshold (greyed)
    step["args"][0]["visible"][i*2 + 1] = True  # Meeting threshold (colored)
    # Always show prediction lines
    step["args"][0]["visible"][-2] = True  # Low NCSOS line
    step["args"][0]["visible"][-1] = True  # High NCSOS line
    steps.append(step)

# Add slider
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Minimum Postseason Level: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    title={
        'text': "Interactive Dashboard: Transfer Portal Effects by Postseason Performance",
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_title="Transfer Minutes",
    yaxis_title="Segments",
    xaxis=dict(range=[0, 100]),
    width=900,
    height=700,
    showlegend=True,
    legend=dict(
        x=0.02,
        y=0.98,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1
    ),
    sliders=sliders,
    margin=dict(l=80, r=80, t=100, b=80)
)

# Save and display the dashboard
fig.write_html("transfer_portal_dashboard.html")
print("Dashboard saved as 'transfer_portal_dashboard.html'")
print("Open this file in your browser to view the interactive dashboard.")
