import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import sys
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


# Add the directory containing Smoothing+Regression.py to the path
sys.path.append('.')

# Load the data
def load_data():
    try:
        df = pd.read_csv('team_adjust_times_and_eos_similarity.csv')
        return df
    except FileNotFoundError:
        print("Could not find 'team_adjust_times_and_eos_similarity.csv'. Please ensure the file is in the same directory.")
        return None

def create_season_plot(team_name, year, df):
    # Filter the DataFrame for the selected team and year
    filtered_data = df[(df['team_id'] == team_name) & (df['season'] == year)]
    if len(filtered_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text=f"No data found for {team_name} in {year}",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    sample_row = filtered_data.iloc[0]
    # Parse string representations back to arrays
    import ast
    residuals_str = sample_row['residuals'].strip('[]')
    smoothed_str = sample_row['smoothed'].strip('[]')
    change_points_str = sample_row['change_points'].strip('[]')
    
    residuals = np.array([float(x) for x in residuals_str.split()])
    smoothed = np.array([float(x) for x in smoothed_str.split()])
    change_points = np.array([int(x) for x in change_points_str.split(',')])
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add residuals trace
    fig.add_trace(go.Scatter(
        x=list(range(len(residuals))),
        y=residuals,
        mode='lines',
        name='Residuals',
        opacity=0.5
    ))
    
    # Add smoothed residuals trace
    fig.add_trace(go.Scatter(
        x=list(range(len(smoothed))),
        y=smoothed,
        mode='lines',
        name='Smoothed Residuals',
        line=dict(width=3)
    ))
    
    # Add change points as vertical lines
    for i, cp in enumerate(change_points[:-1]):  # Exclude the last change point
        fig.add_vline(x=cp, line_dash="dash", line_color="red", opacity=0.7)
    
    # Add segment averages
    prev_cp = 0
    for i, cp in enumerate(change_points):
        segment = smoothed[prev_cp:cp]
        if len(segment) > 0:
            avg_val = np.mean(segment)
            fig.add_shape(
                type="line",
                x0=prev_cp, x1=cp, y0=avg_val, y1=avg_val,
                line=dict(color="green", width=2),
                opacity=0.7
            )
            fig.add_annotation(
                x=(prev_cp + cp) / 2, y=avg_val,
                text=f"{avg_val:.2f}",
                showarrow=False,
                bgcolor="white",
                bordercolor="green"
            )
        prev_cp = cp
    
    # Add important game markers
    fig.add_vline(x=filtered_data['last_regular_season_game'].values[0]+1, line_dash="dash", line_color="black")
    fig.add_vline(x=13, line_dash="dash", line_color="blue")
    
    # Update layout
    fig.update_layout(
        title=f"{team_name} - {year} Season Analysis",
        xaxis_title="Game Index",
        yaxis_title="Residual",
        xaxis=dict(range=[0, len(residuals)-1]),
        hovermode='x',
        height=600
    )
    
    # Add legend entries for the vertical lines
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Change Point',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='Conference Play',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Postseason Play',
        showlegend=True
    ))
    
    return fig

# Initialize the Dash app
app = dash.Dash(__name__)

# Load data
df = load_data()
if df is None:
    teams = []
    years = []
else:
    teams = sorted(df['team_id'].unique()) if 'team_id' in df.columns else []
    years = sorted(df['season'].unique()) if 'season' in df.columns else []

# Define the layout
app.layout = html.Div([
    html.H1("Team Season Similarity Dashboard", style={'textAlign': 'center'}),
    html.P("Interactive dashboard to visualize season similarity plots by team and year", 
           style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.Label("Select Team:"),
            dcc.Dropdown(
                id='team-dropdown',
                options=[{'label': team, 'value': team} for team in teams],
                value=teams[0] if teams else None,
                style={'marginBottom': '20px'}
            ),
            
            html.Label("Select Year:"),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': year, 'value': year} for year in years],
                value=years[0] if years else None,
                style={'marginBottom': '20px'}
            ),
            
            html.Button('Generate Season Plot', id='generate-button', n_clicks=0,
                       style={'backgroundColor': '#007CBA', 'color': 'white', 'padding': '10px'})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        
        html.Div([
            dcc.Graph(id='season-plot'),
            html.Div(id='data-preview')
        ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px'})
    ])
])

@callback([Output('season-plot', 'figure'),
           Output('data-preview', 'children')],
          [Input('generate-button', 'n_clicks')],
          [Input('team-dropdown', 'value'),
           Input('year-dropdown', 'value')])
def update_plot(n_clicks, selected_team, selected_year):
    if df is None or selected_team is None or selected_year is None:
        return go.Figure(), html.P("No data available")
    
    # Filter data based on selection
    filtered_data = df[(df['team_id'] == selected_team) & (df['season'] == selected_year)]

    if len(filtered_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text=f"No data found for {selected_team} in {selected_year}",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        data_preview = html.P(f"No data found for {selected_team} in {selected_year}")
        return fig, data_preview
    
    # Generate plot
    try:
        # You'll need to modify create_season_plot to return a Plotly figure
        # or create a new function that generates Plotly plots
        fig = create_season_plot(selected_team, selected_year, df)

        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error generating plot: {str(e)}",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Create data preview
    data_preview = html.Div([
        html.P(f"Showing data for {selected_team} - {selected_year}"),
        dcc.Graph(
            figure={
                'data': [go.Table(
                    header=dict(values=list(filtered_data.columns)),
                    cells=dict(values=[filtered_data[col] for col in filtered_data.columns])
                )],
                'layout': {'height': 400}
            }
        )
    ])
    
    return fig, data_preview

if __name__ == "__main__":
    app.run(debug=True)