import pandas as pd
from sqlalchemy import create_engine
from db_connect import get_connection
import dash
from dash.dependencies import Input, Output
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score


### This portion of the project is using the ind_player_dev_predict_2024 ###

table_name = 'ind_player_dev_predict_2024'

#Load data from PostgreSQL
engine = get_connection()
df = pd.read_sql(f"SELECT * FROM {table_name};", con = engine)

#Initialize Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Transfer Portal Player Development Impact Dashboard"),

    dcc.Input(
        id='player-input',
        type='text',
        placeholder='Search player...',
        style={'marginBottom': '15px', 'width': '40%'}
    ),

    dcc.Dropdown(
        id='team-dropdown',
        options=[{'label': team, 'value': team} for team in sorted(df['team'].dropna().unique())],
        placeholder='Filter by team',
        style={'marginBottom': '15px', 'width': '40%'}
    ),

    dcc.Dropdown(
        id='transfer-filter',
        options=[
            {'label': 'Transferred', 'value': True},
            {'label': 'Did Not Transfer', 'value': False}
        ],
        placeholder='Filter by transfer status',
        style={'marginBottom': '20px', 'width': '40%'}
    ),

    dcc.Graph(id='model-chart1'),
    dcc.Graph(id='model-chart2'),
    dcc.Graph(id='team-impact-chart')
])

def model_results_plotly(df):
    mse = mean_squared_error(df["Actual Composite Score (2024)"], df["Predicted"])
    r2 = r2_score(df["Actual Composite Score (2024)"], df["Predicted"])

    # Scatter
    fig1 = px.scatter(
        df,
        x="Predicted",
        y="Actual Composite Score (2024)",
        title=f"Actual vs. Predicted Composite Score (2024) | R²={r2:.3f}",
        labels={"Predicted": "Predicted Score", "Actual Composite Score (2024)": "Actual Score"},
        hover_name="player"
    )
    fig1.add_trace(go.Scatter(
        x=[df["Predicted"].min(), df["Predicted"].max()],
        y=[df["Predicted"].min(), df["Predicted"].max()],
        mode='lines',
        line=dict(dash='dash', color='red'),
        showlegend=False
    ))

    # Histogram
    fig2 = px.histogram(
        df,
        x="Estimated Difference",
        nbins=20,
        title="Estimated Difference on Composite Score",
        opacity=0.75,
        marginal="rug"
    )
    fig2.add_vline(x=0, line_dash='dash', line_color='red')

    return fig1, fig2

@app.callback(
    [Output('model-chart1', 'figure'),
     Output('model-chart2', 'figure'),
     Output('team-impact-chart', 'figure')],
    [Input('player-input', 'value'),
     Input('team-dropdown', 'value'),
     Input('transfer-filter', 'value')]
)

def update_charts(player_name, selected_team, transfer_status):
    filtered = df.copy()

    # Apply filters
    if player_name:
        filtered = filtered[filtered['player'].str.lower().str.contains(player_name.lower())]

    if selected_team:
        filtered = filtered[filtered['team'] == selected_team]

    if transfer_status is not None:
        filtered = filtered[filtered['transfer'] == transfer_status]

    if filtered.empty:
        return (
            px.scatter(title= "No data matches the filters"),
            px.histogram(title= "No data matches the filters"),
            px.bar(title= "No data for team impact")
        )

    model_fig1, model_fig2 = model_results_plotly(filtered)

    team_sum = (
        filtered.groupby('team')['Estimated Difference']
        .sum()
        .sort_values(ascending = False)
        .head(5)
    )

    team_fig = px.bar(
        x=team_sum.index,
        y=team_sum.values,
        labels={'x': 'Team', 'y': 'Net Estimated Difference'},
        title='Top Teams by Net Estimated Impact'
    )

    return model_fig1, model_fig2, team_fig

#Run the Dash App
if __name__ == '__main__':
    app.run(debug = True)