import pandas as pd
from sqlalchemy import create_engine
from db_connect import get_connection
import dash
from dash import html, dcc
import plotly.express as px
from dash.dependencies import Input, Output

### This portion of the project is using the ind_player_dev_predict_2024 ###

table_name = 'ind_player_dev_predict_2024'

#Load data from PostgreSQL
engine = get_connection()
df = pd.read_sql(f"SELECT * FROM {table_name};", con = engine)

#Initialize Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Input(
        id='player-input', type='text', placeholder='Enter player name...',
        style={'marginBottom': '20px'}
    ),
    dcc.Graph(id='chart1'),
    dcc.Graph(id='chart2'),
    dcc.Graph(id='chart3')
])

@app.callback(
    [Output('chart1', 'figure'),
     Output('chart2', 'figure'),
     Output('chart3', 'figure')],
    [Input('player-input', 'value')]
)

# Create Dashboard Plots
def update_charts(player_name):
    if player_name:
        filtered = df[df['player'].str.lower().str.contains(player_name.lower())]
    else:
        filtered = df

    # 1. Basic Scatterplot
    fig1 = px.scatter(
        filtered,
        x='Predicted If Stayed',
        y='Actual Offensive Rating (2024)',
        hover_name='player',
        title='Predicted vs. Actual Ratings',
        labels={'Predicted If Stayed': 'Predicted (No Transfer)', 'Actual Offensive Rating (2024)': 'Actual Rating'}
    )

    # 2. KDE-style Residual Histogram
    fig2 = px.histogram(
        filtered,
        x='Residual',
        nbins=20,
        title='Distribution of Prediction Residuals',
        marginal="rug",
        opacity=0.75
    )
    fig2.add_vline(x=0, line_dash='dash', line_color='red')

    # 3. Bar Chart: Top 10 largest estimated transfer impacts
    if not filtered.empty:
        top_10 = filtered.copy()
        top_10['abs_effect'] = top_10['Estimated Transfer Effect'].abs()
        top_10 = top_10.sort_values('abs_effect', ascending=False).head(10)

        fig3 = px.bar(
            top_10,
            x='player',
            y='Estimated Transfer Effect',
            color='Estimated Transfer Effect',
            title='Top 10 Most Impactful Transfer Effects',
            labels={'Estimated Transfer Effect': 'Effect Size'},
            text='Estimated Transfer Effect'
        )
        fig3.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig3.update_layout(xaxis_tickangle=-45)
    else:
        fig3 = px.bar(title="No data available for selected player.")

    return fig1, fig2, fig3

#Run the Dash App
if __name__ == '__main__':
    app.run(debug = True)