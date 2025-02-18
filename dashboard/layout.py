import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

def create_layout(app):
    """Create Dash layout integrated with Flask"""
    app.title = "Fraud Detection Dashboard"
    
    return html.Div([
        html.H1("Fraud Detection Analytics", style={'textAlign': 'center'}),
        
        # Summary Cards Row
        html.Div([
            html.Div(id='total-transactions-card', className='card'),
            html.Div(id='fraud-cases-card', className='card'),
            html.Div(id='fraud-percentage-card', className='card')
        ], className='row'),
        
        # Main Charts Row
        html.Div([
            dcc.Graph(id='fraud-trend-chart', className='six columns'),
            dcc.Graph(id='geo-map', className='six columns')
        ], className='row'),
        
        # Device/Browser Row
        html.Div([
            dcc.Graph(id='device-fraud-chart', className='six columns'),
            dcc.Graph(id='browser-fraud-chart', className='six columns')
        ], className='row'),
        
        # Interval component for live updates
        dcc.Interval(
            id='interval-component',
            interval=60*1000,  # 1 minute
            n_intervals=0
        )
    ])