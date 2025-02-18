from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import requests

def register_callbacks(app):
    """Register Dash callbacks"""
    
    @app.callback(
        [Output('total-transactions-card', 'children'),
         Output('fraud-cases-card', 'children'),
         Output('fraud-percentage-card', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_summary(_):
        # Get data from Flask API
        response = requests.get('http://localhost:5000/api/summary')
        data = response.json()
        
        cards = [
            html.Div([
                html.H3(f"{data['total_transactions']:,}"),
                html.P("Total Transactions")
            ], className='card-content'),
            
            html.Div([
                html.H3(f"{data['total_fraud']:,}", style={'color': '#FF4B4B'}),
                html.P("Fraud Cases")
            ], className='card-content'),
            
            html.Div([
                html.H3(f"{data['fraud_percentage']:.2f}%"),
                html.P("Fraud Percentage")
            ], className='card-content')
        ]
        
        return cards

    @app.callback(
        Output('fraud-trend-chart', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_trend(_):
        response = requests.get('http://localhost:5000/api/fraud_trend')
        df = pd.DataFrame(response.json())
        fig = px.line(df, x='date', y='fraud_cases', 
                     title="Fraud Cases Over Time")
        return fig

    @app.callback(
        Output('geo-map', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_geo(_):
        response = requests.get('http://localhost:5000/api/geo_fraud')
        df = pd.DataFrame(response.json())
        fig = px.choropleth(df, locations='country', locationmode='country names',
                           color='fraud_cases', title="Fraud by Country")
        return fig

    @app.callback(
        [Output('device-fraud-chart', 'figure'),
         Output('browser-fraud-chart', 'figure')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_device_browser(_):
        # Device Fraud
        device_response = requests.get('http://localhost:5000/api/device_fraud')
        device_df = pd.DataFrame(device_response.json())
        device_fig = px.bar(device_df, x='device', y='fraud_cases', 
                           title="Fraud by Device")
        
        # Browser Fraud
        browser_response = requests.get('http://localhost:5000/api/browser_fraud')
        browser_df = pd.DataFrame(browser_response.json())
        browser_fig = px.bar(browser_df, x='browser', y='fraud_cases',
                            title="Fraud by Browser")
        
        return device_fig, browser_fig