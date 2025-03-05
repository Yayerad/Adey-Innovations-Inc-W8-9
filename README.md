<div align="center">
  <h1>📈 GMF Investments Portfolio Optimization</h1>
  <p>Time Series Forecasting & Portfolio Management System</p>
</div>


## 📂 Repository Structure
<details>
<summary>Click to expand folder structure</summary>

```bash
.
├── .venv/                   # Python virtual environment
├── notebooks/               # Jupyter notebooks and analysis
│   ├── forecasts/           # Forecast results
│   │   ├── BND_forecast.csv
│   │   ├── SPY_forecast.csv
│   │   ├── TSLA_forecast.csv
│   │   └── combined_forecasts.png
│   ├── models/              # Trained model binaries
│   ├── portfolio_metrics/   # Optimization results
│   │   └── optimization_report.txt
│   ├── future_market_trends.ipynb
│   ├── portfolio_optimization.ipynb
│   ├── preprocessing_and_eda.ipynb
│   └── time_series_forecasting.ipynb
├── scripts/                 # Utility scripts
├── src/                     # Source code
│   └── data/                # Historical price data
│       ├── BND_historical_data.csv
│       ├── SPY_historical_data.csv
│       └── TSLA_historical_data.csv
├── tests/                   # Unit tests
├── .gitignore
├── README.md
└── requirements.txt
</details>
<details>
🚀 Getting Started
Prerequisites
Python 3.8+
Jupyter Notebook
Installation
bash
Copy
Edit
git clone https://github.com/yourusername/gmf-investments.git
cd gmf-investments
pip install -r requirements.txt
</details>
📊 Key Features
📈 Time Series Forecasting
SARIMA models for TSLA, BND, SPY
6-month market trend predictions
95% confidence intervals
📊 Portfolio Optimization
Sharpe ratio maximization
Efficient frontier analysis
Value at Risk (VaR) calculation
🧠 Analysis Workflow
preprocessing_and_eda.ipynb - Data cleaning and exploration
time_series_forecasting.ipynb - Model training and validation
future_market_trends.ipynb - 6-month forecasts generation
portfolio_optimization.ipynb - Optimal asset allocation
📄 License
MIT License - See LICENSE for details
