# LSTM Stock Price Prediction Dashboard

An interactive Streamlit dashboard for exploring individual test set predictions from the trained LSTM model.

## Features

- **Searchable Ticker Selection**: Browse all 80 tickers from the test set
- **Quarter Selection**: View all available prediction quarters for each ticker
- **Detailed Predictions**: See predicted vs actual prices with error metrics
- **Attention Visualization**: View which input quarters the model focused on
- **Sequence Explorer**: Examine the full 8-quarter sequence data
- **JSON Export**: Download predictions for further analysis

## Quick Start

Run the dashboard from the project root directory:

```bash
streamlit run inference/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Data Source

The dashboard uses **test data only** from `data_pipeline/data/sequences_8q_test.parquet` to ensure proper evaluation on unseen examples.

## Usage

1. **Select a Ticker**: Choose from the dropdown menu (e.g., AAPL, MSFT, GOOGL)
2. **Select a Quarter**: Pick which quarter's prediction you want to view
3. **View Results**:
   - Prediction metrics (predicted price, actual price, error)
   - Attention weights showing model focus
   - Full sequence data for Q1-Q7 (input) and Q8 (target)
4. **Download**: Export the prediction as JSON if needed

## Requirements

- Python 3.8+
- streamlit
- torch
- pandas
- numpy
- sklearn

All dependencies should already be installed if you've run the training pipeline.
