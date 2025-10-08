import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from live_predictor import LivePredictor

# ----------------------------- Logging ---------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dashboard")

# ----------------------------- Model (from train_lstm.py) ---------------------------
class CausalAttention(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = hidden_size ** 0.5

    def forward(self, outputs: torch.Tensor, last_h: torch.Tensor):
        q = last_h.unsqueeze(1)
        k = outputs
        v = outputs
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / self.scale
        attn = torch.softmax(attn_logits, dim=-1)
        ctx = torch.bmm(attn, v).squeeze(1)
        return ctx, attn.squeeze(1)

class UniLSTMAttnDelta(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 192, num_layers: int = 2, dropout: float = 0.25, use_attn: bool = True, legacy_fc1: bool = False):
        super().__init__()
        self.use_attn = use_attn
        self.legacy_fc1 = legacy_fc1
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attn = CausalAttention(hidden_size) if use_attn else None
        rep_dim = hidden_size * (2 if use_attn else 1)
        self.ln = torch.nn.LayerNorm(rep_dim)
        # Legacy models concatenate y7_log after fc1, new models before
        fc1_input_dim = rep_dim if legacy_fc1 else rep_dim + 1
        self.fc1 = torch.nn.Linear(fc1_input_dim, 128)
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, y7_log: torch.Tensor):
        outputs, (h, _) = self.lstm(x)
        last_h = h[-1]
        if self.use_attn:
            ctx, attn_weights = self.attn(outputs, last_h)
            rep = torch.cat([last_h, ctx], dim=1)
        else:
            rep = last_h
            attn_weights = None
        rep = self.ln(rep)
        if not self.legacy_fc1:
            rep = torch.cat([rep, y7_log], dim=1)
        z = self.fc1(rep)
        z = self.act(z)
        z = self.drop(z)
        delta_log = self.fc2(z)
        y8_log_hat = y7_log + delta_log
        return y8_log_hat, delta_log, attn_weights

class UniLSTMAttnDirect(torch.nn.Module):
    """LSTM that predicts Q8 price DIRECTLY (no Q7 price input, no residual)"""
    def __init__(self, input_size: int, hidden_size: int = 192, num_layers: int = 2, dropout: float = 0.25, use_attn: bool = True):
        super().__init__()
        self.use_attn = use_attn
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attn = CausalAttention(hidden_size) if use_attn else None
        rep_dim = hidden_size * (2 if use_attn else 1)
        self.ln = torch.nn.LayerNorm(rep_dim)
        self.fc1 = torch.nn.Linear(rep_dim, 128)
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, y7_log: torch.Tensor = None):
        outputs, (h, _) = self.lstm(x)
        last_h = h[-1]
        if self.use_attn:
            ctx, attn_weights = self.attn(outputs, last_h)
            rep = torch.cat([last_h, ctx], dim=1)
        else:
            rep = last_h
            attn_weights = None
        rep = self.ln(rep)
        z = self.fc1(rep)
        z = self.act(z)
        z = self.drop(z)
        y8_log_hat = self.fc2(z)
        # For compatibility, return same format as Delta model
        delta_log = y8_log_hat - y7_log if y7_log is not None else torch.zeros_like(y8_log_hat)
        return y8_log_hat, delta_log, attn_weights

# ----------------------------- Dashboard ----------------------------------
@st.cache_resource
def load_model(model_path: str):
    """Load the model and return it along with feature columns and scaler."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(model_path)
    if not model_path.exists():
        st.error(f"Model checkpoint not found: {model_path}")
        return None, None, None, device

    logger.info(f"Loading model from {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Detect model type by checking fc1.weight shape
    fc1_weight_shape = ckpt["model_state_dict"]["fc1.weight"].shape
    use_attn = ckpt.get("use_attn", True)
    hidden_size = 192  # Default from checkpoint
    rep_dim = hidden_size * (2 if use_attn else 1)

    # Determine model type:
    # - UniLSTMAttnDirect: fc1 input = rep_dim (no y7_log)
    # - UniLSTMAttnDelta (legacy): fc1 input = rep_dim (y7_log added after fc1)
    # - UniLSTMAttnDelta (new): fc1 input = rep_dim + 1 (y7_log added before fc1)
    fc1_input_size = fc1_weight_shape[1]

    if fc1_input_size == rep_dim:
        # Could be Direct or legacy Delta - check if model predicts directly
        # Direct models don't use residual connection, so we use Direct model
        model = UniLSTMAttnDirect(
            input_size=len(ckpt["feature_cols"]),
            use_attn=use_attn
        )
    else:  # fc1_input_size == rep_dim + 1
        # New Delta model
        model = UniLSTMAttnDelta(
            input_size=len(ckpt["feature_cols"]),
            use_attn=use_attn,
            legacy_fc1=False
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load scaler
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]

    logger.info(f"Model loaded. Features: {len(ckpt['feature_cols'])}")
    return model, ckpt["feature_cols"], scaler, device

@st.cache_data
def load_data(data_dir: str):
    """Load and combine data from train, val, and test sets."""
    data_dir = Path(data_dir)

    datasets = []
    for split in ['train', 'val', 'test']:
        data_path = data_dir / f"sequences_8q_{split}.parquet"
        if data_path.exists():
            logger.info(f"Loading {split} data from {data_path}")
            df = pd.read_parquet(data_path)
            df['dataset'] = split
            datasets.append(df)
        else:
            logger.warning(f"Data file not found: {data_path}")

    if not datasets:
        st.error(f"No data files found in {data_dir}")
        return None

    combined_df = pd.concat(datasets, ignore_index=True)
    combined_df['fiscal_quarter_end'] = pd.to_datetime(combined_df['fiscal_quarter_end'])
    num_sequences = combined_df['sequence_id'].nunique()
    logger.info(f"Loaded {num_sequences} total sequences ({len(combined_df)} rows) from {len(datasets)} datasets")
    return combined_df

def _fetch_single_ticker_name(ticker):
    """Fetch a single ticker's company name."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Try multiple fields in order of preference
        name = (info.get('longName') or
                info.get('shortName') or
                info.get('name') or
                ticker)

        # If we only got the ticker back, it likely failed
        if name == ticker and not info:
            logger.warning(f"No info available for {ticker}")

        return ticker, name
    except Exception as e:
        logger.warning(f"Could not fetch name for {ticker}: {e}")
        return ticker, ticker

@st.cache_data
def get_company_names(tickers):
    """Fetch company names for given tickers using yfinance with parallel requests."""
    cache_file = Path("data_pipeline/data/ticker_names_cache.json")

    # Load existing cache if available
    ticker_to_name = {}
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                ticker_to_name = json.load(f)
            logger.info(f"Loaded {len(ticker_to_name)} cached ticker names")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")

    # Find tickers that need to be fetched
    tickers_to_fetch = [t for t in tickers if t not in ticker_to_name]

    if tickers_to_fetch:
        logger.info(f"Fetching names for {len(tickers_to_fetch)} tickers in parallel...")

        # Fetch in parallel with progress, reduced max_workers to be more conservative
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_fetch_single_ticker_name, ticker): ticker
                      for ticker in tickers_to_fetch}

            completed = 0
            for future in as_completed(futures):
                try:
                    ticker, name = future.result()
                    ticker_to_name[ticker] = name
                    completed += 1
                    if completed % 50 == 0:
                        logger.info(f"Fetched {completed}/{len(tickers_to_fetch)} ticker names...")
                except Exception as e:
                    logger.error(f"Failed to fetch ticker name: {e}")

        logger.info(f"Completed fetching {len(tickers_to_fetch)} ticker names")

        # Save updated cache
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(ticker_to_name, f, indent=2)
            logger.info(f"Saved cache with {len(ticker_to_name)} ticker names")
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")

    return ticker_to_name

def get_prediction(model, feature_cols, scaler, device, sequence_df):
    """Make a prediction for a given sequence."""
    if len(sequence_df) != 8:
        st.error(f"Sequence does not have 8 quarters. Found {len(sequence_df)}.")
        return None

    # Prepare the input data (first 7 quarters)
    input_df = sequence_df.iloc[:7].copy()

    # Scale features
    input_df[feature_cols] = scaler.transform(input_df[feature_cols])

    X = input_df[feature_cols].ffill().bfill().fillna(0.0).values.astype(np.float32)
    X_tensor = torch.from_numpy(X).unsqueeze(0).to(device)

    # Get y7_log (current_price of the 7th quarter)
    y7_raw = sequence_df.iloc[6]['current_price']

    if pd.isna(y7_raw):
        st.error("The required input price ('current_price') for the last quarter is missing (NaN).")
        return None

    y7_log = np.log1p(float(y7_raw))
    y7_log_tensor = torch.tensor([[y7_log]], dtype=torch.float32).to(device)

    # Run prediction
    with torch.no_grad():
        y8_log_hat, delta_log, attn_weights = model(X_tensor, y7_log_tensor)

    # Inverse transform the prediction
    predicted_log_price = y8_log_hat.item()
    predicted_price = np.expm1(predicted_log_price)

    # Get actual Q8 price for comparison (this is what we're trying to predict)
    # Q7's target_price_next_q should equal Q8's current_price
    actual_price_q8 = sequence_df.iloc[7]['current_price']

    q7_date = sequence_df.iloc[6]['fiscal_quarter_end'].strftime('%Y-%m-%d')
    q8_date = sequence_df.iloc[7]['fiscal_quarter_end'].strftime('%Y-%m-%d')

    result = {
        "ticker": sequence_df.iloc[0]['ticker'],
        "prediction_for_quarter_ending": q8_date,
        "input_sequence_ends_on": q7_date,
        "last_known_price": float(y7_raw),  # Q7 current_price
        "predicted_price": float(predicted_price),  # Model's prediction for Q8
        "actual_price": float(actual_price_q8) if pd.notna(actual_price_q8) else None,  # Q8 current_price
        "sequence_id": sequence_df.iloc[0]['sequence_id'],
        "delta_log": delta_log.item(),
        "attn_weights": attn_weights[0].cpu().numpy() if attn_weights is not None else None,
        "sequence_data": sequence_df
    }

    return result

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_live_prediction(_model, feature_cols, _scaler, _device, ticker: str):
    """Make a live prediction for current quarter using recent data from yfinance"""
    try:
        predictor = LivePredictor()

        # Make prediction
        result = predictor.predict_next_quarter(
            ticker=ticker,
            model=_model,
            feature_cols=feature_cols,
            scaler=_scaler,
            device=_device
        )

        return result

    except Exception as e:
        logger.error(f"Error in live prediction: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    st.set_page_config(page_title="LSTM Stock Price Prediction Dashboard", layout="wide")

    st.title("📈 LSTM Stock Price Prediction Dashboard")
    st.markdown("Explore individual test set predictions by ticker and quarter")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Model selector
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["With Price", "No Price"],
        help="With Price: Uses price features. No Price: Excludes price features."
    )

    model_path = "models/best_model.pt" if model_choice == "With Price" else "models_no_price/best_model_no_price.pt"
    st.sidebar.text(f"📁 {model_path}")

    data_dir = "data_pipeline/data"

    # Add prediction mode selector
    prediction_mode = st.sidebar.radio(
        "Prediction Mode",
        ["Historical (Test Data)", "Live (Current Quarter)"],
        help="Historical: Use test set data. Live: Predict current quarter for any S&P 500 stock"
    )

    st.sidebar.info(f"📊 Loading data from train, val, and test sets")

    # Load model and data
    model, feature_cols, scaler, device = load_model(model_path)
    df = load_data(data_dir)

    if model is None or df is None or feature_cols is None or scaler is None:
        st.error("Failed to load model or data. Please check the paths.")
        return

    st.sidebar.success(f"✓ Model loaded ({len(feature_cols)} features)")

    # Show dataset breakdown
    dataset_counts = df.groupby('dataset')['sequence_id'].nunique()
    st.sidebar.success(f"✓ Data loaded ({df['sequence_id'].nunique()} total sequences)")
    for dataset, count in dataset_counts.items():
        st.sidebar.text(f"  • {dataset}: {count} sequences")

    # Main content based on mode
    if prediction_mode == "Live (Current Quarter)":
        st.header("🔴 Live Prediction - Current Quarter")
        st.markdown("**Predict stock prices for the upcoming quarter using the most recent 7 quarters of data**")

        # Get company names for all unique tickers
        unique_tickers = sorted(df['ticker'].unique())
        ticker_to_name = get_company_names(unique_tickers)

        # Create ticker options with company names
        ticker_options_live = {f"{t} - {ticker_to_name[t]}": t for t in unique_tickers}
        ticker_display_names_live = sorted(ticker_options_live.keys())

        selected_display_live = st.selectbox(
            "Select Company for Live Prediction",
            ticker_display_names_live,
            index=0
        )
        selected_ticker_live = ticker_options_live[selected_display_live]

        # Get next quarter info
        next_quarter = pd.Period.now(freq='Q') + 1
        next_quarter_str = next_quarter.strftime('%Y Q%q')

        st.info(f"📅 Target Quarter: **{next_quarter_str}**")
        st.info(f"📊 Input Data: Most recent 7 quarters")

        if st.button("🚀 Generate Live Prediction", type="primary"):
            with st.spinner(f"Fetching live data for {selected_ticker_live}..."):
                result = get_live_prediction(model, feature_cols, scaler, device, selected_ticker_live)

            if result.get('success'):
                st.success(f"✅ Successfully generated prediction for {result['ticker']}")

                # Display prediction results
                st.header("Live Prediction Results")

                # Key metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)

                with metric_col1:
                    st.metric("Current Price", f"${result['current_price']:.2f}")

                with metric_col2:
                    st.metric(
                        "Predicted Price",
                        f"${result['predicted_price']:.2f}",
                        f"{result['price_change_pct']:+.2f}%"
                    )

                with metric_col3:
                    st.metric(
                        "Expected Change",
                        f"${result['price_change']:+.2f}",
                        f"{result['price_change_pct']:+.2f}%"
                    )

                # Details
                st.subheader("Prediction Details")
                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:
                    st.write(f"**Ticker:** {result['ticker']}")
                    st.write(f"**Input Sequence Ends:** {result['input_sequence_ends']}")
                    st.write(f"**Prediction For:** {result['prediction_for_quarter']}")

                with detail_col2:
                    st.write(f"**Delta (log-space):** {result['delta_log']:.6f}")
                    st.write(f"**Features Used:** {result['features_available']}/{result['features_total']}")

                    if result['features_available'] < result['features_total']:
                        st.warning(f"⚠️ Only {result['features_available']}/{result['features_total']} features available. Missing features filled with zeros.")

                # Attention weights
                if result['attn_weights'] is not None:
                    st.subheader("Attention Weights Across Input Sequence")
                    st.markdown("*Shows which quarters the model focused on when making the prediction*")

                    attn_df = pd.DataFrame({
                        'Quarter': [f"Q{i+1}" for i in range(7)],
                        'Date': result['quarters_used'],
                        'Attention Weight': result['attn_weights']
                    })
                    st.bar_chart(attn_df.set_index('Quarter')['Attention Weight'])

                    with st.expander("View Quarter Dates"):
                        st.dataframe(attn_df[['Quarter', 'Date']], use_container_width=True)

                # Disclaimer
                st.divider()
                st.warning("""
                ⚠️ **Disclaimer**: This is a predictive model and not financial advice.
                - Live predictions use data from yfinance which may be incomplete
                - Model was trained on historical S&P 500 data through 2024
                - Actual results may vary significantly from predictions
                - Do not use this for actual trading decisions
                """)

            else:
                st.error(f"❌ Failed to generate prediction")
                st.error(f"Error: {result.get('error', 'Unknown error')}")

        return

    # Historical mode (original functionality)
    st.header("Select a Test Example")

    # Get unique ticker-dataset combinations
    ticker_dataset_df = df.groupby(['ticker', 'dataset']).size().reset_index()[['ticker', 'dataset']]

    # Get company names for all unique tickers
    unique_tickers = sorted(df['ticker'].unique())
    ticker_to_name = get_company_names(unique_tickers)

    # Create ticker options with company names and dataset labels
    ticker_options = {}
    for _, row in ticker_dataset_df.iterrows():
        ticker = row['ticker']
        dataset = row['dataset']
        company_name = ticker_to_name[ticker]
        display_name = f"{ticker} - {company_name} [{dataset}]"
        ticker_options[display_name] = (ticker, dataset)

    ticker_display_names = sorted(ticker_options.keys())

    # Create two columns for filters
    col1, col2 = st.columns(2)

    with col1:
        # Searchable selectbox with company names and dataset labels
        selected_display = st.selectbox(
            "Select Ticker",
            ticker_display_names,
            index=0
        )
        selected_ticker, selected_dataset = ticker_options[selected_display]

    # Filter data by ticker and dataset
    ticker_df = df[(df['ticker'] == selected_ticker) & (df['dataset'] == selected_dataset)]

    # Get Q8 quarters for this ticker (prediction targets)
    q8_quarters = ticker_df[ticker_df['quarter_in_sequence'] == 7].sort_values('fiscal_quarter_end')
    quarter_options = q8_quarters['fiscal_quarter_end'].dt.strftime('%Y-%m-%d').tolist()

    with col2:
        if quarter_options:
            selected_quarter = st.selectbox(
                f"Select Target Quarter (Q8) - {len(quarter_options)} available",
                quarter_options,
                index=len(quarter_options)-1  # Default to most recent
            )
        else:
            st.warning(f"No complete sequences found for {selected_ticker}")
            return

    # Get the sequence for the selected quarter
    selected_q8 = q8_quarters[q8_quarters['fiscal_quarter_end'] == pd.to_datetime(selected_quarter)].iloc[0]
    sequence_id = selected_q8['sequence_id']
    sequence_df = df[df['sequence_id'] == sequence_id].sort_values('quarter_in_sequence')

    # Make prediction
    result = get_prediction(model, feature_cols, scaler, device, sequence_df)

    if result is None:
        return

    # Display results
    st.header("Prediction Results")

    # Key metrics in columns
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Last Known Price", f"${result['last_known_price']:.2f}")

    with metric_col2:
        change = result['predicted_price'] - result['last_known_price']
        percent_change = (change / result['last_known_price']) * 100 if result['last_known_price'] != 0 else 0
        st.metric("Predicted Price", f"${result['predicted_price']:.2f}",
                 f"{percent_change:+.2f}%")

    with metric_col3:
        if result['actual_price'] is not None:
            actual_change = result['actual_price'] - result['last_known_price']
            actual_percent = (actual_change / result['last_known_price']) * 100 if result['last_known_price'] != 0 else 0
            st.metric("Actual Price", f"${result['actual_price']:.2f}",
                     f"{actual_percent:+.2f}%")
        else:
            st.metric("Actual Price", "N/A")

    with metric_col4:
        if result['actual_price'] is not None:
            error = abs(result['predicted_price'] - result['actual_price'])
            error_pct = (error / result['actual_price']) * 100 if result['actual_price'] != 0 else 0
            st.metric("Prediction Error", f"${error:.2f}", f"{error_pct:.2f}%")
        else:
            st.metric("Prediction Error", "N/A")

    # Sequence information
    st.subheader("Sequence Details")
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write(f"**Ticker:** {result['ticker']}")
        st.write(f"**Sequence ID:** {result['sequence_id']}")
        st.write(f"**Input Sequence Ends:** {result['input_sequence_ends_on']}")

    with info_col2:
        st.write(f"**Prediction Target Quarter:** {result['prediction_for_quarter_ending']}")
        st.write(f"**Delta (log-space):** {result['delta_log']:.6f}")
        if result['actual_price'] is not None:
            direction_correct = (change > 0) == (actual_change > 0)
            st.write(f"**Direction Correct:** {'✓ Yes' if direction_correct else '✗ No'}")

    # Attention weights visualization (if available)
    if result['attn_weights'] is not None:
        st.subheader("Attention Weights Across Input Sequence")
        st.markdown("*Shows which quarters the model focused on when making the prediction*")

        attn_df = pd.DataFrame({
            'Quarter': [f"Q{i+1}" for i in range(7)],
            'Attention Weight': result['attn_weights']
        })
        st.bar_chart(attn_df.set_index('Quarter'))

if __name__ == "__main__":
    main()
