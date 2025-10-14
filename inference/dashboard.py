import argparse
import json
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
import requests
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
    """Fetch a single ticker's company name from FMP API."""
    try:
        api_key = os.environ.get('FMP_API_KEY', 'FlVMN0WC4KpV1f9OWTNVamh39gW8wx7L')
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data and len(data) > 0:
            name = data[0].get('companyName', ticker)
            logger.debug(f"Fetched name for {ticker}: {name}")
            return ticker, name
        else:
            logger.warning(f"No company data returned for {ticker}")
            return ticker, ticker

    except Exception as e:
        logger.warning(f"Could not fetch name for {ticker}: {e}")
        return ticker, ticker

@st.cache_data
def get_company_names(tickers):
    """Fetch company names for given tickers using FMP API with parallel requests."""
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
                ticker = futures[future]  # Get the ticker even if future fails
                try:
                    ticker, name = future.result()
                    ticker_to_name[ticker] = name
                    completed += 1
                    if completed % 50 == 0:
                        logger.info(f"Fetched {completed}/{len(tickers_to_fetch)} ticker names...")
                except Exception as e:
                    logger.error(f"Failed to fetch ticker name for {ticker}: {e}")
                    # Fallback: use ticker as name if fetch fails
                    ticker_to_name[ticker] = ticker

        logger.info(f"Completed fetching {len(tickers_to_fetch)} ticker names")

        # Ensure ALL requested tickers are in the mapping (final safety check)
        for ticker in tickers:
            if ticker not in ticker_to_name:
                ticker_to_name[ticker] = ticker

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

    # Add explanation about company-specific period dates
    with st.expander("ℹ️ About Information Availability Dates (Period Start/End)", expanded=False):
        st.markdown("""
        ### How Period Dates Work - Eliminating Look-Ahead Bias

        This model uses **actual information availability dates** to ensure realistic, bias-free predictions:

        #### Key Dates for Each Quarter
        - 📊 **Fiscal Quarter End**: The end of the company's fiscal quarter (e.g., Mar 31, Jun 30, Sep 30, Dec 31)
        - 📞 **Earnings Call Transcript Release**: When the transcript actually becomes publicly available (15-90 days after quarter end)
        - 📅 **Period Start Date**: The actual date when quarter Q's transcript was filed/released - this is when we enter trading positions
        - 📅 **Period End Date**: The actual date when quarter Q+1's transcript/earnings were released - this is when we exit trading positions
        - 📈 **Holding Period**: Actual number of days between period start and period end (varies by company)

        #### Why This Matters - No Look-Ahead Bias
        **OLD (INCORRECT) Approach:** Assumed information available on fixed dates (e.g., 45 days after quarter end)
        - Problem: This creates look-ahead bias by using information before it's actually available
        - Problem: Assumes all companies report on the same schedule

        **NEW (CORRECT) Approach:** Uses actual transcript filing dates from historical data
        - ✅ Each quarter has its own actual information release date
        - ✅ Trading periods based on when information actually became public
        - ✅ Each company has company-specific reporting delays
        - ✅ Holding periods vary realistically (50-150 days depending on company reporting speed)

        #### Company-Specific Timing Statistics
        - Average holding period: ~80-90 days (varies by company)
        - Fast reporters: ~50-60 days between transcripts
        - Slow reporters: ~120-150 days between transcripts
        - Each company has unique, realistic trading windows

        _Example: Company XYZ's Q1 ends Mar 31. Their Q1 transcript was released May 8 (period_start_date).
        Their Q2 transcript was released Aug 12 (period_end_date). The model predicts prices for the
        holding period from May 8 to Aug 12 (96 days actual)._
        """)

    st.divider()

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Model selector
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["Final (All Data)", "Best (Validation)", "No Price"],
        help="Final: Trained on all data after validation. Best: Best validation model. No Price: Excludes price features."
    )

    if model_choice == "Final (All Data)":
        model_path = "models/final_model_all_data.pt"
    elif model_choice == "Best (Validation)":
        model_path = "models/best_model.pt"
    else:
        model_path = "models_no_price/best_model_no_price.pt"

    st.sidebar.text(f"📁 {model_path}")

    data_dir = "data_pipeline/data"

    # Add prediction mode selector
    prediction_mode = st.sidebar.radio(
        "Prediction Mode",
        ["Historical (Test Data)", "Live (Q3 2025)"],
        help="Historical: Use test set data. Live: Predict Q3 2025 transcript release for S&P 500 stocks"
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
    if prediction_mode == "Live (Q3 2025)":
        st.header("🔴 Live Earnings Predictions")

        # Simple, clear explanation
        st.markdown("""
        ### What This Does
        Predicts stock prices at the **next earnings transcript release** using **live data** fetched in real-time.

        - **Live Financial Data**: Fetches the most recent 7 quarters from FMP API (income statements, balance sheets, cash flow)
        - **Live Transcripts**: Downloads earnings call transcripts for all 7 quarters (~325KB of text)
        - **AI Embeddings**: Uses E5-Mistral-7B (state-of-the-art financial embeddings) with 4-bit quantization
        - **62 Features**: Calculates 30 financial metrics + 32 PCA-reduced embedding dimensions
        - **Processing Time**: ~30-40 seconds per prediction (model inference on transcripts)
        - **Prediction Target**: Stock price when next earnings transcript is released
        """)

        # Get eligible tickers using LIVE data with transcript embeddings
        predictor = LivePredictor(use_cached_data=False, use_transcript_embeddings=True)
        eligible_tickers = predictor.get_eligible_tickers()

        st.success(f"✅ **{len(eligible_tickers)} S&P 500 companies** available with complete data")

        # Get company names for eligible tickers
        ticker_to_name = get_company_names(eligible_tickers)

        # Create ticker options with company names
        ticker_options_live = {f"{t} - {ticker_to_name.get(t, t)}": t for t in eligible_tickers}
        ticker_display_names_live = sorted(ticker_options_live.keys())

        selected_display_live = st.selectbox(
            "Select Company",
            ticker_display_names_live,
            index=0
        )
        selected_ticker_live = ticker_options_live[selected_display_live]

        if st.button("🚀 Generate Live Prediction", type="primary"):
            # Show progress steps with detailed messaging
            progress_bar = st.progress(0, text="Initializing prediction pipeline...")
            status_container = st.container()

            with status_container:
                st.markdown("### 🔄 Processing Steps")
                step1 = st.empty()
                step2 = st.empty()
                step3 = st.empty()
                step4 = st.empty()

            try:
                # Step 1: Financial Data
                step1.info("📊 **Step 1/4**: Fetching quarterly financial data from FMP API...")
                step1.markdown("📊 **Step 1/4**: Fetching income statements, balance sheets, and cash flow data for the last 7 quarters...")
                progress_bar.progress(10, text="Fetching financial data...")

                step1.success("✅ **Step 1/4**: Financial data retrieved successfully")

                # Step 2: Feature Engineering
                step2.info("📈 **Step 2/4**: Calculating financial features...")
                step2.markdown("📈 **Step 2/4**: Computing 30 financial metrics (profitability ratios, growth rates, leverage ratios, efficiency metrics)...")
                progress_bar.progress(30, text="Engineering financial features...")

                step2.success("✅ **Step 2/4**: Financial features calculated")

                # Step 3: Transcripts and Embeddings (longest step)
                step3.info("📞 **Step 3/4**: Processing earnings call transcripts...")
                step3.markdown("""
                📞 **Step 3/4**: Downloading and processing earnings transcripts...
                - Fetching ~325KB of transcript text for 7 quarters
                - Loading E5-Mistral-7B model (4-bit quantization)
                - Generating semantic embeddings from transcript text
                - Reducing embeddings to 32 PCA dimensions

                ⏱️ *This step takes 25-35 seconds (AI model inference)*
                """)
                progress_bar.progress(40, text="Processing transcripts with AI model (this may take 30-40 seconds)...")

                # Make the actual prediction
                result = get_live_prediction(model, feature_cols, scaler, device, selected_ticker_live)

                step3.success("✅ **Step 3/4**: Transcript embeddings generated")
                progress_bar.progress(85, text="Finalizing prediction...")

                # Step 4: Model Prediction
                step4.info("🤖 **Step 4/4**: Running LSTM prediction model...")
                step4.markdown("🤖 **Step 4/4**: Feeding 62 features through trained LSTM network with attention mechanism...")
                progress_bar.progress(95, text="Running final prediction...")

                step4.success("✅ **Step 4/4**: Prediction complete!")
                progress_bar.progress(100, text="✅ All steps completed successfully!")

            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ **Prediction Failed**: {error_msg}")
                if "transcript" in error_msg.lower():
                    st.warning("💡 **Tip**: This error often occurs when transcript data is unavailable. Try selecting a different company.")
                elif "api" in error_msg.lower() or "rate" in error_msg.lower():
                    st.warning("💡 **Tip**: API rate limit may have been reached. Please wait a moment and try again.")
                result = {'success': False, 'error': error_msg}

            if result.get('success'):
                st.success(f"✅ Prediction generated for **{result['ticker']}**")

                st.divider()

                # Parse dates for display
                input_end_date = pd.to_datetime(result['input_sequence_ends'])
                prediction_date = pd.to_datetime(result['prediction_for_quarter'])
                days_ahead = (prediction_date - input_end_date).days

                # Determine which quarter this is
                latest_quarter = pd.to_datetime(result['quarters_used'][-1])
                pred_quarter = prediction_date

                # Simple quarter labels
                def format_quarter(date):
                    q = (date.month - 1) // 3 + 1
                    return f"Q{q} {date.year}"

                latest_q_label = format_quarter(latest_quarter)
                pred_q_label = format_quarter(pred_quarter)

                # Calculate days until prediction target
                today = pd.Timestamp.now()
                days_until_target = (prediction_date - today).days

                st.markdown(f"""
                ### 📅 Timeline
                - **Latest Data Available**: {input_end_date.strftime('%B %d, %Y')} ({latest_q_label} transcript released)
                - **Prediction Target**: {prediction_date.strftime('%B %d, %Y')} ({pred_q_label} transcript expected)
                - **Days Ahead**: {days_ahead} days (time between data cutoff and prediction target)
                - **Days Until Prediction Target**: {days_until_target} days (time from today to prediction target)
                """)

                # Simple price display
                st.markdown("### 💰 Price Prediction")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Quarter Start Price",
                        f"${result['quarter_start_price']:.2f}",
                        help=f"Price on {input_end_date.strftime('%b %d, %Y')} when transcript was released"
                    )

                with col2:
                    st.metric(
                        "Current Price (Today)",
                        f"${result['current_price']:.2f}",
                        help="Live market price as of today"
                    )

                with col3:
                    st.metric(
                        "Predicted Price",
                        f"${result['predicted_price']:.2f}",
                        f"{result['price_change_pct']:+.2f}%",
                        help=f"Model prediction for {prediction_date.strftime('%b %d, %Y')}"
                    )

                with col4:
                    st.metric(
                        "Expected Change",
                        f"${result['price_change']:+.2f}",
                        f"{result['price_change_pct']:+.2f}%",
                        help="Change from current price to predicted price"
                    )

                # Show if this is past prediction (for backtesting comparison)
                if 'actual_price' in result:
                    st.divider()
                    st.warning("⚠️ Note: This prediction target date has already passed. Showing comparison with actual results.")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Actual Price",
                            f"${result['actual_price']:.2f}",
                            help=f"Actual price on {result['actual_price_date']}"
                        )

                    with col2:
                        direction_emoji = "✅" if result['direction_correct'] else "❌"
                        st.metric(
                            "Direction Correct",
                            direction_emoji,
                            help="Did the model predict the right direction?"
                        )

                    with col3:
                        st.metric(
                            "Prediction Error",
                            f"${result['prediction_error']:.2f}",
                            f"{result['prediction_error_pct']:.2f}%",
                            help="How far off was the prediction?"
                        )

                # Additional details in expandable section
                with st.expander("📊 View Model Details"):
                    st.markdown(f"""
                    - **Quarters Used**: {len(result['quarters_used'])} quarters
                    - **Quarter Range**: {result['quarters_used'][0]} to {result['quarters_used'][-1]}
                    - **Features Available**: {result['features_available']}/{result['features_total']}
                    - **Model Delta (log-space)**: {result['delta_log']:.6f}
                    """)

                    if result['attn_weights'] is not None:
                        st.markdown("**Attention Weights** (which quarters the model focused on):")
                        attn_df = pd.DataFrame({
                            'Quarter': [f"Q{i+1}" for i in range(7)],
                            'Date': result['quarters_used'],
                            'Weight': result['attn_weights']
                        })
                        st.bar_chart(attn_df.set_index('Quarter')['Weight'])

                # Disclaimer
                st.divider()
                st.info(f"""
                **ℹ️ About This Prediction**

                This model predicts stock prices at the next earnings transcript release ({pred_q_label}).
                It uses 7 quarters of historical data and was trained on S&P 500 data through Q2 2024.

                **Data availability varies by company** - this company's data goes through {latest_q_label}.

                **Not financial advice** - for educational purposes only.
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
        company_name = ticker_to_name.get(ticker, ticker)
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
    # Use actual quarter end dates as they appear in company data (not standardized)
    q8_quarters = ticker_df[ticker_df['quarter_in_sequence'] == 7].sort_values('fiscal_quarter_end')
    quarter_options = q8_quarters['fiscal_quarter_end'].tolist()

    with col2:
        if quarter_options:
            selected_quarter = st.selectbox(
                f"Select Quarter End Date (Q8) - {len(quarter_options)} available",
                quarter_options,
                format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'),
                index=len(quarter_options)-1,  # Default to most recent
                help="Company-specific fiscal quarter end date"
            )
        else:
            st.warning(f"No complete sequences found for {selected_ticker}")
            return

    # Get the sequence for the selected quarter
    # Ensure selected_quarter is a datetime for comparison
    selected_quarter_dt = pd.to_datetime(selected_quarter) if not isinstance(selected_quarter, pd.Timestamp) else selected_quarter
    selected_q8 = q8_quarters[q8_quarters['fiscal_quarter_end'] == selected_quarter_dt].iloc[0]
    sequence_id = selected_q8['sequence_id']
    sequence_df = df[df['sequence_id'] == sequence_id].sort_values('quarter_in_sequence')

    # Make prediction
    result = get_prediction(model, feature_cols, scaler, device, sequence_df)

    if result is None:
        return

    # Display results
    st.header("Prediction Results")

    # Add prediction timeline info
    seq_data = result['sequence_data']
    q7_data = seq_data.iloc[6]
    q8_data = seq_data.iloc[7]

    # Use period_start_date (actual transcript availability) instead of fiscal quarter ends (generic)
    # Fallback to rebalance_date for backwards compatibility with older data
    q7_period_start = pd.to_datetime(q7_data.get('period_start_date', q7_data.get('rebalance_date', q7_data['fiscal_quarter_end'])))
    q8_period_start = pd.to_datetime(q8_data.get('period_start_date', q8_data.get('rebalance_date', q8_data['fiscal_quarter_end'])))

    # Get period_end_date if available
    q7_period_end = pd.to_datetime(q7_data['period_end_date']) if pd.notna(q7_data.get('period_end_date')) else q8_period_start

    # Calculate prediction target date (Q8 period start = when Q8 transcript available)
    prediction_target_date = q8_period_start

    # Calculate holding period
    holding_period_days = (q7_period_end - q7_period_start).days

    st.info(f"🎯 **Trading Period**: Enter position on {q7_period_start.strftime('%B %d, %Y')} "
            f"(Q7 transcript available) → Exit on {q7_period_end.strftime('%B %d, %Y')} "
            f"(Q8 transcript available) | **Holding Period: {holding_period_days} days**")

    # Key metrics in columns
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            "Entry Price (Period Start)",
            f"${result['last_known_price']:.2f}",
            help=f"Price on {q7_period_start.strftime('%Y-%m-%d')} when Q7 transcript available"
        )

    with metric_col2:
        change = result['predicted_price'] - result['last_known_price']
        percent_change = (change / result['last_known_price']) * 100 if result['last_known_price'] != 0 else 0
        st.metric(
            f"Predicted Exit Price",
            f"${result['predicted_price']:.2f}",
            f"{percent_change:+.2f}%",
            help=f"Model prediction for {q7_period_end.strftime('%Y-%m-%d')} when Q8 transcript available"
        )

    with metric_col3:
        if result['actual_price'] is not None:
            actual_change = result['actual_price'] - result['last_known_price']
            actual_percent = (actual_change / result['last_known_price']) * 100 if result['last_known_price'] != 0 else 0
            st.metric(
                f"Actual Exit Price",
                f"${result['actual_price']:.2f}",
                f"{actual_percent:+.2f}%",
                help=f"Actual price on {q7_period_end.strftime('%Y-%m-%d')} when Q8 transcript available"
            )
        else:
            st.metric("Actual Exit Price", "N/A", help="Actual future price not yet available")

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

    # Calculate days from fiscal quarter end to period start (transcript availability)
    q7_fiscal_end = pd.to_datetime(q7_data['fiscal_quarter_end'])
    days_to_period_start = (q7_period_start - q7_fiscal_end).days

    with info_col1:
        st.write(f"**Ticker:** {result['ticker']}")
        st.write(f"**Sequence ID:** {result['sequence_id']}")
        st.write(f"**Q7 Fiscal Quarter End:** {q7_fiscal_end.strftime('%Y-%m-%d')}")
        st.write(f"**Q7 Period Start (Entry):** {q7_period_start.strftime('%Y-%m-%d')} ({days_to_period_start} days after quarter)")
        st.caption(f"_Actual transcript release date for Q7_")

    with info_col2:
        q8_fiscal_end = pd.to_datetime(q8_data['fiscal_quarter_end'])
        days_to_q8_period_start = (q8_period_start - q8_fiscal_end).days

        st.write(f"**Q8 Fiscal Quarter End:** {q8_fiscal_end.strftime('%Y-%m-%d')}")
        st.write(f"**Q7 Period End (Exit):** {q7_period_end.strftime('%Y-%m-%d')} = Q8 Period Start")
        st.write(f"**Q8 Period Start:** {q8_period_start.strftime('%Y-%m-%d')} ({days_to_q8_period_start} days after Q8 end)")
        st.caption(f"_Actual transcript release dates - no look-ahead bias_")
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
