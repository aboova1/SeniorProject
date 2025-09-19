1. Data Collection
Quantitative (FMP API): pull stock price history, fundamentals (PE, EPS, Market Cap, etc.), balance sheet, income statement, and sector classification for all S&P 500 tickers.
Qualitative (FMP API):
10-K MD&A → use as baseline text (annual, carry forward to all quarters).
Earnings-call transcripts → when available, replace MD&A for that quarter.
(Optional) Press releases/news → only for 2019+; exclude from long backtests.
Sector encoding: add sector as categorical feature (one-hot or embedding).
2. Text Processing
Use FinBERT to embed transcripts/MD&A.
Reduce embeddings from 768 → 64 dims with PCA
Store one 64-dim embedding per stock per quarter.
3. Feature Engineering
Combine numeric features (~13–18) + text embedding (64 dims) into a single feature vector.
Normalize/scale features.
Control for overlap, don’t include highly collinear ratios twice.
4. Model Training
Build an LSTM that takes the past 8 quarters of features per stock as input.
Train to predict the next quarter’s stock return (or price % change).
Training size: ~36,000 sequences
5. Portfolio Construction
Use predicted returns (and optionally model-estimated variance).
Apply PyPortfolioOpt to build optimized portfolios at different volatility levels.
User selects a risk preference, portfolio weights are output.
6. Evaluation
Backtest against historical data.
Report:
Accuracy of return forecasts.
Portfolio performance vs benchmarks.
Confidence intervals (99%, 95%, 90%) from model output variance.
7. Deliverables
Interface (simple web app): user inputs risk profile → outputs expected return, portfolio allocation, and confidence intervals.
(Optional) Final report: show backtest results, highlight whether qualitative + quantitative beats quantitative-only baselines.


STEP 1, DETAILED:
Perfect — here’s Step 1: Data Gathering rewritten so that the output files are already in model-ready format, with each row = one stock × one quarter.

⸻

Time Horizon
	•	Collect 2005 → latest quarter.
	•	Use 10-K MD&A (annual, carried forward) for all years.
	•	Use earnings-call transcripts (quarterly) where available.

⸻

Data to Pull from FMP
	1.	Daily prices (stock + SPY) → used to compute returns, momentum, volatility, beta.
	2.	Quarterly fundamentals → income, balance sheet, cash flow (rev, EPS, margins, debt, equity, CFO, FCF, etc.).
	3.	Earnings metadata → actual vs. estimated EPS (for surprise).
	4.	Text sources → MD&A (annual), transcripts (quarterly).
	5.	Reference → ticker list (S&P 500 snapshot), sector classification.

⸻

Model-ready Format (per stock × quarter)

Create a master table called quarters.parquet.
Each row = one stock × one fiscal quarter.
Columns:
	•	Keys / Dates
	•	ticker
	•	fiscal_quarter_end
	•	rebalance_date (lagged release date)
	•	sector
	•	Numeric features (~15–18 total)
	•	earnings_yield
	•	operating_margin
	•	roe
	•	revenue_yoy
	•	eps_yoy
	•	debt_to_equity
	•	current_ratio
	•	accruals_scaled
	•	fcf_margin
	•	momentum_12m
	•	vol_60d
	•	beta_1y
	•	earnings_surprise_pct
	•	Text features
	•	txt_01 … txt_64 (PCA-compressed FinBERT embeddings; 64 dims)
	•	Target label
	•	target_excess_return_next_q (next quarter stock log return – SPY log return)

⸻

Supporting Tables
	•	prices_daily.parquet: daily adj_close for all tickers + SPY.
	•	text_raw.parquet: raw MD&A / transcripts with filing dates.
	•	universe.parquet: metadata with ticker, sector, first_obs, last_obs, eligibility flag.