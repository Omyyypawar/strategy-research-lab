# Strategy Research Lab

An end-to-end AI-powered quantitative trading platform for cryptocurrency markets. Ingests real-time Binance klines, engineers microstructure features, trains ML classifiers for short-horizon return prediction, and executes trades with confidence-aware sizing and institutional-grade risk controls.

## Architecture

```
Binance 1m Klines (BTCUSDT)
        |
        v
+-------------------+     +------------------------+
| Rolling Buffer    | --> | Feature Factory V2     |
| (2000 candles)    |     | 19 features: entropy,  |
+-------------------+     | VPIN, vol regime, mom  |
                          +------------------------+
                                    |
                                    v
                          +------------------------+
                          | RandomForest Classifier|
                          | 500 trees, depth=12    |
                          | predict_proba() output |
                          +------------------------+
                                    |
                                    v
                          +------------------------+
                          | Ensemble Filter        |
                          | model_prob (0.6)       |
                          | + vol regime (0.2)     |
                          | + micro consensus(0.2) |
                          +------------------------+
                                    |
                                    v
                          +------------------------+     +-----------------+
                          | Execution Engine       | --> | CSV Logs        |
                          | Slippage model         |     | Trades file     |
                          | Confidence sizing      |     | Metrics file    |
                          | Circuit breakers       |     +-----------------+
                          +------------------------+            |
                                                                v
                                                      +-----------------+
                                                      | Streamlit       |
                                                      | Live Dashboard  |
                                                      +-----------------+
```

## Project Structure

```
strategy_research_lab/
|-- src/
|   |-- feature_factory_v2.py     # Production: 19 microstructure features
|   |-- incremental_features.py   # Streaming feature engine (tick-by-tick)
|   |-- feature_engineering.py    # Legacy Phase 2 features (deprecated)
|   |-- indicators.py             # Technical indicators (SMA, EMA, RSI, MACD, BB)
|   |-- dataset_builder_v2.py     # Parquet dataset builder from Binance klines
|   |-- model_training.py         # RF model training
|   |-- experiment_runner_v2.py   # Full training pipeline with metadata export
|   |-- ensemble_model.py         # LightGBM + RF stacking ensemble with threshold opt
|   |-- ml_signal_engine.py       # Legacy RF trainer with Plotly dashboards
|   |-- walk_forward.py           # Walk-forward backtest with Monte Carlo CI
|   |-- trade_analytics.py        # Trade-level P&L, MAE/MFE, streaks, rolling metrics
|   |-- executor_v3.py            # Latency/slippage tracking executor
|   |-- executor_v4.py            # Spread-aware VWAP + ensemble executor
|   |-- executor_v5.py            # Env-aligned executor with signal smoothing
|   |-- live_executor.py          # Production executor (probability + MAE/MFE)
|   |-- live_trading.py           # Paper trading via REST polling
|   |-- live_stream_trading.py    # WebSocket real-time paper trading
|   |-- alt_data.py               # Funding rate / OI / liquidations / L-S ratio features
|   |-- online_learner.py         # River-based streaming model (Hoeffding, AMF)
|   |-- feature_store.py          # SQLite feature audit + drift detection
|   |-- backtesting.py            # Backtesting engine (CAGR, Sharpe, Sortino, DD)
|   |-- visualization.py          # Plotly 4-panel crypto dashboard
|   |-- live_dashboard.py         # Streamlit institutional live monitor
|   |-- comparison_dashboard.py   # Hybrid vs ML comparison dashboard
|   |-- binance_client.py         # Binance API wrapper
|   `-- main.py                   # Phase 2 hybrid strategy orchestrator
|
|-- models/
|   |-- phase3_rf_v2.pkl              # Trained RandomForest model
|   |-- features_columns_phase3_v2.json  # Feature manifest (19 columns)
|   `-- phase3_model_meta.json        # Model metadata & hyperparameters
|
|-- data/
|   `-- features_v2.parquet           # V2 training dataset
|
|-- tests/                            # pytest test suite (85 tests)
|   |-- test_feature_factory_v2.py    # Feature builder tests (17 tests)
|   |-- test_backtesting.py           # Backtesting engine tests (7 tests)
|   |-- test_indicators.py            # Technical indicator tests (9 tests)
|   |-- test_incremental_features.py  # Streaming engine tests (7 tests)
|   |-- test_walk_forward.py          # Walk-forward backtest tests (11 tests)
|   |-- test_trade_analytics.py       # Trade analytics tests (12 tests)
|   |-- test_ensemble_model.py        # Ensemble model tests (6 tests)
|   |-- test_alt_data.py              # Alt-data state tests (7 tests)
|   |-- test_feature_store.py         # Feature store tests (8 tests)
|   |-- test_online_learner.py        # Online learner tests (8 tests, skipped without River)
|   `-- conftest.py                   # Shared fixtures
|
|-- docs/
|   `-- phase3_v2_feature_reference.md  # Feature documentation
|
|-- Dockerfile                        # Python 3.11-slim image for executor
|-- docker-compose.yml                # Executor + Streamlit dashboard services
|-- .github/workflows/ci.yml          # GitHub Actions CI (pytest on push/PR)
|-- .env.example                      # Configuration template
`-- requirements.txt                  # Python dependencies
```

## Features (V2 Pipeline)

The production ML pipeline uses 19 features engineered in `feature_factory_v2.py`:

| Feature | Description |
|---------|-------------|
| `ret_1m`, `ret_5m` | Backward-looking returns (1-min, 5-min) |
| `vol_short`, `vol_long` | Rolling volatility (10-bar, 60-bar std) |
| `vol_regime` | Short/long vol ratio (regime detection) |
| `entropy_50` | Rolling entropy of returns (market randomness) |
| `norm_range_20` | Normalized high-low range |
| `vol_imbalance_20` | Bullish vs bearish volume imbalance |
| `vpin_50` | VPIN-like directional flow proxy |
| `zscore_ret5` | Z-score of 5-min returns |
| `mom1`, `mom2`, `mom2_smooth` | 1st & 2nd order momentum + smoothed |
| `avg_vol_20` | 20-bar average volume (liquidity proxy) |

All features are leak-safe (backward-looking only).

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and set your Binance API credentials:

```
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BINANCE_TESTNET=True
EXECUTE_ORDERS=False
```

### Build Dataset & Train Model

```bash
# 1. Fetch klines and build feature dataset
python -m src.dataset_builder_v2

# 2a. Train single RandomForest (legacy)
python -m src.experiment_runner_v2

# 2b. Train stacking ensemble (RF + LightGBM) — recommended
python -m src.ensemble_model
```

### Validate with Walk-Forward Backtest

```bash
# Run 5-fold walk-forward with Monte Carlo significance test
python -m src.walk_forward
```

### Run Paper Trading

```bash
# Production executor (recommended)
python -m src.live_executor

# Or with Streamlit dashboard
streamlit run src/live_dashboard.py
```

### Analyze Trade Performance

```bash
# Generate trade analytics report from executor logs
python -m src.trade_analytics
```

### Run Tests

```bash
python -m pytest tests/ -v    # 70 tests
```

## Risk Controls

| Control | Default | Description |
|---------|---------|-------------|
| Daily loss limit | 10% | Stops trading if daily P&L drops below threshold |
| Max drawdown | 20% | Emergency stop on peak-to-trough decline |
| Consecutive losses | 5 | Pauses after N losing trades in a row |
| Order cooldown | 30s | Minimum time between orders |
| Slippage model | 0.05% base | Adaptive slippage scaled by order size and volatility |

## Tech Stack

- **Data**: pandas, NumPy, Parquet
- **ML**: scikit-learn (RandomForest), scipy
- **Exchange**: python-binance (AsyncClient, WebSocket)
- **Viz**: Plotly, Streamlit, Matplotlib
- **Async**: asyncio for real-time execution

## Author

**Om Pawar** | Strategy Research Lab

## License

See [LICENSE](LICENSE) for details.
