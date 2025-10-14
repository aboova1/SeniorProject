#!/usr/bin/env python3
"""
Test script for live prediction pipeline
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from live_predictor import LivePredictor

# Import model classes
import sys
sys.path.append(str(Path(__file__).parent))
from dashboard import UniLSTMAttnDelta, UniLSTMAttnDirect

def test_live_prediction():
    """Test a complete live prediction"""

    # Load model
    model_path = Path("../models/final_model_all_data.pt")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Detect model type
    fc1_weight_shape = ckpt["model_state_dict"]["fc1.weight"].shape
    use_attn = ckpt.get("use_attn", True)
    hidden_size = 192
    rep_dim = hidden_size * (2 if use_attn else 1)
    fc1_input_size = fc1_weight_shape[1]

    if fc1_input_size == rep_dim:
        model = UniLSTMAttnDirect(
            input_size=len(ckpt["feature_cols"]),
            use_attn=use_attn
        )
    else:
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
    feature_cols = ckpt["feature_cols"]

    print(f"Model loaded. Features: {len(feature_cols)}")

    # Test with AAPL
    ticker = "AAPL"
    print(f"\n{'='*60}")
    print(f"Testing live prediction for {ticker}")
    print(f"{'='*60}")

    predictor = LivePredictor(use_cached_data=False)
    result = predictor.predict_next_quarter(
        ticker=ticker,
        model=model,
        feature_cols=feature_cols,
        scaler=scaler,
        device=device
    )

    if result['success']:
        print(f"\nPrediction successful!")
        print(f"Ticker: {result['ticker']}")
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"Predicted Price: ${result['predicted_price']:.2f}")
        print(f"Expected Change: ${result['price_change']:+.2f} ({result['price_change_pct']:+.2f}%)")
        print(f"Input sequence ends: {result['input_sequence_ends']}")
        print(f"Prediction target: {result['prediction_for_quarter']}")
        print(f"Features available: {result['features_available']}/{result['features_total']}")

        if 'actual_price' in result:
            print(f"\nActual price available for comparison:")
            print(f"Actual Price: ${result['actual_price']:.2f}")
            print(f"Prediction Error: ${result['prediction_error']:.2f} ({result['prediction_error_pct']:.2f}%)")
            print(f"Direction Correct: {result['direction_correct']}")
    else:
        print(f"\nPrediction failed: {result.get('error')}")

if __name__ == "__main__":
    test_live_prediction()
