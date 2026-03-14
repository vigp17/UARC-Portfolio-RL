# Uncertainty-Aware Regime-Conditioned RL for Portfolio Management
# UARC Project — Stage 1: Bayesian HMM Regime Detection

## Project Structure

```
regime-rl-portfolio/
  src/
    data/
      data_loader.py       ← Download + feature engineering (Stage 1)
    hmm/
      hmm_model.py         ← Bayesian HMM with posterior inference (Stage 1)
      regime_visualizer.py ← All visualization plots (Stage 1)
    encoder/               ← iTransformer encoder (Stage 2)
    agent/                 ← IQN agent (Stage 3)
    backtest/              ← Backtesting engine (Stage 4)
  tests/
    test_hmm.py            ← Unit tests for HMM (run: pytest tests/ -v)
  outputs/
    models/                ← Saved model weights
    figures/               ← Generated plots
  run_stage1.py            ← Stage 1 entry point
  requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Stage 1
python run_stage1.py

# Run tests
pytest tests/test_hmm.py -v
```

## Stage 1 Success Criteria
- [ ] Transition matrix diagonal > 0.85 (regimes are persistent)
- [ ] Bull/Bear/Sideways regimes visually distinct in emission plots
- [ ] Online vs batch posterior max diff < 1e-4 (no lookahead bias)
- [ ] All unit tests pass
