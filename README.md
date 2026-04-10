# UARC: Uncertainty-Aware Regime-Conditioned RL for Portfolio Management

Target venue: ACM ICAIF 2026

A research framework combining Bayesian HMM regime detection, iTransformer cross-asset representation learning, and IQN distributional reinforcement learning.

---

## Overview

Financial markets are non-stationary and exhibit regime shifts (bull, bear, sideways). This project investigates whether explicitly modeling regimes improves reinforcement learning-based portfolio allocation.

We propose UARC (Uncertainty-Aware Regime-Conditioned RL):
- Uses full HMM posterior instead of hard labels
- Learns cross-asset dependencies via iTransformer
- Optimizes risk-aware policies using distributional RL (IQN)

---

## Results

Evaluated on a 20-asset diversified portfolio across equities, bonds, commodities, and defensive assets.

Holdout period: 2021–2024

System | Sharpe | Ann. Return | Max Drawdown
-------|--------|------------|--------------
Buy & Hold | 0.454 | +4.6% | -22.2%
HMM Hard + DQN | 0.576 | +5.9% | -22.2%
No Regime + IQN | 0.576 | +5.9% | -22.2%
HMM Posterior + DQN | 0.576 | +5.9% | -22.2%
UARC (Ours) | 0.576 | +5.9% | -22.2%

---

## Key Finding

All learned systems converge to near-identical performance and allocations, closely matching a diversified baseline. Even with 20 assets and regime signals, policies collapse toward equal-weight-like allocations.

---

## Interpretation

This is a negative but important result:
- Explicit regime conditioning does not improve performance
- Distributional RL (IQN) is the primary driver
- Policies converge to diversified, high-entropy allocations

Insight: strong representation learning (iTransformer) may already encode regime structure implicitly.

---

## Architecture

<img src="outputs/figures/backtest/fig0_architecture.png" width="800"/>

State = [Embedding (64) + Posterior (3) + Previous Weights (20)] → IQN Agent → Portfolio Weights (20 assets)

---

## Installation

git clone https://github.com/YOUR_USERNAME/UARC-Portfolio-RL.git
cd UARC-Portfolio-RL
pip install -r requirements.txt

Requirements: Python 3.10+, PyTorch 2.0+, hmmlearn, yfinance, pandas, numpy, matplotlib

---

## Run Pipeline

python run_stage1.py
python run_stage2.py
python run_stage3.py --seeds 42
python run_stage4.py

Quick test:
python run_stage3.py --agent uarc --seeds 42

---

## Project Structure

src/
  agent/
  encoder/
  hmm/
  backtest/
  data/

outputs/
  figures/
  UARC_paper.tex
  uarc.bib

paper/
  UARC_ICAIF2026_VigneshPai.pdf

run_stage1.py
run_stage2.py
run_stage3.py
run_stage4.py

---

## Data

Data is automatically downloaded via yfinance.

Splits:
Train: 2000–2017
Val: 2018–2020
Test: 2021–2024

---

## Key Hyperparameters

Component | Parameter | Value
----------|----------|------
HMM | States K | 3
Encoder | Lookback | 60 days
Encoder | d_model | 64
IQN | State dim | 87
IQN | Hidden dim | 256
Training | Episodes | 1000
Training | LR (enc/agent) | 1e-5 / 1e-4
Reward | CVaR lambda | 0.1
Hardware | Device | Apple M1 (MPS)

---

## Diagnostics

- High entropy allocations (near-uniform)
- Very low turnover
- Minimal deviation from equal weight baseline

---

## Reproducibility

pytest tests/

Outputs:
outputs/backtest_results.csv
outputs/figures/

---

## Future Work

- Fix encoder–replay buffer mismatch
- Scale to larger asset universe
- Add realistic transaction costs
- Adaptive risk aversion
- Options-based tail hedging

---

## License

MIT