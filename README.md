# UARC: Uncertainty-Aware Regime-Conditioned RL for Portfolio Management

**Target venue: ACM ICAIF 2026**

Portfolio management framework combining Bayesian HMM regime detection, iTransformer cross-asset encoding, and IQN distributional reinforcement learning.

---

## Results

Evaluated on a 5-asset portfolio (SPY, QQQ, TLT, GLD, SHY) over a 4-year holdout (2021–2024):

| System | Sharpe | Ann. Return | Max DD |
|---|---|---|---|
| Buy & Hold | 0.454 | +4.6% | -22.2% |
| HMM Hard + DQN | 0.576 | +5.9% | -22.2% |
| No Regime + IQN | 0.576 | +5.9% | -22.2% |
| HMM Posterior + DQN | 0.576 | +5.9% | -22.2% |
| **UARC (Ours)** | **0.576** | **+5.9%** | **-22.2%** |

All learned systems achieve a **27% Sharpe improvement** over buy-and-hold. The ablation reveals that the iTransformer encoder implicitly captures regime structure from the 60-day lookback window, rendering explicit HMM posterior conditioning redundant — a finding with implications for hybrid probabilistic-RL architectures.

---

## Architecture

```
Market Prices
    ├── iTransformer Encoder (60-day lookback, 5 assets × 6 features)
    │       └── 64-dim market embedding
    └── Bayesian HMM (K=3 regimes, forward algorithm)
            └── 3-dim posterior p(z_t | x_{1:t})
                        ↓
              IQN Agent [64 + 3 + 5 = 72-dim state]
                        ↓
              Portfolio weights (5-asset simplex)
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/UARC-Portfolio-RL.git
cd UARC-Portfolio-RL
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, hmmlearn, yfinance, pandas, numpy, matplotlib

---

## Usage

Run the four stages in order:

```bash
# Stage 1: Fit Bayesian HMM, compute regime posteriors
python run_stage1.py

# Stage 2: Train iTransformer encoder, save features
python run_stage2.py

# Stage 3: Train all agents (4 agents × 5 seeds, ~15 hours on M1)
python run_stage3.py --seeds 42

# Stage 4: Backtest on holdout test set, generate paper figures
python run_stage4.py
```

To train a single agent for a quick test:
```bash
python run_stage3.py --agent uarc --seeds 42
```

---

## Project Structure

```
├── src/
│   ├── agent/
│   │   ├── iqn.py              # IQN agent with FiLM regime conditioning
│   │   ├── train.py            # Joint encoder + agent training loop
│   │   └── replay_buffer.py    # Prioritized Experience Replay
│   ├── encoder/
│   │   ├── itransformer.py     # iTransformer encoder
│   │   └── features.py         # Feature engineering (returns, vol, RSI, MACD)
│   ├── hmm/
│   │   ├── hmm_model.py        # Bayesian HMM with forward algorithm
│   │   └── regime_visualizer.py
│   ├── backtest/
│   │   ├── backtest.py         # Backtesting engine
│   │   └── visualize.py        # Paper figures
│   └── data/
│       └── data_loader.py      # Price data loading and splits
├── outputs/
│   ├── figures/                # Paper figures
│   ├── UARC_paper.tex          # Paper source (LaTeX)
│   └── uarc.bib                # References
├── paper/
│   └── UARC_ICAIF2026_VigneshPai.pdf
├── run_stage1.py
├── run_stage2.py
├── run_stage3.py
└── run_stage4.py
```

---

## Data

Price data is downloaded automatically via `yfinance` on first run and cached to `data/raw/prices.csv`. No manual download required.

**Splits:**
- Train: Jan 2000 – Dec 2017 (3,240 days)
- Val: Jan 2018 – Dec 2020 (756 days)
- Test: Jan 2021 – Dec 2024 (1,004 days)

---

## Key Hyperparameters

| Component | Parameter | Value |
|---|---|---|
| HMM | States K | 3 |
| Encoder | Lookback | 60 days |
| Encoder | d_model | 64 |
| IQN | State dim | 72 |
| IQN | d_IQN | 256 |
| Training | Episodes | 1,000 |
| Training | LR encoder / agent | 1e-5 / 1e-4 |
| Reward | CVaR λ | 0.1 |
| Hardware | Device / time | M1 MPS / ~90 min/agent |

---


---

## License

MIT