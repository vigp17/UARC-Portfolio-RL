"""
train.py  —  UARC Stage 3 Training (v4 — joint encoder+agent training)
=======================================================================

Key change from v3: encoder is NO LONGER FROZEN.

Problem with frozen encoder: all agents received identical pre-computed
embeddings, making regime conditioning redundant. FiLM and concatenation
both failed because the 64-dim encoder output already implicitly contained
regime information from Stage 2 pretraining.

Fix: joint training with two learning rates:
  - Encoder:    lr = 1e-5  (fine-tune, slow — preserve Stage 2 features)
  - IQN/DQN:   lr = 1e-4  (train from scratch, fast)

Each agent now shapes its OWN encoder via RL gradients. Over 1000 episodes:
  - NO_REGIME_IQN:     encoder learns regime-agnostic features
  - HMM_HARD_DQN:      encoder learns to emphasize hard regime boundaries
  - HMM_POSTERIOR_DQN: encoder learns to use soft regime uncertainty
  - UARC:              encoder + IQN jointly learn distributional regime policy

Buffer change: stores [t_idx | regime(K) | weights(N)] instead of
[enc(64) | regime(K) | weights(N)]. During _learn, encoder is called
on the actual lookback window for each transition — gradients flow.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from src.encoder.itransformer import iTransformerEncoder
from src.agent.iqn import IQNAgent, huber_quantile_loss
from src.agent.replay_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


class AgentType(Enum):
    NO_REGIME_IQN     = "no_regime_iqn"
    HMM_HARD_DQN      = "hmm_hard_dqn"
    HMM_POSTERIOR_DQN = "hmm_posterior_dqn"
    UARC              = "uarc"


@dataclass
class TrainConfig:
    n_assets:        int   = 5
    encoder_dim:     int   = 64
    n_regimes:       int   = 3
    lookback:        int   = 60
    n_features:      int   = 6
    d_model_enc:     int   = 64
    n_heads:         int   = 4
    n_layers_enc:    int   = 2
    d_model_iqn:     int   = 256
    n_cos:           int   = 64
    agent_type:      AgentType = AgentType.UARC
    use_iqn:         bool  = True
    regime_mode:     str   = "posterior"
    n_tau_train:     int   = 32
    n_tau_eval:      int   = 64
    n_episodes:      int   = 1000
    episode_len:     int   = 252
    batch_size:      int   = 256
    lr_agent:        float = 1e-4
    lr_encoder:      float = 1e-5
    weight_decay:    float = 1e-5
    grad_clip:       float = 1.0
    warmup_steps:    int   = 2000
    gamma:           float = 0.99
    tau_soft:        float = 0.005
    update_freq:     int   = 4
    lambda_cvar:     float = 0.1
    transaction_cost: float = 0.001
    buffer_capacity: int   = 100_000
    alpha_per:       float = 0.6
    beta_start:      float = 0.4
    log_freq:        int   = 100
    eval_freq:       int   = 100
    model_dir:       str   = "outputs/models"
    device:          str   = "auto"
    seed:            int   = 42

    @classmethod
    def for_agent(cls, agent_type: AgentType, seed: int = 42, **kwargs) -> "TrainConfig":
        base = dict(agent_type=agent_type, seed=seed)
        if agent_type == AgentType.NO_REGIME_IQN:
            base.update(use_iqn=True,  regime_mode="uniform")
        elif agent_type == AgentType.HMM_HARD_DQN:
            base.update(use_iqn=False, regime_mode="hard")
        elif agent_type == AgentType.HMM_POSTERIOR_DQN:
            base.update(use_iqn=False, regime_mode="posterior")
        elif agent_type == AgentType.UARC:
            base.update(use_iqn=True,  regime_mode="posterior")
        base.update(kwargs)
        return cls(**base)


def preprocess_regime(posterior: np.ndarray, mode: str, n_regimes: int) -> np.ndarray:
    if mode == "uniform":
        return np.full(n_regimes, 1.0 / n_regimes, dtype=np.float32)
    elif mode == "hard":
        one_hot = np.zeros(n_regimes, dtype=np.float32)
        one_hot[int(np.argmax(posterior))] = 1.0
        return one_hot
    elif mode == "posterior":
        return posterior.astype(np.float32)
    else:
        raise ValueError(f"Unknown regime_mode: {mode}")


class TradingEnvironment:
    def __init__(
        self,
        enc_features:   np.ndarray,
        posteriors:     np.ndarray,
        prices:         np.ndarray,
        regime_mode:    str,
        n_regimes:      int,
        episode_len:    int = 252,
        lookback:       int = 60,
    ):
        self.enc_features = enc_features
        self.posteriors   = posteriors
        self.regime_mode  = regime_mode
        self.n_regimes    = n_regimes
        self.episode_len  = episode_len
        self.lookback     = lookback
        self.n_assets     = prices.shape[1]

        self.returns     = np.zeros_like(prices)
        self.returns[1:] = np.log(prices[1:] / (prices[:-1] + 1e-10))

        self.T            = len(enc_features)
        self.t            = lookback
        self.episode_t    = 0
        self.prev_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

    def reset(self, random_start: bool = True) -> Dict:
        if random_start:
            max_start = max(self.lookback + 1, self.T - self.episode_len - 1)
            self.t    = np.random.randint(self.lookback, max_start)
        else:
            self.t = self.lookback
        self.episode_t    = 0
        self.prev_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        return self._get_state()

    def step(self, weights: np.ndarray) -> Tuple[Dict, float, bool]:
        port_return    = float(np.dot(weights, self.returns[self.t]))
        self.t        += 1
        self.episode_t += 1
        self.prev_weights = weights.copy()
        done = self.episode_t >= self.episode_len or self.t >= self.T - 1
        return self._get_state(), port_return, done

    def _get_state(self) -> Dict:
        window = self.enc_features[self.t - self.lookback: self.t]
        return {
            "t_idx":            self.t,
            "enc_window":       window.astype(np.float32),
            "regime_posterior": preprocess_regime(
                self.posteriors[self.t], self.regime_mode, self.n_regimes
            ),
            "prev_weights":     self.prev_weights.astype(np.float32),
        }

    def state_to_vector(self, state: Dict) -> np.ndarray:
        return np.concatenate([
            np.array([state["t_idx"]], dtype=np.float32),
            state["regime_posterior"],
            state["prev_weights"],
        ]).astype(np.float32)


class UARCTrainer:
    def __init__(
        self,
        config:                  TrainConfig,
        enc_features_train:      np.ndarray,
        posteriors_train:        np.ndarray,
        prices_train:            np.ndarray,
        enc_features_val:        np.ndarray,
        posteriors_val:          np.ndarray,
        prices_val:              np.ndarray,
        pretrained_encoder_path: str = "outputs/models/encoder_stage2.pt",
    ):
        self.config = config
        _set_seed(config.seed)

        if config.device == "auto":
            self.device = torch.device(
                "mps"  if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(config.device)

        # Load encoder — NOT frozen, joint training
        self.encoder = iTransformerEncoder(
            n_assets = config.n_assets,
            lookback = config.lookback * config.n_features,
            d_model  = config.d_model_enc,
            n_heads  = config.n_heads,
            n_layers = config.n_layers_enc,
            dropout  = 0.1,
        ).to(self.device)
        self.encoder.load_state_dict(
            torch.load(pretrained_encoder_path, map_location=self.device)
        )
        self.encoder.train()

        logger.info(
            f"[{config.agent_type.value} | seed={config.seed}] "
            f"Encoder JOINT (lr={config.lr_encoder}) | device={self.device}"
        )

        # Store raw features for lookup during _learn
        self.enc_features_train = enc_features_train
        self.enc_features_val   = enc_features_val

        # Pre-compute val embeddings once for fast eval
        self.enc_emb_val = self._compute_embeddings(enc_features_val)
        logger.info(f"  Val embeddings: {self.enc_emb_val.shape}")

        # IQN / DQN agent
        n_tau_train = config.n_tau_train if config.use_iqn else 1
        n_tau_eval  = config.n_tau_eval  if config.use_iqn else 1

        def _make_agent():
            return IQNAgent(
                n_assets    = config.n_assets,
                encoder_dim = config.encoder_dim,
                n_regimes   = config.n_regimes,
                d_model     = config.d_model_iqn,
                n_cos       = config.n_cos,
                n_tau_train = n_tau_train,
                n_tau_eval  = n_tau_eval,
            ).to(self.device)

        self.agent        = _make_agent()
        self.agent_target = _make_agent()
        self.agent_target.load_state_dict(self.agent.state_dict())
        self.agent_target.eval()

        # Frozen encoder for stable target network
        self.encoder_target = iTransformerEncoder(
            n_assets = config.n_assets,
            lookback = config.lookback * config.n_features,
            d_model  = config.d_model_enc,
            n_heads  = config.n_heads,
            n_layers = config.n_layers_enc,
            dropout  = 0.0,
        ).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.encoder_target.eval()
        for p in self.encoder_target.parameters():
            p.requires_grad_(False)

        # Joint optimizer: encoder 10x slower than agent
        self.optimizer = optim.Adam([
            {"params": self.encoder.parameters(), "lr": config.lr_encoder},
            {"params": self.agent.parameters(),   "lr": config.lr_agent},
        ], weight_decay=config.weight_decay)

        env_kwargs = dict(
            regime_mode = config.regime_mode,
            n_regimes   = config.n_regimes,
            episode_len = config.episode_len,
            lookback    = config.lookback,
        )
        self.train_env = TradingEnvironment(
            enc_features_train, posteriors_train, prices_train, **env_kwargs
        )
        self.val_env = TradingEnvironment(
            enc_features_val, posteriors_val, prices_val, **env_kwargs
        )

        self.buffer = PrioritizedReplayBuffer(
            capacity   = config.buffer_capacity,
            alpha      = config.alpha_per,
            beta_start = config.beta_start,
        )

        self.global_step     = 0
        self.best_val_sharpe = -np.inf
        Path(config.model_dir).mkdir(parents=True, exist_ok=True)

    def _compute_embeddings(self, enc_features: np.ndarray) -> np.ndarray:
        T          = len(enc_features)
        embeddings = np.zeros((T, self.config.encoder_dim), dtype=np.float32)
        with torch.no_grad():
            for t in range(self.config.lookback, T):
                window = enc_features[t - self.config.lookback: t]
                enc_in = window.transpose(1, 0, 2).reshape(
                    1, self.config.n_assets,
                    self.config.lookback * self.config.n_features
                )
                embeddings[t] = self.encoder(
                    torch.from_numpy(enc_in).to(self.device)
                ).cpu().numpy()
        return embeddings

    def _window_to_tensor(
        self, enc_features: np.ndarray, t_indices: np.ndarray
    ) -> torch.Tensor:
        B = len(t_indices)
        L = self.config.lookback
        windows = np.zeros(
            (B, self.config.n_assets, L * self.config.n_features),
            dtype=np.float32
        )
        for i, t in enumerate(t_indices):
            t = max(int(t), L)  
            window = enc_features[t - L: t]
            windows[i] = window.transpose(1, 0, 2).reshape(
                self.config.n_assets, L * self.config.n_features
            )
        return torch.from_numpy(windows).to(self.device)

    def _select_action(self, state: Dict, epsilon: float = 0.0) -> np.ndarray:
        window = state["enc_window"].transpose(1, 0, 2).reshape(
            1, self.config.n_assets,
            self.config.lookback * self.config.n_features
        )
        self.encoder.eval()
        self.agent.eval()
        with torch.no_grad():
            enc = self.encoder(torch.from_numpy(window).to(self.device))
            reg = torch.from_numpy(state["regime_posterior"]).unsqueeze(0).to(self.device)
            wts = torch.from_numpy(state["prev_weights"]).unsqueeze(0).to(self.device)
            w   = self.agent.get_portfolio_weights(enc, reg, wts, risk_aversion=0.0).squeeze(0).cpu().numpy()
        self.encoder.train()

        if epsilon > 0:
            w = np.abs(w + np.random.normal(0, epsilon, size=w.shape))
            s = w.sum()
            w = w / s if s > 1e-8 else np.ones_like(w) / len(w)
        return w.astype(np.float32)

    def _learn(self, batch: Dict) -> Dict[str, float]:
        states      = batch["states"]
        next_states = batch["next_states"]
        rewards     = batch["rewards"].to(self.device)
        dones       = batch["dones"].to(self.device)
        is_weights  = batch["weights"].to(self.device)

        K = self.config.n_regimes
        N = self.config.n_assets

        t_idx_c = states[:, 0].cpu().numpy().astype(int)
        reg_c   = states[:, 1:1+K].to(self.device)
        wts_c   = states[:, 1+K:1+K+N].to(self.device)

        t_idx_n = next_states[:, 0].cpu().numpy().astype(int)
        reg_n   = next_states[:, 1:1+K].to(self.device)
        wts_n   = next_states[:, 1+K:1+K+N].to(self.device)

        # Encoder forward — gradients flow through encoder
        windows_c = self._window_to_tensor(self.enc_features_train, t_idx_c)
        enc_c     = self.encoder(windows_c)

        with torch.no_grad():
            windows_n = self._window_to_tensor(self.enc_features_train, t_idx_n)
            enc_n     = self.encoder_target(windows_n)

        self.agent.train()
        q_vals, tau = self.agent(enc_c, reg_c, wts_c, training=True)

        with torch.no_grad():
            q_next, _ = self.agent_target(enc_n, reg_n, wts_n, training=True)
            targets   = (
                rewards.unsqueeze(1).unsqueeze(2) +
                self.config.gamma * q_next * (1 - dones).unsqueeze(1).unsqueeze(2)
            )

        loss = (huber_quantile_loss(q_vals, targets.detach(), tau) * is_weights.mean()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.agent.parameters()),
            self.config.grad_clip
        )
        self.optimizer.step()

        with torch.no_grad():
            td_err = (q_vals.mean(1) - targets.mean(1)).abs().mean(-1).cpu().numpy()

        return {"loss": float(loss.item()), "td_error": float(td_err.mean())}

    def _soft_update_target(self):
        tau = self.config.tau_soft
        for p, tp in zip(self.agent.parameters(), self.agent_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for p, tp in zip(self.encoder.parameters(), self.encoder_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def evaluate(self, n_episodes: int = 5) -> Dict[str, float]:
        all_r = []
        for _ in range(n_episodes):
            s    = self.val_env.reset(random_start=True)
            done = False
            while not done:
                w          = self._select_action(s, epsilon=0.0)
                s, r, done = self.val_env.step(w)
                all_r.append(r)
        arr = np.array(all_r)
        return {
            "val_sharpe":    _compute_sharpe(arr),
            "val_total_ret": float(arr.sum()),
            "val_max_dd":    _compute_max_drawdown(arr),
        }

    def train(self) -> Dict[str, float]:
        config = self.config
        losses = []

        logger.info(
            f"\n{'='*60}\n"
            f"  Agent : {config.agent_type.value}\n"
            f"  Type  : {'IQN' if config.use_iqn else 'DQN (n_tau=1)'}\n"
            f"  Regime: {config.regime_mode}\n"
            f"  Seed  : {config.seed}\n"
            f"  Enc LR: {config.lr_encoder}  Agent LR: {config.lr_agent}\n"
            f"{'='*60}"
        )

        for ep in range(config.n_episodes):
            state   = self.train_env.reset(random_start=True)
            ep_ret  = []
            ep_loss = []
            done    = False
            eps     = max(0.01, 0.3 * (1 - ep / config.n_episodes))

            while not done:
                w               = self._select_action(state, epsilon=eps)
                nstate, r, done = self.train_env.step(w)

                sv  = self.train_env.state_to_vector(state)
                nsv = self.train_env.state_to_vector(nstate)
                self.buffer.add(sv, w, r, nsv, done)

                ep_ret.append(r)
                state = nstate
                self.global_step += 1

                if (
                    self.global_step > config.warmup_steps and
                    len(self.buffer) >= config.batch_size and
                    self.global_step % config.update_freq == 0
                ):
                    b = self.buffer.sample(config.batch_size)
                    m = self._learn(b)
                    self.buffer.update_priorities(
                        b["indices"],
                        np.full(config.batch_size, m["td_error"])
                    )
                    self._soft_update_target()
                    ep_loss.append(m["loss"])

            avg_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
            losses.append(avg_loss)

            if ep % config.log_freq == 0:
                logger.info(
                    f"  Ep {ep:4d}/{config.n_episodes} | "
                    f"Ret: {sum(ep_ret):+.4f} | "
                    f"Sharpe: {_compute_sharpe(np.array(ep_ret)):.3f} | "
                    f"Loss: {avg_loss:.5f} | "
                    f"Buf: {len(self.buffer):,} | eps: {eps:.3f}"
                )

            if ep % config.eval_freq == 0 and ep > 0:
                val = self.evaluate()
                logger.info(
                    f"  [VAL] Sharpe: {val['val_sharpe']:.3f} | "
                    f"Ret: {val['val_total_ret']:+.4f} | "
                    f"DD: {val['val_max_dd']:.4f}"
                )
                if val["val_sharpe"] > self.best_val_sharpe:
                    self.best_val_sharpe = val["val_sharpe"]
                    self._save("best")
                    logger.info(f"  * New best: {self.best_val_sharpe:.3f}")

        self._save("final")
        logger.info(f"  Done. Best val Sharpe: {self.best_val_sharpe:.3f}")
        return {
            "agent_type":      config.agent_type.value,
            "seed":            config.seed,
            "best_val_sharpe": self.best_val_sharpe,
            "final_loss":      float(np.mean(losses[-50:])) if losses else 0.0,
        }

    def _save(self, tag: str):
        name = f"{self.config.agent_type.value}_seed{self.config.seed}"
        torch.save(
            self.agent.state_dict(),
            Path(self.config.model_dir) / f"agent_{name}_{tag}.pt"
        )
        torch.save(
            self.encoder.state_dict(),
            Path(self.config.model_dir) / f"encoder_{name}_{tag}.pt"
        )

    def load_best(self):
        name = f"{self.config.agent_type.value}_seed{self.config.seed}"
        self.agent.load_state_dict(
            torch.load(
                Path(self.config.model_dir) / f"agent_{name}_best.pt",
                map_location=self.device,
            )
        )
        self.encoder.load_state_dict(
            torch.load(
                Path(self.config.model_dir) / f"encoder_{name}_best.pt",
                map_location=self.device,
            )
        )
        self.agent.eval()
        self.encoder.eval()


def _compute_sharpe(returns: np.ndarray, annualize: bool = True) -> float:
    if len(returns) < 2 or returns.std() < 1e-10:
        return 0.0
    sr = returns.mean() / returns.std()
    return float(sr * np.sqrt(252)) if annualize else float(sr)


def _compute_max_drawdown(returns: np.ndarray) -> float:
    cum = np.exp(np.cumsum(returns))
    dd  = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
    return float(dd.min())


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)