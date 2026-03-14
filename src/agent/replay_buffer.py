"""
replay_buffer.py
----------------
Prioritized Experience Replay (PER) buffer for IQN training.

Stores transitions (state, action, reward, next_state, done) and
samples them with probability proportional to TD error magnitude.
High-error transitions are replayed more often, improving sample efficiency.

For financial RL, PER is especially useful because:
  - Crisis periods (high TD error) are rare but critical to learn from
  - Standard uniform replay undersamples these important transitions
  - PER ensures the agent sees enough bear market / regime-switch examples
"""

import numpy as np
from typing import Tuple, Dict, Optional
import torch


class SumTree:
    """
    Binary sum tree for O(log n) priority sampling.

    Leaves store individual transition priorities.
    Internal nodes store sum of subtree priorities.
    Enables efficient weighted sampling and priority updates.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_ptr = 0
        self.n_entries = 0

    @property
    def total_priority(self) -> float:
        return float(self.tree[0])

    def update(self, idx: int, priority: float):
        """Update priority of leaf at position idx."""
        tree_idx   = idx + self.capacity - 1
        delta      = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate change up to root
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += delta
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def add(self, priority: float) -> int:
        """Add a new entry with given priority. Returns leaf index."""
        idx = self.data_ptr
        self.update(idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return idx

    def sample(self, value: float) -> Tuple[int, float]:
        """
        Sample a leaf index for a given cumulative priority value.
        Returns (leaf_idx, priority).
        """
        idx = 0  # Start at root
        while idx < self.capacity - 1:
            left  = 2 * idx + 1
            right = 2 * idx + 2
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx    = right

        leaf_idx = idx - (self.capacity - 1)
        return leaf_idx, float(self.tree[idx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Stores transitions and samples them proportionally to TD error,
    with importance sampling weights to correct for the bias.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    alpha : float
        Priority exponent. 0 = uniform, 1 = full prioritization.
    beta_start : float
        Initial importance sampling exponent (0 = no correction).
    beta_end : float
        Final IS exponent (1 = full correction).
    beta_steps : int
        Steps over which beta is annealed from beta_start to beta_end.
    epsilon : float
        Small constant added to priorities to ensure all transitions
        have non-zero probability.
    """

    def __init__(
        self,
        capacity:    int   = 100_000,
        alpha:       float = 0.6,
        beta_start:  float = 0.4,
        beta_end:    float = 1.0,
        beta_steps:  int   = 100_000,
        epsilon:     float = 1e-6,
    ):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.beta_steps = beta_steps
        self.epsilon    = epsilon
        self.step       = 0

        self.tree  = SumTree(capacity)

        # Transition storage (numpy arrays for efficiency)
        # All dimensions set at first add() call
        self._initialized = False
        self.states       = None
        self.next_states  = None
        self.actions      = None
        self.rewards      = None
        self.dones        = None

        self.max_priority = 1.0

    def _initialize(self, state_dim: int, action_dim: int):
        """Lazy initialization of storage arrays."""
        self.states      = np.zeros((self.capacity, state_dim),  dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros(self.capacity,               dtype=np.float32)
        self.dones       = np.zeros(self.capacity,               dtype=np.float32)
        self._initialized = True

    @property
    def beta(self) -> float:
        """Linearly annealed beta for importance sampling correction."""
        frac = min(1.0, self.step / self.beta_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    @property
    def size(self) -> int:
        return self.tree.n_entries

    def add(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        """Add a transition with maximum current priority."""
        if not self._initialized:
            self._initialize(state.shape[0], action.shape[0])

        idx = self.tree.add(self.max_priority ** self.alpha)

        self.states[idx]      = state
        self.next_states[idx] = next_state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.dones[idx]       = float(done)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions with priority-weighted probabilities.

        Returns importance sampling weights to correct for sampling bias.

        Returns
        -------
        dict with keys: states, actions, rewards, next_states, dones,
                        weights (IS weights), indices (for priority update)
        """
        assert self.size >= batch_size, \
            f"Buffer has {self.size} transitions, need {batch_size}"

        indices    = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float32)

        # Stratified sampling: divide priority range into equal segments
        segment = self.tree.total_priority / batch_size
        for i in range(batch_size):
            lo    = segment * i
            hi    = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            idx, priority = self.tree.sample(value)
            indices[i]    = idx
            priorities[i] = priority + self.epsilon

        # Importance sampling weights
        sampling_probs = priorities / (self.tree.total_priority + self.epsilon)
        is_weights     = (self.size * sampling_probs) ** (-self.beta)
        is_weights    /= is_weights.max()   # Normalize

        self.step += 1

        return {
            "states":      torch.from_numpy(self.states[indices]),
            "actions":     torch.from_numpy(self.actions[indices]),
            "rewards":     torch.from_numpy(self.rewards[indices]),
            "next_states": torch.from_numpy(self.next_states[indices]),
            "dones":       torch.from_numpy(self.dones[indices]),
            "weights":     torch.from_numpy(is_weights.astype(np.float32)),
            "indices":     indices,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on new TD errors after a learning step.

        Called after each gradient update with the computed TD errors
        for the sampled batch.
        """
        for idx, error in zip(indices, td_errors):
            priority = (abs(float(error)) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.size