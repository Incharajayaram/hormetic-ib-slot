"""
Beta schedules for hormetic Information-Bottleneck training.

Six conditions from the proposal:
  1. HormeticSigmoid  - normalised sigmoid ramp (the primary intervention)
  2. HormeticCosine   - cosine ramp (secondary hormetic baseline)
  3. Linear           - monotone linear increase
  4. Reverse          - monotone linear decrease (starts compressed)
  5. RandomPermutation - shuffled linear values (breaks monotonicity)
  6. FixedBeta        - constant (standard VIB / beta-VAE baseline)
"""

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class BetaSchedule(ABC):
    """Abstract base class for all beta schedules."""

    @abstractmethod
    def get_beta(self, step: int, total_steps: int) -> float:
        """Return the beta value at the given training step."""
        ...

    def __call__(self, step: int, total_steps: int) -> float:
        return self.get_beta(step, total_steps)

    def get_trajectory(self, total_steps: int) -> np.ndarray:
        """Return the full beta trajectory as a numpy array."""
        return np.array([self.get_beta(s, total_steps) for s in range(total_steps)])


class HormeticSigmoid(BetaSchedule):
    """
    β(t) = β_max * sigmoid(steepness * (t − midpoint))

    Normalised so β(0) = 0 and β(1) = β_max exactly.
    Represents gradually increasing compression pressure — the hormetic hypothesis.
    """

    def __init__(
        self,
        beta_max: float = 1.0,
        steepness: float = 10.0,
        midpoint: float = 0.5,
    ) -> None:
        self.beta_max = beta_max
        self.steepness = steepness
        self.midpoint = midpoint

        # Pre-compute normalisation constants so the endpoints are exact.
        self._low = 1.0 / (1.0 + math.exp(-steepness * (0.0 - midpoint)))
        self._high = 1.0 / (1.0 + math.exp(-steepness * (1.0 - midpoint)))
        self._range = self._high - self._low

    def get_beta(self, step: int, total_steps: int) -> float:
        t = step / max(total_steps - 1, 1)
        raw = 1.0 / (1.0 + math.exp(-self.steepness * (t - self.midpoint)))
        normalised = (raw - self._low) / self._range
        return float(self.beta_max * normalised)


class HormeticCosine(BetaSchedule):
    """
    β(t) = β_max * 0.5 * (1 − cos(π·t))

    Smooth increasing ramp: starts at 0, ends at β_max, with gentle curvature
    at both ends.
    """

    def __init__(self, beta_max: float = 1.0) -> None:
        self.beta_max = beta_max

    def get_beta(self, step: int, total_steps: int) -> float:
        t = step / max(total_steps - 1, 1)
        return float(self.beta_max * 0.5 * (1.0 - math.cos(math.pi * t)))


class Linear(BetaSchedule):
    """
    β(t) = β_min + (β_max − β_min) · t

    Monotone linear ramp, the simplest increasing schedule.
    """

    def __init__(self, beta_max: float = 1.0, beta_min: float = 0.0) -> None:
        self.beta_max = beta_max
        self.beta_min = beta_min

    def get_beta(self, step: int, total_steps: int) -> float:
        t = step / max(total_steps - 1, 1)
        return float(self.beta_min + (self.beta_max - self.beta_min) * t)


class Reverse(BetaSchedule):
    """
    β(t) = β_max − (β_max − β_min) · t

    Monotone linear *decrease*: starts at full compression, ends at minimum.
    If hormetic scheduling works, this should perform worse than the increasing
    schedules.
    """

    def __init__(self, beta_max: float = 1.0, beta_min: float = 0.0) -> None:
        self.beta_max = beta_max
        self.beta_min = beta_min

    def get_beta(self, step: int, total_steps: int) -> float:
        t = step / max(total_steps - 1, 1)
        return float(self.beta_max - (self.beta_max - self.beta_min) * t)


class RandomPermutation(BetaSchedule):
    """
    Permuted linear schedule.

    Generates the same β values as Linear but in a uniformly random order.
    Isolates whether monotonicity (not just the set of β values seen) matters.
    """

    def __init__(
        self,
        beta_max: float = 1.0,
        beta_min: float = 0.0,
        seed: int = 42,
        num_steps: int = 100_000,
    ) -> None:
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.seed = seed
        # Pre-compute the permuted linear grid.
        rng = np.random.default_rng(seed)
        linear = np.linspace(beta_min, beta_max, num_steps)
        self._values = rng.permutation(linear)

    def get_beta(self, step: int, total_steps: int) -> float:
        # Map step → index in the pre-computed array, clamped to its length.
        idx = min(step, len(self._values) - 1)
        return float(self._values[idx])


class FixedBeta(BetaSchedule):
    """
    β(t) = β_max (constant).

    Standard VIB / β-VAE baseline.  Same final β as the hormetic schedules,
    applied from the first step.
    """

    def __init__(self, beta_max: float = 1.0) -> None:
        self.beta_max = beta_max

    def get_beta(self, step: int, total_steps: int) -> float:
        return float(self.beta_max)


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

SCHEDULE_REGISTRY: dict = {
    "hormetic_sigmoid": HormeticSigmoid,
    "hormetic_cosine": HormeticCosine,
    "linear": Linear,
    "reverse": Reverse,
    "random_permutation": RandomPermutation,
    "fixed_beta": FixedBeta,
    "fixed": FixedBeta,  # alias
}


def make_schedule(name: str, **kwargs) -> BetaSchedule:
    """
    Instantiate a BetaSchedule by name.  Extra kwargs are silently ignored so
    callers can pass a superset of parameters without per-schedule branching.

    Example::
        schedule = make_schedule("hormetic_sigmoid", beta_max=1.0, steepness=12.0)
    """
    import inspect
    if name not in SCHEDULE_REGISTRY:
        raise ValueError(
            f"Unknown schedule '{name}'. Available: {list(SCHEDULE_REGISTRY.keys())}"
        )
    cls = SCHEDULE_REGISTRY[name]
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(**filtered)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

_SCHEDULE_PLOT_CONFIGS = [
    ("hormetic_sigmoid", {"beta_max": 1.0}, "Hormetic-Sigmoid", "#e41a1c"),
    ("hormetic_cosine", {"beta_max": 1.0}, "Hormetic-Cosine", "#377eb8"),
    ("linear", {"beta_max": 1.0}, "Linear", "#4daf4a"),
    ("reverse", {"beta_max": 1.0}, "Reverse", "#ff7f00"),
    ("random_permutation", {"beta_max": 1.0, "num_steps": 1000}, "Random Permutation", "#984ea3"),
    ("fixed_beta", {"beta_max": 1.0}, "Fixed-β", "#a65628"),
]


def plot_schedules(
    total_steps: int = 1000,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot all six β schedules on a single figure.

    Args:
        total_steps: Number of training steps to simulate.
        save_path: If provided, save the figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    steps = np.arange(total_steps)
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, kwargs, label, color in _SCHEDULE_PLOT_CONFIGS:
        schedule = make_schedule(name, **kwargs)
        betas = np.array([schedule.get_beta(s, total_steps) for s in steps])
        ax.plot(steps / total_steps, betas, label=label, color=color, linewidth=1.8)

    ax.set_xlabel("Training progress t = step / total_steps", fontsize=12)
    ax.set_ylabel("β(t)", fontsize=12)
    ax.set_title("Beta Schedules for Hormetic IB Training", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.08)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig
