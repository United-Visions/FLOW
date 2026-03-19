"""Temperature schedule for the Annealing Engine.

Implements the physics-inspired cooling schedule:

    T(t) = T₀ · e^(-λt) + T_floor

Where:
    T₀      = initial temperature   (high flexibility, exploratory)
    λ        = cooling rate          (how fast the system stiffens)
    T_floor  = minimum temperature  (system never fully freezes)
    t        = continuous time       (advances via step())

Properties:
    - Early:  High T → geometry is highly flexible, coarse structure forms
    - Later:  Low T  → geometry stiffens where experience is dense
    - Always: T_floor > 0 → unknown territory stays flexible forever
"""

from __future__ import annotations

import math


class TemperatureSchedule:
    """Exponential cooling schedule with a guaranteed minimum floor.

    Parameters
    ----------
    T0 : float
        Initial temperature.  Typical value: 1.0.
    lambda_ : float
        Cooling rate (decay constant).  Larger = cools faster.  Default 0.01.
    T_floor : float
        Minimum temperature; system never falls below this.  Default 0.05.
    dt : float
        Time increment per call to step().  Default 1.0.
    """

    def __init__(
        self,
        T0: float = 1.0,
        lambda_: float = 0.01,
        T_floor: float = 0.05,
        dt: float = 1.0,
    ) -> None:
        if T0 <= 0:
            raise ValueError(f"T0 must be positive, got {T0}")
        if lambda_ < 0:
            raise ValueError(f"lambda_ must be ≥ 0, got {lambda_}")
        if T_floor < 0:
            raise ValueError(f"T_floor must be ≥ 0, got {T_floor}")
        if T_floor >= T0:
            raise ValueError(
                f"T_floor ({T_floor}) must be strictly less than T0 ({T0})"
            )
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.T0 = T0
        self.lambda_ = lambda_
        self.T_floor = T_floor
        self.dt = dt
        self._t: float = 0.0

    # ------------------------------------------------------------------ #
    # Core interface                                                       #
    # ------------------------------------------------------------------ #

    def temperature(self, t: float | None = None) -> float:
        """Return T at time *t*.  If *t* is None, use the internal clock.

        T(t) = T₀ · e^(-λt) + T_floor
        """
        if t is None:
            t = self._t
        return self.T0 * math.exp(-self.lambda_ * t) + self.T_floor

    def step(self) -> float:
        """Advance the internal clock by dt and return the new temperature."""
        self._t += self.dt
        return self.temperature()

    def reset(self) -> None:
        """Reset the internal clock to t=0 (returns to initial temperature)."""
        self._t = 0.0

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def t(self) -> float:
        """Current internal time."""
        return self._t

    @property
    def current_temperature(self) -> float:
        """Temperature at the current internal time."""
        return self.temperature(self._t)

    @property
    def initial_temperature(self) -> float:
        """Temperature at t=0: T₀ + T_floor."""
        return self.T0 + self.T_floor

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def is_cold(self, threshold: float = 0.1) -> bool:
        """Return True when current temperature is within threshold of T_floor."""
        return (self.current_temperature - self.T_floor) < threshold

    def locality_radius(
        self,
        base_radius: float,
        t: float | None = None,
    ) -> float:
        """Derive the deformation radius from the current temperature.

        Higher temperature → wider deformation radius.
        Lower  temperature → narrower, more local deformation.

        r(t) = base_radius · T(t) / (T₀ + T_floor)
        """
        T = self.temperature(t)
        T_max = self.T0 + self.T_floor
        return base_radius * (T / T_max)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TemperatureSchedule("
            f"T0={self.T0}, λ={self.lambda_}, "
            f"T_floor={self.T_floor}, t={self._t:.2f}, "
            f"T_current={self.current_temperature:.4f})"
        )
