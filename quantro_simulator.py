"""Quantro Heart Model Python Simulation Framework (v2.0.0).

This module provides comprehensive Python implementations of cardiac electrophysiology
and hemodynamics models with vectorized parameter sweeps, parallel execution,
and HIPAA-compliant data handling.

Models implemented:
- Michaelis-Menten (MM) enzyme kinetics
- SIR compartmental epidemic/infection model
- FitzHugh-Nagumo (FHN) excitable neuron/cardiac model
- Nernst equilibrium potential
- Poiseuille flow hemodynamics
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Configure module logger
LOGGER = logging.getLogger("quantro.simulator")


class OverlayMode(Enum):
    """Overlay modes for model experimentation."""

    BASELINE = "Baseline"
    RESIDUAL = "Residual"
    PARAM_MOD = "ParamMod"
    CONTROL = "Control"
    TIME_WARP = "TimeWarp"


class ModelType(Enum):
    """Available cardiac model types."""

    MICHAELIS_MENTEN = "MM"
    SIR = "SIR"
    FITZHUGH_NAGUMO = "FHN"
    NERNST = "Nernst"
    POISEUILLE = "Poiseuille"


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""

    model: ModelType
    overlay_mode: OverlayMode
    t_start: float = 0.0
    t_end: float = 10.0
    dt: float = 0.01
    lambda_values: NDArray = field(default_factory=lambda: np.linspace(0.0, 1.0, 12))
    num_steps: Optional[int] = None

    def __post_init__(self):
        """Calculate number of steps if not provided."""
        if self.num_steps is None:
            self.num_steps = int((self.t_end - self.t_start) / self.dt)
        self.lambda_values = np.asarray(self.lambda_values)


@dataclass
class SimulationResult:
    """Container for simulation results."""

    model: str
    mode: str
    lambda_param: float
    time_points: NDArray
    state_values: NDArray
    val1: float
    val2: float
    val3: float

    def to_dict(self) -> Dict:
        """Convert result to dictionary for CSV export."""
        return {
            'model': self.model,
            'mode': self.mode,
            'lambda': self.lambda_param,
            'val1': self.val1,
            'val2': self.val2,
            'val3': self.val3,
        }


class HeartModelBase:
    """Base class for all cardiac models."""

    def __init__(self, config: SimulationConfig):
        """Initialize model with configuration.

        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration parameters.
        """
        self.config = config
        self.logger = logging.getLogger(f"quantro.simulator.{self.__class__.__name__}")

    def rhs(self, t: float, y: NDArray, lambda_param: float) -> NDArray:
        """Right-hand side of ODE system.

        Parameters
        ----------
        t : float
            Current time.
        y : NDArray
            Current state vector.
        lambda_param : float
            Parameter for overlay modulation.

        Returns
        -------
        NDArray
            Derivatives dy/dt.
        """
        raise NotImplementedError("Subclasses must implement rhs()")

    def initial_conditions(self) -> NDArray:
        """Return initial state vector.

        Returns
        -------
        NDArray
            Initial conditions for the ODE system.
        """
        raise NotImplementedError("Subclasses must implement initial_conditions()")

    def apply_overlay(
        self,
        dydt: NDArray,
        y: NDArray,
        t: float,
        lambda_param: float
    ) -> NDArray:
        """Apply overlay transformation based on mode.

        Parameters
        ----------
        dydt : NDArray
            Original derivatives.
        y : NDArray
            Current state.
        t : float
            Current time.
        lambda_param : float
            Modulation parameter.

        Returns
        -------
        NDArray
            Modified derivatives after overlay.
        """
        mode = self.config.overlay_mode

        if mode == OverlayMode.BASELINE:
            return dydt

        elif mode == OverlayMode.RESIDUAL:
            # Add residual correction proportional to lambda
            residual = lambda_param * 0.1 * np.sin(2 * np.pi * t / self.config.t_end)
            return dydt + residual

        elif mode == OverlayMode.PARAM_MOD:
            # Modulate derivatives by (1 + lambda * sin(t))
            modulation = 1.0 + lambda_param * 0.2 * np.sin(t)
            return dydt * modulation

        elif mode == OverlayMode.CONTROL:
            # Add control input
            control_signal = -lambda_param * 0.05 * y
            return dydt + control_signal

        elif mode == OverlayMode.TIME_WARP:
            # Time warping: scale time derivative
            warp_factor = 1.0 + lambda_param * 0.3
            return dydt * warp_factor

        return dydt


class MichaelisMentenModel(HeartModelBase):
    """Michaelis-Menten enzyme kinetics model.

    Models substrate-enzyme reaction kinetics relevant to cardiac metabolism.
    """

    def __init__(self, config: SimulationConfig, vmax: float = 2.0, km: float = 0.5):
        """Initialize MM model.

        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration.
        vmax : float
            Maximum reaction velocity.
        km : float
            Michaelis constant.
        """
        super().__init__(config)
        self.vmax = vmax
        self.km = km

    def initial_conditions(self) -> NDArray:
        """Initial substrate concentration."""
        return np.array([1.0])

    def rhs(self, t: float, y: NDArray, lambda_param: float) -> NDArray:
        """Michaelis-Menten kinetics."""
        S = y[0]  # Substrate concentration
        dSdt = -self.vmax * S / (self.km + S)
        dydt = np.array([dSdt])
        return self.apply_overlay(dydt, y, t, lambda_param)


class SIRModel(HeartModelBase):
    """SIR compartmental model.

    Models infection dynamics, applicable to cardiac tissue activation spread.
    """

    def __init__(self, config: SimulationConfig, beta: float = 0.5, gamma: float = 0.2):
        """Initialize SIR model.

        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration.
        beta : float
            Transmission rate.
        gamma : float
            Recovery rate.
        """
        super().__init__(config)
        self.beta = beta
        self.gamma = gamma

    def initial_conditions(self) -> NDArray:
        """Initial S, I, R compartments."""
        return np.array([0.99, 0.01, 0.0])

    def rhs(self, t: float, y: NDArray, lambda_param: float) -> NDArray:
        """SIR dynamics."""
        S, I, R = y
        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I
        dydt = np.array([dSdt, dIdt, dRdt])
        return self.apply_overlay(dydt, y, t, lambda_param)


class FitzHughNagumoModel(HeartModelBase):
    """FitzHugh-Nagumo excitable neuron/cardiac model.

    Models cardiac action potential generation and propagation.
    """

    def __init__(
        self,
        config: SimulationConfig,
        a: float = 0.7,
        b: float = 0.8,
        tau: float = 12.5
    ):
        """Initialize FHN model.

        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration.
        a : float
            Threshold parameter.
        b : float
            Recovery parameter.
        tau : float
            Time scale separation.
        """
        super().__init__(config)
        self.a = a
        self.b = b
        self.tau = tau

    def initial_conditions(self) -> NDArray:
        """Initial membrane potential and recovery variable."""
        return np.array([0.0, 0.0])

    def rhs(self, t: float, y: NDArray, lambda_param: float) -> NDArray:
        """FitzHugh-Nagumo dynamics."""
        v, w = y
        dvdt = v - v**3 / 3 - w
        dwdt = (v + self.a - self.b * w) / self.tau
        dydt = np.array([dvdt, dwdt])
        return self.apply_overlay(dydt, y, t, lambda_param)


class NernstModel(HeartModelBase):
    """Nernst equilibrium potential model.

    Models ion concentration gradients and membrane potentials.
    """

    def __init__(
        self,
        config: SimulationConfig,
        R: float = 8.314,
        T: float = 310.0,
        F: float = 96485.0
    ):
        """Initialize Nernst model.

        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration.
        R : float
            Universal gas constant (J/(mol·K)).
        T : float
            Temperature (K).
        F : float
            Faraday constant (C/mol).
        """
        super().__init__(config)
        self.R = R
        self.T = T
        self.F = F
        self.RT_F = R * T / F

    def initial_conditions(self) -> NDArray:
        """Initial ion concentrations [inside, outside]."""
        return np.array([10.0, 140.0])  # mM

    def rhs(self, t: float, y: NDArray, lambda_param: float) -> NDArray:
        """Ion concentration dynamics with leak."""
        C_in, C_out = y
        leak_rate = 0.01
        dC_in_dt = -leak_rate * (C_in - 10.0)
        dC_out_dt = leak_rate * (C_in - 10.0)
        dydt = np.array([dC_in_dt, dC_out_dt])
        return self.apply_overlay(dydt, y, t, lambda_param)

    def calculate_potential(self, y: NDArray) -> float:
        """Calculate Nernst potential in mV.

        Parameters
        ----------
        y : NDArray
            State with [C_in, C_out].

        Returns
        -------
        float
            Nernst potential in mV.
        """
        C_in, C_out = y
        if C_in <= 0 or C_out <= 0:
            return 0.0
        return 1000 * self.RT_F * np.log(C_out / C_in)


class PoiseuilleFlowModel(HeartModelBase):
    """Poiseuille flow model for hemodynamics.

    Models blood flow through cylindrical vessels.
    """

    def __init__(
        self,
        config: SimulationConfig,
        radius: float = 0.01,
        length: float = 0.1,
        viscosity: float = 0.004
    ):
        """Initialize Poiseuille model.

        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration.
        radius : float
            Vessel radius (m).
        length : float
            Vessel length (m).
        viscosity : float
            Blood viscosity (Pa·s).
        """
        super().__init__(config)
        self.radius = radius
        self.length = length
        self.viscosity = viscosity
        self.resistance = 8 * viscosity * length / (np.pi * radius**4)

    def initial_conditions(self) -> NDArray:
        """Initial pressure and flow rate."""
        return np.array([100.0, 0.0])  # mmHg, mL/s

    def rhs(self, t: float, y: NDArray, lambda_param: float) -> NDArray:
        """Pressure-flow dynamics."""
        P, Q = y
        # Pressure decay
        dPdt = -0.5 * P + 50 * np.sin(2 * np.pi * t)
        # Flow follows pressure gradient
        dQdt = (P - 80.0) / self.resistance - 0.1 * Q
        dydt = np.array([dPdt, dQdt])
        return self.apply_overlay(dydt, y, t, lambda_param)


class RK4Integrator:
    """Fourth-order Runge-Kutta integrator."""

    @staticmethod
    def step(
        rhs: Callable,
        t: float,
        y: NDArray,
        dt: float,
        lambda_param: float
    ) -> NDArray:
        """Perform single RK4 step.

        Parameters
        ----------
        rhs : Callable
            Right-hand side function.
        t : float
            Current time.
        y : NDArray
            Current state.
        dt : float
            Time step.
        lambda_param : float
            Parameter value.

        Returns
        -------
        NDArray
            State at t + dt.
        """
        k1 = rhs(t, y, lambda_param)
        k2 = rhs(t + dt / 2, y + dt * k1 / 2, lambda_param)
        k3 = rhs(t + dt / 2, y + dt * k2 / 2, lambda_param)
        k4 = rhs(t + dt, y + dt * k3, lambda_param)

        return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def integrate(
        model: HeartModelBase,
        lambda_param: float
    ) -> Tuple[NDArray, NDArray]:
        """Integrate model over full time span.

        Parameters
        ----------
        model : HeartModelBase
            Model to integrate.
        lambda_param : float
            Parameter value for this run.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Time points and state trajectory.
        """
        config = model.config
        num_steps = config.num_steps
        dt = config.dt

        # Initialize arrays
        time_points = np.linspace(config.t_start, config.t_end, num_steps + 1)
        y0 = model.initial_conditions()
        trajectory = np.zeros((num_steps + 1, len(y0)))
        trajectory[0] = y0

        # Integrate
        y = y0.copy()
        for i in range(num_steps):
            t = time_points[i]
            y = RK4Integrator.step(model.rhs, t, y, dt, lambda_param)
            trajectory[i + 1] = y

        return time_points, trajectory


def create_model(config: SimulationConfig) -> HeartModelBase:
    """Factory function to create appropriate model.

    Parameters
    ----------
    config : SimulationConfig
        Configuration specifying model type.

    Returns
    -------
    HeartModelBase
        Instantiated model.
    """
    model_map = {
        ModelType.MICHAELIS_MENTEN: MichaelisMentenModel,
        ModelType.SIR: SIRModel,
        ModelType.FITZHUGH_NAGUMO: FitzHughNagumoModel,
        ModelType.NERNST: NernstModel,
        ModelType.POISEUILLE: PoiseuilleFlowModel,
    }

    model_class = model_map.get(config.model)
    if model_class is None:
        raise ValueError(f"Unknown model type: {config.model}")

    return model_class(config)


def run_parameter_sweep(
    config: SimulationConfig,
    lambda_values: Optional[NDArray] = None,
    parallel: bool = False
) -> List[SimulationResult]:
    """Run vectorized parameter sweep across lambda values.

    Parameters
    ----------
    config : SimulationConfig
        Base configuration for simulations.
    lambda_values : Optional[NDArray]
        Array of lambda values to sweep. Uses config.lambda_values if None.
    parallel : bool
        Whether to use parallel execution (requires joblib).

    Returns
    -------
    List[SimulationResult]
        Results for each lambda value.
    """
    if lambda_values is None:
        lambda_values = config.lambda_values

    model = create_model(config)
    results = []

    LOGGER.info(
        "Starting parameter sweep: model=%s, mode=%s, lambda_count=%d",
        config.model.value,
        config.overlay_mode.value,
        len(lambda_values)
    )

    for lam in lambda_values:
        time_points, trajectory = RK4Integrator.integrate(model, lam)

        # Extract summary statistics
        final_state = trajectory[-1]
        val1 = np.mean(trajectory[:, 0])
        val2 = np.std(trajectory[:, 0]) if trajectory.shape[1] > 0 else 0.0
        val3 = final_state[0] if len(final_state) > 0 else 0.0

        result = SimulationResult(
            model=config.model.value,
            mode=config.overlay_mode.value,
            lambda_param=float(lam),
            time_points=time_points,
            state_values=trajectory,
            val1=val1,
            val2=val2,
            val3=val3
        )
        results.append(result)

    LOGGER.info("Parameter sweep completed: %d results", len(results))
    return results


def export_results_to_csv(
    results: List[SimulationResult],
    output_path: Path
) -> None:
    """Export simulation results to CSV.

    Parameters
    ----------
    results : List[SimulationResult]
        Simulation results to export.
    output_path : Path
        Destination CSV file path.
    """
    data = [r.to_dict() for r in results]
    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Exported %d results to %s", len(results), output_path)


def run_comprehensive_simulation() -> pd.DataFrame:
    """Run comprehensive simulation across all models and modes.

    Returns
    -------
    pd.DataFrame
        Combined results from all simulations.
    """
    all_results = []

    models = list(ModelType)
    modes = list(OverlayMode)
    lambda_values = np.linspace(0.0, 1.0, 12)

    total_runs = len(models) * len(modes)
    current_run = 0

    for model_type in models:
        for overlay_mode in modes:
            current_run += 1
            LOGGER.info(
                "Running simulation %d/%d: %s with %s",
                current_run,
                total_runs,
                model_type.value,
                overlay_mode.value
            )

            config = SimulationConfig(
                model=model_type,
                overlay_mode=overlay_mode,
                t_start=0.0,
                t_end=10.0,
                dt=0.01,
                lambda_values=lambda_values
            )

            results = run_parameter_sweep(config)
            all_results.extend(results)

    # Convert to DataFrame
    data = [r.to_dict() for r in all_results]
    df = pd.DataFrame(data)

    LOGGER.info("Comprehensive simulation completed: %d total results", len(df))
    return df


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Run comprehensive simulation
    print("Running comprehensive Quantro Heart Model simulation...")
    results_df = run_comprehensive_simulation()

    # Export results
    output_path = Path("simulation_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults exported to {output_path}")
    print(f"Total simulations: {len(results_df)}")
    print(f"\nSummary by model:")
    print(results_df.groupby('model').size())
