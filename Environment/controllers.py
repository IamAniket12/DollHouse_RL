"""
Thermal Control Controllers Module.

This module provides various control strategies for the dollhouse thermal environment,
including rule-based, PID, and fuzzy logic controllers for comparative analysis.
"""

import argparse
import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from dollhouse_env import DollhouseThermalEnv
from train_sindy_model import train_sindy_model


class BaseController(ABC):
    """
    Abstract base class for thermal controllers.

    Defines the interface that all thermal controllers must implement
    for consistent evaluation and comparison.
    """

    def __init__(self, name: str):
        """
        Initialize the base controller.

        Args:
            name: Human-readable name for the controller
        """
        self.name = name
        self.reset()

    @abstractmethod
    def control(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute control actions based on current observation.

        Args:
            observation: Current environment observation

        Returns:
            Action array [ground_light, ground_window, top_light, top_window]
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset controller internal state for new episode."""
        pass

    def get_parameters(self) -> Dict:
        """
        Get controller parameters for logging and reproducibility.

        Returns:
            Dictionary of controller parameters
        """
        return {"name": self.name}


class RuleBasedController(BaseController):
    """
    Simple rule-based thermal controller with hysteresis.

    Uses temperature thresholds with hysteresis to prevent oscillation
    and make basic heating/cooling decisions for each floor.
    """

    def __init__(self, hysteresis: float = 0.5):
        """
        Initialize rule-based controller.

        Args:
            hysteresis: Temperature buffer to prevent control oscillation
        """
        self.hysteresis = hysteresis
        super().__init__(f"Rule-Based (hysteresis={hysteresis})")

    def control(self, observation: np.ndarray) -> np.ndarray:
        """
        Implement rule-based control logic.

        Args:
            observation: Environment observation array

        Returns:
            Binary control actions for lights and windows
        """
        ground_temp = observation[0]
        top_temp = observation[1]
        heating_setpoint = observation[7]
        cooling_setpoint = observation[8]

        avg_setpoint = (heating_setpoint + cooling_setpoint) / 2
        action = np.zeros(4, dtype=int)

        # Ground floor control
        if ground_temp < avg_setpoint - self.hysteresis:
            action[0] = 1  # Turn on light for heating
            action[1] = 0  # Close window
        else:
            action[0] = 0  # Turn off light
            action[1] = 1  # Open window for cooling

        # Top floor control (same logic)
        if top_temp < avg_setpoint - self.hysteresis:
            action[2] = 1  # Turn on light for heating
            action[3] = 0  # Close window
        else:
            action[2] = 0  # Turn off light
            action[3] = 1  # Open window for cooling

        return action

    def reset(self) -> None:
        """Reset controller state (no internal state for rule-based)."""
        pass

    def get_parameters(self) -> Dict:
        """Get controller parameters."""
        return {
            **super().get_parameters(),
            "hysteresis": self.hysteresis,
            "type": "rule_based",
        }


class PIDController(BaseController):
    """
    PID thermal controller with separate controllers for each floor.

    Implements classical PID control with proportional, integral, and
    derivative terms for precise temperature regulation.
    """

    def __init__(
        self,
        ground_kp: float = 2.0,
        ground_ki: float = 0.1,
        ground_kd: float = 0.05,
        top_kp: float = 2.0,
        top_ki: float = 0.1,
        top_kd: float = 0.05,
        sample_time: float = 30.0,
        output_limits: Tuple[float, float] = (-1.0, 1.0),
    ):
        """
        Initialize PID controller.

        Args:
            ground_kp: Proportional gain for ground floor
            ground_ki: Integral gain for ground floor
            ground_kd: Derivative gain for ground floor
            top_kp: Proportional gain for top floor
            top_ki: Integral gain for top floor
            top_kd: Derivative gain for top floor
            sample_time: Control loop sample time in seconds
            output_limits: Min and max output values
        """
        self.ground_params = {"kp": ground_kp, "ki": ground_ki, "kd": ground_kd}
        self.top_params = {"kp": top_kp, "ki": top_ki, "kd": top_kd}
        self.sample_time = sample_time
        self.output_limits = output_limits

        super().__init__(f"PID (Kp={ground_kp}, Ki={ground_ki}, Kd={ground_kd})")

    def control(self, observation: np.ndarray) -> np.ndarray:
        """
        Implement PID control logic.

        Args:
            observation: Environment observation array

        Returns:
            Binary control actions based on PID outputs
        """
        ground_temp = observation[0]
        top_temp = observation[1]
        heating_setpoint = observation[7]
        cooling_setpoint = observation[8]

        target_temp = (heating_setpoint + cooling_setpoint) / 2
        current_time = self.step_count * self.sample_time
        self.step_count += 1

        ground_output = self._compute_pid_output(
            target_temp,
            ground_temp,
            current_time,
            self.ground_state,
            self.ground_params,
        )
        top_output = self._compute_pid_output(
            target_temp, top_temp, current_time, self.top_state, self.top_params
        )

        return self._pid_outputs_to_actions(ground_output, top_output)

    def _compute_pid_output(
        self,
        setpoint: float,
        current_value: float,
        current_time: float,
        state: Dict,
        params: Dict,
    ) -> float:
        """Compute PID output for a single control loop."""
        error = setpoint - current_value

        dt = (
            current_time - state["last_time"]
            if state["last_time"] is not None
            else self.sample_time
        )
        if dt <= 0:
            dt = self.sample_time

        # Proportional term
        proportional = params["kp"] * error

        # Integral term
        state["integral"] += error * dt
        integral = params["ki"] * state["integral"]

        # Derivative term
        derivative = params["kd"] * (error - state["previous_error"]) / dt

        # Calculate output
        output = proportional + integral + derivative
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Update state
        state["previous_error"] = error
        state["last_time"] = current_time

        return output

    def _pid_outputs_to_actions(
        self, ground_output: float, top_output: float
    ) -> np.ndarray:
        """Convert PID outputs to binary actions."""
        action = np.zeros(4, dtype=int)

        # Ground floor actions
        if ground_output > 0.1:  # Need heating
            action[0] = 1  # Turn on light
            action[1] = 0  # Close window
        elif ground_output < -0.1:  # Need cooling
            action[0] = 0  # Turn off light
            action[1] = 1  # Open window
        else:  # Maintain
            action[0] = 0
            action[1] = 0

        # Top floor actions
        if top_output > 0.1:  # Need heating
            action[2] = 1  # Turn on light
            action[3] = 0  # Close window
        elif top_output < -0.1:  # Need cooling
            action[2] = 0  # Turn off light
            action[3] = 1  # Open window
        else:  # Maintain
            action[2] = 0
            action[3] = 0

        return action

    def reset(self) -> None:
        """Reset PID controller state."""
        self.ground_state = {"previous_error": 0.0, "integral": 0.0, "last_time": None}
        self.top_state = {"previous_error": 0.0, "integral": 0.0, "last_time": None}
        self.step_count = 0

    def get_parameters(self) -> Dict:
        """Get controller parameters."""
        return {
            **super().get_parameters(),
            "ground_params": self.ground_params,
            "top_params": self.top_params,
            "sample_time": self.sample_time,
            "output_limits": self.output_limits,
            "type": "pid",
        }


class FuzzyLogicController(BaseController):
    """
    Fuzzy logic thermal controller with linguistic rules.

    Uses fuzzy sets and linguistic rules to make control decisions
    based on temperature errors and rates of change.
    """

    def __init__(self, heating_threshold: float = 0.2, cooling_threshold: float = 0.2):
        """
        Initialize fuzzy logic controller.

        Args:
            heating_threshold: Threshold for activating heating actions
            cooling_threshold: Threshold for activating cooling actions
        """
        self.heating_threshold = heating_threshold
        self.cooling_threshold = cooling_threshold

        self._setup_membership_functions()
        self._setup_control_rules()

        super().__init__(
            f"Fuzzy Logic (heat_th={heating_threshold}, cool_th={cooling_threshold})"
        )

    def _setup_membership_functions(self) -> None:
        """Define fuzzy membership functions."""
        self.temp_error_sets = {
            "very_cold": (2.0, 4.0, 6.0),
            "cold": (0.5, 1.5, 3.0),
            "ok": (-1.0, 0.0, 1.0),
            "warm": (-3.0, -1.5, -0.5),
            "very_warm": (-6.0, -4.0, -2.0),
        }

        self.temp_rate_sets = {
            "falling_fast": (-1.0, -0.3, -0.1),
            "falling": (-0.2, -0.1, 0.0),
            "stable": (-0.05, 0.0, 0.05),
            "rising": (0.0, 0.1, 0.2),
            "rising_fast": (0.1, 0.3, 1.0),
        }

    def _setup_control_rules(self) -> None:
        """Define fuzzy control rules."""
        self.heating_rules = [
            ("very_cold", "any", 1.0),
            ("very_cold", "falling", 1.0),
            ("very_cold", "rising", 0.8),
            ("cold", "any", 0.8),
            ("cold", "falling", 0.9),
            ("cold", "stable", 0.7),
            ("cold", "rising", 0.5),
            ("ok", "falling_fast", 0.6),
            ("ok", "falling", 0.4),
            ("warm", "falling_fast", 0.3),
        ]

        self.cooling_rules = [
            ("very_warm", "any", 1.0),
            ("very_warm", "rising", 1.0),
            ("very_warm", "falling", 0.8),
            ("warm", "any", 0.8),
            ("warm", "rising", 0.9),
            ("warm", "stable", 0.7),
            ("warm", "falling", 0.5),
            ("ok", "rising_fast", 0.6),
            ("ok", "rising", 0.4),
            ("cold", "rising_fast", 0.3),
        ]

    def control(self, observation: np.ndarray) -> np.ndarray:
        """
        Implement fuzzy logic control.

        Args:
            observation: Environment observation array

        Returns:
            Binary control actions based on fuzzy inference
        """
        ground_temp = observation[0]
        top_temp = observation[1]
        heating_setpoint = observation[7]
        cooling_setpoint = observation[8]

        target_temp = (heating_setpoint + cooling_setpoint) / 2

        ground_error = target_temp - ground_temp
        top_error = target_temp - top_temp

        if hasattr(self, "previous_temps"):
            ground_rate = ground_temp - self.previous_temps["ground"]
            top_rate = top_temp - self.previous_temps["top"]
        else:
            ground_rate = 0.0
            top_rate = 0.0

        self.previous_temps = {"ground": ground_temp, "top": top_temp}

        ground_heat, ground_cool = self._evaluate_fuzzy_rules(ground_error, ground_rate)
        top_heat, top_cool = self._evaluate_fuzzy_rules(top_error, top_rate)

        return self._fuzzy_outputs_to_actions(
            ground_heat, ground_cool, top_heat, top_cool
        )

    def _triangular_membership(
        self, x: float, left: float, center: float, right: float
    ) -> float:
        """Calculate triangular membership function value."""
        if x <= left or x >= right:
            return 0.0
        elif x == center:
            return 1.0
        elif x < center:
            return max(0.0, (x - left) / (center - left))
        else:
            return max(0.0, (right - x) / (right - center))

    def _get_memberships(self, value: float, fuzzy_sets: Dict) -> Dict:
        """Get membership values for all fuzzy sets."""
        memberships = {}
        for name, (left, center, right) in fuzzy_sets.items():
            memberships[name] = self._triangular_membership(value, left, center, right)
        return memberships

    def _evaluate_fuzzy_rules(
        self, temp_error: float, temp_rate: float
    ) -> Tuple[float, float]:
        """Evaluate fuzzy rules and return heating and cooling strengths."""
        temp_memberships = self._get_memberships(temp_error, self.temp_error_sets)
        rate_memberships = self._get_memberships(temp_rate, self.temp_rate_sets)

        heating_strength = 0.0
        cooling_strength = 0.0

        for temp_cond, rate_cond, strength in self.heating_rules:
            temp_membership = temp_memberships[temp_cond]
            rate_membership = 1.0 if rate_cond == "any" else rate_memberships[rate_cond]

            rule_activation = min(temp_membership, rate_membership) * strength
            heating_strength = max(heating_strength, rule_activation)

        for temp_cond, rate_cond, strength in self.cooling_rules:
            temp_membership = temp_memberships[temp_cond]
            rate_membership = 1.0 if rate_cond == "any" else rate_memberships[rate_cond]

            rule_activation = min(temp_membership, rate_membership) * strength
            cooling_strength = max(cooling_strength, rule_activation)

        return heating_strength, cooling_strength

    def _fuzzy_outputs_to_actions(
        self, ground_heat: float, ground_cool: float, top_heat: float, top_cool: float
    ) -> np.ndarray:
        """Convert fuzzy outputs to binary actions."""
        action = np.zeros(4, dtype=int)

        # Ground floor control
        if ground_heat > self.heating_threshold and ground_heat > ground_cool:
            action[0] = 1  # Turn on light
            action[1] = 0  # Close window
        elif ground_cool > self.cooling_threshold and ground_cool > ground_heat:
            action[0] = 0  # Turn off light
            action[1] = 1  # Open window
        else:
            action[0] = 0  # Energy saving mode
            action[1] = 0

        # Top floor control
        if top_heat > self.heating_threshold and top_heat > top_cool:
            action[2] = 1  # Turn on light
            action[3] = 0  # Close window
        elif top_cool > self.cooling_threshold and top_cool > top_heat:
            action[2] = 0  # Turn off light
            action[3] = 1  # Open window
        else:
            action[2] = 0  # Energy saving mode
            action[3] = 0

        return action

    def reset(self) -> None:
        """Reset fuzzy controller state."""
        if hasattr(self, "previous_temps"):
            delattr(self, "previous_temps")

    def get_parameters(self) -> Dict:
        """Get controller parameters."""
        return {
            **super().get_parameters(),
            "heating_threshold": self.heating_threshold,
            "cooling_threshold": self.cooling_threshold,
            "type": "fuzzy_logic",
        }


class ControllerEvaluator:
    """
    Evaluator for comparing different thermal controllers.

    Provides standardized evaluation, visualization, and comparison
    capabilities for different control strategies.
    """

    def __init__(self, env: DollhouseThermalEnv):
        """
        Initialize controller evaluator.

        Args:
            env: Dollhouse thermal environment for evaluation
        """
        self.env = env
        self.results = {}

    def evaluate_controller(
        self,
        controller: BaseController,
        num_episodes: int = 5,
        render: bool = False,
        verbose: bool = True,
    ) -> Dict:
        """
        Evaluate a controller on the environment.

        Args:
            controller: Controller instance to evaluate
            num_episodes: Number of evaluation episodes
            render: Whether to render during evaluation
            verbose: Whether to print progress information

        Returns:
            Dictionary containing evaluation results
        """
        if hasattr(self.env, "episode_history"):
            self.env.episode_history = []

        total_rewards = []
        episode_data = {
            "temperatures": [],
            "external_temps": [],
            "actions": [],
            "rewards": [],
            "setpoints": [],
        }

        for episode in range(num_episodes):
            controller.reset()
            episode_result = self._run_single_episode(
                controller, render, verbose, episode, num_episodes
            )

            total_rewards.append(episode_result["total_reward"])
            for key in episode_data:
                episode_data[key].append(episode_result[key])

        performance = self._calculate_performance_metrics(total_rewards, episode_data)
        performance["controller_params"] = controller.get_parameters()

        self.results[controller.name] = performance

        if verbose:
            self._print_evaluation_summary(controller.name, performance)

        return performance

    def _run_single_episode(
        self,
        controller: BaseController,
        render: bool,
        verbose: bool,
        episode: int,
        num_episodes: int,
    ) -> Dict:
        """Run a single evaluation episode."""
        obs, info = self.env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        temps = []
        ext_temps = []
        actions = []
        rewards = []
        setpoints = []

        while not terminated and not truncated:
            if len(obs) > 8:
                heating_sp = obs[7]
                cooling_sp = obs[8]
            else:
                heating_sp = self.env.initial_heating_setpoint
                cooling_sp = self.env.initial_cooling_setpoint

            setpoints.append([heating_sp, cooling_sp])

            action = controller.control(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)

            episode_reward += reward
            temps.append([obs[0], obs[1]])
            ext_temps.append(obs[2])
            actions.append(action)
            rewards.append(reward)

            if render:
                self.env.render()

        if verbose:
            print(
                f"Episode {episode+1}/{num_episodes}: Total Reward = {episode_reward:.2f}"
            )

        return {
            "total_reward": episode_reward,
            "temperatures": temps,
            "external_temps": ext_temps,
            "actions": actions,
            "rewards": rewards,
            "setpoints": setpoints,
        }

    def _calculate_performance_metrics(
        self, total_rewards: List[float], episode_data: Dict
    ) -> Dict:
        """Calculate comprehensive performance metrics."""
        if hasattr(self.env, "get_performance_summary"):
            performance = self.env.get_performance_summary()
        else:
            performance = {
                "avg_total_reward": np.mean(total_rewards),
                "std_total_reward": np.std(total_rewards),
            }

        performance["episode_data"] = episode_data
        performance["num_episodes"] = len(total_rewards)

        if episode_data["setpoints"]:
            first_setpoints = episode_data["setpoints"][0]
            if first_setpoints:
                performance["heating_setpoint"] = first_setpoints[0][0]
                performance["cooling_setpoint"] = first_setpoints[0][1]
                performance["has_dynamic_setpoints"] = True
            else:
                performance["heating_setpoint"] = self.env.initial_heating_setpoint
                performance["cooling_setpoint"] = self.env.initial_cooling_setpoint
                performance["has_dynamic_setpoints"] = False

        return performance

    def _print_evaluation_summary(
        self, controller_name: str, performance: Dict
    ) -> None:
        """Print evaluation summary to console."""
        print(f"\n{controller_name} Evaluation Summary:")
        print(f"Average Total Reward: {performance.get('avg_total_reward', 0):.2f}")
        print(
            f"Ground Floor Comfort %: {performance.get('avg_ground_comfort_pct', 0):.2f}%"
        )
        print(f"Top Floor Comfort %: {performance.get('avg_top_comfort_pct', 0):.2f}%")
        print(f"Average Light Hours: {performance.get('avg_light_hours', 0):.2f}")

    def compare_controllers(self) -> Dict:
        """
        Compare all evaluated controllers.

        Returns:
            Dictionary containing comparative analysis
        """
        if not self.results:
            return {"error": "No controllers have been evaluated yet"}

        comparison = {"controllers": list(self.results.keys()), "metrics": {}}

        metrics_to_compare = [
            "avg_total_reward",
            "avg_ground_comfort_pct",
            "avg_top_comfort_pct",
            "avg_light_hours",
        ]

        for metric in metrics_to_compare:
            comparison["metrics"][metric] = {
                name: result.get(metric, 0) for name, result in self.results.items()
            }

        # Find best performing controller for each metric
        comparison["best_performers"] = {}
        for metric in metrics_to_compare:
            values = comparison["metrics"][metric]
            if metric == "avg_light_hours":  # Lower is better for energy
                best_controller = min(values, key=values.get)
            else:  # Higher is better for rewards and comfort
                best_controller = max(values, key=values.get)
            comparison["best_performers"][metric] = best_controller

        return comparison

    def save_results(
        self, output_dir: str, filename: str = "controller_comparison.json"
    ) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            output_dir: Directory to save results
            filename: Name of the output file
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for controller_name, result in self.results.items():
            serializable_result = {}
            for key, value in result.items():
                if key == "episode_data":
                    serializable_result[key] = {
                        sub_key: (
                            [
                                [
                                    (
                                        [float(x) for x in item]
                                        if isinstance(item, (list, np.ndarray))
                                        else float(item)
                                    )
                                    for item in episode
                                ]
                                for episode in sub_value
                            ]
                            if isinstance(sub_value, list)
                            else sub_value
                        )
                        for sub_key, sub_value in value.items()
                    }
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_result[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value

            serializable_results[controller_name] = serializable_result

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=4)

        print(f"Results saved to {filepath}")

    def plot_comparison(self, output_dir: str) -> None:
        """
        Create comparison plots for all evaluated controllers.

        Args:
            output_dir: Directory to save plots
        """
        if not self.results:
            print("No results to plot")
            return

        os.makedirs(output_dir, exist_ok=True)

        self._plot_performance_comparison(output_dir)
        self._plot_temperature_trajectories(output_dir)
        self._plot_action_patterns(output_dir)

    def _plot_performance_comparison(self, output_dir: str) -> None:
        """Plot performance metrics comparison."""
        controllers = list(self.results.keys())
        metrics = [
            "avg_total_reward",
            "avg_ground_comfort_pct",
            "avg_top_comfort_pct",
            "avg_light_hours",
        ]
        metric_labels = [
            "Total Reward",
            "Ground Comfort %",
            "Top Comfort %",
            "Light Hours",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [
                self.results[controller].get(metric, 0) for controller in controllers
            ]

            bars = axes[i].bar(controllers, values)
            axes[i].set_title(label)
            axes[i].set_ylabel(label)
            axes[i].tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(values),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "controller_performance_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_temperature_trajectories(self, output_dir: str) -> None:
        """Plot temperature trajectories for first episode of each controller."""
        fig, axes = plt.subplots(
            len(self.results), 1, figsize=(12, 4 * len(self.results))
        )
        if len(self.results) == 1:
            axes = [axes]

        for i, (controller_name, result) in enumerate(self.results.items()):
            episode_temps = result["episode_data"]["temperatures"][0]
            setpoints = result["episode_data"]["setpoints"][0]

            ground_temps = [temp[0] for temp in episode_temps]
            top_temps = [temp[1] for temp in episode_temps]

            axes[i].plot(ground_temps, label="Ground Floor", linewidth=2)
            axes[i].plot(top_temps, label="Top Floor", linewidth=2)

            if setpoints:
                heating_sp = [sp[0] for sp in setpoints]
                cooling_sp = [sp[1] for sp in setpoints]
                axes[i].plot(heating_sp, "r--", label="Heating Setpoint", alpha=0.7)
                axes[i].plot(cooling_sp, "b--", label="Cooling Setpoint", alpha=0.7)

            axes[i].set_title(f"{controller_name} - Temperature Control")
            axes[i].set_ylabel("Temperature (°C)")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time Steps")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "temperature_trajectories_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_action_patterns(self, output_dir: str) -> None:
        """Plot action patterns for each controller."""
        fig, axes = plt.subplots(
            len(self.results), 1, figsize=(12, 3 * len(self.results))
        )
        if len(self.results) == 1:
            axes = [axes]

        action_names = ["Ground Light", "Ground Window", "Top Light", "Top Window"]
        colors = ["red", "blue", "orange", "cyan"]

        for i, (controller_name, result) in enumerate(self.results.items()):
            episode_actions = result["episode_data"]["actions"][0]
            actions_array = np.array(episode_actions)

            for j, (name, color) in enumerate(zip(action_names, colors)):
                axes[i].plot(
                    actions_array[:, j], label=name, linewidth=2, color=color, alpha=0.8
                )

            axes[i].set_title(f"{controller_name} - Control Actions")
            axes[i].set_ylabel("Action State")
            axes[i].set_ylim(-0.1, 1.1)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time Steps")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "action_patterns_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def create_environment_from_data(
    data_file: str, env_params_path: Optional[str] = None
) -> DollhouseThermalEnv:
    """
    Create dollhouse environment from data file and optional parameters.

    Args:
        data_file: Path to data file for SINDy model training
        env_params_path: Optional path to environment parameters JSON

    Returns:
        Configured dollhouse thermal environment
    """
    print(f"Training SINDy model on {data_file}...")
    start_time = time.time()
    sindy_model = train_sindy_model(file_path=data_file)
    training_time = time.time() - start_time
    print(f"SINDy model training completed in {training_time:.2f} seconds")

    if env_params_path and os.path.exists(env_params_path):
        with open(env_params_path, "r") as f:
            env_params = json.load(f)
        env_params["sindy_model"] = sindy_model
        print(f"Loaded environment parameters from {env_params_path}")
    else:
        env_params = {
            "sindy_model": sindy_model,
            "use_reward_shaping": True,
            "random_start_time": False,
            "shaping_weight": 0.3,
            "episode_length": 5760,
            "time_step_seconds": 30,
            "heating_setpoint": 26.0,
            "cooling_setpoint": 28.0,
            "external_temp_pattern": "sine",
            "setpoint_pattern": "schedule",
            "reward_type": "balanced",
            "energy_weight": 0.0,
            "comfort_weight": 1.0,
        }
        print("Using default environment parameters")

    env = DollhouseThermalEnv(**env_params)

    print(f"Environment Configuration:")
    print(f"  Setpoint Pattern: {env.setpoint_pattern}")
    print(f"  Base Heating Setpoint: {env.initial_heating_setpoint}")
    print(f"  Base Cooling Setpoint: {env.initial_cooling_setpoint}")
    print(f"  Random Start Time: {env.random_start_time}")
    print(f"  Reward Shaping: {env.use_reward_shaping}")

    return env


def run_controller_comparison(
    data_file: str,
    output_dir: Optional[str] = None,
    num_episodes: int = 5,
    render: bool = False,
    env_params_path: Optional[str] = None,
    controllers_config: Optional[Dict] = None,
) -> Dict:
    """
    Run comprehensive comparison of thermal controllers.

    Args:
        data_file: Path to data file for SINDy model training
        output_dir: Directory to save results
        num_episodes: Number of episodes for each controller evaluation
        render: Whether to render during evaluation
        env_params_path: Optional path to environment parameters
        controllers_config: Optional controller configuration overrides

    Returns:
        Dictionary containing comparison results
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"controller_comparison_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")

    env = create_environment_from_data(data_file, env_params_path)
    evaluator = ControllerEvaluator(env)

    # Create controllers with default or provided configurations
    controllers = create_controllers(controllers_config)

    print(f"\nEvaluating {len(controllers)} controllers...")

    # Evaluate each controller
    for controller in controllers:
        print(f"\nEvaluating {controller.name}...")
        start_time = time.time()

        evaluator.evaluate_controller(
            controller=controller,
            num_episodes=num_episodes,
            render=render,
            verbose=True,
        )

        evaluation_time = time.time() - start_time
        print(f"Evaluation completed in {evaluation_time:.2f} seconds")

    # Generate comparison analysis
    comparison = evaluator.compare_controllers()

    print("\n" + "=" * 60)
    print("CONTROLLER COMPARISON SUMMARY")
    print("=" * 60)

    for metric, best_controller in comparison["best_performers"].items():
        print(f"Best {metric}: {best_controller}")

    # Save results and generate plots
    evaluator.save_results(output_dir)
    evaluator.plot_comparison(output_dir)

    # Save comparison summary
    with open(os.path.join(output_dir, "comparison_summary.json"), "w") as f:
        json.dump(comparison, f, indent=4)

    # Save environment configuration
    env_config = {
        "data_file": data_file,
        "episode_length": env.episode_length,
        "setpoint_pattern": env.setpoint_pattern,
        "reward_type": env.reward_type,
        "use_reward_shaping": env.use_reward_shaping,
        "evaluation_episodes": num_episodes,
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(output_dir, "evaluation_config.json"), "w") as f:
        json.dump(env_config, f, indent=4)

    print(f"\nComparison completed. Results saved to {output_dir}")
    return comparison


def create_controllers(
    controllers_config: Optional[Dict] = None,
) -> List[BaseController]:
    """
    Create controller instances with specified configurations.

    Args:
        controllers_config: Optional configuration dictionary for controllers

    Returns:
        List of configured controller instances
    """
    if controllers_config is None:
        controllers_config = {}

    controllers = []

    # Rule-based controller
    rule_config = controllers_config.get("rule_based", {})
    rule_controller = RuleBasedController(hysteresis=rule_config.get("hysteresis", 0.5))
    controllers.append(rule_controller)

    # PID controller
    pid_config = controllers_config.get("pid", {})
    pid_controller = PIDController(
        ground_kp=pid_config.get("ground_kp", 2.0),
        ground_ki=pid_config.get("ground_ki", 0.1),
        ground_kd=pid_config.get("ground_kd", 0.05),
        top_kp=pid_config.get("top_kp", 2.0),
        top_ki=pid_config.get("top_ki", 0.1),
        top_kd=pid_config.get("top_kd", 0.05),
    )
    controllers.append(pid_controller)

    # Fuzzy logic controller
    fuzzy_config = controllers_config.get("fuzzy", {})
    fuzzy_controller = FuzzyLogicController(
        heating_threshold=fuzzy_config.get("heating_threshold", 0.2),
        cooling_threshold=fuzzy_config.get("cooling_threshold", 0.2),
    )
    controllers.append(fuzzy_controller)

    return controllers


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Compare thermal control strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        default="../Data/dollhouse-data-2025-03-24.csv",
        help="Path to data file for training SINDy model",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes for evaluation"
    )
    parser.add_argument(
        "--render", action="store_true", help="Enable rendering during evaluation"
    )
    parser.add_argument(
        "--env-params",
        type=str,
        default=None,
        help="Path to environment parameters JSON file",
    )
    parser.add_argument(
        "--controllers-config",
        type=str,
        default=None,
        help="Path to controllers configuration JSON file",
    )

    # Controller-specific parameters
    parser.add_argument(
        "--rule-hysteresis",
        type=float,
        default=0.5,
        help="Hysteresis for rule-based controller",
    )
    parser.add_argument(
        "--pid-kp",
        type=float,
        default=2.0,
        help="Proportional gain for PID controllers",
    )
    parser.add_argument(
        "--pid-ki", type=float, default=0.1, help="Integral gain for PID controllers"
    )
    parser.add_argument(
        "--pid-kd", type=float, default=0.05, help="Derivative gain for PID controllers"
    )
    parser.add_argument(
        "--fuzzy-heat-threshold",
        type=float,
        default=0.2,
        help="Heating threshold for fuzzy controller",
    )
    parser.add_argument(
        "--fuzzy-cool-threshold",
        type=float,
        default=0.2,
        help="Cooling threshold for fuzzy controller",
    )

    args = parser.parse_args()

    # Load or create controllers configuration
    if args.controllers_config and os.path.exists(args.controllers_config):
        with open(args.controllers_config, "r") as f:
            controllers_config = json.load(f)
    else:
        controllers_config = {
            "rule_based": {"hysteresis": args.rule_hysteresis},
            "pid": {
                "ground_kp": args.pid_kp,
                "ground_ki": args.pid_ki,
                "ground_kd": args.pid_kd,
                "top_kp": args.pid_kp,
                "top_ki": args.pid_ki,
                "top_kd": args.pid_kd,
            },
            "fuzzy": {
                "heating_threshold": args.fuzzy_heat_threshold,
                "cooling_threshold": args.fuzzy_cool_threshold,
            },
        }

    # Run comparison
    comparison_results = run_controller_comparison(
        data_file=args.data,
        output_dir=args.output,
        num_episodes=args.episodes,
        render=args.render,
        env_params_path=args.env_params,
        controllers_config=controllers_config,
    )

    return comparison_results


if __name__ == "__main__":
    main()
