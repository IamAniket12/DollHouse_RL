"""
Dollhouse Thermal Environment for Reinforcement Learning.

This module provides a Gymnasium-compatible environment for thermal control
in a two-floor dollhouse system using a pre-trained SINDy model for dynamics.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces


class DollhouseThermalEnv(gym.Env):
    """
    A Gymnasium environment for dollhouse thermal control using SINDy dynamics.

    This environment simulates a two-floor dollhouse with controllable lights and windows
    on each floor. The goal is to maintain comfortable temperatures while minimizing
    energy consumption.

    Features:
    - Random start time for diverse training scenarios
    - Andrew Ng style reward shaping for accelerated learning
    - Multiple setpoint patterns (fixed, schedule, adaptive, challenging)
    - Configurable external temperature patterns

    Attributes:
        metadata: Environment metadata for Gymnasium compatibility
        action_space: MultiDiscrete space for binary controls
        observation_space: Box space for continuous observations
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        sindy_model,
        external_temp_pattern: str = "fixed",
        episode_length: int = 2880,
        time_step_seconds: int = 30,
        heating_setpoint: float = 30.0,
        cooling_setpoint: float = 35.0,
        initial_ground_temp: float = 22.0,
        initial_top_temp: float = 23.0,
        reward_type: str = "comfort",
        energy_weight: float = 0.5,
        comfort_weight: float = 1.0,
        random_seed: Optional[int] = None,
        setpoint_pattern: str = "fixed",
        render_mode: Optional[str] = None,
        random_start_time: bool = False,
        start_time_range: Tuple[float, float] = (0, 24),
        use_reward_shaping: bool = False,
        shaping_gamma: float = 0.99,
        shaping_weight: float = 0.3,
        comfort_potential_weight: float = 1.0,
        energy_potential_weight: float = 0.5,
        comfort_decay_rate: float = 0.4,
        custom_observation_config: Optional[str] = None,
    ):
        """
        Initialize the dollhouse thermal environment.

        Args:
            sindy_model: Pre-trained SINDy model for system dynamics
            external_temp_pattern: Pattern for external temperature ("fixed", "sine", "real_data", "random_walk")
            episode_length: Number of timesteps per episode
            time_step_seconds: Duration of each timestep in seconds
            heating_setpoint: Lower temperature bound in Celsius
            cooling_setpoint: Upper temperature bound in Celsius
            initial_ground_temp: Starting ground floor temperature in Celsius
            initial_top_temp: Starting top floor temperature in Celsius
            reward_type: Type of reward function ("comfort", "energy", "balanced")
            energy_weight: Weight for energy penalty in reward
            comfort_weight: Weight for comfort penalty in reward
            random_seed: Random seed for reproducibility
            setpoint_pattern: Pattern for setpoint variation ("fixed", "schedule", "adaptive", "challenging")
            render_mode: Rendering mode ("human", "rgb_array", or None)
            random_start_time: Whether to randomize episode start times
            start_time_range: Hours range for random start time
            use_reward_shaping: Whether to apply potential-based reward shaping
            shaping_gamma: Discount factor for reward shaping
            shaping_weight: Overall influence of reward shaping
            comfort_potential_weight: Weight for comfort potential function
            energy_potential_weight: Weight for energy potential function
            comfort_decay_rate: Exponential decay rate for temperature deviations
            custom_observation_config: Path to JSON file defining custom observation space
        """
        super().__init__()

        self._validate_parameters(render_mode, external_temp_pattern, setpoint_pattern)
        self._initialize_environment_parameters(
            sindy_model,
            external_temp_pattern,
            episode_length,
            time_step_seconds,
            heating_setpoint,
            cooling_setpoint,
            initial_ground_temp,
            initial_top_temp,
            reward_type,
            energy_weight,
            comfort_weight,
            setpoint_pattern,
            render_mode,
        )
        self._initialize_random_start_parameters(random_start_time, start_time_range)
        self._initialize_reward_shaping_parameters(
            use_reward_shaping,
            shaping_gamma,
            shaping_weight,
            comfort_potential_weight,
            energy_potential_weight,
            comfort_decay_rate,
        )
        self._load_custom_observation_config(custom_observation_config)
        self._setup_spaces()
        self._initialize_state_variables()
        self._setup_rendering()

        self.np_random = None
        self._rng = np.random.default_rng(random_seed)

    def _validate_parameters(
        self,
        render_mode: Optional[str],
        external_temp_pattern: str,
        setpoint_pattern: str,
    ) -> None:
        """Validate input parameters."""
        valid_render_modes = self.metadata["render_modes"] + [None]
        if render_mode not in valid_render_modes:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. Choose from {valid_render_modes}"
            )

        valid_temp_patterns = ["fixed", "sine", "real_data", "random_walk"]
        if external_temp_pattern not in valid_temp_patterns:
            raise ValueError(f"Invalid external_temp_pattern: {external_temp_pattern}")

        valid_setpoint_patterns = ["fixed", "schedule", "adaptive", "challenging"]
        if setpoint_pattern not in valid_setpoint_patterns:
            raise ValueError(f"Invalid setpoint_pattern: {setpoint_pattern}")

    def _initialize_environment_parameters(
        self,
        sindy_model,
        external_temp_pattern: str,
        episode_length: int,
        time_step_seconds: int,
        heating_setpoint: float,
        cooling_setpoint: float,
        initial_ground_temp: float,
        initial_top_temp: float,
        reward_type: str,
        energy_weight: float,
        comfort_weight: float,
        setpoint_pattern: str,
        render_mode: Optional[str],
    ) -> None:
        """Initialize core environment parameters."""
        self.sindy_model = sindy_model
        self.episode_length = episode_length
        self.time_step_seconds = time_step_seconds
        self.initial_heating_setpoint = heating_setpoint
        self.initial_cooling_setpoint = cooling_setpoint
        self.initial_ground_temp = initial_ground_temp
        self.initial_top_temp = initial_top_temp
        self.reward_type = reward_type
        self.energy_weight = energy_weight
        self.comfort_weight = comfort_weight
        self.external_temp_pattern = external_temp_pattern
        self.setpoint_pattern = setpoint_pattern
        self.render_mode = render_mode

    def _initialize_random_start_parameters(
        self, random_start_time: bool, start_time_range: Tuple[float, float]
    ) -> None:
        """Initialize random start time parameters."""
        self.random_start_time = random_start_time
        self.start_time_range = start_time_range

    def _initialize_reward_shaping_parameters(
        self,
        use_reward_shaping: bool,
        shaping_gamma: float,
        shaping_weight: float,
        comfort_potential_weight: float,
        energy_potential_weight: float,
        comfort_decay_rate: float,
    ) -> None:
        """Initialize reward shaping parameters."""
        self.use_reward_shaping = use_reward_shaping
        self.shaping_gamma = shaping_gamma
        self.shaping_weight = shaping_weight
        self.comfort_potential_weight = comfort_potential_weight
        self.energy_potential_weight = energy_potential_weight
        self.comfort_decay_rate = comfort_decay_rate

    def _load_custom_observation_config(
        self, custom_observation_config: Optional[str]
    ) -> None:
        """Load custom observation space configuration from JSON file."""
        if custom_observation_config and os.path.exists(custom_observation_config):
            with open(custom_observation_config, "r") as f:
                self.custom_observation_config = json.load(f)
                print(
                    f"Loaded custom observation configuration from {custom_observation_config}"
                )
                self._validate_observation_config()
        else:
            self._setup_default_observation_config()

    def _validate_observation_config(self) -> None:
        """Validate the custom observation configuration."""
        if "variables" not in self.custom_observation_config:
            raise ValueError("Missing 'variables' field in observation config")

        variables = self.custom_observation_config["variables"]
        if not isinstance(variables, list) or len(variables) == 0:
            raise ValueError("'variables' must be a non-empty list")

        for i, var in enumerate(variables):
            if not all(field in var for field in ["name", "low", "high"]):
                raise ValueError(
                    f"Variable {i} missing required fields: name, low, high"
                )

            if var["low"] >= var["high"]:
                raise ValueError(f"Variable {i}: 'low' must be less than 'high'")

        self.observation_variables = [var["name"] for var in variables]
        print(f"Custom observation space: {self.observation_variables}")

    def _setup_default_observation_config(self) -> None:
        """Setup default observation configuration."""
        self.observation_variables = [
            "ground_temp",
            "top_temp",
            "external_temp",
            "ground_light",
            "ground_window",
            "top_light",
            "top_window",
            "heating_setpoint",
            "cooling_setpoint",
            "hour_of_day",
            "time_step",
        ]

        self.custom_observation_config = {
            "variables": [
                {"name": "ground_temp", "low": -10.0, "high": 50.0},
                {"name": "top_temp", "low": -10.0, "high": 50.0},
                {"name": "external_temp", "low": -30.0, "high": 50.0},
                {"name": "ground_light", "low": 0, "high": 1},
                {"name": "ground_window", "low": 0, "high": 1},
                {"name": "top_light", "low": 0, "high": 1},
                {"name": "top_window", "low": 0, "high": 1},
                {"name": "heating_setpoint", "low": 10.0, "high": 35.0},
                {"name": "cooling_setpoint", "low": 10.0, "high": 35.0},
                {"name": "hour_of_day", "low": 0.0, "high": 23.0},
                {"name": "time_step", "low": 0, "high": self.episode_length},
            ]
        }

    def _setup_spaces(self) -> None:
        """Setup action and observation spaces."""
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2])

        # Create observation space from configuration
        variables = self.custom_observation_config["variables"]
        lows = [var["low"] for var in variables]
        highs = [var["high"] for var in variables]

        self.observation_space = spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
            dtype=np.float32,
        )

    def _initialize_state_variables(self) -> None:
        """Initialize state variables."""
        self.current_step = 0
        self.episode_start_time_offset = 0
        self.ground_temp = None
        self.top_temp = None
        self.external_temperatures = None
        self.heating_setpoint = None
        self.cooling_setpoint = None
        self.current_action = None
        self.ground_temp_history = None
        self.top_temp_history = None
        self.external_temp_history = None
        self.history = None
        self.previous_potential = 0.0
        self.shaping_history = []

    def _setup_rendering(self) -> None:
        """Setup rendering components."""
        self.fig = None
        self.ax = None
        self.episode_history = []

    def _generate_external_temperature(
        self, start_offset_hours: float = 0.0
    ) -> np.ndarray:
        """
        Generate external temperature pattern for the entire episode.

        Args:
            start_offset_hours: Hour offset to start the pattern from

        Returns:
            Array of external temperatures for the episode
        """
        time_steps = self.episode_length
        start_offset_steps = int(start_offset_hours * 3600 / self.time_step_seconds)

        if self.external_temp_pattern == "sine":
            return self._generate_sine_temperature(time_steps, start_offset_steps)
        elif self.external_temp_pattern == "real_data":
            return self._generate_realistic_temperature(time_steps, start_offset_hours)
        elif self.external_temp_pattern == "random_walk":
            return self._generate_random_walk_temperature(
                time_steps, start_offset_hours
            )
        else:
            return self._generate_fixed_temperature(time_steps)

    def _generate_sine_temperature(
        self, time_steps: int, start_offset_steps: int
    ) -> np.ndarray:
        """Generate sinusoidal temperature pattern."""
        total_steps = time_steps + start_offset_steps
        time = np.linspace(0, 2 * np.pi * total_steps / 2880, total_steps)
        base_temp = 20.0
        amplitude = 2.0

        all_temperatures = (
            base_temp
            + amplitude * np.sin(time - np.pi / 2)
            + self.np_random.normal(0, 0.5, total_steps)
        )

        return all_temperatures[start_offset_steps : start_offset_steps + time_steps]

    def _generate_realistic_temperature(
        self, time_steps: int, start_offset_hours: float
    ) -> np.ndarray:
        """Generate realistic daily temperature pattern."""
        temperatures = []
        base_temp = 15.0

        for i in range(time_steps):
            actual_hour = (
                start_offset_hours + (i * self.time_step_seconds / 3600)
            ) % 24

            if actual_hour < 6:
                temp = base_temp - 5.0 + actual_hour * 0.3
            elif actual_hour < 12:
                temp = base_temp - 3.0 + (actual_hour - 6) * 1.5
            elif actual_hour < 18:
                temp = base_temp + 7.0 - (actual_hour - 12) * 0.5
            else:
                temp = base_temp + 4.0 - (actual_hour - 18) * 1.5

            temp += self.np_random.normal(0, 1.0)
            temperatures.append(temp)

        return np.array(temperatures)

    def _generate_random_walk_temperature(
        self, time_steps: int, start_offset_hours: float
    ) -> np.ndarray:
        """Generate random walk temperature pattern."""
        start_hour = start_offset_hours % 24
        initial_temp = 18.0 if 6 <= start_hour < 18 else 12.0

        temperatures = np.zeros(time_steps)
        temperatures[0] = initial_temp

        for i in range(1, time_steps):
            step = self.np_random.normal(0, 1.0)
            actual_hour = (
                start_offset_hours + (i * self.time_step_seconds / 3600)
            ) % 24

            if 9 <= actual_hour < 18:
                step += 0.1
            else:
                step -= 0.1

            temperatures[i] = temperatures[i - 1] + step
            temperatures[i] = np.clip(temperatures[i], -5.0, 35.0)

        return temperatures

    def _generate_fixed_temperature(self, time_steps: int) -> np.ndarray:
        """Generate fixed temperature with small variations."""
        base_temp = 20.0
        return base_temp + self.np_random.normal(0, 0.3, time_steps)

    def _update_setpoints(self, time_step: int) -> Tuple[float, float]:
        """
        Update heating and cooling setpoints based on the chosen pattern.

        Args:
            time_step: Current timestep in the episode

        Returns:
            Tuple of (heating_setpoint, cooling_setpoint)
        """
        actual_hour = (
            self.episode_start_time_offset + (time_step * self.time_step_seconds / 3600)
        ) % 24

        if self.setpoint_pattern == "fixed":
            return self.heating_setpoint, self.cooling_setpoint
        elif self.setpoint_pattern == "schedule":
            return self._get_scheduled_setpoints(actual_hour)
        elif self.setpoint_pattern == "adaptive":
            return self._get_adaptive_setpoints(time_step)
        elif self.setpoint_pattern == "challenging":
            return self._get_challenging_setpoints(time_step, actual_hour)
        else:
            return self.heating_setpoint, self.cooling_setpoint

    def _get_scheduled_setpoints(self, actual_hour: float) -> Tuple[float, float]:
        """Get setpoints based on time schedule."""
        if 11 <= actual_hour < 18:
            return 22.0, 24.0
        elif 8 <= actual_hour < 11:
            return 26.0, 28.0
        else:
            return 20.0, 24.0

    def _get_adaptive_setpoints(self, time_step: int) -> Tuple[float, float]:
        """Get setpoints that adapt to external temperature."""
        ext_temp = self.external_temperatures[time_step]
        if ext_temp < 5:
            return 19.0, 24.0
        elif ext_temp > 25:
            return 21.0, 26.0
        else:
            return 20.0, 25.0

    def _get_challenging_setpoints(
        self, time_step: int, actual_hour: float
    ) -> Tuple[float, float]:
        """Get challenging setpoints that test controller adaptability."""
        minutes_elapsed = (time_step * self.time_step_seconds) / 60

        if minutes_elapsed < 30:
            return 22.0, 24.0
        elif 30 <= minutes_elapsed < 90:
            return 20.0, 27.0
        elif 90 <= minutes_elapsed < 240:
            cycle_position = (minutes_elapsed - 90) % 20
            return (22.5, 23.5) if cycle_position < 10 else (19.0, 28.0)
        elif 240 <= minutes_elapsed < 480:
            return self._get_time_dependent_setpoints(actual_hour, time_step)
        else:
            return self._get_moving_window_setpoints(minutes_elapsed)

    def _get_time_dependent_setpoints(
        self, actual_hour: float, time_step: int
    ) -> Tuple[float, float]:
        """Get setpoints that depend on time of day and external temperature."""
        ext_temp = self.external_temperatures[
            min(time_step, len(self.external_temperatures) - 1)
        ]

        if 6 <= actual_hour < 12:
            base_heating, base_cooling = 21.0, 25.0
        elif 12 <= actual_hour < 18:
            base_heating, base_cooling = 23.0, 26.0
        elif 18 <= actual_hour < 22:
            base_heating, base_cooling = 22.0, 24.0
        else:
            base_heating, base_cooling = 20.0, 26.0

        if ext_temp < 15:
            return base_heating - 1.0, base_cooling - 1.0
        elif ext_temp > 25:
            return base_heating + 1.0, base_cooling + 1.0
        else:
            return base_heating, base_cooling

    def _get_moving_window_setpoints(
        self, minutes_elapsed: float
    ) -> Tuple[float, float]:
        """Get setpoints with a moving comfort window."""
        minutes_in_phase = minutes_elapsed - 480
        center_temp = 23.5 + 2.0 * np.sin(2 * np.pi * minutes_in_phase / 60)

        heating_sp = np.clip(center_temp - 0.75, 18.0, 26.0)
        cooling_sp = np.clip(center_temp + 0.75, 20.0, 30.0)

        return heating_sp, cooling_sp

    def _prepare_sindy_features(
        self, state: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        """
        Prepare input features for the SINDy model.

        Args:
            state: Current state [ground_temp, top_temp, external_temp]
            action: Current action [ground_light, ground_window, top_light, top_window]

        Returns:
            Feature vector for SINDy model input
        """
        ground_temp, top_temp, external_temp = state
        ground_light, ground_window, top_light, top_window = action
        time_diff_seconds = self.time_step_seconds

        floor_temp_diff = top_temp - ground_temp
        ground_ext_temp_diff = ground_temp - external_temp
        top_ext_temp_diff = top_temp - external_temp
        ground_window_ext_effect = ground_window * ground_ext_temp_diff
        top_window_ext_effect = top_window * top_ext_temp_diff

        ground_temp_lag1 = self.ground_temp_history[-1]
        top_temp_lag1 = self.top_temp_history[-1]
        ext_temp_lag1 = self.external_temp_history[-1]
        ground_temp_lag2 = self.ground_temp_history[-2]
        top_temp_lag2 = self.top_temp_history[-2]
        ext_temp_lag2 = self.external_temp_history[-2]

        if len(self.ground_temp_history) >= 2:
            ground_temp_rate = (ground_temp - ground_temp_lag1) / time_diff_seconds
            top_temp_rate = (top_temp - top_temp_lag1) / time_diff_seconds
        else:
            ground_temp_rate = 0.0
            top_temp_rate = 0.0

        features = np.array(
            [
                ground_light,
                ground_window,
                top_light,
                top_window,
                external_temp,
                time_diff_seconds,
                floor_temp_diff,
                ground_ext_temp_diff,
                top_ext_temp_diff,
                ground_window_ext_effect,
                top_window_ext_effect,
                ground_temp_lag1,
                top_temp_lag1,
                ext_temp_lag1,
                ground_temp_lag2,
                top_temp_lag2,
                ext_temp_lag2,
                ground_temp_rate,
                top_temp_rate,
            ]
        )

        return features.reshape(1, -1)

    def _calculate_base_reward(
        self, ground_temp: float, top_temp: float, action: np.ndarray
    ) -> float:
        """
        Calculate base reward based on comfort and energy efficiency.

        Args:
            ground_temp: Ground floor temperature
            top_temp: Top floor temperature
            action: Action array

        Returns:
            Base reward value
        """
        ground_comfortable = (
            self.heating_setpoint <= ground_temp <= self.cooling_setpoint
        )
        top_comfortable = self.heating_setpoint <= top_temp <= self.cooling_setpoint

        if ground_comfortable and top_comfortable:
            comfort_reward = 1.0 * self.comfort_weight
            lights_on = action[0] + action[2]
            energy_bonus = self.energy_weight * (1.0 - lights_on / 2.0)
            return comfort_reward + energy_bonus
        else:
            return 0.0

    def _comfort_zone_potential(
        self, ground_temp: float, top_temp: float, heating_sp: float, cooling_sp: float
    ) -> float:
        """
        Calculate comfort zone potential for reward shaping.

        Args:
            ground_temp: Ground floor temperature
            top_temp: Top floor temperature
            heating_sp: Heating setpoint
            cooling_sp: Cooling setpoint

        Returns:
            Comfort potential value
        """

        def zone_potential(temp):
            if heating_sp <= temp <= cooling_sp:
                return 1.0
            else:
                distance = min(abs(heating_sp - temp), abs(temp - cooling_sp))
                return np.exp(-self.comfort_decay_rate * distance)

        ground_potential = zone_potential(ground_temp)
        top_potential = zone_potential(top_temp)
        return (ground_potential + top_potential) / 2.0

    def _energy_efficiency_potential(
        self,
        ground_temp: float,
        top_temp: float,
        action: np.ndarray,
        heating_sp: float,
        cooling_sp: float,
    ) -> float:
        """
        Calculate energy efficiency potential for reward shaping.

        Args:
            ground_temp: Ground floor temperature
            top_temp: Top floor temperature
            action: Action array
            heating_sp: Heating setpoint
            cooling_sp: Cooling setpoint

        Returns:
            Energy potential value
        """
        ground_in_comfort = heating_sp <= ground_temp <= cooling_sp
        top_in_comfort = heating_sp <= top_temp <= cooling_sp
        both_comfortable = ground_in_comfort and top_in_comfort

        lights_on = action[0] + action[2]

        if both_comfortable:
            return 1.0 - 0.3 * (lights_on / 2.0)
        else:
            return 0.2 - 0.1 * (lights_on / 2.0)

    def _calculate_total_potential(
        self, ground_temp: float, top_temp: float, action: np.ndarray
    ) -> float:
        """Calculate combined potential function for reward shaping."""
        comfort_potential = self._comfort_zone_potential(
            ground_temp, top_temp, self.heating_setpoint, self.cooling_setpoint
        )
        energy_potential = self._energy_efficiency_potential(
            ground_temp, top_temp, action, self.heating_setpoint, self.cooling_setpoint
        )

        return (
            self.comfort_potential_weight * comfort_potential
            + self.energy_potential_weight * energy_potential
        )

    def _calculate_reward(
        self, ground_temp: float, top_temp: float, action: np.ndarray
    ) -> float:
        """
        Calculate reward with optional potential-based shaping.

        Args:
            ground_temp: Ground floor temperature
            top_temp: Top floor temperature
            action: Action array

        Returns:
            Total reward value
        """
        base_reward = self._calculate_base_reward(ground_temp, top_temp, action)

        if not self.use_reward_shaping:
            return base_reward

        current_potential = self._calculate_total_potential(
            ground_temp, top_temp, action
        )
        shaping_term = self.shaping_gamma * current_potential - self.previous_potential
        shaped_reward = base_reward + self.shaping_weight * shaping_term

        self.previous_potential = current_potential
        self.shaping_history.append(
            {
                "step": self.current_step,
                "base_reward": base_reward,
                "potential": current_potential,
                "shaped_component": self.shaping_weight * shaping_term,
                "total_reward": shaped_reward,
            }
        )

        return shaped_reward

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for episode
            options: Additional options (unused)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._reset_episode_state()
        self._generate_episode_conditions()
        self._initialize_episode_history()

        if self.use_reward_shaping:
            initial_action = np.zeros(4)
            self.previous_potential = self._calculate_total_potential(
                self.ground_temp, self.top_temp, initial_action
            )
            self.shaping_history = []

        hour_of_day = (
            self.episode_start_time_offset
            + (self.current_step * self.time_step_seconds / 3600)
        ) % 24

        observation = self._build_custom_observation(np.zeros(4))

        info = {"episode_start_time_offset": self.episode_start_time_offset}
        return observation, info

    def _reset_episode_state(self) -> None:
        """Reset episode state variables."""
        self.current_step = 0
        self.ground_temp = self.initial_ground_temp
        self.top_temp = self.initial_top_temp
        self.heating_setpoint = self.initial_heating_setpoint
        self.cooling_setpoint = self.initial_cooling_setpoint
        self.current_action = np.zeros(4)

    def _generate_episode_conditions(self) -> None:
        """Generate episode-specific conditions."""
        if self.random_start_time:
            start_hour_min, start_hour_max = self.start_time_range
            self.episode_start_time_offset = self._rng.uniform(
                start_hour_min, start_hour_max
            )
        else:
            self.episode_start_time_offset = 0.0

        self.external_temperatures = self._generate_external_temperature(
            start_offset_hours=self.episode_start_time_offset
        )

    def _initialize_episode_history(self) -> None:
        """Initialize history tracking for the episode."""
        initial_temp = [self.initial_ground_temp] * 3
        initial_top_temp = [self.initial_top_temp] * 3
        initial_ext_temp = [self.external_temperatures[0]] * 3

        self.ground_temp_history = initial_temp
        self.top_temp_history = initial_top_temp
        self.external_temp_history = initial_ext_temp

        self.history = {
            "ground_temp": [self.ground_temp],
            "top_temp": [self.top_temp],
            "external_temp": [self.external_temperatures[0]],
            "heating_setpoint": [self.heating_setpoint],
            "cooling_setpoint": [self.cooling_setpoint],
            "ground_light": [0],
            "ground_window": [0],
            "top_light": [0],
            "top_window": [0],
            "reward": [0],
            "episode_start_time_offset": self.episode_start_time_offset,
        }

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Action array [ground_light, ground_window, top_light, top_window]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_action = action
        current_state = np.array([self.ground_temp, self.top_temp])

        self._update_temperature_history()

        u_features = self._prepare_sindy_features(
            state=np.array(
                [
                    self.ground_temp,
                    self.top_temp,
                    self.external_temperatures[self.current_step],
                ]
            ),
            action=action,
        )

        next_state = self.sindy_model.predict(
            current_state.reshape(1, -1), u=u_features
        )[0]
        self.ground_temp, self.top_temp = next_state

        self.heating_setpoint, self.cooling_setpoint = self._update_setpoints(
            self.current_step
        )
        reward = self._calculate_reward(self.ground_temp, self.top_temp, action)
        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.episode_length

        observation = self._build_custom_observation(action)
        info = self._build_info()
        self._update_episode_history(action, reward)

        if truncated:
            self.episode_history.append(self.history)

        return observation, reward, terminated, truncated, info

    def _update_temperature_history(self) -> None:
        """Update temperature history for feature calculation."""
        self.ground_temp_history.append(self.ground_temp)
        self.top_temp_history.append(self.top_temp)
        self.external_temp_history.append(
            self.external_temperatures[min(self.current_step, self.episode_length - 1)]
        )

        if len(self.ground_temp_history) > 3:
            self.ground_temp_history.pop(0)
            self.top_temp_history.pop(0)
            self.external_temp_history.pop(0)

    def _get_full_observation_dict(
        self, action: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Get dictionary of all possible observation variables."""
        if action is None:
            action = np.zeros(4)

        hour_of_day = (
            self.episode_start_time_offset
            + (self.current_step * self.time_step_seconds / 3600)
        ) % 24

        return {
            "ground_temp": self.ground_temp,
            "top_temp": self.top_temp,
            "external_temp": self.external_temperatures[
                min(self.current_step, self.episode_length - 1)
            ],
            "ground_light": float(action[0]),
            "ground_window": float(action[1]),
            "top_light": float(action[2]),
            "top_window": float(action[3]),
            "heating_setpoint": self.heating_setpoint,
            "cooling_setpoint": self.cooling_setpoint,
            "hour_of_day": hour_of_day,
            "time_step": float(self.current_step),
        }

    def _build_custom_observation(self, action: np.ndarray) -> np.ndarray:
        """Build observation array based on configuration."""
        full_obs_dict = self._get_full_observation_dict(action)

        observation_values = []
        for var_name in self.observation_variables:
            if var_name in full_obs_dict:
                observation_values.append(full_obs_dict[var_name])
            else:
                raise ValueError(f"Unknown observation variable: {var_name}")

        return np.array(observation_values, dtype=np.float32)

    def _build_info(self) -> Dict:
        """Build info dictionary."""
        return {
            "ground_temp": self.ground_temp,
            "top_temp": self.top_temp,
            "external_temp": self.external_temperatures[
                min(self.current_step, self.episode_length - 1)
            ],
            "ground_comfort_violation": max(self.heating_setpoint - self.ground_temp, 0)
            + max(self.ground_temp - self.cooling_setpoint, 0),
            "top_comfort_violation": max(self.heating_setpoint - self.top_temp, 0)
            + max(self.top_temp - self.cooling_setpoint, 0),
            "energy_use": self.current_action[0] + self.current_action[2],
            "episode_start_time_offset": self.episode_start_time_offset,
        }

    def _update_episode_history(self, action: np.ndarray, reward: float) -> None:
        """Update episode history with current step data."""
        self.history["ground_temp"].append(self.ground_temp)
        self.history["top_temp"].append(self.top_temp)
        self.history["external_temp"].append(
            self.external_temperatures[min(self.current_step, self.episode_length - 1)]
        )
        self.history["heating_setpoint"].append(self.heating_setpoint)
        self.history["cooling_setpoint"].append(self.cooling_setpoint)
        self.history["ground_light"].append(action[0])
        self.history["ground_window"].append(action[1])
        self.history["top_light"].append(action[2])
        self.history["top_window"].append(action[3])
        self.history["reward"].append(reward)

    def render(self):
        """
        Render the environment state.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            return None

    def _render_frame(self):
        """Render a single frame of the environment."""
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            if self.render_mode == "human":
                plt.ion()

        for a in self.ax:
            a.clear()

        time_steps = range(len(self.history["ground_temp"]))
        time_hours = [
            (self.episode_start_time_offset + t * self.time_step_seconds / 3600) % 24
            for t in time_steps
        ]

        self._plot_temperatures(time_hours)
        self._plot_actions(time_hours)
        self._plot_rewards(time_hours)

        plt.tight_layout()

        if self.render_mode == "human":
            self.fig.canvas.draw()
            plt.pause(0.1)
            return None
        else:
            self.fig.canvas.draw()
            return np.transpose(
                np.array(self.fig.canvas.renderer.buffer_rgba()), (2, 0, 1)
            )

    def _plot_temperatures(self, time_hours: List[float]) -> None:
        """Plot temperature data."""
        self.ax[0].plot(
            time_hours, self.history["ground_temp"], "b-", label="Ground Floor Temp"
        )
        self.ax[0].plot(
            time_hours, self.history["top_temp"], "r-", label="Top Floor Temp"
        )
        self.ax[0].plot(
            time_hours, self.history["external_temp"], "g-", label="External Temp"
        )
        self.ax[0].plot(
            time_hours,
            self.history["heating_setpoint"],
            "k--",
            label="Heating Setpoint",
        )
        self.ax[0].plot(
            time_hours,
            self.history["cooling_setpoint"],
            "k-.",
            label="Cooling Setpoint",
        )
        self.ax[0].set_ylabel("Temperature (°C)")
        self.ax[0].legend(loc="best")
        self.ax[0].grid(True)

        title = f"Dollhouse Thermal Environment (Start: {self.episode_start_time_offset:.1f}h)"
        self.ax[0].set_title(title)

    def _plot_actions(self, time_hours: List[float]) -> None:
        """Plot control actions."""
        self.ax[1].step(
            time_hours, self.history["ground_light"], "b-", label="Ground Light"
        )
        self.ax[1].step(
            time_hours, self.history["ground_window"], "b--", label="Ground Window"
        )
        self.ax[1].step(time_hours, self.history["top_light"], "r-", label="Top Light")
        self.ax[1].step(
            time_hours, self.history["top_window"], "r--", label="Top Window"
        )
        self.ax[1].set_ylabel("Control State")
        self.ax[1].set_yticks([0, 1])
        self.ax[1].set_yticklabels(["OFF/CLOSED", "ON/OPEN"])
        self.ax[1].legend(loc="best")
        self.ax[1].grid(True)

    def _plot_rewards(self, time_hours: List[float]) -> None:
        """Plot reward signal."""
        self.ax[2].plot(time_hours, self.history["reward"], "k-")
        self.ax[2].set_xlabel("Time (hours)")
        self.ax[2].set_ylabel("Reward")
        self.ax[2].grid(True)

    def close(self):
        """Clean up rendering resources."""
        if self.fig is not None:
            plt.close(self.fig)
            plt.ioff()
            self.fig = None
            self.ax = None

    def get_performance_summary(self) -> Dict:
        """
        Get performance summary across all completed episodes.

        Returns:
            Dictionary containing performance metrics
        """
        if not self.episode_history:
            return {"error": "No episodes completed yet"}

        ground_comfort_pcts = []
        top_comfort_pcts = []
        total_rewards = []
        light_hours = []
        start_times = []

        for episode in self.episode_history:
            ground_temps = np.array(episode["ground_temp"])
            top_temps = np.array(episode["top_temp"])
            heating_sp = np.array(episode["heating_setpoint"])
            cooling_sp = np.array(episode["cooling_setpoint"])

            ground_comfort = (
                np.mean((ground_temps >= heating_sp) & (ground_temps <= cooling_sp))
                * 100
            )
            top_comfort = (
                np.mean((top_temps >= heating_sp) & (top_temps <= cooling_sp)) * 100
            )

            ground_comfort_pcts.append(ground_comfort)
            top_comfort_pcts.append(top_comfort)
            total_rewards.append(np.sum(episode["reward"]))

            light_hours.append(
                (
                    np.sum(np.array(episode["ground_light"]))
                    + np.sum(np.array(episode["top_light"]))
                )
                * self.time_step_seconds
                / 3600
            )

            if "episode_start_time_offset" in episode:
                start_times.append(episode["episode_start_time_offset"])

        summary = {
            "num_episodes": len(self.episode_history),
            "avg_ground_comfort_pct": np.mean(ground_comfort_pcts),
            "avg_top_comfort_pct": np.mean(top_comfort_pcts),
            "avg_total_comfort_pct": np.mean([ground_comfort_pcts, top_comfort_pcts]),
            "avg_total_reward": np.mean(total_rewards),
            "avg_light_hours": np.mean(light_hours),
            "std_total_reward": np.std(total_rewards),
            "min_total_reward": np.min(total_rewards),
            "max_total_reward": np.max(total_rewards),
        }

        if start_times:
            summary.update(
                {
                    "avg_start_time": np.mean(start_times),
                    "std_start_time": np.std(start_times),
                    "min_start_time": np.min(start_times),
                    "max_start_time": np.max(start_times),
                }
            )

        return summary

    def get_shaping_analysis(self) -> Dict:
        """
        Analyze reward shaping effectiveness.

        Returns:
            Dictionary containing shaping analysis metrics
        """
        if not self.use_reward_shaping or not self.shaping_history:
            return {"error": "No shaping data available"}

        base_rewards = [h["base_reward"] for h in self.shaping_history]
        shaped_components = [h["shaped_component"] for h in self.shaping_history]
        total_rewards = [h["total_reward"] for h in self.shaping_history]
        potentials = [h["potential"] for h in self.shaping_history]

        return {
            "episode_length": len(self.shaping_history),
            "avg_base_reward": np.mean(base_rewards),
            "avg_shaped_component": np.mean(shaped_components),
            "avg_total_reward": np.mean(total_rewards),
            "avg_potential": np.mean(potentials),
            "shaping_contribution_pct": 100
            * abs(np.mean(shaped_components))
            / (abs(np.mean(total_rewards)) + 1e-6),
            "potential_std": np.std(potentials),
            "base_reward_range": (min(base_rewards), max(base_rewards)),
            "shaped_component_range": (min(shaped_components), max(shaped_components)),
        }

    def plot_shaping_analysis(self) -> None:
        """Plot reward shaping analysis charts."""
        if not self.use_reward_shaping or not self.shaping_history:
            print("No shaping data to plot")
            return

        steps = [h["step"] for h in self.shaping_history]
        base_rewards = [h["base_reward"] for h in self.shaping_history]
        shaped_components = [h["shaped_component"] for h in self.shaping_history]
        total_rewards = [h["total_reward"] for h in self.shaping_history]
        potentials = [h["potential"] for h in self.shaping_history]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(steps, base_rewards, label="Base Reward", alpha=0.7)
        axes[0, 0].plot(steps, total_rewards, label="Total Reward", alpha=0.7)
        axes[0, 0].set_title("Base vs Total Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(
            steps, shaped_components, label="Shaped Component", color="green", alpha=0.7
        )
        axes[0, 1].set_title("Reward Shaping Contribution")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(
            steps, potentials, label="State Potential Φ(s)", color="purple", alpha=0.7
        )
        axes[1, 0].set_title("State Potential Over Time")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        cum_base = np.cumsum(base_rewards)
        cum_total = np.cumsum(total_rewards)
        axes[1, 1].plot(steps, cum_base, label="Cumulative Base", alpha=0.7)
        axes[1, 1].plot(steps, cum_total, label="Cumulative Total", alpha=0.7)
        axes[1, 1].set_title("Cumulative Rewards")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def save_results(self, filepath: str, controller_name: str = "Unknown") -> Dict:
        """
        Save episode results to JSON file.

        Args:
            filepath: Path to save results
            controller_name: Name of the controller for identification

        Returns:
            Dictionary containing the saved results
        """
        results = {
            "controller_name": controller_name,
            "num_episodes": len(self.episode_history),
            "environment_params": {
                "episode_length": self.episode_length,
                "time_step_seconds": self.time_step_seconds,
                "heating_setpoint": self.initial_heating_setpoint,
                "cooling_setpoint": self.initial_cooling_setpoint,
                "setpoint_pattern": self.setpoint_pattern,
                "reward_type": self.reward_type,
                "energy_weight": self.energy_weight,
                "comfort_weight": self.comfort_weight,
                "random_start_time": self.random_start_time,
                "use_reward_shaping": self.use_reward_shaping,
                "shaping_weight": (
                    self.shaping_weight if self.use_reward_shaping else None
                ),
            },
            "episodes": [],
        }

        for i, episode in enumerate(self.episode_history):
            ground_temps = np.array(episode["ground_temp"])
            top_temps = np.array(episode["top_temp"])
            heating_sp = np.array(episode["heating_setpoint"])
            cooling_sp = np.array(episode["cooling_setpoint"])

            ground_cold_violations = np.maximum(heating_sp - ground_temps, 0)
            ground_hot_violations = np.maximum(ground_temps - cooling_sp, 0)
            top_cold_violations = np.maximum(heating_sp - top_temps, 0)
            top_hot_violations = np.maximum(top_temps - cooling_sp, 0)

            ground_light_energy = np.sum(np.array(episode["ground_light"]))
            top_light_energy = np.sum(np.array(episode["top_light"]))

            episode_data = {
                "episode_id": i,
                "total_reward": np.sum(episode["reward"]),
                "avg_reward": np.mean(episode["reward"]),
                "episode_start_time_offset": episode.get(
                    "episode_start_time_offset", 0.0
                ),
                "comfort_metrics": {
                    "ground_floor_avg_cold_violation": np.mean(ground_cold_violations),
                    "ground_floor_avg_hot_violation": np.mean(ground_hot_violations),
                    "ground_floor_max_violation": max(
                        np.max(ground_cold_violations), np.max(ground_hot_violations)
                    ),
                    "top_floor_avg_cold_violation": np.mean(top_cold_violations),
                    "top_floor_avg_hot_violation": np.mean(top_hot_violations),
                    "top_floor_max_violation": max(
                        np.max(top_cold_violations), np.max(top_hot_violations)
                    ),
                    "time_in_comfort_band_ground_pct": 100
                    * np.mean(
                        (ground_temps >= heating_sp) & (ground_temps <= cooling_sp)
                    ),
                    "time_in_comfort_band_top_pct": 100
                    * np.mean((top_temps >= heating_sp) & (top_temps <= cooling_sp)),
                },
                "energy_metrics": {
                    "ground_light_hours": ground_light_energy
                    * self.time_step_seconds
                    / 3600,
                    "top_light_hours": top_light_energy * self.time_step_seconds / 3600,
                    "total_light_hours": (ground_light_energy + top_light_energy)
                    * self.time_step_seconds
                    / 3600,
                    "ground_window_open_hours": np.sum(
                        np.array(episode["ground_window"])
                    )
                    * self.time_step_seconds
                    / 3600,
                    "top_window_open_hours": np.sum(np.array(episode["top_window"]))
                    * self.time_step_seconds
                    / 3600,
                },
            }

            results["episodes"].append(episode_data)

        if len(self.episode_history) > 0:
            results["overall_metrics"] = {
                "avg_total_reward": np.mean(
                    [ep["total_reward"] for ep in results["episodes"]]
                ),
                "avg_ground_comfort_pct": np.mean(
                    [
                        ep["comfort_metrics"]["time_in_comfort_band_ground_pct"]
                        for ep in results["episodes"]
                    ]
                ),
                "avg_top_comfort_pct": np.mean(
                    [
                        ep["comfort_metrics"]["time_in_comfort_band_top_pct"]
                        for ep in results["episodes"]
                    ]
                ),
                "avg_light_energy": np.mean(
                    [
                        ep["energy_metrics"]["total_light_hours"]
                        for ep in results["episodes"]
                    ]
                ),
                "avg_start_time_offset": np.mean(
                    [ep["episode_start_time_offset"] for ep in results["episodes"]]
                ),
            }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {filepath}")
        return results

    def get_observation_space_info(self) -> Dict:
        """Get information about the current observation space configuration."""
        return {
            "variables": self.observation_variables,
            "space_shape": self.observation_space.shape,
            "space_low": self.observation_space.low.tolist(),
            "space_high": self.observation_space.high.tolist(),
        }
