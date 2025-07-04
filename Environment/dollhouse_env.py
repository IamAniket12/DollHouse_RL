import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
import json
import os


class DollhouseThermalEnv(gym.Env):
    """
    A Gymnasium environment for the dollhouse thermal control problem using a pre-trained SINDy model.

    Enhanced with:
    - Random start time feature for diverse training scenarios
    - Andrew Ng style reward shaping for accelerated learning

    The environment simulates a two-floor dollhouse with:
    - Controllable lights (ON/OFF) on each floor
    - Controllable windows (OPEN/CLOSED) on each floor
    - Temperature states for ground floor and top floor
    - External temperature (time-varying)

    The goal is to maintain temperatures within desired setpoints for both floors.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        sindy_model,  # Pre-trained SINDy model
        external_temp_pattern: str = "fixed",
        episode_length: int = 2880,  # 24 hours with 30-second timesteps (24*60*60/30)
        time_step_seconds: int = 30,
        heating_setpoint: float = 30.0,  # °C
        cooling_setpoint: float = 35.0,  # °C
        initial_ground_temp: float = 22.0,  # °C
        initial_top_temp: float = 23.0,  # °C
        reward_type: str = "comfort",
        energy_weight: float = 0.5,
        comfort_weight: float = 1.0,
        random_seed: Optional[int] = None,
        setpoint_pattern: str = "fixed",
        render_mode: Optional[str] = None,
        # NEW: Random start time parameters
        random_start_time: bool = False,
        start_time_range: tuple = (0, 24),  # Hours range for random start
        # NEW: Reward shaping parameters
        use_reward_shaping: bool = False,
        shaping_gamma: float = 0.99,  # Discount factor for shaping
        shaping_weight: float = 0.3,  # Overall shaping influence
        comfort_potential_weight: float = 1.0,  # Weight for comfort potential
        energy_potential_weight: float = 0.5,  # Weight for energy potential
        comfort_decay_rate: float = 0.4,  # Exponential decay for temperature deviations
    ):
        super().__init__()

        # Validate render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Store parameters
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

        # Random start time parameters
        self.random_start_time = random_start_time
        self.start_time_range = start_time_range

        # Reward shaping parameters
        self.use_reward_shaping = use_reward_shaping
        self.shaping_gamma = shaping_gamma
        self.shaping_weight = shaping_weight
        self.comfort_potential_weight = comfort_potential_weight
        self.energy_potential_weight = energy_potential_weight
        self.comfort_decay_rate = comfort_decay_rate

        # Store the pre-trained SINDy model
        self.sindy_model = sindy_model

        # Action space (binary variables for lights and windows on both floors)
        # [ground_light, ground_window, top_light, top_window]
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2])

        # Observation space
        # [ground_temp, top_temp, external_temp, ground_light, ground_window, top_light, top_window,
        #  heating_setpoint, cooling_setpoint, hour_of_day, time_step]
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -10.0, -30.0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 0]),
            high=np.array(
                [50.0, 50.0, 50.0, 1, 1, 1, 1, 35.0, 35.0, 23.0, episode_length]
            ),
            dtype=np.float32,
        )

        # For rendering
        self.fig = None
        self.ax = None

        # Store temperature and action history for all episodes
        self.episode_history = []

        # Initialize RNG
        self.np_random = None
        self._rng = np.random.default_rng(random_seed)

        # Will be initialized in reset()
        self.current_step = 0
        self.episode_start_time_offset = 0  # NEW: For random start time
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

        # Reward shaping state variables
        self.previous_potential = 0.0
        self.shaping_history = []

    def _generate_external_temperature(
        self, start_offset_hours: float = 0.0
    ) -> np.ndarray:
        """
        Generate external temperature pattern for the entire episode.

        Args:
            start_offset_hours: Hour offset to start the pattern from (for random start time)
        """
        time_steps = self.episode_length

        # Convert start offset to time steps
        start_offset_steps = int(start_offset_hours * 3600 / self.time_step_seconds)

        if self.external_temp_pattern == "sine":
            # Sinusoidal pattern: cooler at night, warmer during the day
            # Generate a longer pattern to account for offset
            total_steps = time_steps + start_offset_steps
            time = np.linspace(
                0, 2 * np.pi * total_steps / 2880, total_steps
            )  # 2880 steps = 24 hours
            base_temp = 20.0
            amplitude = 2.0

            all_temperatures = (
                base_temp
                + amplitude * np.sin(time - np.pi / 2)
                + self.np_random.normal(0, 0.5, total_steps)
            )

            # Extract the episode portion starting from offset
            temperatures = all_temperatures[
                start_offset_steps : start_offset_steps + time_steps
            ]

        elif self.external_temp_pattern == "real_data":
            temperatures = []
            base_temp = 15.0

            for i in range(time_steps):
                # Calculate actual hour considering offset
                actual_hour = (
                    start_offset_hours + (i * self.time_step_seconds / 3600)
                ) % 24

                if actual_hour < 6:  # Night
                    temp = base_temp - 5.0 + actual_hour * 0.3
                elif actual_hour < 12:  # Morning
                    temp = base_temp - 3.0 + (actual_hour - 6) * 1.5
                elif actual_hour < 18:  # Afternoon
                    temp = base_temp + 7.0 - (actual_hour - 12) * 0.5
                else:  # Evening
                    temp = base_temp + 4.0 - (actual_hour - 18) * 1.5

                temp += self.np_random.normal(0, 1.0)
                temperatures.append(temp)

            temperatures = np.array(temperatures)

        elif self.external_temp_pattern == "random_walk":
            # For random walk, start from a temperature appropriate for the start time
            start_hour = start_offset_hours % 24
            if 6 <= start_hour < 18:  # Daytime
                initial_temp = 18.0
            else:  # Nighttime
                initial_temp = 12.0

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
                temperatures[i] = max(min(temperatures[i], 35.0), -5.0)

        else:  # fixed
            base_temp = 20.0
            temperatures = base_temp + self.np_random.normal(0, 0.3, time_steps)

        return temperatures

    def _update_setpoints(self, time_step: int) -> Tuple[float, float]:
        """Update heating and cooling setpoints based on the chosen pattern."""
        # Calculate actual hour considering the episode start offset
        actual_hour = (
            self.episode_start_time_offset + (time_step * self.time_step_seconds / 3600)
        ) % 24

        if self.setpoint_pattern == "fixed":
            return self.heating_setpoint, self.cooling_setpoint

        elif self.setpoint_pattern == "schedule":
            if 11 <= actual_hour < 18:  # Daytime
                return 22.0, 24.0
            elif 8 <= actual_hour < 11:
                # return 20.0, 22.0
                return 26.0, 28.0
            else:  # Night time
                return 20.0, 24.0

        elif self.setpoint_pattern == "adaptive":
            ext_temp = self.external_temperatures[time_step]
            if ext_temp < 5:
                return 19.0, 24.0
            elif ext_temp > 25:
                return 21.0, 26.0
            else:
                return 20.0, 25.0
        elif self.setpoint_pattern == "challenging":
            """
            Challenging pattern designed to test RL agents vs rule-based controllers:
            - Starts with tight control (22-24°C) for initial period
            - Expands to broader range (20-27°C) to test adaptability
            - Includes periodic tightening to test responsiveness
            """
            # Convert time_step to minutes from episode start
            minutes_elapsed = (time_step * self.time_step_seconds) / 60

            # Phase 1: Initial tight control (first 30 minutes)
            if minutes_elapsed < 30:
                return 22.0, 24.0

            # Phase 2: Expanded range (30-90 minutes)
            elif 30 <= minutes_elapsed < 90:
                return 20.0, 27.0

            # Phase 3: Periodic cycling between tight and loose control
            elif 90 <= minutes_elapsed < 240:  # 90-240 minutes (2.5-4 hours)
                # 20-minute cycles: 10 min tight, 10 min loose
                cycle_position = (minutes_elapsed - 90) % 20
                if cycle_position < 10:
                    return 22.5, 23.5  # Very tight control
                else:
                    return 19.0, 28.0  # Very loose control

            # Phase 4: Time-of-day dependent with external temperature influence
            elif 240 <= minutes_elapsed < 480:  # 4-8 hours
                ext_temp = self.external_temperatures[
                    min(time_step, len(self.external_temperatures) - 1)
                ]

                # Base setpoints vary by time of day
                if 6 <= actual_hour < 12:  # Morning
                    base_heating, base_cooling = 21.0, 25.0
                elif 12 <= actual_hour < 18:  # Afternoon
                    base_heating, base_cooling = 23.0, 26.0
                elif 18 <= actual_hour < 22:  # Evening
                    base_heating, base_cooling = 22.0, 24.0
                else:  # Night
                    base_heating, base_cooling = 20.0, 26.0

                # Adjust based on external temperature
                if ext_temp < 15:
                    return base_heating - 1.0, base_cooling - 1.0
                elif ext_temp > 25:
                    return base_heating + 1.0, base_cooling + 1.0
                else:
                    return base_heating, base_cooling

            # Phase 5: Final challenge - narrow moving window
            else:  # After 8 hours
                # Create a "moving comfort zone" that shifts over time
                minutes_in_phase = minutes_elapsed - 480

                # Sinusoidal variation with 60-minute period
                center_temp = 23.5 + 2.0 * np.sin(2 * np.pi * minutes_in_phase / 60)

                # Narrow 1.5°C window around the moving center
                heating_sp = center_temp - 0.75
                cooling_sp = center_temp + 0.75

                # Clamp to reasonable bounds
                heating_sp = max(18.0, min(heating_sp, 26.0))
                cooling_sp = max(20.0, min(cooling_sp, 30.0))

                return heating_sp, cooling_sp
        else:
            return self.heating_setpoint, self.cooling_setpoint

    def _prepare_sindy_features(
        self, state: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        """Prepare the input features for the SINDy model."""
        ground_temp = state[0]
        top_temp = state[1]
        external_temp = state[2]
        ground_light = action[0]
        ground_window = action[1]
        top_light = action[2]
        top_window = action[3]

        time_diff_seconds = self.time_step_seconds

        # Physics-informed features
        floor_temp_diff = top_temp - ground_temp
        ground_ext_temp_diff = ground_temp - external_temp
        top_ext_temp_diff = top_temp - external_temp

        ground_window_ext_effect = ground_window * ground_ext_temp_diff
        top_window_ext_effect = top_window * top_ext_temp_diff

        # Lag features
        ground_temp_lag1 = self.ground_temp_history[-1]
        top_temp_lag1 = self.top_temp_history[-1]
        ext_temp_lag1 = self.external_temp_history[-1]

        ground_temp_lag2 = self.ground_temp_history[-2]
        top_temp_lag2 = self.top_temp_history[-2]
        ext_temp_lag2 = self.external_temp_history[-2]

        # Temperature rate of change
        if len(self.ground_temp_history) >= 2:
            ground_temp_rate = (ground_temp - ground_temp_lag1) / time_diff_seconds
            top_temp_rate = (top_temp - top_temp_lag1) / time_diff_seconds
        else:
            ground_temp_rate = 0.0
            top_temp_rate = 0.0

        u = np.array(
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

        return u.reshape(1, -1)

    def _calculate_base_reward(
        self, ground_temp: float, top_temp: float, action: np.ndarray
    ) -> float:
        # Comfort component
        ground_comfortable = (
            self.heating_setpoint <= ground_temp <= self.cooling_setpoint
        )
        top_comfortable = self.heating_setpoint <= top_temp <= self.cooling_setpoint

        if ground_comfortable and top_comfortable:
            comfort_reward = 1.0 * self.comfort_weight

            # Energy efficiency bonus when comfortable
            lights_on = action[0] + action[2]
            # energy_bonus = 0.5 * (1.0 - lights_on / 2.0)  # 0.0 to 0.3 bonus
            energy_bonus = self.energy_weight * (1.0 - lights_on / 2.0)
            return comfort_reward + energy_bonus  # 1.0 to 1.3
        else:
            return 0.0

    def _comfort_zone_potential(
        self, ground_temp: float, top_temp: float, heating_sp: float, cooling_sp: float
    ) -> float:
        """
        Comfort zone potential - flat within zone, exponential gradient outside.
        Does NOT bias toward center, allows strategic positioning within comfort zone.
        """

        def zone_potential(temp):
            if heating_sp <= temp <= cooling_sp:
                return 1.0  # Flat reward within comfort zone (no center bias)
            else:
                if temp < heating_sp:
                    distance = heating_sp - temp
                    return np.exp(-self.comfort_decay_rate * distance)
                else:  # temp > cooling_sp
                    distance = temp - cooling_sp
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
        Energy efficiency potential - rewards achieving comfort with minimal energy.
        """
        ground_in_comfort = heating_sp <= ground_temp <= cooling_sp
        top_in_comfort = heating_sp <= top_temp <= cooling_sp
        both_comfortable = ground_in_comfort and top_in_comfort

        lights_on = action[0] + action[2]

        if both_comfortable:
            return 1.0 - 0.3 * (lights_on / 2.0)  # Linear penalty for energy use
        else:
            return 0.2 - 0.1 * (
                lights_on / 2.0
            )  # Small baseline with slight energy penalty

    def _calculate_total_potential(
        self, ground_temp: float, top_temp: float, action: np.ndarray
    ) -> float:
        """
        Combined potential function: Φ(s) = Φ_comfort(s) + Φ_energy(s)
        """
        # Comfort zone potential (uniform within zone, gradient outside)
        comfort_potential = self._comfort_zone_potential(
            ground_temp, top_temp, self.heating_setpoint, self.cooling_setpoint
        )

        # Energy efficiency potential
        energy_potential = self._energy_efficiency_potential(
            ground_temp, top_temp, action, self.heating_setpoint, self.cooling_setpoint
        )

        # Weighted combination
        return (
            self.comfort_potential_weight * comfort_potential
            + self.energy_potential_weight * energy_potential
        )

    def _calculate_reward(
        self, ground_temp: float, top_temp: float, action: np.ndarray
    ) -> float:
        """
        Andrew Ng style reward shaping: R'(s,a,s') = R(s,a,s') + F(s,a,s')
        where F(s,a,s') = γ * Φ(s') - Φ(s)
        """
        # Simplified base reward (binary comfort + basic energy check)
        R_base = self._calculate_base_reward(ground_temp, top_temp, action)

        if not self.use_reward_shaping:
            return R_base

        # Calculate current state potential Φ(s')
        current_potential = self._calculate_total_potential(
            ground_temp, top_temp, action
        )

        # Potential-based shaping function: F(s,a,s') = γ * Φ(s') - Φ(s)
        F_shaping = self.shaping_gamma * current_potential - self.previous_potential

        # Total shaped reward: R'(s,a,s') = R(s,a,s') + λ * F(s,a,s')
        R_shaped = R_base + self.shaping_weight * F_shaping

        # Update potential for next timestep
        self.previous_potential = current_potential

        # Store shaping history for analysis
        self.shaping_history.append(
            {
                "step": self.current_step,
                "base_reward": R_base,
                "potential": current_potential,
                "shaped_component": self.shaping_weight * F_shaping,
                "total_reward": R_shaped,
            }
        )

        return R_shaped

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        # Set up RNG
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset state
        self.current_step = 0
        self.ground_temp = self.initial_ground_temp
        self.top_temp = self.initial_top_temp

        # NEW: Generate random start time if enabled
        if self.random_start_time:
            start_hour_min, start_hour_max = self.start_time_range
            self.episode_start_time_offset = self._rng.uniform(
                start_hour_min, start_hour_max
            )
        else:
            self.episode_start_time_offset = 0.0

        # Generate external temperatures with start offset
        self.external_temperatures = self._generate_external_temperature(
            start_offset_hours=self.episode_start_time_offset
        )

        # Set initial setpoints
        self.heating_setpoint = self.initial_heating_setpoint
        self.cooling_setpoint = self.initial_cooling_setpoint

        # Init action
        self.current_action = np.zeros(4)

        # Reset temperature history
        self.ground_temp_history = [
            self.initial_ground_temp,
            self.initial_ground_temp,
            self.initial_ground_temp,
        ]
        self.top_temp_history = [
            self.initial_top_temp,
            self.initial_top_temp,
            self.initial_top_temp,
        ]
        self.external_temp_history = [
            self.external_temperatures[0],
            self.external_temperatures[0],
            self.external_temperatures[0],
        ]

        # NEW: Initialize reward shaping state
        if self.use_reward_shaping:
            # Calculate initial potential for first state
            initial_action = np.zeros(4)
            self.previous_potential = self._calculate_total_potential(
                self.ground_temp, self.top_temp, initial_action
            )
            self.shaping_history = []

        # Track history
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
            "episode_start_time_offset": self.episode_start_time_offset,  # NEW: Track start time
        }

        # Calculate hour of day (considering offset)
        hour_of_day = (
            self.episode_start_time_offset
            + (self.current_step * self.time_step_seconds / 3600)
        ) % 24

        observation = np.array(
            [
                self.ground_temp,
                self.top_temp,
                self.external_temperatures[0],
                0,  # ground_light
                0,  # ground_window
                0,  # top_light
                0,  # top_window
                self.heating_setpoint,
                self.cooling_setpoint,
                hour_of_day,
                self.current_step,
            ],
            dtype=np.float32,
        )

        info = {
            "episode_start_time_offset": self.episode_start_time_offset,  # NEW: Include in info
        }

        return observation, info

    def step(self, action):
        """Take a step in the environment."""
        # Store action
        self.current_action = action

        # Get current state
        current_state = np.array([self.ground_temp, self.top_temp])

        # Update temperature history
        self.ground_temp_history.append(self.ground_temp)
        self.top_temp_history.append(self.top_temp)
        self.external_temp_history.append(
            self.external_temperatures[min(self.current_step, self.episode_length - 1)]
        )

        # Keep only latest 3 entries
        if len(self.ground_temp_history) > 3:
            self.ground_temp_history.pop(0)
            self.top_temp_history.pop(0)
            self.external_temp_history.pop(0)

        # Prepare features for SINDy
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

        # Predict next state
        next_state = self.sindy_model.predict(
            current_state.reshape(1, -1), u=u_features
        )[0]

        # Update temperatures
        self.ground_temp, self.top_temp = next_state

        # Update setpoints (now considers random start time)
        self.heating_setpoint, self.cooling_setpoint = self._update_setpoints(
            self.current_step
        )

        # Calculate reward (with optional shaping)
        reward = self._calculate_reward(self.ground_temp, self.top_temp, action)

        # Update step counter
        self.current_step += 1

        # Check termination
        terminated = False  # Episode doesn't end due to failure
        truncated = self.current_step >= self.episode_length  # Time limit reached

        # Calculate hour of day (considering offset)
        hour_of_day = (
            self.episode_start_time_offset
            + (self.current_step * self.time_step_seconds / 3600)
        ) % 24

        # Prepare observation
        obs = np.array(
            [
                self.ground_temp,
                self.top_temp,
                self.external_temperatures[
                    min(self.current_step, self.episode_length - 1)
                ],
                action[0],  # ground_light
                action[1],  # ground_window
                action[2],  # top_light
                action[3],  # top_window
                self.heating_setpoint,
                self.cooling_setpoint,
                hour_of_day,
                self.current_step,
            ],
            dtype=np.float32,
        )

        # Update history
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

        # Store episode history if done
        if truncated:
            self.episode_history.append(self.history)

        # Additional info
        info = {
            "ground_temp": self.ground_temp,
            "top_temp": self.top_temp,
            "external_temp": self.external_temperatures[
                min(self.current_step, self.episode_length - 1)
            ],
            "ground_comfort_violation": max(self.heating_setpoint - self.ground_temp, 0)
            + max(self.ground_temp - self.cooling_setpoint, 0),
            "top_comfort_violation": max(self.heating_setpoint - self.top_temp, 0)
            + max(self.top_temp - self.cooling_setpoint, 0),
            "energy_use": action[0] + action[2],  # lights
            "episode_start_time_offset": self.episode_start_time_offset,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment state."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            return None

    def _render_frame(self):
        """Internal method to render a frame."""
        if self.fig is None or self.ax is None:
            # Create figure and axes
            self.fig, self.ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            if self.render_mode == "human":
                plt.ion()  # Interactive mode

        # Clear previous plots
        for a in self.ax:
            a.clear()

        # Time steps for x-axis (adjusted for start time offset)
        time_steps = range(len(self.history["ground_temp"]))
        time_hours = [
            (self.episode_start_time_offset + t * self.time_step_seconds / 3600) % 24
            for t in time_steps
        ]

        # Plot temperatures
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

        # Update title to show start time
        title = f"Dollhouse Thermal Environment (Start: {self.episode_start_time_offset:.1f}h)"
        self.ax[0].set_title(title)

        # Plot control actions
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

        # Plot reward
        self.ax[2].plot(time_hours, self.history["reward"], "k-")
        self.ax[2].set_xlabel("Time (hours)")
        self.ax[2].set_ylabel("Reward")
        self.ax[2].grid(True)

        # Adjust layout
        plt.tight_layout()

        if self.render_mode == "human":
            # Draw plot
            self.fig.canvas.draw()
            plt.pause(0.1)
            return None
        else:  # rgb_array
            # Return rgb array
            self.fig.canvas.draw()
            return np.transpose(
                np.array(self.fig.canvas.renderer.buffer_rgba()), (2, 0, 1)
            )

    def close(self):
        """Clean up resources."""
        if self.fig is not None:
            plt.close(self.fig)
            plt.ioff()
            self.fig = None
            self.ax = None

    def get_performance_summary(self):
        """Return a summary of the environment's performance."""
        if not self.episode_history:
            return {"error": "No episodes completed yet"}

        # Calculate comfort metrics
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

            # Comfort percentages
            ground_comfort = (
                np.mean((ground_temps >= heating_sp) & (ground_temps <= cooling_sp))
                * 100
            )
            top_comfort = (
                np.mean((top_temps >= heating_sp) & (top_temps <= cooling_sp)) * 100
            )

            ground_comfort_pcts.append(ground_comfort)
            top_comfort_pcts.append(top_comfort)

            # Reward
            total_rewards.append(np.sum(episode["reward"]))

            # Energy (light hours)
            light_hours.append(
                (
                    np.sum(np.array(episode["ground_light"]))
                    + np.sum(np.array(episode["top_light"]))
                )
                * self.time_step_seconds
                / 3600
            )

            # Start times (if available)
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

        # Add start time statistics if available
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

    def get_shaping_analysis(self):
        """Analyze the contribution and effectiveness of reward shaping."""
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

    def plot_shaping_analysis(self):
        """Plot the components of reward shaping over time."""
        if not self.use_reward_shaping or not self.shaping_history:
            print("No shaping data to plot")
            return

        steps = [h["step"] for h in self.shaping_history]
        base_rewards = [h["base_reward"] for h in self.shaping_history]
        shaped_components = [h["shaped_component"] for h in self.shaping_history]
        total_rewards = [h["total_reward"] for h in self.shaping_history]
        potentials = [h["potential"] for h in self.shaping_history]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Base vs Total Reward
        axes[0, 0].plot(steps, base_rewards, label="Base Reward", alpha=0.7)
        axes[0, 0].plot(steps, total_rewards, label="Total Reward", alpha=0.7)
        axes[0, 0].set_title("Base vs Total Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Shaped Component
        axes[0, 1].plot(
            steps, shaped_components, label="Shaped Component", color="green", alpha=0.7
        )
        axes[0, 1].set_title("Reward Shaping Contribution")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Potential Function
        axes[1, 0].plot(
            steps, potentials, label="State Potential Φ(s)", color="purple", alpha=0.7
        )
        axes[1, 0].set_title("State Potential Over Time")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Cumulative rewards
        cum_base = np.cumsum(base_rewards)
        cum_total = np.cumsum(total_rewards)
        axes[1, 1].plot(steps, cum_base, label="Cumulative Base", alpha=0.7)
        axes[1, 1].plot(steps, cum_total, label="Cumulative Total", alpha=0.7)
        axes[1, 1].set_title("Cumulative Rewards")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def save_results(self, filepath, controller_name="Unknown"):
        """Save episode results to a JSON file."""
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
                "random_start_time": self.random_start_time,  # NEW
                "use_reward_shaping": self.use_reward_shaping,  # NEW
                "shaping_weight": (
                    self.shaping_weight if self.use_reward_shaping else None
                ),  # NEW
            },
            "episodes": [],
        }

        # Process each episode
        for i, episode in enumerate(self.episode_history):
            # Calculate comfort and energy metrics
            ground_temps = np.array(episode["ground_temp"])
            top_temps = np.array(episode["top_temp"])
            heating_sp = np.array(episode["heating_setpoint"])
            cooling_sp = np.array(episode["cooling_setpoint"])

            # Comfort violations
            ground_cold_violations = np.maximum(heating_sp - ground_temps, 0)
            ground_hot_violations = np.maximum(ground_temps - cooling_sp, 0)
            top_cold_violations = np.maximum(heating_sp - top_temps, 0)
            top_hot_violations = np.maximum(top_temps - cooling_sp, 0)

            # Energy use
            ground_light_energy = np.sum(np.array(episode["ground_light"]))
            top_light_energy = np.sum(np.array(episode["top_light"]))

            # Episode summary
            episode_data = {
                "episode_id": i,
                "total_reward": np.sum(episode["reward"]),
                "avg_reward": np.mean(episode["reward"]),
                "episode_start_time_offset": episode.get(
                    "episode_start_time_offset", 0.0
                ),  # NEW
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

        # Calculate overall averages
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
                ),  # NEW
            }

        # Save results
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {filepath}")
        return results


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
# Example 1: Original behavior (no new features)
env = DollhouseThermalEnv(
    sindy_model=your_model,
    random_start_time=False,
    use_reward_shaping=False,
    # ... your other parameters
)

# Example 2: With random start time only
env = DollhouseThermalEnv(
    sindy_model=your_model,
    random_start_time=True,
    start_time_range=(0, 24),  # Full 24-hour range
    use_reward_shaping=False,
    # ... your other parameters
)

# Example 3: With reward shaping only
env = DollhouseThermalEnv(
    sindy_model=your_model,
    random_start_time=False,
    use_reward_shaping=True,
    shaping_weight=0.3,
    comfort_potential_weight=1.0,
    energy_potential_weight=0.5,
    comfort_decay_rate=0.4,
    # ... your other parameters
)

# Example 4: With both features enabled
env = DollhouseThermalEnv(
    sindy_model=your_model,
    random_start_time=True,
    start_time_range=(0, 24),
    use_reward_shaping=True,
    shaping_weight=0.3,
    comfort_potential_weight=1.0,
    energy_potential_weight=0.5,
    comfort_decay_rate=0.4,
    # ... your other parameters
)

# Analyze shaping performance
analysis = env.get_shaping_analysis()
print(f"Shaping contributes {analysis['shaping_contribution_pct']:.1f}% of total reward")

# Visualize shaping components
env.plot_shaping_analysis()

# Check performance summary with start time stats
summary = env.get_performance_summary()
print(f"Average start time: {summary.get('avg_start_time', 'N/A'):.2f} hours")
"""
