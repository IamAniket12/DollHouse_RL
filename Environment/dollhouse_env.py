import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
import json
import os


class DollhouseThermalEnv(gym.Env):
    """
    A Gym environment for the dollhouse thermal control problem using a pre-trained SINDy model.

    The environment simulates a two-floor dollhouse with:
    - Controllable lights (ON/OFF) on each floor
    - Controllable windows (OPEN/CLOSED) on each floor
    - Temperature states for ground floor and top floor
    - External temperature (time-varying)

    The goal is to maintain temperatures within desired setpoints for both floors.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

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
    ):
        super(DollhouseThermalEnv, self).__init__()

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

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

        # Initialize state variables
        self.reset()

        # For rendering
        self.fig = None
        self.ax = None

        # Store temperature and action history for all episodes
        self.episode_history = []

        # Create lag variables for temperature history (needed for SINDy features)
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

    def _generate_external_temperature(self) -> np.ndarray:
        """
        Generate external temperature pattern for the entire episode.

        Different patterns can be chosen:
        - sine: sinusoidal pattern with one full cycle per day
        - real_data: simulated pattern based on real data
        - random_walk: random walk with boundaries and time-of-day tendency
        - fixed: constant temperature with some noise

        Args:
            None

        Returns:
            temperatures: A NumPy array of length episode_length with the
                external temperatures for each time step.
        """
        time_steps = self.episode_length

        if self.external_temp_pattern == "sine":
            # Sinusoidal pattern: cooler at night, warmer during the day
            # One full cycle per day
            time = np.linspace(0, 2 * np.pi, time_steps)
            base_temp = 20.0  # Base temperature
            amplitude = 2.0  # Amplitude of the sine wave

            # Generate temperatures with a sine wave plus some noise
            temperatures = (
                base_temp
                + amplitude * np.sin(time - np.pi / 2)
                + np.random.normal(0, 0.5, time_steps)
            )

        elif self.external_temp_pattern == "real_data":
            # In a real implementation, you could load actual temperature data
            # For now, we'll simulate with a more complex pattern
            temperatures = []
            base_temp = 15.0

            for i in range(time_steps):
                hour = (i * self.time_step_seconds / 60) % 24

                # Daily cycle with morning and evening variations
                if hour < 6:  # Night (midnight to 6am)
                    temp = base_temp - 5.0 + hour * 0.3
                elif hour < 12:  # Morning (6am to noon)
                    temp = base_temp - 3.0 + (hour - 6) * 1.5
                elif hour < 18:  # Afternoon (noon to 6pm)
                    temp = base_temp + 7.0 - (hour - 12) * 0.5
                else:  # Evening (6pm to midnight)
                    temp = base_temp + 4.0 - (hour - 18) * 1.5

                # Add some noise
                temp += np.random.normal(0, 1.0)
                temperatures.append(temp)

            temperatures = np.array(temperatures)

        elif self.external_temp_pattern == "random_walk":
            # Random walk with boundaries
            temperatures = np.zeros(time_steps)
            temperatures[0] = 15.0  # Start at 15°C

            for i in range(1, time_steps):
                # Random step with momentum
                step = np.random.normal(0, 1.0)
                # Add time-of-day tendency
                hour = (i * self.time_step_seconds / 60) % 24
                if 9 <= hour < 18:  # Daytime tendency to warm up
                    step += 0.1
                else:  # Nighttime tendency to cool down
                    step -= 0.1

                temperatures[i] = temperatures[i - 1] + step

                # Enforce reasonable boundaries
                temperatures[i] = max(min(temperatures[i], 35.0), -5.0)

        elif self.external_temp_pattern == "fixed":
            # Default to constant temperature with some noise
            base_temp = 20.0
            # temperatures = base_temp
            temperatures = base_temp + np.random.normal(0, 0.2, time_steps)

        return temperatures

    def _update_setpoints(self, time_step: int) -> Tuple[float, float]:
        """
        Update heating and cooling setpoints based on the chosen pattern.

        The setpoint pattern can be one of the following:
        - fixed: fixed setpoints throughout the day
        - schedule: scheduled setpoints based on time of day
        - adaptive: adaptive setpoints that respond to external temperature

        Args:
            time_step: The current time step in the episode

        Returns:
            Tuple[float, float]: The updated heating and cooling setpoints
        """

        # Calculate current hour of the day
        hour = (time_step * self.time_step_seconds / 60) % 24

        if self.setpoint_pattern == "fixed":
            # Fixed setpoints throughout the day
            return self.heating_setpoint, self.cooling_setpoint

        elif self.setpoint_pattern == "schedule":
            # Scheduled setpoints based on time of day
            if 8 <= hour < 18:  # Daytime (8am to 6pm)
                return 21.0, 24.0  # Narrower comfort band during day
            else:  # Night time
                return 19.0, 26.0  # Wider comfort band at night

        elif self.setpoint_pattern == "adaptive":
            # Adaptive setpoints that respond to external temperature
            ext_temp = self.external_temperatures[time_step]

            # Adjust heating setpoint down and cooling setpoint up when external temp is extreme
            if ext_temp < 5:
                return 19.0, 24.0  # Energy-saving during very cold weather
            elif ext_temp > 25:
                return 21.0, 26.0  # Energy-saving during very hot weather
            else:
                return 20.0, 25.0  # Normal setpoints

        else:  # Default to fixed
            return self.heating_setpoint, self.cooling_setpoint

    def _prepare_sindy_features(
        self, state: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        """
        Prepare the input features for the SINDy model.

        The SINDy model expects specific features including the physics-informed ones.

        Args:
            state: Current state of the environment [ground_temp, top_temp, external_temp]
            action: Current action [ground_light, ground_window, top_light, top_window]

        Returns:
            np.ndarray: Prepared features for the SINDy model, reshaped to fit model input
        """
        # Extract values from state and action
        ground_temp = state[0]
        top_temp = state[1]
        external_temp = state[2]
        ground_light = action[0]
        ground_window = action[1]
        top_light = action[2]
        top_window = action[3]

        # Time difference in seconds
        time_diff_seconds = self.time_step_seconds

        # Create physics-informed features
        # Temperature differences (heat transfer drivers)
        floor_temp_diff = top_temp - ground_temp
        ground_ext_temp_diff = ground_temp - external_temp
        top_ext_temp_diff = top_temp - external_temp

        # Window effects (accelerated heat transfer when open)
        ground_window_ext_effect = ground_window * ground_ext_temp_diff
        top_window_ext_effect = top_window * top_ext_temp_diff

        # Lag features for thermal inertia (using history)
        ground_temp_lag1 = self.ground_temp_history[-1]
        top_temp_lag1 = self.top_temp_history[-1]
        ext_temp_lag1 = self.external_temp_history[-1]

        ground_temp_lag2 = self.ground_temp_history[-2]
        top_temp_lag2 = self.top_temp_history[-2]
        ext_temp_lag2 = self.external_temp_history[-2]

        # Temperature rate of change (based on history)
        if len(self.ground_temp_history) >= 2:
            ground_temp_rate = (ground_temp - ground_temp_lag1) / time_diff_seconds
            top_temp_rate = (top_temp - top_temp_lag1) / time_diff_seconds
        else:
            ground_temp_rate = 0.0
            top_temp_rate = 0.0

        # Assemble all features - match the exact order used during SINDy training
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

        # Reshape for the SINDy model
        return u.reshape(1, -1)

    def _calculate_reward(
        self, ground_temp: float, top_temp: float, action: np.ndarray
    ) -> float:
        """
        Calculate the reward based on comfort and energy use.

        The reward function is a weighted sum of comfort and energy penalties.
        Comfort is penalized when temperatures are far from their setpoints.
        Energy is penalized when lights are used, and when windows are open during
        hot or cold weather.

        Args:
            ground_temp: Current ground floor temperature
            top_temp: Current top floor temperature
            action: Current action (used to calculate energy penalties)

        Returns:
            float: The calculated reward
        """
        # Comfort penalty (how far temperatures are from setpoints)
        ground_comfort_penalty = 0
        top_comfort_penalty = 0

        # For heating mode (temp below heating setpoint)
        if ground_temp < self.heating_setpoint:
            ground_comfort_penalty = abs(self.heating_setpoint - ground_temp) ** 2
        # For cooling mode (temp above cooling setpoint)
        elif ground_temp > self.cooling_setpoint:
            ground_comfort_penalty = abs(ground_temp - self.cooling_setpoint) ** 2

        # Same for top floor
        if top_temp < self.heating_setpoint:
            top_comfort_penalty = abs(self.heating_setpoint - top_temp) ** 2
        elif top_temp > self.cooling_setpoint:
            top_comfort_penalty = abs(top_temp - self.cooling_setpoint) ** 2

        # Total comfort penalty
        comfort_penalty = (
            ground_comfort_penalty + top_comfort_penalty
        ) * self.comfort_weight

        # Energy penalty (lights consume energy)
        energy_use = action[0] + action[2]  # ground_light + top_light
        energy_penalty = energy_use * self.energy_weight

        # Opening windows may have energy implications depending on outside temperature
        ext_temp = self.external_temperatures[self.current_step]

        window_penalty = 0
        # If it's cold outside and windows are open (heating wasted)
        if ext_temp < self.heating_setpoint:
            window_penalty += (
                (action[1] + action[3])
                * abs(ext_temp - self.heating_setpoint)
                * self.energy_weight
            )
        # If it's hot outside and windows are open (cooling wasted)
        elif ext_temp > self.cooling_setpoint:
            window_penalty += (
                (action[1] + action[3])
                * abs(ext_temp - self.cooling_setpoint)
                * self.energy_weight
            )

        # Calculate final reward (negative penalties)
        if self.reward_type == "comfort":
            # Prioritize comfort over energy use
            reward = -comfort_penalty - 0.1 * (energy_penalty + window_penalty)
        elif self.reward_type == "energy":
            # Prioritize energy savings over strict comfort
            reward = -0.1 * comfort_penalty - (energy_penalty + window_penalty)
        elif self.reward_type == "balanced":
            # Equal weighting
            reward = -comfort_penalty - (energy_penalty + window_penalty)
        else:
            # Default to balanced
            reward = -comfort_penalty - (energy_penalty + window_penalty)

        return reward

    def reset(self):
        """Reset the environment to initial state."""
        # Reset time step counter
        self.current_step = 0

        # Set initial temperatures
        self.ground_temp = self.initial_ground_temp
        self.top_temp = self.initial_top_temp

        # Generate external temperature pattern for the episode
        self.external_temperatures = self._generate_external_temperature()

        # Set initial setpoints
        self.heating_setpoint = self.initial_heating_setpoint
        self.cooling_setpoint = self.initial_cooling_setpoint

        # Init action to all OFF/CLOSED
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

        # Track history for plotting
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
        }

        # Calculate hour of day (0-23)
        # For instance, when calculating hour_of_day:
        hour_of_day = (self.current_step * self.time_step_seconds / 3600) % 24

        # Return the initial observation
        return np.array(
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

    def step(self, action):
        """
        Take a step in the environment given the action.

        Args:
            action: [ground_light, ground_window, top_light, top_window]
                   Each value is 0 (OFF/CLOSED) or 1 (ON/OPEN)

        Returns:
            observation: The new state
            reward: The reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Store action for reference
        self.current_action = action

        # Get current state
        current_state = np.array([self.ground_temp, self.top_temp])

        # Update temperature history
        self.ground_temp_history.append(self.ground_temp)
        self.top_temp_history.append(self.top_temp)
        self.external_temp_history.append(
            self.external_temperatures[min(self.current_step, self.episode_length - 1)]
        )

        # Keep only the latest 3 entries (current + 2 lags)
        if len(self.ground_temp_history) > 3:
            self.ground_temp_history.pop(0)
            self.top_temp_history.pop(0)
            self.external_temp_history.pop(0)

        # Prepare input features for SINDy model
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

        # Predict next state with SINDy model

        next_state = self.sindy_model.predict(
            current_state.reshape(1, -1), u=u_features
        )[0]

        # Update temperatures
        self.ground_temp, self.top_temp = next_state

        # Update setpoints based on time
        self.heating_setpoint, self.cooling_setpoint = self._update_setpoints(
            self.current_step
        )

        # Calculate reward
        reward = self._calculate_reward(self.ground_temp, self.top_temp, action)

        # Update step counter
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= self.episode_length

        # Calculate hour of day (0-23)
        # For instance, when calculating hour_of_day:
        hour_of_day = (self.current_step * self.time_step_seconds / 3600) % 24

        # Prepare the observation
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

        # Store episode history if episode is done
        if done:
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
        }

        return obs, reward, done, info

    def seed(self, seed=None):
        """Set the seed for this environment's random number generator."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def render(self, mode="human"):
        """
        Render the environment state.

        Args:
            mode: 'human' for displaying plots, 'rgb_array' for returning an image
        """
        if self.fig is None or self.ax is None:
            # Create figure and axes
            self.fig, self.ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            plt.ion()  # Interactive mode

        # Clear previous plots
        for a in self.ax:
            a.clear()

        # Time steps for x-axis
        time_steps = range(len(self.history["ground_temp"]))
        time_hours = [t * self.time_step_seconds / 3600 for t in time_steps]

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
        self.ax[0].set_title("Dollhouse Thermal Environment")

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

        # Draw plot
        self.fig.canvas.draw()
        plt.pause(0.1)

        # Return rgb array if needed
        if mode == "rgb_array":
            return np.transpose(
                np.array(self.fig.canvas.renderer.buffer_rgba()), (2, 0, 1)
            )

    def close(self):
        """Clean up resources."""
        if self.fig is not None:
            plt.close(self.fig)
            plt.ioff()  # Turn off interactive mode
            self.fig = None
            self.ax = None

    def get_performance_summary(self):
        """
        Return a summary of the environment's performance.

        Returns:
            dict: Performance metrics across all episodes
        """
        if not self.episode_history:
            return {"error": "No episodes completed yet"}

        # Calculate comfort metrics
        ground_comfort_pcts = []
        top_comfort_pcts = []
        total_rewards = []
        light_hours = []

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
            # When calculating energy metrics in hours:
            light_hours.append(
                (
                    np.sum(np.array(episode["ground_light"]))
                    + np.sum(np.array(episode["top_light"]))
                )
                * self.time_step_seconds
                / 3600
            )

        return {
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

    def save_results(self, filepath, controller_name="Unknown"):
        """
        Save episode results to a JSON file.

        Args:
            filepath: Path to save the results
            controller_name: Name of the controller for reference
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
            }

        # Save results
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {filepath}")
        return results
