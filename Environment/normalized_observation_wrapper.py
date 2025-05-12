import gym
import numpy as np


class NormalizedObservationWrapper(gym.Wrapper):
    """
    A wrapper that normalizes observations using predefined min/max ranges.
    Includes built-in default ranges for dollhouse environment observations.
    """

    # Default ranges for dollhouse environment observations
    DEFAULT_DOLLHOUSE_RANGES = {
        0: (0.0, 40.0),  # Ground floor temperature (°C)
        1: (0.0, 40.0),  # Top floor temperature (°C)
        2: (-10.0, 40.0),  # External temperature (°C)
        3: (0.0, 1.0),  # Ground light (binary)
        4: (0.0, 1.0),  # Ground window (binary)
        5: (0.0, 1.0),  # Top light (binary)
        6: (0.0, 1.0),  # Top window (binary)
        7: (10.0, 35.0),  # Heating setpoint (°C)
        8: (10.0, 35.0),  # Cooling setpoint (°C)
        9: (0.0, 23.0),  # Hour of day
        10: (0, 2880),  # Time step
    }

    def __init__(self, env, ranges=None, zero_center=True):
        """
        Initialize the NormalizedObservationWrapper.

        Args:
            env: The environment to wrap
            ranges: Dictionary mapping observation indices to (min, max) tuples
                   If None, will use the DEFAULT_DOLLHOUSE_RANGES
            zero_center: If True, normalize to [-1, 1], otherwise to [0, 1]
        """
        super(NormalizedObservationWrapper, self).__init__(env)

        # Store settings
        self.zero_center = zero_center

        # Store original observation for reference
        self.original_obs = None

        # Set up normalization ranges
        if ranges is None:
            # Use default dollhouse ranges
            self.ranges = self.DEFAULT_DOLLHOUSE_RANGES
        else:
            self.ranges = ranges

        # Ensure all observation dimensions have a range
        for i in range(self.observation_space.shape[0]):
            if i not in self.ranges:
                # Use observation space bounds if available, otherwise default to (-1, 1)
                if hasattr(self.observation_space, "low") and hasattr(
                    self.observation_space, "high"
                ):
                    self.ranges[i] = (
                        self.observation_space.low[i],
                        self.observation_space.high[i],
                    )
                else:
                    self.ranges[i] = (-1.0, 1.0)

    def reset(self, **kwargs):
        """Reset the environment and normalize the observation."""
        obs = self.env.reset(**kwargs)
        self.original_obs = obs.copy()
        return self._normalize_observation(obs)

    def step(self, action):
        """Take a step and normalize the resulting observation."""
        obs, reward, done, info = self.env.step(action)
        self.original_obs = obs.copy()
        return self._normalize_observation(obs), reward, done, info

    def _normalize_observation(self, obs):
        """Normalize observation using predefined min/max ranges."""
        normalized_obs = np.zeros_like(obs)

        for i in range(len(obs)):
            min_val, max_val = self.ranges[i]

            # Skip normalization if min and max are the same (avoid division by zero)
            if min_val == max_val:
                normalized_obs[i] = 0.0 if self.zero_center else 0.5
                continue

            # Apply min-max normalization
            normalized_obs[i] = (obs[i] - min_val) / (max_val - min_val)

            # Scale to [-1, 1] if zero-centered
            if self.zero_center:
                normalized_obs[i] = 2.0 * normalized_obs[i] - 1.0

        return normalized_obs

    def get_original_obs(self):
        """Get the original non-normalized observation."""
        return self.original_obs


# Example usage:
# if __name__ == "__main__":
# Basic usage with default ranges:
# env = DollhouseThermalEnv(**env_params)
# env = NormalizedObservationWrapper(env)  # Uses default dollhouse ranges

# Custom ranges:
# custom_ranges = {0: (-10, 50), 1: (-10, 50)}  # Only override what you need
# env = NormalizedObservationWrapper(env, ranges=custom_ranges)
