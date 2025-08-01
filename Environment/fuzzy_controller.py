import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import argparse
import json

# Import our modules
from train_sindy_model import train_sindy_model
from dollhouse_env import DollhouseThermalEnv


class FuzzyLogic:
    """
    Fixed Fuzzy Logic controller with corrected logic and better tuning.
    """

    def __init__(self):
        """Initialize fuzzy logic controller with membership functions and rules."""
        self.setup_membership_functions()
        self.setup_control_rules()

    def setup_membership_functions(self):
        """Define fuzzy sets based on realistic building control needs."""

        # Temperature error fuzzy sets (setpoint - current temperature)
        # Positive error = too cold (need heating), Negative error = too warm (need cooling)
        self.temp_error_sets = {
            "very_cold": (2.0, 4.0, 6.0),  # Much too cold - need strong heating
            "cold": (0.5, 1.5, 3.0),  # Too cold - need heating
            "ok": (-1.0, 0.0, 1.0),  # Near setpoint - minimal action
            "warm": (-3.0, -1.5, -0.5),  # Too warm - need cooling
            "very_warm": (-6.0, -4.0, -2.0),  # Much too warm - need strong cooling
        }

        # Temperature change rate (current - previous)
        # Positive = rising, Negative = falling
        self.temp_rate_sets = {
            "falling_fast": (-1.0, -0.3, -0.1),
            "falling": (-0.2, -0.1, 0.0),
            "stable": (-0.05, 0.0, 0.05),
            "rising": (0.0, 0.1, 0.2),
            "rising_fast": (0.1, 0.3, 1.0),
        }

    def triangular_membership(self, x, left, center, right):
        """Calculate triangular membership function."""
        if x <= left or x >= right:
            return 0.0
        elif x == center:
            return 1.0
        elif x < center:
            return max(0.0, (x - left) / (center - left))
        else:
            return max(0.0, (right - x) / (right - center))

    def get_memberships(self, value, fuzzy_sets):
        """Get membership values for all sets."""
        memberships = {}
        for name, (left, center, right) in fuzzy_sets.items():
            memberships[name] = self.triangular_membership(value, left, center, right)
        return memberships

    def setup_control_rules(self):
        """Define control rules with corrected logic."""

        # HEATING rules - when we need to add heat (positive error = too cold)
        self.heating_rules = [
            # Strong heating - when very cold
            ("very_cold", "any", 1.0),
            ("very_cold", "falling", 1.0),
            ("very_cold", "rising", 0.8),
            # Moderate heating - when cold
            ("cold", "any", 0.8),
            ("cold", "falling", 0.9),
            ("cold", "stable", 0.7),
            ("cold", "rising", 0.5),
            # Light heating - when OK but falling
            ("ok", "falling_fast", 0.6),
            ("ok", "falling", 0.4),
            # Preventive heating
            ("warm", "falling_fast", 0.3),
        ]

        # COOLING rules - when we need to remove heat (negative error = too warm)
        self.cooling_rules = [
            # Strong cooling - when very warm
            ("very_warm", "any", 1.0),
            ("very_warm", "rising", 1.0),
            ("very_warm", "falling", 0.8),
            # Moderate cooling - when warm
            ("warm", "any", 0.8),
            ("warm", "rising", 0.9),
            ("warm", "stable", 0.7),
            ("warm", "falling", 0.5),
            # Light cooling - when OK but rising
            ("ok", "rising_fast", 0.6),
            ("ok", "rising", 0.4),
            # Preventive cooling
            ("cold", "rising_fast", 0.3),
        ]

    def evaluate_rules(self, temp_error, temp_rate):
        """
        Evaluate fuzzy rules to get heating and cooling strengths.

        Args:
            temp_error: Temperature error (setpoint - current)
                       Positive = too cold, Negative = too warm
            temp_rate: Rate of temperature change (current - previous)
                      Positive = rising, Negative = falling

        Returns:
            tuple: (heating_strength, cooling_strength)
        """
        # Get membership values
        temp_memberships = self.get_memberships(temp_error, self.temp_error_sets)
        rate_memberships = self.get_memberships(temp_rate, self.temp_rate_sets)

        heating_strength = 0.0
        cooling_strength = 0.0

        # Evaluate heating rules (for when temp_error > 0, i.e., too cold)
        for temp_cond, rate_cond, strength in self.heating_rules:
            temp_membership = temp_memberships[temp_cond]

            if rate_cond == "any":
                rate_membership = 1.0
            else:
                rate_membership = rate_memberships[rate_cond]

            # Rule activation = min(antecedents) * strength
            rule_activation = min(temp_membership, rate_membership) * strength
            heating_strength = max(heating_strength, rule_activation)

        # Evaluate cooling rules (for when temp_error < 0, i.e., too warm)
        for temp_cond, rate_cond, strength in self.cooling_rules:
            temp_membership = temp_memberships[temp_cond]

            if rate_cond == "any":
                rate_membership = 1.0
            else:
                rate_membership = rate_memberships[rate_cond]

            rule_activation = min(temp_membership, rate_membership) * strength
            cooling_strength = max(cooling_strength, rule_activation)

        return heating_strength, cooling_strength


def create_fuzzy_logic_controller(
    heating_threshold=0.2, cooling_threshold=0.2, debug=False
):
    """
    Create a fixed fuzzy logic controller.

    Args:
        heating_threshold: Threshold for activating heating (lower = more sensitive)
        cooling_threshold: Threshold for activating cooling (lower = more sensitive)
        debug: Enable debug output

    Returns:
        function: Fuzzy logic controller function
    """
    # Create fuzzy controller instance
    fuzzy_ctrl = FuzzyLogic()

    # Store previous temperatures for rate calculation
    previous_temps = {"ground": None, "top": None}
    step_counter = {"count": 0}

    def controller(observation):
        """
        Fixed fuzzy controller function.

        Args:
            observation: Environment observation

        Returns:
            np.array: Action array [ground_light, ground_window, top_light, top_window]
        """
        # Extract state variables
        ground_temp = observation[0]
        top_temp = observation[1]
        external_temp = observation[2]
        heating_setpoint = observation[7]
        cooling_setpoint = observation[8]

        # Calculate target temperature (average of setpoints)
        target_temp = (heating_setpoint + cooling_setpoint) / 2

        # FIXED: Calculate temperature errors correctly
        # Positive error = too cold (need heating), Negative error = too warm (need cooling)
        ground_error = target_temp - ground_temp  # This is correct
        top_error = target_temp - top_temp  # This is correct

        # Calculate temperature rates (positive = rising temperature)
        if previous_temps["ground"] is not None:
            ground_rate = ground_temp - previous_temps["ground"]
            top_rate = top_temp - previous_temps["top"]
        else:
            ground_rate = 0.0
            top_rate = 0.0

        # Update previous temperatures
        previous_temps["ground"] = ground_temp
        previous_temps["top"] = top_temp

        # Get fuzzy control strengths
        ground_heat, ground_cool = fuzzy_ctrl.evaluate_rules(ground_error, ground_rate)
        top_heat, top_cool = fuzzy_ctrl.evaluate_rules(top_error, top_rate)

        # FIXED: Better action logic with lower thresholds
        action = np.zeros(4, dtype=int)

        # Ground floor control - clearer logic
        if ground_heat > heating_threshold and ground_heat > ground_cool:
            # Need heating: turn on light, close window
            action[0] = 1  # Ground light ON
            action[1] = 0  # Ground window CLOSED
        elif ground_cool > cooling_threshold and ground_cool > ground_heat:
            # Need cooling: turn off light, open window
            action[0] = 0  # Ground light OFF
            action[1] = 1  # Ground window OPEN
        else:
            # Neutral: energy saving mode
            action[0] = 0  # Ground light OFF
            action[1] = 0  # Ground window CLOSED

        # Top floor control - same logic
        if top_heat > heating_threshold and top_heat > top_cool:
            # Need heating: turn on light, close window
            action[2] = 1  # Top light ON
            action[3] = 0  # Top window CLOSED
        elif top_cool > cooling_threshold and top_cool > top_heat:
            # Need cooling: turn off light, open window
            action[2] = 0  # Top light OFF
            action[3] = 1  # Top window OPEN
        else:
            # Neutral: energy saving mode
            action[2] = 0  # Top light OFF
            action[3] = 0  # Top window CLOSED

        step_counter["count"] += 1

        if debug and step_counter["count"] % 100 == 0:
            print(f"Step {step_counter['count']}:")
            print(
                f"  Ground: T={ground_temp:.1f}°C, Target={target_temp:.1f}°C, Error={ground_error:.2f}°C, Rate={ground_rate:.3f}"
            )
            print(
                f"          Heat={ground_heat:.3f}, Cool={ground_cool:.3f} -> Actions=[{action[0]}, {action[1]}]"
            )
            print(
                f"  Top:    T={top_temp:.1f}°C, Target={target_temp:.1f}°C, Error={top_error:.2f}°C, Rate={top_rate:.3f}"
            )
            print(
                f"          Heat={top_heat:.3f}, Cool={top_cool:.3f} -> Actions=[{action[2]}, {action[3]}]"
            )
            print()

        return action

    # Store fuzzy controller and state for reset capability
    controller.fuzzy_ctrl = fuzzy_ctrl
    controller.previous_temps = previous_temps
    controller.step_counter = step_counter
    controller.heating_threshold = heating_threshold
    controller.cooling_threshold = cooling_threshold

    return controller


def calculate_control_stability(episode_actions):
    """
    Calculate control stability metric as total state changes divided by episode length.

    Args:
        episode_actions: List of action arrays for the episode

    Returns:
        float: Control stability metric (state changes per timestep)
    """
    if len(episode_actions) <= 1:
        return 0.0

    actions = np.array(episode_actions)
    total_state_changes = 0

    # Count state changes for each action dimension
    for action_dim in range(actions.shape[1]):
        action_series = actions[:, action_dim]
        state_changes = np.sum(np.diff(action_series) != 0)
        total_state_changes += state_changes

    # Normalize by episode length
    control_stability = total_state_changes / len(episode_actions)

    return control_stability


def evaluate_fuzzy_logic_controller(
    env,
    num_episodes=5,
    render=True,
    heating_threshold=0.2,
    cooling_threshold=0.2,
    debug=False,
    output_dir=None,
):
    """
    Evaluate the fixed fuzzy logic controller on the environment.

    Args:
        env: The environment to evaluate on
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        heating_threshold: Threshold for heating action
        cooling_threshold: Threshold for cooling action
        debug: Enable debug output
        output_dir: Directory to save results (if None, uses default)

    Returns:
        dict: Evaluation results including control stability
    """
    # Create the fuzzy logic controller
    controller = create_fuzzy_logic_controller(
        heating_threshold=heating_threshold,
        cooling_threshold=cooling_threshold,
        debug=debug,
    )

    # Get the original environment if wrapped
    if hasattr(env, "unwrapped"):
        orig_env = env.unwrapped
    else:
        orig_env = env

    # Reset the environment's episode history
    if hasattr(orig_env, "episode_history"):
        orig_env.episode_history = []

    total_rewards = []
    actions_taken = {
        "ground_light_on": 0,
        "ground_window_open": 0,
        "top_light_on": 0,
        "top_window_open": 0,
    }

    # Store episode data for visualization and analysis
    episode_temperatures = []
    episode_external_temps = []
    episode_actions = []
    episode_rewards = []
    episode_setpoints = []
    episode_fuzzy_outputs = []
    control_stability_scores = []

    for episode in range(num_episodes):
        # Reset controller state for new episode
        controller.previous_temps = {"ground": None, "top": None}
        controller.step_counter = {"count": 0}

        # Reset returns (obs, info) tuple in gymnasium
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0

        # For tracking performance
        comfort_violations = 0

        # For this episode
        temps = []
        ext_temps = []
        actions = []
        rewards = []
        setpoints = []
        fuzzy_outputs = []

        while not terminated and not truncated:
            # Extract and store current setpoints from observation
            if len(obs) > 8:  # Check if setpoints are in observation
                heating_sp = obs[7]
                cooling_sp = obs[8]
            else:
                # Fallback to environment attributes
                heating_sp = orig_env.initial_heating_setpoint
                cooling_sp = orig_env.initial_cooling_setpoint

            setpoints.append([heating_sp, cooling_sp])

            # Calculate fuzzy outputs for analysis
            target_temp = (heating_sp + cooling_sp) / 2
            ground_error = target_temp - obs[0]  # Positive = too cold
            top_error = target_temp - obs[1]  # Positive = too cold

            # Get rate information
            if controller.previous_temps["ground"] is not None:
                ground_rate = obs[0] - controller.previous_temps["ground"]
                top_rate = obs[1] - controller.previous_temps["top"]
            else:
                ground_rate = 0.0
                top_rate = 0.0

            # Get fuzzy reasoning outputs
            ground_heat, ground_cool = controller.fuzzy_ctrl.evaluate_rules(
                ground_error, ground_rate
            )
            top_heat, top_cool = controller.fuzzy_ctrl.evaluate_rules(
                top_error, top_rate
            )

            fuzzy_outputs.append([ground_heat, ground_cool, top_heat, top_cool])

            # If the environment is normalized, get the original observation
            if hasattr(env, "get_original_obs"):
                orig_obs = env.get_original_obs()
                action = controller(orig_obs)
            else:
                action = controller(obs)

            # Update action counter
            actions_taken["ground_light_on"] += action[0]
            actions_taken["ground_window_open"] += action[1]
            actions_taken["top_light_on"] += action[2]
            actions_taken["top_window_open"] += action[3]

            # Take the action - now returns 5 values
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            # Record data for analysis
            temps.append([obs[0], obs[1]])  # ground_temp, top_temp
            ext_temps.append(obs[2])  # external_temp
            actions.append(action)
            rewards.append(reward)

            # Check for comfort violations
            if (
                "ground_comfort_violation" in info
                and info["ground_comfort_violation"] > 0
            ):
                comfort_violations += 1
            if "top_comfort_violation" in info and info["top_comfort_violation"] > 0:
                comfort_violations += 1

            if render:
                orig_env.render()

        # Calculate control stability for this episode
        episode_control_stability = calculate_control_stability(actions)
        control_stability_scores.append(episode_control_stability)

        # Store episode data
        episode_temperatures.append(temps)
        episode_external_temps.append(ext_temps)
        episode_actions.append(actions)
        episode_rewards.append(rewards)
        episode_setpoints.append(setpoints)
        episode_fuzzy_outputs.append(fuzzy_outputs)

        avg_actions = {k: v / steps for k, v in actions_taken.items()}

        print(
            f"Episode {episode+1}/{num_episodes}: Total Reward = {episode_reward:.2f}, "
            f"Control Stability = {episode_control_stability:.3f}"
        )
        print(f"  Steps: {steps}, Comfort Violations: {comfort_violations}")
        print(
            f"  Actions: Ground Light: {avg_actions['ground_light_on']:.2f}, Ground Window: {avg_actions['ground_window_open']:.2f}"
        )
        print(
            f"           Top Light: {avg_actions['top_light_on']:.2f}, Top Window: {avg_actions['top_window_open']:.2f}"
        )

        total_rewards.append(episode_reward)

    # Get performance summary
    if hasattr(orig_env, "get_performance_summary"):
        performance = orig_env.get_performance_summary()

        # Add control stability metrics
        performance["control_stability"] = np.mean(control_stability_scores)
        performance["control_stability_std"] = np.std(control_stability_scores)
        performance["control_stability_scores"] = control_stability_scores

        print(f"\nFixed Fuzzy Logic Controller Evaluation Summary:")
        print(f"Average Total Reward: {performance['avg_total_reward']:.2f}")
        print(f"Ground Floor Comfort %: {performance['avg_ground_comfort_pct']:.2f}%")
        print(f"Top Floor Comfort %: {performance['avg_top_comfort_pct']:.2f}%")
        print(f"Average Light Hours: {performance['avg_light_hours']:.2f}")
        print(
            f"Control Stability: {performance['control_stability']:.3f} ± {performance['control_stability_std']:.3f}"
        )

        # Add episode data to performance dict
        performance["episode_data"] = {
            "temperatures": episode_temperatures,
            "external_temps": episode_external_temps,
            "actions": episode_actions,
            "rewards": episode_rewards,
            "total_rewards": total_rewards,
            "setpoints": episode_setpoints,
            "fuzzy_outputs": episode_fuzzy_outputs,
        }

        # Add fuzzy parameters to performance
        performance["heating_threshold"] = heating_threshold
        performance["cooling_threshold"] = cooling_threshold

        # Handle dynamic setpoints properly
        if episode_setpoints and len(episode_setpoints[0]) > 0:
            # Use the first setpoint as fallback for static displays
            performance["heating_setpoint"] = episode_setpoints[0][0][0]
            performance["cooling_setpoint"] = episode_setpoints[0][0][1]
            performance["has_dynamic_setpoints"] = True
        else:
            # Fallback to environment attributes
            performance["heating_setpoint"] = orig_env.initial_heating_setpoint
            performance["cooling_setpoint"] = orig_env.initial_cooling_setpoint
            performance["has_dynamic_setpoints"] = False

        performance["reward_type"] = orig_env.reward_type
        performance["energy_weight"] = orig_env.energy_weight
        performance["comfort_weight"] = orig_env.comfort_weight

        # Use provided output_dir or default to fuzzy_results
        save_dir = output_dir if output_dir else "fuzzy_results"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            save_dir, f"fuzzy_logic_controller_results_{timestamp}.json"
        )

        orig_env.save_results(
            filepath,
            controller_name="Fixed Fuzzy Logic Controller",
        )
        print(f"Results saved to {filepath}")

        # Save performance data with episode details
        results_path = os.path.join(
            save_dir, f"fuzzy_logic_detailed_results_{timestamp}.json"
        )

        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, dict):
                return {
                    key: convert_to_serializable(value) for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, "item"):
                return obj.item()
            else:
                return obj

        with open(results_path, "w") as f:
            serializable_perf = convert_to_serializable(performance)
            json.dump(serializable_perf, f, indent=4)
        print(f"Detailed results saved to {results_path}")

        # Create visualizations
        visualize_fuzzy_logic_performance(
            performance, save_dir, "Fixed Fuzzy Logic Controller"
        )
    else:
        # Simple performance metrics if the environment doesn't provide detailed ones
        performance = {
            "avg_total_reward": np.mean(total_rewards),
            "std_total_reward": np.std(total_rewards),
            "control_stability": np.mean(control_stability_scores),
            "control_stability_std": np.std(control_stability_scores),
        }
        print(
            f"\nAverage Total Reward: {performance['avg_total_reward']:.2f} ± {performance['std_total_reward']:.2f}"
        )
        print(
            f"Control Stability: {performance['control_stability']:.3f} ± {performance['control_stability_std']:.3f}"
        )

    return performance


def visualize_fuzzy_logic_performance(
    performance, output_dir, controller_name="Fixed Fuzzy Logic Controller"
):
    """
    Create visualizations of fuzzy logic controller performance with control stability metric.

    Args:
        performance: Performance dictionary from evaluate_controller
        output_dir: Directory to save visualizations
        controller_name: Name of the controller for plot titles
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract episode data
    episode_temperatures = performance["episode_data"]["temperatures"]
    episode_external_temps = performance["episode_data"]["external_temps"]
    episode_actions = performance["episode_data"]["actions"]
    episode_rewards = performance["episode_data"]["rewards"]
    episode_setpoints = performance["episode_data"].get("setpoints", [])
    episode_fuzzy_outputs = performance["episode_data"].get("fuzzy_outputs", [])

    # Check if we have dynamic setpoints
    has_dynamic_setpoints = len(episode_setpoints) > 0 and len(episode_setpoints[0]) > 0

    # Plot temperatures, fuzzy outputs, actions, setpoints, and rewards for the first episode
    plt.figure(figsize=(15, 20))

    # Temperature plot with dynamic setpoints
    plt.subplot(6, 1, 1)
    ground_temps = [temp[0] for temp in episode_temperatures[0]]
    top_temps = [temp[1] for temp in episode_temperatures[0]]
    plt.plot(ground_temps, label="Ground Floor Temperature", linewidth=2)
    plt.plot(top_temps, label="Top Floor Temperature", linewidth=2)

    if has_dynamic_setpoints:
        # Plot dynamic setpoints
        heating_setpoints = [sp[0] for sp in episode_setpoints[0]]
        cooling_setpoints = [sp[1] for sp in episode_setpoints[0]]
        plt.plot(heating_setpoints, "r--", label="Heating Setpoint", linewidth=1.5)
        plt.plot(cooling_setpoints, "b--", label="Cooling Setpoint", linewidth=1.5)
    else:
        # Plot static setpoints
        heating_setpoint = performance.get("heating_setpoint", 20.0)
        cooling_setpoint = performance.get("cooling_setpoint", 24.0)
        plt.axhline(
            y=heating_setpoint,
            color="r",
            linestyle="--",
            label=f"Heating Setpoint ({heating_setpoint}°C)",
        )
        plt.axhline(
            y=cooling_setpoint,
            color="b",
            linestyle="--",
            label=f"Cooling Setpoint ({cooling_setpoint}°C)",
        )

    plt.title(f"{controller_name} - Temperatures (Episode 1)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Fuzzy outputs plot
    plt.subplot(6, 1, 2)
    if episode_fuzzy_outputs:
        fuzzy_data = np.array(episode_fuzzy_outputs[0])
        plt.plot(
            fuzzy_data[:, 0],
            label="Ground Heating",
            linewidth=2,
            color="red",
            alpha=0.7,
        )
        plt.plot(
            fuzzy_data[:, 1],
            label="Ground Cooling",
            linewidth=2,
            color="blue",
            alpha=0.7,
        )
        plt.plot(
            fuzzy_data[:, 2],
            label="Top Heating",
            linewidth=2,
            color="orange",
            alpha=0.7,
        )
        plt.plot(
            fuzzy_data[:, 3], label="Top Cooling", linewidth=2, color="cyan", alpha=0.7
        )

        heating_thresh = performance.get("heating_threshold", 0.2)
        cooling_thresh = performance.get("cooling_threshold", 0.2)
        plt.axhline(
            y=heating_thresh,
            color="red",
            linestyle=":",
            alpha=0.7,
            label=f"Heating Threshold ({heating_thresh})",
        )
        plt.axhline(
            y=cooling_thresh,
            color="blue",
            linestyle=":",
            alpha=0.7,
            label=f"Cooling Threshold ({cooling_thresh})",
        )
    plt.title(f"{controller_name} - Fuzzy Outputs (Episode 1)")
    plt.ylabel("Fuzzy Strength [0,1]")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # External temperature plot
    plt.subplot(6, 1, 3)
    ext_temps = episode_external_temps[0]
    plt.plot(ext_temps, label="External Temperature", color="purple", linewidth=2)

    if has_dynamic_setpoints:
        heating_setpoints = [sp[0] for sp in episode_setpoints[0]]
        cooling_setpoints = [sp[1] for sp in episode_setpoints[0]]
        plt.plot(heating_setpoints, "r--", alpha=0.7, label="Heating Setpoint")
        plt.plot(cooling_setpoints, "b--", alpha=0.7, label="Cooling Setpoint")
    else:
        heating_setpoint = performance.get("heating_setpoint", 20.0)
        cooling_setpoint = performance.get("cooling_setpoint", 24.0)
        plt.axhline(
            y=heating_setpoint,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Heating Setpoint ({heating_setpoint}°C)",
        )
        plt.axhline(
            y=cooling_setpoint,
            color="b",
            linestyle="--",
            alpha=0.7,
            label=f"Cooling Setpoint ({cooling_setpoint}°C)",
        )

    plt.title(f"{controller_name} - External Temperature (Episode 1)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Actions plot with state change indicators
    plt.subplot(6, 1, 4)
    actions = np.array(episode_actions[0])
    action_names = ["Ground Light", "Ground Window", "Top Light", "Top Window"]
    colors = ["red", "blue", "orange", "cyan"]

    for i, (name, color) in enumerate(zip(action_names, colors)):
        action_series = actions[:, i]
        plt.plot(action_series, label=name, linewidth=2, color=color, alpha=0.8)

        # Mark state changes with red dots
        changes = np.where(np.diff(action_series) != 0)[0]
        if len(changes) > 0:
            plt.scatter(
                changes, action_series[changes], color="red", s=20, alpha=0.7, zorder=5
            )

    # Calculate and display control stability for this episode
    episode_control_stability = calculate_control_stability(episode_actions[0])
    plt.title(
        f"{controller_name} - Actions (Episode 1) - Control Stability: {episode_control_stability:.3f}"
    )
    plt.ylabel("Action State (0/1)")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Rewards plot
    plt.subplot(6, 1, 5)
    plt.plot(episode_rewards[0], label="Step Reward", linewidth=2, color="green")
    plt.title(f"{controller_name} - Rewards (Episode 1)")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Fuzzy decision analysis
    plt.subplot(6, 1, 6)
    if episode_fuzzy_outputs:
        fuzzy_data = np.array(episode_fuzzy_outputs[0])
        # Show net heating/cooling decision
        ground_decision = fuzzy_data[:, 0] - fuzzy_data[:, 1]  # heating - cooling
        top_decision = fuzzy_data[:, 2] - fuzzy_data[:, 3]
        plt.plot(
            ground_decision,
            label="Ground Net Decision (Heat-Cool)",
            linewidth=2,
            color="red",
        )
        plt.plot(
            top_decision,
            label="Top Net Decision (Heat-Cool)",
            linewidth=2,
            color="orange",
        )
        plt.axhline(y=0, color="k", linestyle="-", alpha=0.5, label="Neutral")
    plt.title(f"{controller_name} - Net Decision Analysis (Episode 1)")
    plt.xlabel("Timestep")
    plt.ylabel("Net Decision")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"fixed_fuzzy_logic_controller_episode_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Summary metrics plot (now includes control stability)
    plt.figure(figsize=(15, 8))
    metrics = [
        ("avg_total_reward", "Total Reward"),
        ("avg_ground_comfort_pct", "Ground Floor Comfort %"),
        ("avg_top_comfort_pct", "Top Floor Comfort %"),
        ("avg_light_hours", "Light Hours"),
        ("control_stability", "Control Stability\n(State Changes/Timestep)"),
    ]

    for i, (metric, label) in enumerate(metrics):
        plt.subplot(1, 5, i + 1)
        value = performance.get(metric, 0)
        plt.bar([controller_name], [value], color="lightgreen")
        plt.title(label)
        plt.grid(True, alpha=0.3)

        # Add value label
        plt.text(
            0,
            value + 0.01,
            f"{value:.3f}" if metric == "control_stability" else f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"fixed_fuzzy_logic_controller_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Create control stability detailed analysis plot
    plt.figure(figsize=(14, 10))

    # Plot control stability per episode
    plt.subplot(2, 2, 1)
    episodes = range(1, len(performance["control_stability_scores"]) + 1)
    plt.bar(episodes, performance["control_stability_scores"])
    plt.title(f"{controller_name} - Control Stability per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Control Stability (State Changes/Timestep)")
    plt.grid(True, alpha=0.3)

    # Add average line
    avg_stability = performance["control_stability"]
    plt.axhline(
        y=avg_stability,
        color="red",
        linestyle="--",
        label=f"Average: {avg_stability:.3f}",
    )
    plt.legend()

    # Action switching breakdown for first episode
    plt.subplot(2, 2, 2)
    actions = np.array(episode_actions[0])
    action_names = ["Ground Light", "Ground Window", "Top Light", "Top Window"]

    switches_per_action = []
    for i in range(actions.shape[1]):
        action_series = actions[:, i]
        switches = np.sum(np.diff(action_series) != 0)
        switches_per_action.append(switches)

    bars = plt.bar(action_names, switches_per_action)
    plt.title(f"{controller_name} - State Changes by Action (Episode 1)")
    plt.ylabel("Number of State Changes")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, switches_per_action):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(value),
            ha="center",
            va="bottom",
        )

    # Control stability distribution across episodes
    plt.subplot(2, 2, 3)
    plt.hist(
        performance["control_stability_scores"],
        bins=min(10, len(performance["control_stability_scores"])),
        edgecolor="black",
        alpha=0.7,
    )
    plt.axvline(
        x=avg_stability, color="red", linestyle="--", label=f"Mean: {avg_stability:.3f}"
    )
    plt.title(f"{controller_name} - Control Stability Distribution")
    plt.xlabel("Control Stability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Action usage summary
    plt.subplot(2, 2, 4)
    all_actions = np.concatenate(episode_actions)
    duty_cycles = np.mean(all_actions, axis=0)

    bars = plt.bar(action_names, duty_cycles)
    plt.title(f"{controller_name} - Action Duty Cycles")
    plt.ylabel("Fraction of Time Active")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, duty_cycles):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"fixed_fuzzy_logic_controller_control_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Membership functions and rules visualization
    plt.figure(figsize=(15, 10))

    # Temperature error membership functions
    plt.subplot(2, 3, 1)
    x = np.linspace(-8, 8, 200)
    fuzzy_ctrl = FuzzyLogic()
    colors = ["purple", "blue", "green", "orange", "red"]
    for i, (name, (left, center, right)) in enumerate(
        fuzzy_ctrl.temp_error_sets.items()
    ):
        y = [fuzzy_ctrl.triangular_membership(xi, left, center, right) for xi in x]
        plt.plot(x, y, label=name, linewidth=2, color=colors[i])
    plt.title("Temperature Error Membership Functions")
    plt.xlabel("Temperature Error (°C)")
    plt.ylabel("Membership Degree")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Temperature rate membership functions
    plt.subplot(2, 3, 2)
    x = np.linspace(-1.5, 1.5, 200)
    for i, (name, (left, center, right)) in enumerate(
        fuzzy_ctrl.temp_rate_sets.items()
    ):
        y = [fuzzy_ctrl.triangular_membership(xi, left, center, right) for xi in x]
        plt.plot(x, y, label=name, linewidth=2, color=colors[i])
    plt.title("Temperature Rate Membership Functions")
    plt.xlabel("Temperature Rate (°C/step)")
    plt.ylabel("Membership Degree")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Rules visualization
    plt.subplot(2, 3, 3)
    heating_rules_text = "HEATING RULES (Fixed):\n"
    for i, (temp_cond, rate_cond, strength) in enumerate(
        fuzzy_ctrl.heating_rules[:6]
    ):  # Show first 6
        heating_rules_text += (
            f"{i+1}. IF temp={temp_cond} AND rate={rate_cond} THEN heat={strength}\n"
        )

    plt.text(
        0.05,
        0.95,
        heating_rules_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )
    plt.title("Sample Heating Rules (Fixed)")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    cooling_rules_text = "COOLING RULES (Fixed):\n"
    for i, (temp_cond, rate_cond, strength) in enumerate(
        fuzzy_ctrl.cooling_rules[:6]
    ):  # Show first 6
        cooling_rules_text += (
            f"{i+1}. IF temp={temp_cond} AND rate={rate_cond} THEN cool={strength}\n"
        )

    plt.text(
        0.05,
        0.95,
        cooling_rules_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )
    plt.title("Sample Cooling Rules (Fixed)")
    plt.axis("off")

    # Fuzzy output distribution
    plt.subplot(2, 3, 5)
    if episode_fuzzy_outputs:
        all_fuzzy_data = np.concatenate(episode_fuzzy_outputs)
        labels = ["Ground Heating", "Ground Cooling", "Top Heating", "Top Cooling"]
        colors = ["red", "blue", "orange", "cyan"]
        for i, (label, color) in enumerate(zip(labels, colors)):
            plt.hist(
                all_fuzzy_data[:, i],
                alpha=0.6,
                bins=20,
                label=label,
                density=True,
                color=color,
            )
    plt.title("Fuzzy Output Distribution")
    plt.xlabel("Fuzzy Output Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Control surface (simplified 2D view)
    plt.subplot(2, 3, 6)
    temp_errors = np.linspace(-6, 6, 50)
    temp_rates = np.linspace(-1.0, 1.0, 50)
    X, Y = np.meshgrid(temp_errors, temp_rates)
    Z = np.zeros_like(X)

    for i in range(len(temp_errors)):
        for j in range(len(temp_rates)):
            heat, cool = fuzzy_ctrl.evaluate_rules(temp_errors[i], temp_rates[j])
            Z[j, i] = heat - cool  # Net heating action

    plt.contourf(X, Y, Z, levels=20, cmap="RdBu_r")
    plt.colorbar(label="Net Heating Action")
    plt.xlabel("Temperature Error (°C)")
    plt.ylabel("Temperature Rate (°C/step)")
    plt.title("Fixed Fuzzy Control Surface")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"fixed_fuzzy_logic_membership_and_rules.png"),
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Visualizations saved to {output_dir}")


def run_fuzzy_logic_evaluation(
    data_file,
    output_dir=None,
    num_episodes=5,
    render=True,
    env_params_path=None,
    heating_threshold=0.2,
    cooling_threshold=0.2,
    debug=False,
):
    """
    Train a SINDy model and evaluate the fixed fuzzy logic controller.

    Args:
        data_file: Path to data file for training SINDy model
        output_dir: Directory to save results
        num_episodes: Number of episodes for evaluation
        render: Whether to render during evaluation
        env_params_path: Path to the saved environment parameters
        heating_threshold: Threshold for heating action (lowered default)
        cooling_threshold: Threshold for cooling action (lowered default)
        debug: Enable debug output

    Returns:
        dict: Evaluation results
    """
    # Set output directory - use fuzzy_results as default
    if output_dir is None:
        output_dir = "fuzzy_results"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")

    print(f"Fixed Fuzzy Logic Controller Parameters:")
    print(f"Heating Threshold: {heating_threshold} (lowered for better sensitivity)")
    print(f"Cooling Threshold: {cooling_threshold} (lowered for better sensitivity)")
    print(f"Debug Mode: {debug}")

    # Create environment
    if env_params_path:
        # Load environment parameters
        with open(env_params_path, "r") as f:
            env_params = json.load(f)

        # Train SINDy model
        print(f"Training SINDy model on {data_file}...")
        start_time = time.time()
        sindy_model = train_sindy_model(file_path=data_file)
        training_time = time.time() - start_time
        print(f"SINDy model training completed in {training_time:.2f} seconds")

        # Replace SINDy model placeholder
        env_params["sindy_model"] = sindy_model

        # Create environment with saved parameters
        env = DollhouseThermalEnv(**env_params)
    else:
        # Train SINDy model
        print(f"Training SINDy model on {data_file}...")
        start_time = time.time()
        sindy_model = train_sindy_model(file_path=data_file)
        training_time = time.time() - start_time
        print(f"SINDy model training completed in {training_time:.2f} seconds")

        # Create environment with default parameters
        env_params = {
            "sindy_model": sindy_model,
            "use_reward_shaping": True,
            "random_start_time": False,
            "shaping_weight": 0.3,
            "episode_length": 5760,  # 24 hours with 30-second timesteps
            "time_step_seconds": 30,
            "heating_setpoint": 26.0,
            "cooling_setpoint": 28.0,
            "external_temp_pattern": "sine",
            "setpoint_pattern": "schedule",
            "reward_type": "balanced",
            "energy_weight": 0.0,
            "comfort_weight": 1.0,
        }

        env = DollhouseThermalEnv(**env_params)

    # Print environment setpoint configuration
    print(f"\nEnvironment Configuration:")
    print(f"Setpoint Pattern: {env.setpoint_pattern}")
    print(f"Base Heating Setpoint: {env.initial_heating_setpoint}")
    print(f"Base Cooling Setpoint: {env.initial_cooling_setpoint}")
    print(f"Random Start Time: {env.random_start_time}")
    print(f"Reward Shaping: {env.use_reward_shaping}")

    # Save environment parameters and fuzzy configuration
    with open(os.path.join(output_dir, "env_params.json"), "w") as f:
        # Convert non-serializable parameters to strings
        serializable_params = env_params.copy()
        serializable_params["sindy_model"] = "SINDy model object (not serializable)"
        serializable_params["fuzzy_heating_threshold"] = heating_threshold
        serializable_params["fuzzy_cooling_threshold"] = cooling_threshold
        serializable_params["fuzzy_debug"] = debug
        json.dump(serializable_params, f, indent=4)

    # Evaluate fuzzy logic controller
    print(f"\nEvaluating Fixed Fuzzy Logic Controller...")
    start_time = time.time()
    performance = evaluate_fuzzy_logic_controller(
        env=env,
        num_episodes=num_episodes,
        render=render,
        heating_threshold=heating_threshold,
        cooling_threshold=cooling_threshold,
        debug=debug,
        output_dir=output_dir,
    )
    evaluation_time = time.time() - start_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")

    # Print control stability interpretation
    print(f"\nControl Stability Summary:")
    print(
        f"  Average Control Stability: {performance['control_stability']:.3f} ± {performance['control_stability_std']:.3f}"
    )
    print(f"  Interpretation: Lower values indicate more stable control")

    # Provide interpretation
    control_stability = performance["control_stability"]
    if control_stability < 0.1:
        interpretation = "Excellent - Very stable control with minimal switching"
    elif control_stability < 0.2:
        interpretation = "Good - Reasonable control stability"
    elif control_stability < 0.4:
        interpretation = "Fair - Moderate switching frequency"
    else:
        interpretation = "Poor - High switching frequency, potentially inefficient"

    print(f"  Assessment: {interpretation}")

    return performance


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train SINDy model and evaluate fixed fuzzy logic controller with control stability metric"
    )

    # Required arguments
    parser.add_argument(
        "--data",
        type=str,
        default="../Data/dollhouse-data-2025-03-24.csv",
        help="Path to data file for training SINDy model",
    )

    # Optional arguments
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes for evaluation"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Disable rendering during evaluation"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--env-params",
        type=str,
        default=None,
        help="Path to the saved environment parameters",
    )

    # Fixed fuzzy logic parameters (with better defaults)
    parser.add_argument(
        "--heating-threshold",
        type=float,
        default=0.2,
        help="Threshold for activating heating (0.0-1.0, lowered for better sensitivity)",
    )
    parser.add_argument(
        "--cooling-threshold",
        type=float,
        default=0.2,
        help="Threshold for activating cooling (0.0-1.0, lowered for better sensitivity)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Run evaluation
    run_fuzzy_logic_evaluation(
        data_file=args.data,
        output_dir=args.output,
        num_episodes=args.episodes,
        render=not args.no_render,
        env_params_path=args.env_params,
        heating_threshold=args.heating_threshold,
        cooling_threshold=args.cooling_threshold,
        debug=args.debug,
    )

# Example usage:
# python fuzzy_controller.py --data "../Data/dollhouse-data-2025-03-24.csv" --episodes 5 --debug
# python fuzzy_controller.py --data "../Data/dollhouse-data-2025-03-24.csv" --episodes 5 --heating-threshold 0.15 --cooling-threshold 0.15
# python fuzzy_controller.py --data "../Data/dollhouse-data-2025-03-24.csv" --env-params "results/ppo_20250513_151705/env_params.json" --output "fuzzy_results"
