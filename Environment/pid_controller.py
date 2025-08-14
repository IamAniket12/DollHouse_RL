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


class PIDController:
    """
    PID Controller for temperature control in the dollhouse environment.
    """

    def __init__(self, kp=1.0, ki=0.1, kd=0.01, output_limits=(-1, 1), sample_time=30):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: Tuple of (min, max) output limits
            sample_time: Sample time in seconds
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.sample_time = sample_time

        # Internal state
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def reset(self):
        """Reset the PID controller state."""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def compute(self, setpoint, current_value, current_time=None):
        """
        Compute PID output.

        Args:
            setpoint: Desired value
            current_value: Current measured value
            current_time: Current time (optional, uses sample_time if None)

        Returns:
            float: PID output
        """
        # Calculate error
        error = setpoint - current_value

        # Calculate time delta
        if current_time is None or self.last_time is None:
            dt = self.sample_time
        else:
            dt = current_time - self.last_time

        if dt <= 0:
            dt = self.sample_time

        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral

        # Derivative term
        derivative = self.kd * (error - self.previous_error) / dt

        # Calculate output
        output = proportional + integral + derivative

        # Apply output limits
        if self.output_limits:
            output = max(min(output, self.output_limits[1]), self.output_limits[0])

        # Store values for next iteration
        self.previous_error = error
        self.last_time = current_time

        return output


def get_state_from_observation(env, obs):
    """
    Extract meaningful state values from observation, handling custom observations.

    Args:
        env: Environment (may have custom observations)
        obs: Current observation vector

    Returns:
        dict: Dictionary with meaningful state values
    """
    # Get observation info
    obs_info = env.get_observation_info()
    obs_list = obs_info["observation_list"]

    # Create dictionary mapping observation names to values
    obs_dict = {}
    for i, obs_name in enumerate(obs_list):
        if i < len(obs):
            obs_dict[obs_name] = obs[i]

    # Extract state values, falling back to environment state if not in observations
    state = {}

    # Temperature values
    if "ground_temp" in obs_dict:
        state["ground_temp"] = obs_dict["ground_temp"]
    else:
        state["ground_temp"] = env.ground_temp

    if "top_temp" in obs_dict:
        state["top_temp"] = obs_dict["top_temp"]
    else:
        state["top_temp"] = env.top_temp

    if "external_temp" in obs_dict:
        state["external_temp"] = obs_dict["external_temp"]
    else:
        current_step = min(env.current_step, len(env.external_temperatures) - 1)
        state["external_temp"] = env.external_temperatures[current_step]

    # Setpoints
    if "heating_setpoint" in obs_dict:
        state["heating_setpoint"] = obs_dict["heating_setpoint"]
    else:
        state["heating_setpoint"] = env.heating_setpoint

    if "cooling_setpoint" in obs_dict:
        state["cooling_setpoint"] = obs_dict["cooling_setpoint"]
    else:
        state["cooling_setpoint"] = env.cooling_setpoint

    # Current actions (if available)
    state["ground_light"] = obs_dict.get("ground_light", 0)
    state["top_light"] = obs_dict.get("top_light", 0)
    state["ground_window"] = obs_dict.get("ground_window", 0)
    state["top_window"] = obs_dict.get("top_window", 0)

    return state


def create_pid_controller(ground_params=None, top_params=None):
    """
    Create a PID-based controller for the dollhouse environment that works with any observation space.

    Args:
        ground_params: Dict with PID parameters for ground floor (kp, ki, kd)
        top_params: Dict with PID parameters for top floor (kp, ki, kd)

    Returns:
        function: PID controller function
    """
    # Default PID parameters
    default_ground_params = {"kp": 2.0, "ki": 0.1, "kd": 0.05}
    default_top_params = {"kp": 2.0, "ki": 0.1, "kd": 0.05}

    ground_params = ground_params or default_ground_params
    top_params = top_params or default_top_params

    # Create PID controllers for each zone
    ground_pid = PIDController(
        kp=ground_params["kp"],
        ki=ground_params["ki"],
        kd=ground_params["kd"],
        output_limits=(-1, 1),
        sample_time=30,
    )

    top_pid = PIDController(
        kp=top_params["kp"],
        ki=top_params["ki"],
        kd=top_params["kd"],
        output_limits=(-1, 1),
        sample_time=30,
    )

    # Store step counter for time tracking
    step_counter = {"count": 0}

    def controller(env, observation):
        """
        PID controller function that works with any observation space.

        Args:
            env: Environment instance
            observation: Environment observation

        Returns:
            np.array: Action array [ground_light, ground_window, top_light, top_window]
        """
        # Extract state variables using the helper function
        state = get_state_from_observation(env, observation)

        ground_temp = state["ground_temp"]
        top_temp = state["top_temp"]
        external_temp = state["external_temp"]
        heating_setpoint = state["heating_setpoint"]
        cooling_setpoint = state["cooling_setpoint"]

        # Calculate target temperature (midpoint between heating and cooling setpoints)
        ground_target = (heating_setpoint + cooling_setpoint) / 2
        top_target = (heating_setpoint + cooling_setpoint) / 2

        # Get current time for PID computation
        current_time = step_counter["count"] * 30  # 30 seconds per step
        step_counter["count"] += 1

        # Compute PID outputs
        ground_output = ground_pid.compute(ground_target, ground_temp, current_time)
        top_output = top_pid.compute(top_target, top_temp, current_time)

        # Convert PID outputs to actions
        action = np.zeros(4, dtype=int)

        # Ground floor actions
        if ground_output > 0.1:  # Need heating
            action[0] = 1  # Turn ON ground light
            action[1] = 0  # Close ground window
        elif ground_output < -0.1:  # Need cooling
            action[0] = 0  # Turn OFF ground light
            action[1] = 1  # Open ground window
        else:  # Maintain current state
            action[0] = 0  # Turn OFF ground light
            action[1] = 0  # Close ground window

        # Top floor actions
        if top_output > 0.1:  # Need heating
            action[2] = 1  # Turn ON top light
            action[3] = 0  # Close top window
        elif top_output < -0.1:  # Need cooling
            action[2] = 0  # Turn OFF top light
            action[3] = 1  # Open top window
        else:  # Maintain current state
            action[2] = 0  # Turn OFF top light
            action[3] = 0  # Close top window

        return action

    # Store PID controllers for reset capability
    controller.ground_pid = ground_pid
    controller.top_pid = top_pid
    controller.step_counter = step_counter

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


def evaluate_pid_controller(
    env,
    num_episodes=5,
    render=True,
    ground_params=None,
    top_params=None,
    output_dir=None,
):
    """
    Evaluate a PID controller on the environment.
    Now handles custom observations properly.

    Args:
        env: The environment to evaluate on
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        ground_params: PID parameters for ground floor
        top_params: PID parameters for top floor
        output_dir: Directory to save results (if None, uses default)

    Returns:
        dict: Evaluation results including control stability
    """
    # Create the PID controller
    controller = create_pid_controller(
        ground_params=ground_params, top_params=top_params
    )

    # Reset the environment's episode history
    if hasattr(env, "episode_history"):
        env.episode_history = []

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
    episode_pid_outputs = []
    control_stability_scores = []

    # Print environment and observation info
    obs_info = env.get_observation_info()
    print(f"Environment Observation Configuration:")
    print(f"  Observation Space Shape: {obs_info['observation_space_shape']}")
    print(f"  Observations Used: {obs_info['observation_list']}")

    for episode in range(num_episodes):
        # Reset controller state for new episode
        controller.ground_pid.reset()
        controller.top_pid.reset()
        controller.step_counter["count"] = 0

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
        pid_outputs = []

        while not terminated and not truncated:
            # Extract state information
            state = get_state_from_observation(env, obs)

            # Store current setpoints
            heating_sp = state["heating_setpoint"]
            cooling_sp = state["cooling_setpoint"]
            setpoints.append([heating_sp, cooling_sp])

            # Get action from controller
            action = controller(env, obs)

            # Store PID outputs for analysis
            ground_target = (heating_sp + cooling_sp) / 2
            top_target = (heating_sp + cooling_sp) / 2

            # Get PID outputs (before action conversion)
            current_time = (steps) * 30
            ground_output = controller.ground_pid.compute(
                ground_target, state["ground_temp"], current_time
            )
            top_output = controller.top_pid.compute(
                top_target, state["top_temp"], current_time
            )
            pid_outputs.append([ground_output, top_output])

            # Update action counter
            actions_taken["ground_light_on"] += action[0]
            actions_taken["ground_window_open"] += action[1]
            actions_taken["top_light_on"] += action[2]
            actions_taken["top_window_open"] += action[3]

            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            # Record data for analysis - use actual temperatures from environment
            temps.append([state["ground_temp"], state["top_temp"]])
            ext_temps.append(state["external_temp"])
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
                env.render()

        # Calculate control stability for this episode
        episode_control_stability = calculate_control_stability(actions)
        control_stability_scores.append(episode_control_stability)

        # Store episode data
        episode_temperatures.append(temps)
        episode_external_temps.append(ext_temps)
        episode_actions.append(actions)
        episode_rewards.append(rewards)
        episode_setpoints.append(setpoints)
        episode_pid_outputs.append(pid_outputs)

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
    if hasattr(env, "get_performance_summary"):
        performance = env.get_performance_summary()

        # Add control stability metrics
        performance["control_stability"] = np.mean(control_stability_scores)
        performance["control_stability_std"] = np.std(control_stability_scores)
        performance["control_stability_scores"] = control_stability_scores

        print(f"\nPID Controller Evaluation Summary:")
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
            "pid_outputs": episode_pid_outputs,
        }

        # Add PID parameters to performance
        performance["ground_pid_params"] = ground_params or {
            "kp": 2.0,
            "ki": 0.1,
            "kd": 0.05,
        }
        performance["top_pid_params"] = top_params or {"kp": 2.0, "ki": 0.1, "kd": 0.05}

        # Handle dynamic setpoints properly
        if episode_setpoints and len(episode_setpoints[0]) > 0:
            # Use the first setpoint as fallback for static displays
            performance["heating_setpoint"] = episode_setpoints[0][0][0]
            performance["cooling_setpoint"] = episode_setpoints[0][0][1]
            performance["has_dynamic_setpoints"] = True
        else:
            # Fallback to environment attributes
            performance["heating_setpoint"] = env.initial_heating_setpoint
            performance["cooling_setpoint"] = env.initial_cooling_setpoint
            performance["has_dynamic_setpoints"] = False

        performance["reward_type"] = env.reward_type
        performance["energy_weight"] = env.energy_weight
        performance["comfort_weight"] = env.comfort_weight

        # Add observation information
        performance["observation_space_shape"] = obs_info["observation_space_shape"]
        performance["observations_used"] = obs_info["observation_list"]

        # Use provided output_dir or default to pid_results
        save_dir = output_dir if output_dir else "pid_results"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"pid_controller_results_{timestamp}.json")

        env.save_results(
            filepath,
            controller_name="PID Controller",
        )
        print(f"Results saved to {filepath}")

        # Save performance data with episode details
        results_path = os.path.join(save_dir, f"pid_detailed_results_{timestamp}.json")

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
        visualize_pid_performance(performance, save_dir, "PID Controller")
    else:
        # Simple performance metrics if the environment doesn't provide detailed ones
        performance = {
            "avg_total_reward": np.mean(total_rewards),
            "std_total_reward": np.std(total_rewards),
            "control_stability": np.mean(control_stability_scores),
            "control_stability_std": np.std(control_stability_scores),
            "observation_space_shape": obs_info["observation_space_shape"],
            "observations_used": obs_info["observation_list"],
        }
        print(
            f"\nAverage Total Reward: {performance['avg_total_reward']:.2f} ± {performance['std_total_reward']:.2f}"
        )
        print(
            f"Control Stability: {performance['control_stability']:.3f} ± {performance['control_stability_std']:.3f}"
        )

    return performance


def visualize_pid_performance(
    performance, output_dir, controller_name="PID Controller"
):
    """
    Create visualizations of PID controller performance with control stability metric.
    Updated to handle custom observations.

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
    episode_pid_outputs = performance["episode_data"].get("pid_outputs", [])

    # Check if we have dynamic setpoints
    has_dynamic_setpoints = len(episode_setpoints) > 0 and len(episode_setpoints[0]) > 0

    # Plot temperatures, PID outputs, actions, setpoints, and rewards for the first episode
    fig = plt.figure(figsize=(16, 22))

    # Temperature plot with dynamic setpoints
    plt.subplot(7, 1, 1)
    ground_temps = [temp[0] for temp in episode_temperatures[0]]
    top_temps = [temp[1] for temp in episode_temperatures[0]]
    plt.plot(ground_temps, label="Ground Floor Temperature", linewidth=2, color="blue")
    plt.plot(top_temps, label="Top Floor Temperature", linewidth=2, color="red")

    if has_dynamic_setpoints:
        # Plot dynamic setpoints
        heating_setpoints = [sp[0] for sp in episode_setpoints[0]]
        cooling_setpoints = [sp[1] for sp in episode_setpoints[0]]
        plt.plot(
            heating_setpoints, "r--", label="Heating Setpoint", linewidth=1.5, alpha=0.8
        )
        plt.plot(
            cooling_setpoints, "b--", label="Cooling Setpoint", linewidth=1.5, alpha=0.8
        )
    else:
        # Plot static setpoints
        heating_setpoint = performance.get("heating_setpoint", 20.0)
        cooling_setpoint = performance.get("cooling_setpoint", 24.0)
        plt.axhline(
            y=heating_setpoint,
            color="r",
            linestyle="--",
            label=f"Heating Setpoint ({heating_setpoint}°C)",
            alpha=0.8,
        )
        plt.axhline(
            y=cooling_setpoint,
            color="b",
            linestyle="--",
            label=f"Cooling Setpoint ({cooling_setpoint}°C)",
            alpha=0.8,
        )

    # Add observation info to title
    obs_count = (
        performance.get("observation_space_shape", [0])[0]
        if performance.get("observation_space_shape")
        else "Unknown"
    )
    plt.title(
        f"{controller_name} - Temperatures (Episode 1) - {obs_count} observations",
        fontsize=12,
    )
    plt.ylabel("Temperature (°C)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    # Observation space information subplot
    plt.subplot(7, 1, 2)
    obs_list = performance.get("observations_used", [])
    if obs_list:
        # Create a text display of observations
        obs_text = "Observations: " + ", ".join(obs_list)
        plt.text(
            0.05,
            0.5,
            obs_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            wrap=True,
            verticalalignment="center",
        )
    plt.title(f"Observation Configuration ({len(obs_list)} observations)")
    plt.axis("off")

    # PID outputs plot
    plt.subplot(7, 1, 3)
    if episode_pid_outputs:
        ground_pid_outputs = [pid[0] for pid in episode_pid_outputs[0]]
        top_pid_outputs = [pid[1] for pid in episode_pid_outputs[0]]
        plt.plot(
            ground_pid_outputs,
            label="Ground Floor PID Output",
            linewidth=2,
            color="blue",
        )
        plt.plot(
            top_pid_outputs, label="Top Floor PID Output", linewidth=2, color="red"
        )
        plt.axhline(
            y=0.1, color="r", linestyle=":", alpha=0.7, label="Heating Threshold"
        )
        plt.axhline(
            y=-0.1, color="b", linestyle=":", alpha=0.7, label="Cooling Threshold"
        )
        plt.axhline(y=0, color="k", linestyle="-", alpha=0.5)
    plt.title(f"{controller_name} - PID Outputs (Episode 1)")
    plt.ylabel("PID Output")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # External temperature plot
    plt.subplot(7, 1, 4)
    ext_temps = episode_external_temps[0]
    plt.plot(ext_temps, label="External Temperature", color="purple", linewidth=2)

    if has_dynamic_setpoints:
        heating_setpoints = [sp[0] for sp in episode_setpoints[0]]
        cooling_setpoints = [sp[1] for sp in episode_setpoints[0]]
        plt.plot(heating_setpoints, "r--", alpha=0.5, label="Heating Setpoint")
        plt.plot(cooling_setpoints, "b--", alpha=0.5, label="Cooling Setpoint")
    else:
        heating_setpoint = performance.get("heating_setpoint", 20.0)
        cooling_setpoint = performance.get("cooling_setpoint", 24.0)
        plt.axhline(
            y=heating_setpoint,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Heating Setpoint ({heating_setpoint}°C)",
        )
        plt.axhline(
            y=cooling_setpoint,
            color="b",
            linestyle="--",
            alpha=0.5,
            label=f"Cooling Setpoint ({cooling_setpoint}°C)",
        )

    plt.title(f"{controller_name} - External Temperature (Episode 1)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Actions plot with state change indicators
    plt.subplot(7, 1, 5)
    actions = np.array(episode_actions[0])
    action_names = ["Ground Light", "Ground Window", "Top Light", "Top Window"]
    colors = ["orange", "green", "red", "blue"]

    for i, (name, color) in enumerate(zip(action_names, colors)):
        action_series = actions[:, i]
        plt.plot(action_series, label=name, linewidth=2, color=color)

        # Mark state changes with red dots
        changes = np.where(np.diff(action_series) != 0)[0]
        if len(changes) > 0:
            plt.scatter(
                changes,
                action_series[changes],
                color="red",
                s=30,
                alpha=0.8,
                zorder=5,
                marker="o",
                edgecolors="black",
                linewidth=1,
            )

    # Calculate and display control stability for this episode
    episode_control_stability = calculate_control_stability(episode_actions[0])
    plt.title(
        f"{controller_name} - Actions (Episode 1) - Control Stability: {episode_control_stability:.3f}"
    )
    plt.ylabel("Action State")
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ["OFF/CLOSED", "ON/OPEN"])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Rewards plot
    plt.subplot(7, 1, 6)
    plt.plot(episode_rewards[0], label="Step Reward", linewidth=2, color="darkgreen")
    plt.title(f"{controller_name} - Rewards (Episode 1)")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # PID parameters info
    plt.subplot(7, 1, 7)
    plt.axis("off")
    ground_params = performance.get("ground_pid_params", {})
    top_params = performance.get("top_pid_params", {})

    info_text = f"""PID Controller Parameters:
Ground Floor: Kp={ground_params.get('kp', 'N/A')}, Ki={ground_params.get('ki', 'N/A')}, Kd={ground_params.get('kd', 'N/A')}
Top Floor: Kp={top_params.get('kp', 'N/A')}, Ki={top_params.get('ki', 'N/A')}, Kd={top_params.get('kd', 'N/A')}"""

    plt.text(
        0.1,
        0.5,
        info_text,
        fontsize=12,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
    )

    plt.xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"pid_controller_episode_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Summary metrics plot (now includes observation info)
    plt.figure(figsize=(18, 8))
    metrics = [
        ("avg_total_reward", "Total Reward"),
        ("avg_ground_comfort_pct", "Ground Floor\nComfort %"),
        ("avg_top_comfort_pct", "Top Floor\nComfort %"),
        ("avg_light_hours", "Light Hours"),
        ("control_stability", "Control Stability\n(Changes/Timestep)"),
    ]

    for i, (metric, label) in enumerate(metrics):
        plt.subplot(1, 6, i + 1)
        value = performance.get(metric, 0)
        color = ["skyblue", "lightgreen", "lightcoral", "orange", "plum"][i]
        plt.bar([controller_name], [value], color=color, edgecolor="black", linewidth=1)
        plt.title(label, fontsize=12)
        plt.grid(True, alpha=0.3, axis="y")

        # Add value label
        plt.text(
            0,
            value + abs(value) * 0.02,
            f"{value:.3f}" if metric == "control_stability" else f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add observation space info
    plt.subplot(1, 6, 6)
    obs_count = (
        performance.get("observation_space_shape", [0])[0]
        if performance.get("observation_space_shape")
        else 0
    )
    obs_list = performance.get("observations_used", [])

    plt.bar(
        [controller_name],
        [obs_count],
        color="lightblue",
        edgecolor="black",
        linewidth=1,
    )
    plt.title("Observation\nSpace Size", fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.text(
        0,
        obs_count + obs_count * 0.02,
        str(obs_count),
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    # Add observation list as text below
    if obs_list:
        obs_text = ", ".join(obs_list[:4])  # Show first 4
        if len(obs_list) > 4:
            obs_text += f"\n+ {len(obs_list) - 4} more"
        plt.text(0, -obs_count * 0.15, obs_text, ha="center", va="top", fontsize=8)

    plt.suptitle(
        f"{controller_name} - Performance Summary", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"pid_controller_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create control stability detailed analysis plot
    plt.figure(figsize=(16, 12))

    # Plot control stability per episode
    plt.subplot(2, 3, 1)
    episodes = range(1, len(performance["control_stability_scores"]) + 1)
    bars = plt.bar(
        episodes,
        performance["control_stability_scores"],
        color="lightcoral",
        edgecolor="black",
        linewidth=1,
    )
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
        linewidth=2,
        label=f"Average: {avg_stability:.3f}",
    )
    plt.legend()

    # Action switching breakdown for first episode
    plt.subplot(2, 3, 2)
    actions = np.array(episode_actions[0])
    action_names = ["Ground\nLight", "Ground\nWindow", "Top\nLight", "Top\nWindow"]
    colors = ["orange", "green", "red", "blue"]

    switches_per_action = []
    for i in range(actions.shape[1]):
        action_series = actions[:, i]
        switches = np.sum(np.diff(action_series) != 0)
        switches_per_action.append(switches)

    bars = plt.bar(
        action_names, switches_per_action, color=colors, edgecolor="black", linewidth=1
    )
    plt.title(f"{controller_name} - State Changes by Action (Episode 1)")
    plt.ylabel("Number of State Changes")
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, switches_per_action):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(value),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Control stability distribution across episodes
    plt.subplot(2, 3, 3)
    plt.hist(
        performance["control_stability_scores"],
        bins=min(10, len(performance["control_stability_scores"])),
        edgecolor="black",
        alpha=0.7,
        color="plum",
    )
    plt.axvline(
        x=avg_stability,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {avg_stability:.3f}",
    )
    plt.title(f"{controller_name} - Control Stability Distribution")
    plt.xlabel("Control Stability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Action usage summary
    plt.subplot(2, 3, 4)
    all_actions = np.concatenate(episode_actions)
    duty_cycles = np.mean(all_actions, axis=0)

    bars = plt.bar(
        action_names, duty_cycles, color=colors, edgecolor="black", linewidth=1
    )
    plt.title(f"{controller_name} - Action Duty Cycles")
    plt.ylabel("Fraction of Time Active")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, duty_cycles):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # PID output distribution
    plt.subplot(2, 3, 5)
    if episode_pid_outputs:
        all_ground_outputs = []
        all_top_outputs = []
        for episode_outputs in episode_pid_outputs:
            all_ground_outputs.extend([out[0] for out in episode_outputs])
            all_top_outputs.extend([out[1] for out in episode_outputs])

        plt.hist(
            all_ground_outputs,
            bins=30,
            alpha=0.7,
            label="Ground Floor",
            color="blue",
            edgecolor="black",
        )
        plt.hist(
            all_top_outputs,
            bins=30,
            alpha=0.7,
            label="Top Floor",
            color="red",
            edgecolor="black",
        )
        plt.axvline(
            x=0.1, color="r", linestyle=":", alpha=0.7, label="Heating Threshold"
        )
        plt.axvline(
            x=-0.1, color="b", linestyle=":", alpha=0.7, label="Cooling Threshold"
        )
        plt.axvline(x=0, color="k", linestyle="-", alpha=0.5)
        plt.title(f"{controller_name} - PID Output Distribution")
        plt.xlabel("PID Output")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Temperature performance analysis
    plt.subplot(2, 3, 6)
    comfort_performance = []
    heating_setpoint = performance.get("heating_setpoint", 20.0)
    cooling_setpoint = performance.get("cooling_setpoint", 24.0)

    for episode_idx, episode_temps in enumerate(episode_temperatures):
        ground_temps = np.array([temp[0] for temp in episode_temps])
        top_temps = np.array([temp[1] for temp in episode_temps])

        ground_comfort = (
            np.mean(
                (ground_temps >= heating_setpoint) & (ground_temps <= cooling_setpoint)
            )
            * 100
        )
        top_comfort = (
            np.mean((top_temps >= heating_setpoint) & (top_temps <= cooling_setpoint))
            * 100
        )
        avg_comfort = (ground_comfort + top_comfort) / 2
        comfort_performance.append(avg_comfort)

    episodes = range(1, len(comfort_performance) + 1)
    bars = plt.bar(
        episodes,
        comfort_performance,
        color="lightgreen",
        edgecolor="black",
        linewidth=1,
    )
    plt.title(f"{controller_name} - Comfort Performance by Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Comfort Percentage")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)

    # Add average line
    avg_comfort = np.mean(comfort_performance)
    plt.axhline(
        y=avg_comfort,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Average: {avg_comfort:.1f}%",
    )
    plt.legend()

    plt.suptitle(
        f"{controller_name} - Control Analysis", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"pid_controller_control_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Comprehensive visualizations saved to {output_dir}")


def create_evaluation_report(performance, output_dir, controller_name="PID Controller"):
    """
    Create a comprehensive evaluation report in text format.

    Args:
        performance: Performance dictionary from evaluate_pid_controller
        output_dir: Directory to save the report
        controller_name: Name of the controller
    """
    report_path = os.path.join(
        output_dir, f"{controller_name.lower().replace(' ', '_')}_evaluation_report.txt"
    )

    with open(report_path, "w") as f:
        f.write(f"{'='*80}\n")
        f.write(f"{controller_name.upper()} - COMPREHENSIVE EVALUATION REPORT\n")
        f.write(f"{'='*80}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Environment Configuration
        f.write("ENVIRONMENT CONFIGURATION\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"Observation Space Shape: {performance.get('observation_space_shape', 'N/A')}\n"
        )
        f.write(f"Observations Used: {performance.get('observations_used', 'N/A')}\n")
        f.write(f"Heating Setpoint: {performance.get('heating_setpoint', 'N/A')}°C\n")
        f.write(f"Cooling Setpoint: {performance.get('cooling_setpoint', 'N/A')}°C\n")
        f.write(
            f"Dynamic Setpoints: {'Yes' if performance.get('has_dynamic_setpoints', False) else 'No'}\n"
        )
        f.write(f"Reward Type: {performance.get('reward_type', 'N/A')}\n")
        f.write(f"Energy Weight: {performance.get('energy_weight', 'N/A')}\n")
        f.write(f"Comfort Weight: {performance.get('comfort_weight', 'N/A')}\n\n")

        # PID Controller Configuration
        f.write("PID CONTROLLER CONFIGURATION\n")
        f.write("-" * 50 + "\n")
        ground_params = performance.get("ground_pid_params", {})
        top_params = performance.get("top_pid_params", {})
        f.write(
            f"Ground Floor PID: Kp={ground_params.get('kp', 'N/A')}, Ki={ground_params.get('ki', 'N/A')}, Kd={ground_params.get('kd', 'N/A')}\n"
        )
        f.write(
            f"Top Floor PID: Kp={top_params.get('kp', 'N/A')}, Ki={top_params.get('ki', 'N/A')}, Kd={top_params.get('kd', 'N/A')}\n\n"
        )

        # Performance Summary
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write(f"Number of Episodes: {performance.get('num_episodes', 'N/A')}\n")
        f.write(f"Average Total Reward: {performance.get('avg_total_reward', 0):.3f}\n")
        f.write(
            f"Reward Standard Deviation: {performance.get('std_total_reward', 0):.3f}\n"
        )
        f.write(f"Min Total Reward: {performance.get('min_total_reward', 0):.3f}\n")
        f.write(f"Max Total Reward: {performance.get('max_total_reward', 0):.3f}\n\n")

        # Comfort Performance
        f.write("COMFORT PERFORMANCE\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"Ground Floor Comfort: {performance.get('avg_ground_comfort_pct', 0):.2f}%\n"
        )
        f.write(
            f"Top Floor Comfort: {performance.get('avg_top_comfort_pct', 0):.2f}%\n"
        )
        f.write(
            f"Average Comfort: {performance.get('avg_total_comfort_pct', 0):.2f}%\n\n"
        )

        # Energy Performance
        f.write("ENERGY PERFORMANCE\n")
        f.write("-" * 50 + "\n")
        f.write(f"Average Light Hours: {performance.get('avg_light_hours', 0):.2f}\n")

        # Control Stability
        f.write("CONTROL STABILITY ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"Average Control Stability: {performance.get('control_stability', 0):.3f}\n"
        )
        f.write(
            f"Control Stability Std: {performance.get('control_stability_std', 0):.3f}\n"
        )

        control_stability = performance.get("control_stability", 0)
        if control_stability < 0.1:
            interpretation = "Excellent - Very stable control with minimal switching"
        elif control_stability < 0.2:
            interpretation = "Good - Reasonable control stability"
        elif control_stability < 0.4:
            interpretation = "Fair - Moderate switching frequency"
        else:
            interpretation = "Poor - High switching frequency, potentially inefficient"

        f.write(f"Assessment: {interpretation}\n\n")

        # Individual Episode Results
        f.write("INDIVIDUAL EPISODE RESULTS\n")
        f.write("-" * 50 + "\n")
        episode_rewards = performance["episode_data"]["total_rewards"]
        control_scores = performance.get("control_stability_scores", [])

        for i, (reward, stability) in enumerate(zip(episode_rewards, control_scores)):
            f.write(
                f"Episode {i+1:2d}: Reward = {reward:8.3f}, Stability = {stability:.3f}\n"
            )

        f.write(f"\n{'='*80}\n")
        f.write("END OF REPORT\n")
        f.write(f"{'='*80}\n")

    print(f"Evaluation report saved to {report_path}")


def run_pid_evaluation(
    data_file,
    output_dir=None,
    num_episodes=5,
    render=True,
    env_params_path=None,
    ground_kp=2.0,
    ground_ki=0.1,
    ground_kd=0.05,
    top_kp=2.0,
    top_ki=0.1,
    top_kd=0.05,
):
    """
    Train a SINDy model and evaluate a PID controller.
    Updated to handle custom observations.

    Args:
        data_file: Path to data file for training SINDy model
        output_dir: Directory to save results
        num_episodes: Number of episodes for evaluation
        render: Whether to render during evaluation
        env_params_path: Path to the saved environment parameters
        ground_kp, ground_ki, ground_kd: PID parameters for ground floor
        top_kp, top_ki, top_kd: PID parameters for top floor

    Returns:
        dict: Evaluation results
    """
    # Set output directory - use pid_results as default
    if output_dir is None:
        output_dir = "pid_results"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")

    # PID parameters
    ground_params = {"kp": ground_kp, "ki": ground_ki, "kd": ground_kd}
    top_params = {"kp": top_kp, "ki": top_ki, "kd": top_kd}

    print(f"Ground Floor PID Parameters: {ground_params}")
    print(f"Top Floor PID Parameters: {top_params}")

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
        env_params_copy = env_params.copy()
        env_params_copy["sindy_model"] = sindy_model

        # Remove parameters that aren't for the environment constructor
        # (same approach as other controller scripts)
        env_params_copy.pop("n_envs", None)
        env_params_copy.pop("vec_env_type", None)
        env_params_copy.pop("normalized", None)
        env_params_copy.pop("observation_info", None)

        # Print observation configuration if available
        if "custom_observations" in env_params_copy:
            custom_obs = env_params_copy["custom_observations"]
            if custom_obs:
                print(f"Using custom observations from saved params: {custom_obs}")
            else:
                print("Using default observations from saved params")

        # Create environment with saved parameters
        try:
            env = DollhouseThermalEnv(**env_params_copy)
            print("Successfully created environment with saved parameters")
        except Exception as e:
            print(f"Error creating environment with saved parameters: {e}")
            print(
                "Falling back to environment without custom observation parameters..."
            )

            # Remove custom observation parameters and try again
            fallback_params = env_params_copy.copy()
            fallback_params.pop("custom_observations", None)

            env = DollhouseThermalEnv(**fallback_params)
            print("Using default observations (fallback)")
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
            "episode_length": 2880,  # 24 hours with 30-second timesteps
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

    # Print environment configuration
    print(f"\nEnvironment Configuration:")
    print(f"Setpoint Pattern: {env.setpoint_pattern}")
    print(f"Base Heating Setpoint: {env.initial_heating_setpoint}°C")
    print(f"Base Cooling Setpoint: {env.initial_cooling_setpoint}°C")
    print(f"Random Start Time: {env.random_start_time}")
    print(f"Reward Shaping: {env.use_reward_shaping}")

    # Print observation configuration
    obs_info = env.get_observation_info()
    print(f"Observation Space Shape: {obs_info['observation_space_shape']}")
    print(f"Observations Used: {obs_info['observation_list']}")

    # Save environment parameters and PID configuration
    with open(os.path.join(output_dir, "env_params.json"), "w") as f:
        # Convert non-serializable parameters to strings
        if env_params_path:
            serializable_params = env_params_copy.copy()
        else:
            serializable_params = env_params.copy()
        serializable_params["sindy_model"] = "SINDy model object (not serializable)"
        serializable_params["ground_pid_params"] = ground_params
        serializable_params["top_pid_params"] = top_params
        serializable_params["observation_info"] = obs_info
        json.dump(serializable_params, f, indent=4)

    # Evaluate PID controller
    print(f"\nEvaluating PID Controller for {num_episodes} episodes...")
    start_time = time.time()
    performance = evaluate_pid_controller(
        env=env,
        num_episodes=num_episodes,
        render=render,
        ground_params=ground_params,
        top_params=top_params,
        output_dir=output_dir,
    )
    evaluation_time = time.time() - start_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")

    # Create evaluation report
    print("Generating evaluation report...")
    create_evaluation_report(performance, output_dir, "PID Controller")

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Controller: PID Controller")
    print(f"Episodes Evaluated: {num_episodes}")
    print(f"Evaluation Time: {evaluation_time:.2f} seconds")
    print(f"Ground Floor PID: {ground_params}")
    print(f"Top Floor PID: {top_params}")
    print(f"\nObservation Configuration:")
    print(f"  Observation Space Shape: {performance['observation_space_shape']}")
    print(f"  Observations Used: {performance['observations_used']}")
    print(f"\nPerformance Metrics:")
    print(
        f"  Average Total Reward: {performance['avg_total_reward']:.3f} ± {performance['std_total_reward']:.3f}"
    )
    print(f"  Ground Floor Comfort: {performance['avg_ground_comfort_pct']:.2f}%")
    print(f"  Top Floor Comfort: {performance['avg_top_comfort_pct']:.2f}%")
    print(f"  Average Light Hours: {performance['avg_light_hours']:.2f}")
    print(f"\nControl Stability Analysis:")
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
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}")

    return performance


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train SINDy model and evaluate PID controller with custom observations support"
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

    # PID parameters for ground floor
    parser.add_argument(
        "--ground-kp", type=float, default=2.0, help="Ground floor proportional gain"
    )
    parser.add_argument(
        "--ground-ki", type=float, default=0.1, help="Ground floor integral gain"
    )
    parser.add_argument(
        "--ground-kd", type=float, default=0.05, help="Ground floor derivative gain"
    )

    # PID parameters for top floor
    parser.add_argument(
        "--top-kp", type=float, default=2.0, help="Top floor proportional gain"
    )
    parser.add_argument(
        "--top-ki", type=float, default=0.1, help="Top floor integral gain"
    )
    parser.add_argument(
        "--top-kd", type=float, default=0.05, help="Top floor derivative gain"
    )

    args = parser.parse_args()

    # Run evaluation
    run_pid_evaluation(
        data_file=args.data,
        output_dir=args.output,
        num_episodes=args.episodes,
        render=not args.no_render,
        env_params_path=args.env_params,
        ground_kp=args.ground_kp,
        ground_ki=args.ground_ki,
        ground_kd=args.ground_kd,
        top_kp=args.top_kp,
        top_ki=args.top_ki,
        top_kd=args.top_kd,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

"""
# Basic evaluation with default observations:
python pid_controller.py --data "data/dollhouse-data.csv" --episodes 5

# Evaluate using saved environment parameters (with custom observations):
python pid_controller.py \
  --data "data/dollhouse-data.csv" \
  --env-params "results/ppo_compact_obs/env_params.json" \
  --episodes 10

# Evaluate with custom PID parameters:
python pid_controller.py \
  --data "data/dollhouse-data.csv" \
  --ground-kp 2.5 --ground-ki 0.15 --ground-kd 0.08 \
  --top-kp 1.8 --top-ki 0.12 --top-kd 0.06 \
  --episodes 8 \
  --output "pid_results_tuned"

# Quick evaluation without rendering:
python pid_controller.py \
  --data "data/dollhouse-data.csv" \
  --episodes 3 \
  --no-render


"""
