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


def create_pid_controller(ground_params=None, top_params=None):
    """
    Create a PID-based controller for the dollhouse environment.

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

    def controller(observation):
        """
        PID controller function.

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
    episode_pid_outputs = []
    control_stability_scores = []

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
            # Extract and store current setpoints from observation
            if len(obs) > 8:  # Check if setpoints are in observation
                heating_sp = obs[7]
                cooling_sp = obs[8]
            else:
                # Fallback to environment attributes
                heating_sp = orig_env.initial_heating_setpoint
                cooling_sp = orig_env.initial_cooling_setpoint

            setpoints.append([heating_sp, cooling_sp])

            # If the environment is normalized, get the original observation
            if hasattr(env, "get_original_obs"):
                orig_obs = env.get_original_obs()
                action = controller(orig_obs)
            else:
                action = controller(obs)

            # Store PID outputs for analysis
            ground_target = (heating_sp + cooling_sp) / 2
            top_target = (heating_sp + cooling_sp) / 2

            # Get PID outputs (before action conversion)
            current_time = (steps) * 30
            ground_output = controller.ground_pid.compute(
                ground_target, obs[0], current_time
            )
            top_output = controller.top_pid.compute(top_target, obs[1], current_time)
            pid_outputs.append([ground_output, top_output])

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
    if hasattr(orig_env, "get_performance_summary"):
        performance = orig_env.get_performance_summary()

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
            performance["heating_setpoint"] = orig_env.initial_heating_setpoint
            performance["cooling_setpoint"] = orig_env.initial_cooling_setpoint
            performance["has_dynamic_setpoints"] = False

        performance["reward_type"] = orig_env.reward_type
        performance["energy_weight"] = orig_env.energy_weight
        performance["comfort_weight"] = orig_env.comfort_weight

        # Use provided output_dir or default to pid_results
        save_dir = output_dir if output_dir else "pid_results"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"pid_controller_results_{timestamp}.json")

        orig_env.save_results(
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

    # PID outputs plot
    plt.subplot(6, 1, 2)
    if episode_pid_outputs:
        ground_pid_outputs = [pid[0] for pid in episode_pid_outputs[0]]
        top_pid_outputs = [pid[1] for pid in episode_pid_outputs[0]]
        plt.plot(ground_pid_outputs, label="Ground Floor PID Output", linewidth=2)
        plt.plot(top_pid_outputs, label="Top Floor PID Output", linewidth=2)
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

    for i, name in enumerate(action_names):
        action_series = actions[:, i]
        plt.plot(action_series, label=name, linewidth=2)

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
    plt.plot(episode_rewards[0], label="Step Reward", linewidth=2)
    plt.title(f"{controller_name} - Rewards (Episode 1)")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # PID parameters info
    plt.subplot(6, 1, 6)
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
        plt.bar([controller_name], [value])
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
        os.path.join(output_dir, f"pid_controller_summary.png"),
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
        os.path.join(output_dir, f"pid_controller_control_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Visualizations saved to {output_dir}")


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

    # Save environment parameters and PID configuration
    with open(os.path.join(output_dir, "env_params.json"), "w") as f:
        # Convert non-serializable parameters to strings
        serializable_params = env_params.copy()
        serializable_params["sindy_model"] = "SINDy model object (not serializable)"
        serializable_params["ground_pid_params"] = ground_params
        serializable_params["top_pid_params"] = top_params
        json.dump(serializable_params, f, indent=4)

    # Evaluate PID controller
    print(f"\nEvaluating PID Controller...")
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
        description="Train SINDy model and evaluate PID controller with control stability metric"
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

# Example usage:
# python pid_controller.py --data "../Data/dollhouse-data-2025-03-24.csv" --episodes 10 --ground-kp 2.5 --ground-ki 0.15 --ground-kd 0.08
# python pid_controller.py --data "../Data/dollhouse-data-2025-03-24.csv" --episodes 5 --env-params "results/ppo_20250513_151705/env_params.json" --output "pid_results_tuned"
