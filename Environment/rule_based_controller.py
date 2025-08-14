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


def create_rule_based_controller(hysteresis=0.5):
    """
    Create a simple rule-based controller that works with any observation space.

    Args:
        hysteresis: Temperature buffer to prevent oscillation

    Returns:
        function: Rule-based controller function
    """

    def controller(env, observation):
        # Extract state variables using the helper function
        state = get_state_from_observation(env, observation)

        ground_temp = state["ground_temp"]
        top_temp = state["top_temp"]
        external_temp = state["external_temp"]
        heating_setpoint = state["heating_setpoint"]
        cooling_setpoint = state["cooling_setpoint"]

        # Initialize action
        action = np.zeros(4, dtype=int)

        # Average setpoint for decision boundary
        avg_setpoint = (heating_setpoint + cooling_setpoint) / 2

        # Ground floor control logic
        if ground_temp < avg_setpoint - hysteresis:
            # Too cold - turn on light for heat, close window
            action[0] = 1  # Turn ON ground light
            action[1] = 0  # Close ground window
        else:
            # Too hot - turn off light, open window
            action[0] = 0  # Turn OFF ground light
            action[1] = 1  # Open ground window

        # Top floor control logic (same approach)
        if top_temp < avg_setpoint - hysteresis:
            # Too cold - turn on light for heat, close window
            action[2] = 1  # Turn ON top light
            action[3] = 0  # Close top window
        else:
            # Too hot - turn off light, open window
            action[2] = 0  # Turn OFF top light
            action[3] = 1  # Open top window

        return action

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


def evaluate_rule_based(
    env, num_episodes=5, render=True, hysteresis=0.5, output_dir=None
):
    """
    Evaluate a simple rule-based controller on the environment.
    Now handles custom observations properly.

    Args:
        env: The environment to evaluate on
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        hysteresis: Hysteresis parameter for the rule-based controller
        output_dir: Directory to save results (if None, uses default)

    Returns:
        dict: Evaluation results including control stability
    """
    # Create the rule-based controller
    controller = create_rule_based_controller(hysteresis=hysteresis)

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
    control_stability_scores = []

    # Print environment and observation info
    obs_info = env.get_observation_info()
    print(f"Environment Observation Configuration:")
    print(f"  Observation Space Shape: {obs_info['observation_space_shape']}")
    print(f"  Observations Used: {obs_info['observation_list']}")

    for episode in range(num_episodes):
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

        while not terminated and not truncated:
            # Extract state information
            state = get_state_from_observation(env, obs)

            # Store current setpoints
            heating_sp = state["heating_setpoint"]
            cooling_sp = state["cooling_setpoint"]
            setpoints.append([heating_sp, cooling_sp])

            # Get action from controller
            action = controller(env, obs)

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

        print(f"\nSimple Rule-Based Controller Evaluation Summary:")
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
        }

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

        # Use provided output_dir or default to rule_based_results
        save_dir = output_dir if output_dir else "rule_based_results"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"simple_controller_results_{timestamp}.json")

        env.save_results(
            filepath,
            controller_name="Simple Rule-Based Controller",
        )
        print(f"Results saved to {filepath}")

        # Save performance data with episode details
        results_path = os.path.join(
            save_dir, f"simple_detailed_results_{timestamp}.json"
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
        visualize_performance(performance, save_dir, "Simple Rule-Based Controller")
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


def visualize_performance(
    performance, output_dir, controller_name="Simple Rule-Based Controller"
):
    """
    Create visualizations of controller performance with control stability metric.
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

    # Check if we have dynamic setpoints
    has_dynamic_setpoints = len(episode_setpoints) > 0 and len(episode_setpoints[0]) > 0

    # Plot temperatures, actions, setpoints, and rewards for the first episode
    fig = plt.figure(figsize=(16, 20))

    # Temperature plot with dynamic setpoints
    plt.subplot(6, 1, 1)
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
    plt.subplot(6, 1, 2)
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

    # External temperature plot
    plt.subplot(6, 1, 3)
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
    plt.subplot(6, 1, 4)
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
    plt.subplot(6, 1, 5)
    plt.plot(episode_rewards[0], label="Step Reward", linewidth=2, color="darkgreen")
    plt.title(f"{controller_name} - Rewards (Episode 1)")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Comfort zone analysis
    plt.subplot(6, 1, 6)
    ground_temps = np.array([temp[0] for temp in episode_temperatures[0]])
    top_temps = np.array([temp[1] for temp in episode_temperatures[0]])

    heating_setpoint = performance.get("heating_setpoint", 20.0)
    cooling_setpoint = performance.get("cooling_setpoint", 24.0)

    # Calculate comfort violations
    ground_violations = np.maximum(heating_setpoint - ground_temps, 0) + np.maximum(
        ground_temps - cooling_setpoint, 0
    )
    top_violations = np.maximum(heating_setpoint - top_temps, 0) + np.maximum(
        top_temps - cooling_setpoint, 0
    )

    plt.plot(
        ground_violations, label="Ground Floor Violations", linewidth=2, color="blue"
    )
    plt.plot(top_violations, label="Top Floor Violations", linewidth=2, color="red")
    plt.title(f"{controller_name} - Comfort Zone Violations (Episode 1)")
    plt.xlabel("Timestep")
    plt.ylabel("Temperature Violation (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"simple_controller_episode_analysis.png"),
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
        os.path.join(output_dir, f"simple_controller_summary.png"),
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

    # Temperature performance analysis
    plt.subplot(2, 3, 5)
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

    # Energy efficiency analysis
    plt.subplot(2, 3, 6)
    energy_per_episode = []
    for episode_actions in episode_actions:
        episode_energy = np.sum(
            np.array(episode_actions), axis=0
        )  # Sum for each action type
        total_energy = np.sum(episode_energy[[0, 2]])  # Sum of light actions
        energy_per_episode.append(total_energy)

    episodes = range(1, len(energy_per_episode) + 1)
    bars = plt.bar(
        episodes, energy_per_episode, color="gold", edgecolor="black", linewidth=1
    )
    plt.title(f"{controller_name} - Energy Usage per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Light Usage (timesteps)")
    plt.grid(True, alpha=0.3)

    # Add average line
    avg_energy = np.mean(energy_per_episode)
    plt.axhline(
        y=avg_energy,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Average: {avg_energy:.1f}",
    )
    plt.legend()

    plt.suptitle(
        f"{controller_name} - Control Analysis", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"simple_controller_control_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create temperature distribution plot with dynamic setpoint ranges
    plt.figure(figsize=(16, 10))

    # Combine all temperature data across episodes
    all_ground_temps = []
    all_top_temps = []
    all_heating_setpoints = []
    all_cooling_setpoints = []

    for episode_idx, episode_temps in enumerate(episode_temperatures):
        all_ground_temps.extend([temp[0] for temp in episode_temps])
        all_top_temps.extend([temp[1] for temp in episode_temps])

        if has_dynamic_setpoints and episode_idx < len(episode_setpoints):
            all_heating_setpoints.extend(
                [sp[0] for sp in episode_setpoints[episode_idx]]
            )
            all_cooling_setpoints.extend(
                [sp[1] for sp in episode_setpoints[episode_idx]]
            )

    # Ground floor temperature distribution
    plt.subplot(2, 3, 1)
    n, bins, patches = plt.hist(
        all_ground_temps, bins=30, alpha=0.7, edgecolor="black", color="lightblue"
    )

    if has_dynamic_setpoints and all_heating_setpoints:
        # Show setpoint ranges
        min_heating = min(all_heating_setpoints)
        max_heating = max(all_heating_setpoints)
        min_cooling = min(all_cooling_setpoints)
        max_cooling = max(all_cooling_setpoints)

        plt.axvspan(
            min_heating,
            max_heating,
            alpha=0.3,
            color="red",
            label=f"Heating Range ({min_heating:.1f}-{max_heating:.1f}°C)",
        )
        plt.axvspan(
            min_cooling,
            max_cooling,
            alpha=0.3,
            color="blue",
            label=f"Cooling Range ({min_cooling:.1f}-{max_cooling:.1f}°C)",
        )
    else:
        heating_setpoint = performance.get("heating_setpoint", 20.0)
        cooling_setpoint = performance.get("cooling_setpoint", 24.0)
        plt.axvline(
            x=heating_setpoint,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Heating Setpoint ({heating_setpoint}°C)",
        )
        plt.axvline(
            x=cooling_setpoint,
            color="b",
            linestyle="--",
            linewidth=2,
            label=f"Cooling Setpoint ({cooling_setpoint}°C)",
        )

    plt.title(f"{controller_name} - Ground Floor Temperature Distribution")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Top floor temperature distribution
    plt.subplot(2, 3, 2)
    plt.hist(all_top_temps, bins=30, alpha=0.7, edgecolor="black", color="lightcoral")

    if has_dynamic_setpoints and all_heating_setpoints:
        plt.axvspan(
            min_heating,
            max_heating,
            alpha=0.3,
            color="red",
            label=f"Heating Range ({min_heating:.1f}-{max_heating:.1f}°C)",
        )
        plt.axvspan(
            min_cooling,
            max_cooling,
            alpha=0.3,
            color="blue",
            label=f"Cooling Range ({min_cooling:.1f}-{max_cooling:.1f}°C)",
        )
    else:
        heating_setpoint = performance.get("heating_setpoint", 20.0)
        cooling_setpoint = performance.get("cooling_setpoint", 24.0)
        plt.axvline(
            x=heating_setpoint,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Heating Setpoint ({heating_setpoint}°C)",
        )
        plt.axvline(
            x=cooling_setpoint,
            color="b",
            linestyle="--",
            linewidth=2,
            label=f"Cooling Setpoint ({cooling_setpoint}°C)",
        )

    plt.title(f"{controller_name} - Top Floor Temperature Distribution")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Temperature correlation plot
    plt.subplot(2, 3, 3)
    plt.scatter(all_ground_temps, all_top_temps, alpha=0.5, s=20, color="purple")
    plt.xlabel("Ground Floor Temperature (°C)")
    plt.ylabel("Top Floor Temperature (°C)")
    plt.title(f"{controller_name} - Floor Temperature Correlation")
    plt.grid(True, alpha=0.3)

    # Add correlation coefficient
    if len(all_ground_temps) > 1:
        temp_correlation = np.corrcoef(all_ground_temps, all_top_temps)[0, 1]
        plt.text(
            0.05,
            0.95,
            f"Correlation: {temp_correlation:.3f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # External temperature vs internal temperature
    plt.subplot(2, 3, 4)
    all_external_temps = []
    for episode_ext_temps in episode_external_temps:
        all_external_temps.extend(episode_ext_temps)

    plt.scatter(
        all_external_temps,
        all_ground_temps,
        alpha=0.5,
        s=20,
        color="green",
        label="Ground Floor",
    )
    plt.scatter(
        all_external_temps,
        all_top_temps,
        alpha=0.5,
        s=20,
        color="red",
        label="Top Floor",
    )
    plt.xlabel("External Temperature (°C)")
    plt.ylabel("Internal Temperature (°C)")
    plt.title(f"{controller_name} - External vs Internal Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Action correlation analysis
    plt.subplot(2, 3, 5)
    all_actions = np.concatenate(episode_actions)
    action_names_short = ["GLight", "GWindow", "TLight", "TWindow"]

    # Calculate correlation matrix for actions
    try:
        # Ensure we have enough data and proper shape
        if all_actions.shape[0] > 1 and all_actions.shape[1] == 4:
            action_corr = np.corrcoef(all_actions.T)

            # Check if we got a proper 2D matrix
            if action_corr.ndim == 2 and action_corr.shape == (4, 4):
                # Create correlation heatmap
                im = plt.imshow(action_corr, cmap="coolwarm", vmin=-1, vmax=1)
                plt.colorbar(im, shrink=0.8)
                plt.xticks(range(4), action_names_short, rotation=45)
                plt.yticks(range(4), action_names_short)
                plt.title(f"{controller_name} - Action Correlations")

                # Add correlation values as text
                for i in range(4):
                    for j in range(4):
                        plt.text(
                            j,
                            i,
                            f"{action_corr[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=10,
                        )
            else:
                raise ValueError("Invalid correlation matrix shape")
        else:
            raise ValueError("Insufficient data for correlation analysis")

    except Exception as e:
        print(f"Warning: Could not create action correlation plot: {e}")
        # Fallback: show action usage
        action_usage = np.mean(all_actions, axis=0)

        # Ensure action_usage is always an array with 4 elements
        if np.isscalar(action_usage):
            action_usage = np.array(
                [action_usage, 0, 0, 0]
            )  # Fallback for single action
        elif len(action_usage) != 4:
            # Pad or truncate to ensure we have exactly 4 values
            action_usage = np.pad(
                action_usage, (0, max(0, 4 - len(action_usage))), "constant"
            )[:4]

        bars = plt.bar(
            action_names_short, action_usage, color=["orange", "green", "red", "blue"]
        )
        plt.title(f"{controller_name} - Action Usage")
        plt.ylabel("Average Usage")
        plt.ylim(0, 1)

        # Add value labels
        for bar, value in zip(bars, action_usage):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

    # Reward vs stability analysis
    plt.subplot(2, 3, 6)
    episode_total_rewards = performance["episode_data"]["total_rewards"]
    plt.scatter(
        performance["control_stability_scores"],
        episode_total_rewards,
        color="darkgreen",
        alpha=0.7,
        s=100,
        edgecolors="black",
    )
    plt.xlabel("Control Stability")
    plt.ylabel("Total Episode Reward")
    plt.title(f"{controller_name} - Stability vs Reward")
    plt.grid(True, alpha=0.3)

    # Add correlation coefficient
    if len(episode_total_rewards) > 1:
        correlation = np.corrcoef(
            performance["control_stability_scores"], episode_total_rewards
        )[0, 1]
        plt.text(
            0.05,
            0.95,
            f"Correlation: {correlation:.3f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.suptitle(
        f"{controller_name} - Temperature Analysis", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"simple_controller_temperature_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Comprehensive visualizations saved to {output_dir}")


def create_evaluation_report(
    performance, output_dir, controller_name="Simple Rule-Based Controller"
):
    """
    Create a comprehensive evaluation report in text format.

    Args:
        performance: Performance dictionary from evaluate_rule_based
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


def run_rule_based_evaluation(
    data_file,
    output_dir=None,
    num_episodes=5,
    render=True,
    env_params_path=None,
    hysteresis=0.5,
):
    """
    Train a SINDy model and evaluate a simple rule-based controller.
    Updated to handle custom observations.

    Args:
        data_file: Path to data file for training SINDy model
        output_dir: Directory to save results
        num_episodes: Number of episodes for evaluation
        render: Whether to render during evaluation
        env_params_path: Path to the saved environment parameters
        hysteresis: Hysteresis parameter for the controller

    Returns:
        dict: Evaluation results
    """
    # Set output directory - use rule_based_results as default
    if output_dir is None:
        output_dir = "rule_based_results"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")

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
        # (same approach as evaluate_rl_agent.py)
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

    # Save environment parameters
    with open(os.path.join(output_dir, "env_params.json"), "w") as f:
        # Convert non-serializable parameters to strings
        if env_params_path:
            serializable_params = env_params_copy.copy()
        else:
            serializable_params = env_params.copy()
        serializable_params["sindy_model"] = "SINDy model object (not serializable)"
        serializable_params["observation_info"] = obs_info
        json.dump(serializable_params, f, indent=4)

    # Evaluate rule-based controller
    print(f"\nEvaluating Simple Rule-Based Controller for {num_episodes} episodes...")
    start_time = time.time()
    performance = evaluate_rule_based(
        env=env,
        num_episodes=num_episodes,
        render=render,
        hysteresis=hysteresis,
        output_dir=output_dir,
    )
    evaluation_time = time.time() - start_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")

    # Create evaluation report
    print("Generating evaluation report...")
    create_evaluation_report(performance, output_dir, "Simple Rule-Based Controller")

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Controller: Simple Rule-Based")
    print(f"Episodes Evaluated: {num_episodes}")
    print(f"Evaluation Time: {evaluation_time:.2f} seconds")
    print(f"Hysteresis: {hysteresis}")
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
        description="Train SINDy model and evaluate simple rule-based controller with custom observations support"
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
    parser.add_argument(
        "--hysteresis",
        type=float,
        default=0.5,
        help="Hysteresis parameter for temperature control",
    )

    args = parser.parse_args()

    # Run evaluation
    run_rule_based_evaluation(
        data_file=args.data,
        output_dir=args.output,
        num_episodes=args.episodes,
        render=not args.no_render,
        env_params_path=args.env_params,
        hysteresis=args.hysteresis,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

"""
# Basic evaluation with default observations:
python rule_based_controller.py --data "data/dollhouse-data.csv" --episodes 5

# Evaluate using saved environment parameters (with custom observations):
python rule_based_controller.py \
  --data "data/dollhouse-data.csv" \
  --env-params "results/ppo_compact_obs/env_params.json" \
  --episodes 10

# Evaluate with specific hysteresis and custom output:
python rule_based_controller.py \
  --data "data/dollhouse-data.csv" \
  --hysteresis 1.0 \
  --episodes 8 \
  --output "rule_based_hysteresis_1.0"

# Quick evaluation without rendering:
python rule_based_controller.py \
  --data "data/dollhouse-data.csv" \
  --episodes 3 \
  --no-render

# The script automatically:
# - Detects custom observation configuration from saved params
# - Adapts the rule-based controller to work with any observation space
# - Extracts meaningful state information regardless of observations
# - Generates comprehensive analysis and visualizations
# - Creates detailed performance reports
"""
