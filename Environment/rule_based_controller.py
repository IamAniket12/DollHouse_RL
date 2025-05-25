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


def create_rule_based_controller(hysteresis=0.5):
    """
    Create a simple rule-based controller.

    Args:
        hysteresis: Temperature buffer to prevent oscillation

    Returns:
        function: Rule-based controller function
    """

    def controller(observation):
        # Extract state variables
        ground_temp = observation[0]
        top_temp = observation[1]
        external_temp = observation[2]
        heating_setpoint = observation[7]
        cooling_setpoint = observation[8]

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


def evaluate_rule_based(env, num_episodes=5, render=True, hysteresis=0.5):
    """
    Evaluate a simple rule-based controller on the environment.

    Args:
        env: The environment to evaluate on
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        hysteresis: Hysteresis parameter for the rule-based controller

    Returns:
        dict: Evaluation results
    """
    # Create the rule-based controller
    controller = create_rule_based_controller(hysteresis=hysteresis)

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

        # Store episode data
        episode_temperatures.append(temps)
        episode_external_temps.append(ext_temps)
        episode_actions.append(actions)
        episode_rewards.append(rewards)
        episode_setpoints.append(setpoints)

        avg_actions = {k: v / steps for k, v in actions_taken.items()}

        print(
            f"Episode {episode+1}/{num_episodes}: Total Reward = {episode_reward:.2f}"
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

        print(f"\nSimple Rule-Based Controller Evaluation Summary:")
        print(f"Average Total Reward: {performance['avg_total_reward']:.2f}")
        print(f"Ground Floor Comfort %: {performance['avg_ground_comfort_pct']:.2f}%")
        print(f"Top Floor Comfort %: {performance['avg_top_comfort_pct']:.2f}%")
        print(f"Average Light Hours: {performance['avg_light_hours']:.2f}")

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
            performance["heating_setpoint"] = orig_env.initial_heating_setpoint
            performance["cooling_setpoint"] = orig_env.initial_cooling_setpoint
            performance["has_dynamic_setpoints"] = False

        performance["reward_type"] = orig_env.reward_type
        performance["energy_weight"] = orig_env.energy_weight
        performance["comfort_weight"] = orig_env.comfort_weight

        # Save results
        output_dir = "rule_based_results"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            output_dir, f"simple_controller_results_{timestamp}.json"
        )

        orig_env.save_results(
            filepath,
            controller_name="Simple Rule-Based Controller",
        )
        print(f"Results saved to {filepath}")

        # Save performance data with episode details
        results_path = os.path.join(
            output_dir, f"simple_detailed_results_{timestamp}.json"
        )
        with open(results_path, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_perf = {}
            for key, value in performance.items():
                if key == "episode_data":
                    serializable_perf[key] = {
                        "temperatures": [
                            [[float(t) for t in temp] for temp in episode]
                            for episode in value["temperatures"]
                        ],
                        "external_temps": [
                            [float(t) for t in temps]
                            for temps in value["external_temps"]
                        ],
                        "actions": [
                            [[int(a) for a in action] for action in episode]
                            for episode in value["actions"]
                        ],
                        "rewards": [
                            [float(r) for r in rewards] for rewards in value["rewards"]
                        ],
                        "total_rewards": [float(r) for r in value["total_rewards"]],
                        "setpoints": (
                            [
                                [[float(sp) for sp in setpoint] for setpoint in episode]
                                for episode in value.get("setpoints", [])
                            ]
                            if "setpoints" in value
                            else []
                        ),
                    }
                elif isinstance(value, np.ndarray):
                    serializable_perf[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_perf[key] = (
                        float(value) if isinstance(value, np.floating) else int(value)
                    )
                else:
                    serializable_perf[key] = value

            json.dump(serializable_perf, f, indent=4)
        print(f"Detailed results saved to {results_path}")

        # Create visualizations
        visualize_performance(performance, output_dir, "Simple Rule-Based Controller")
    else:
        # Simple performance metrics if the environment doesn't provide detailed ones
        performance = {
            "avg_total_reward": np.mean(total_rewards),
            "std_total_reward": np.std(total_rewards),
        }
        print(
            f"\nAverage Total Reward: {performance['avg_total_reward']:.2f} ± {performance['std_total_reward']:.2f}"
        )

    return performance


def visualize_performance(
    performance, output_dir, controller_name="Simple Rule-Based Controller"
):
    """
    Create visualizations of controller performance with dynamic setpoint support.

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
    plt.figure(figsize=(15, 16))

    # Temperature plot with dynamic setpoints
    plt.subplot(5, 1, 1)
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

    # Separate setpoint plot for better visibility (only if dynamic)
    if has_dynamic_setpoints:
        plt.subplot(5, 1, 2)
        heating_setpoints = [sp[0] for sp in episode_setpoints[0]]
        cooling_setpoints = [sp[1] for sp in episode_setpoints[0]]
        plt.plot(heating_setpoints, "r-", label="Heating Setpoint", linewidth=2)
        plt.plot(cooling_setpoints, "b-", label="Cooling Setpoint", linewidth=2)
        plt.title(f"{controller_name} - Dynamic Setpoints (Episode 1)")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        subplot_offset = 1
    else:
        subplot_offset = 0

    # External temperature plot
    plt.subplot(5, 1, 2 + subplot_offset)
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

    # Actions plot
    plt.subplot(5, 1, 3 + subplot_offset)
    actions = np.array(episode_actions[0])
    action_names = ["Ground Light", "Ground Window", "Top Light", "Top Window"]
    for i, name in enumerate(action_names):
        plt.plot(actions[:, i], label=name, linewidth=2)
    plt.title(f"{controller_name} - Actions (Episode 1)")
    plt.ylabel("Action State (0/1)")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Rewards plot
    plt.subplot(5, 1, 4 + subplot_offset)
    plt.plot(episode_rewards[0], label="Step Reward", linewidth=2)
    plt.title(f"{controller_name} - Rewards (Episode 1)")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"simple_controller_episode_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Summary metrics plot
    plt.figure(figsize=(12, 8))
    metrics = [
        ("avg_total_reward", "Total Reward"),
        ("avg_ground_comfort_pct", "Ground Floor Comfort %"),
        ("avg_top_comfort_pct", "Top Floor Comfort %"),
        ("avg_light_hours", "Light Hours"),
    ]

    for i, (metric, label) in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.bar([controller_name], [performance.get(metric, 0)])
        plt.title(label)
        plt.grid(True, alpha=0.3)

        # Add value label
        plt.text(
            0,
            performance.get(metric, 0) + 0.1,
            f"{performance.get(metric, 0):.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"simple_controller_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Create temperature distribution plot with dynamic setpoint ranges
    plt.figure(figsize=(14, 6))

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
    plt.subplot(1, 2, 1)
    plt.hist(all_ground_temps, bins=30, alpha=0.7, edgecolor="black")

    if has_dynamic_setpoints and all_heating_setpoints:
        # Show setpoint ranges
        min_heating = min(all_heating_setpoints)
        max_heating = max(all_heating_setpoints)
        min_cooling = min(all_cooling_setpoints)
        max_cooling = max(all_cooling_setpoints)

        plt.axvspan(
            min_heating,
            max_heating,
            alpha=0.2,
            color="red",
            label=f"Heating Range ({min_heating:.1f}-{max_heating:.1f}°C)",
        )
        plt.axvspan(
            min_cooling,
            max_cooling,
            alpha=0.2,
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
            label=f"Heating Setpoint ({heating_setpoint}°C)",
        )
        plt.axvline(
            x=cooling_setpoint,
            color="b",
            linestyle="--",
            label=f"Cooling Setpoint ({cooling_setpoint}°C)",
        )

    plt.title(f"{controller_name} - Ground Floor Temperature Distribution")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Top floor temperature distribution
    plt.subplot(1, 2, 2)
    plt.hist(all_top_temps, bins=30, alpha=0.7, edgecolor="black")

    if has_dynamic_setpoints and all_heating_setpoints:
        plt.axvspan(
            min_heating,
            max_heating,
            alpha=0.2,
            color="red",
            label=f"Heating Range ({min_heating:.1f}-{max_heating:.1f}°C)",
        )
        plt.axvspan(
            min_cooling,
            max_cooling,
            alpha=0.2,
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
            label=f"Heating Setpoint ({heating_setpoint}°C)",
        )
        plt.axvline(
            x=cooling_setpoint,
            color="b",
            linestyle="--",
            label=f"Cooling Setpoint ({cooling_setpoint}°C)",
        )

    plt.title(f"{controller_name} - Top Floor Temperature Distribution")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"simple_controller_temperature_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Visualizations saved to {output_dir}")


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
    # Set output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"simple_controller_results_{timestamp}"

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
            "episode_length": 2880,  # 24 hours with 30-second timesteps
            "time_step_seconds": 30,
            "heating_setpoint": 26.0,
            "cooling_setpoint": 28.0,
            "external_temp_pattern": "fixed",
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

    # Save environment parameters
    with open(os.path.join(output_dir, "env_params.json"), "w") as f:
        # Convert non-serializable parameters to strings
        serializable_params = env_params.copy()
        serializable_params["sindy_model"] = "SINDy model object (not serializable)"
        json.dump(serializable_params, f, indent=4)

    # Evaluate rule-based controller
    print(f"\nEvaluating Simple Rule-Based Controller...")
    start_time = time.time()
    performance = evaluate_rule_based(
        env=env, num_episodes=num_episodes, render=render, hysteresis=hysteresis
    )
    evaluation_time = time.time() - start_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")

    # Plot temperature history if available
    if hasattr(env, "history") and "ground_temp" in env.history:
        plt.figure(figsize=(12, 8))

        # Plot temperatures
        plt.subplot(2, 1, 1)
        time_steps = range(len(env.history["ground_temp"]))
        time_hours = [t * env.time_step_seconds / 3600 for t in time_steps]

        plt.plot(time_hours, env.history["ground_temp"], "b-", label="Ground Floor")
        plt.plot(time_hours, env.history["top_temp"], "r-", label="Top Floor")
        plt.plot(time_hours, env.history["external_temp"], "g-", label="External")

        # Handle dynamic setpoints in history plot
        if "heating_setpoint" in env.history and "cooling_setpoint" in env.history:
            # Dynamic setpoints available in history
            plt.plot(
                time_hours,
                env.history["heating_setpoint"],
                "k--",
                label="Heating Setpoint",
            )
            plt.plot(
                time_hours,
                env.history["cooling_setpoint"],
                "k-.",
                label="Cooling Setpoint",
            )
        else:
            # Static setpoints
            heating_sp = env.initial_heating_setpoint
            cooling_sp = env.initial_cooling_setpoint
            plt.axhline(
                y=heating_sp,
                color="k",
                linestyle="--",
                label=f"Heating Setpoint ({heating_sp}°C)",
            )
            plt.axhline(
                y=cooling_sp,
                color="k",
                linestyle="-.",
                label=f"Cooling Setpoint ({cooling_sp}°C)",
            )

        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.title(f"Simple Rule-Based Controller Performance")
        plt.grid(True)

        # Plot actions
        plt.subplot(2, 1, 2)
        plt.step(time_hours, env.history["ground_light"], "b-", label="Ground Light")
        plt.step(time_hours, env.history["ground_window"], "b--", label="Ground Window")
        plt.step(time_hours, env.history["top_light"], "r-", label="Top Light")
        plt.step(time_hours, env.history["top_window"], "r--", label="Top Window")
        plt.xlabel("Time (hours)")
        plt.ylabel("Control State")
        plt.yticks([0, 1], ["OFF/CLOSED", "ON/OPEN"])
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "simple_controller_performance.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"Performance plot saved to {os.path.join(output_dir, 'simple_controller_performance.png')}"
        )

    return performance


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train SINDy model and evaluate simple rule-based controller"
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

# Example usage:
# python rule_based_controller.py --data "../Data/dollhouse-data-2025-03-24.csv" --episodes 10 --env-params "results/ppo_20250513_151705/env_params
