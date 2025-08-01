import numpy as np
import os
import time
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Import Stable Baselines
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import our modules
from train_sindy_model import train_sindy_model
from dollhouse_env import DollhouseThermalEnv


def load_model(model_path):
    """
    Load a trained RL model.

    Args:
        model_path: Path to the saved model

    Returns:
        model: Loaded model
    """
    # Determine the algorithm from the model path
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path)
    elif "dqn" in model_path.lower():
        model = DQN.load(model_path)
    elif "sac" in model_path.lower():
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm in model path: {model_path}")

    print(f"Successfully loaded model from {model_path}")
    return model


def recreate_environment(env_params_path, data_file=None, model_dir=None):
    """
    Recreate the environment using saved parameters, with normalization if available.

    Args:
        env_params_path: Path to the saved environment parameters
        data_file: Optional path to data file for SINDy model (overrides saved path)
        model_dir: Directory where the model is saved (to look for normalization stats)

    Returns:
        env: Recreated environment (possibly normalized)
    """
    # Load environment parameters
    with open(env_params_path, "r") as f:
        env_params = json.load(f)

    # If data file is provided, retrain SINDy model
    if data_file:
        print(f"Training SINDy model on {data_file}...")
        sindy_model = train_sindy_model(file_path=data_file)
    else:
        # Use the data file from the saved parameters if available
        if "data_file" in env_params:
            data_file = env_params["data_file"]
            print(f"Training SINDy model on {data_file}...")
            sindy_model = train_sindy_model(file_path=data_file)
        else:
            raise ValueError("No data file provided or found in environment parameters")

    # Replace SINDy model placeholder with actual model
    env_params_copy = env_params.copy()
    env_params_copy["sindy_model"] = sindy_model

    # Remove parameters that aren't for the environment constructor
    env_params_copy.pop("n_envs", None)
    env_params_copy.pop("vec_env_type", None)
    env_params_copy.pop("normalized", None)

    # Create environment with saved parameters
    env = DollhouseThermalEnv(**env_params_copy)

    # Check if normalization was used during training
    normalized = env_params.get("normalized", False)

    if normalized and model_dir:
        # Look for normalization statistics
        vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")
        if os.path.exists(vec_normalize_path):
            print(f"Found normalization statistics at {vec_normalize_path}")
            # Wrap environment in DummyVecEnv first
            env = DummyVecEnv([lambda: env])
            # Load and apply normalization
            env = VecNormalize.load(vec_normalize_path, env)
            # Set to evaluation mode (don't update statistics)
            env.training = False
            env.norm_reward = False  # Don't normalize rewards during evaluation
            print("Applied normalization wrapper for evaluation")
        else:
            print(
                f"Warning: Training used normalization but vec_normalize.pkl not found at {vec_normalize_path}"
            )
            print("Proceeding without normalization - results may be suboptimal")
    elif normalized:
        print("Warning: Training used normalization but model_dir not provided")
        print("Proceeding without normalization - results may be suboptimal")

    return env


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


def evaluate_agent(env, model, num_episodes=5, render=False, verbose=True):
    """
    Evaluate a trained agent on the environment using deterministic actions.

    Handles both normalized and non-normalized environments.

    Args:
        env: The environment to evaluate on (may be VecNormalize wrapped)
        model: The trained RL model
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        verbose: Whether to print detailed logs

    Returns:
        dict: Evaluation results including control stability
    """
    # Check if environment is normalized
    is_vec_env = isinstance(env, (DummyVecEnv, VecNormalize))

    # Get the base environment for accessing attributes
    if isinstance(env, VecNormalize):
        base_env = env.venv.envs[0]
    elif isinstance(env, DummyVecEnv):
        base_env = env.envs[0]
    else:
        base_env = env

    total_rewards = []
    episode_temperatures = []
    episode_actions = []
    episode_rewards = []
    episode_external_temps = []
    episode_setpoints = []
    control_stability_scores = []

    for episode in range(num_episodes):
        # Reset environment
        if is_vec_env:
            obs = env.reset()
            # For vec environments, obs is shape (1, obs_dim)
            obs = obs[0] if len(obs.shape) > 1 else obs
        else:
            obs, info = env.reset()

        terminated = False
        truncated = False
        episode_reward = 0

        # Track data for this episode
        temps = []
        actions = []
        rewards = []
        ext_temps = []
        setpoints = []

        while not terminated and not truncated:
            # Get current setpoints from base environment
            heating_sp = base_env.heating_setpoint
            cooling_sp = base_env.cooling_setpoint
            setpoints.append([heating_sp, cooling_sp])

            # Get original observation for logging (before normalization)
            if isinstance(env, VecNormalize):
                # Get unnormalized observation for display
                original_obs = (
                    env.get_original_obs()[0]
                    if hasattr(env, "get_original_obs")
                    else obs
                )
            else:
                original_obs = obs

            # Select action using model
            if is_vec_env:
                # For vectorized environments, we need to add batch dimension
                action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
                action = action[0]  # Remove batch dimension
            else:
                action, _ = model.predict(obs, deterministic=True)

            if verbose and len(temps) % 100 == 0:
                # Use original observation for display
                display_obs = original_obs if isinstance(env, VecNormalize) else obs
                print(
                    f"Step {len(temps)}: Action: {action}, "
                    f"Temps: {display_obs[0]:.1f}/{display_obs[1]:.1f}°C, "
                    f"Setpoints: {heating_sp:.1f}/{cooling_sp:.1f}°C"
                )

            # Take action in environment
            if is_vec_env:
                obs, reward, done, info = env.step([action])
                obs = obs[0]
                reward = reward[0]
                done = done[0]
                info = info[0] if isinstance(info, list) else info

                # VecEnv uses 'done' instead of terminated/truncated
                terminated = done
                truncated = False
            else:
                obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward

            # Record data - use original observation values
            if isinstance(env, VecNormalize):
                # Get unnormalized observation for recording
                original_obs = (
                    env.get_original_obs()[0]
                    if hasattr(env, "get_original_obs")
                    else obs
                )
            else:
                original_obs = obs

            temps.append([original_obs[0], original_obs[1]])
            ext_temps.append(original_obs[2])
            actions.append(action)
            rewards.append(reward)

            if render and not is_vec_env:
                base_env.render()

        # Calculate control stability for this episode
        episode_control_stability = calculate_control_stability(actions)
        control_stability_scores.append(episode_control_stability)

        total_rewards.append(episode_reward)
        episode_temperatures.append(temps)
        episode_external_temps.append(ext_temps)
        episode_actions.append(actions)
        episode_rewards.append(rewards)
        episode_setpoints.append(setpoints)

        if verbose:
            print(
                f"Episode {episode+1}/{num_episodes}: Total Reward = {episode_reward:.2f}, "
                f"Control Stability = {episode_control_stability:.3f}"
            )

    # Get performance summary from base environment
    performance = base_env.get_performance_summary()

    # Add control stability metrics
    performance["control_stability"] = np.mean(control_stability_scores)
    performance["control_stability_std"] = np.std(control_stability_scores)
    performance["control_stability_scores"] = control_stability_scores

    if verbose:
        print("\nAgent Evaluation Summary:")
        print(f"Average Total Reward: {performance['avg_total_reward']:.2f}")
        print(f"Ground Floor Comfort %: {performance['avg_ground_comfort_pct']:.2f}%")
        print(f"Top Floor Comfort %: {performance['avg_top_comfort_pct']:.2f}%")
        print(f"Average Light Hours: {performance['avg_light_hours']:.2f}")
        print(
            f"Control Stability: {performance['control_stability']:.3f} ± {performance['control_stability_std']:.3f}"
        )

    # Add raw episode data to performance dict
    performance["episode_data"] = {
        "temperatures": episode_temperatures,
        "external_temps": episode_external_temps,
        "actions": episode_actions,
        "rewards": episode_rewards,
        "total_rewards": total_rewards,
        "setpoints": episode_setpoints,
    }

    # Set setpoint information
    if episode_setpoints and len(episode_setpoints[0]) > 0:
        performance["heating_setpoint"] = episode_setpoints[0][0][0]
        performance["cooling_setpoint"] = episode_setpoints[0][0][1]
        performance["has_dynamic_setpoints"] = True
    else:
        performance["heating_setpoint"] = base_env.initial_heating_setpoint
        performance["cooling_setpoint"] = base_env.initial_cooling_setpoint
        performance["has_dynamic_setpoints"] = False

    performance["reward_type"] = base_env.reward_type
    performance["energy_weight"] = base_env.energy_weight
    performance["comfort_weight"] = base_env.comfort_weight
    performance["was_normalized"] = is_vec_env

    return performance


def visualize_performance(performance, output_dir, agent_name="RL Agent"):
    """
    Create visualizations of agent performance with control stability metric.

    Args:
        performance: Performance dictionary from evaluate_agent
        output_dir: Directory to save visualizations
        agent_name: Name of the agent for plot titles
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
        heating_setpoints = [sp[0] for sp in episode_setpoints[0]]
        cooling_setpoints = [sp[1] for sp in episode_setpoints[0]]
        plt.plot(heating_setpoints, "r--", label="Heating Setpoint", linewidth=1.5)
        plt.plot(cooling_setpoints, "b--", label="Cooling Setpoint", linewidth=1.5)
    else:
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

    # Add normalization indicator to title
    normalization_status = (
        " (Normalized)" if performance.get("was_normalized", False) else ""
    )
    plt.title(f"{agent_name} - Temperatures (Episode 1){normalization_status}")
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
        plt.title(f"{agent_name} - Dynamic Setpoints (Episode 1)")
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

    plt.title(f"{agent_name} - External Temperature (Episode 1)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Actions plot with state change indicators
    plt.subplot(5, 1, 3 + subplot_offset)
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
        f"{agent_name} - Actions (Episode 1) - Control Stability: {episode_control_stability:.3f}"
    )
    plt.ylabel("Action State (0/1)")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Rewards plot
    plt.subplot(5, 1, 4 + subplot_offset)
    plt.plot(episode_rewards[0], label="Step Reward", linewidth=2)
    plt.title(f"{agent_name} - Rewards (Episode 1)")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir, f'{agent_name.lower().replace(" ", "_")}_episode_analysis.png'
        ),
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
        plt.bar([agent_name], [value])
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
        os.path.join(output_dir, f'{agent_name.lower().replace(" ", "_")}_summary.png'),
        dpi=300,
        bbox_inches="tight",
    )

    # Create control stability detailed analysis plot
    plt.figure(figsize=(14, 10))

    # Plot control stability per episode
    plt.subplot(2, 2, 1)
    episodes = range(1, len(performance["control_stability_scores"]) + 1)
    plt.bar(episodes, performance["control_stability_scores"])
    plt.title(f"{agent_name} - Control Stability per Episode")
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
    plt.title(f"{agent_name} - State Changes by Action (Episode 1)")
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
    plt.title(f"{agent_name} - Control Stability Distribution")
    plt.xlabel("Control Stability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Action usage summary
    plt.subplot(2, 2, 4)
    all_actions = np.concatenate(episode_actions)
    duty_cycles = np.mean(all_actions, axis=0)

    bars = plt.bar(action_names, duty_cycles)
    plt.title(f"{agent_name} - Action Duty Cycles")
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
        os.path.join(
            output_dir, f'{agent_name.lower().replace(" ", "_")}_control_analysis.png'
        ),
        dpi=300,
        bbox_inches="tight",
    )

    # Create temperature distribution plot
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

    plt.title(f"{agent_name} - Ground Floor Temperature Distribution")
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

    plt.title(f"{agent_name} - Top Floor Temperature Distribution")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f'{agent_name.lower().replace(" ", "_")}_temperature_distribution.png',
        ),
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Visualizations saved to {output_dir}")


def main(
    model_path,
    data_file,
    env_params_path=None,
    num_episodes=5,
    render=False,
    output_dir=None,
    verbose=True,
):
    """
    Main function to evaluate a trained agent with control stability metric.

    Args:
        model_path: Path to the trained model
        data_file: Path to data file for training SINDy model
        env_params_path: Path to the saved environment parameters
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        output_dir: Directory to save results
        verbose: Whether to print detailed logs
    """
    # Set output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"eval_results/{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Get model directory
    model_dir = os.path.dirname(model_path)

    # Find env_params_path if not provided
    if env_params_path is None:
        # Try to find it in the same directory as the model
        potential_path = os.path.join(model_dir, "..", "..", "env_params.json")
        if os.path.exists(potential_path):
            env_params_path = potential_path
            print(f"Found environment parameters at {env_params_path}")
        else:
            # Try alternative path structure
            potential_path = os.path.join(model_dir, "..", "env_params.json")
            if os.path.exists(potential_path):
                env_params_path = potential_path
                print(f"Found environment parameters at {env_params_path}")
            else:
                raise ValueError(
                    "env_params_path not provided and not found in model directory"
                )

    # Load model
    model = load_model(model_path)

    # Recreate environment (with normalization if applicable)
    env = recreate_environment(env_params_path, data_file, model_dir)

    # Get base environment for printing info
    if isinstance(env, VecNormalize):
        base_env = env.venv.envs[0]
    elif isinstance(env, DummyVecEnv):
        base_env = env.envs[0]
    else:
        base_env = env

    # Print environment setpoint configuration
    print(f"\nEnvironment Configuration:")
    print(f"Setpoint Pattern: {base_env.setpoint_pattern}")
    print(f"Base Heating Setpoint: {base_env.initial_heating_setpoint}")
    print(f"Base Cooling Setpoint: {base_env.initial_cooling_setpoint}")
    if isinstance(env, VecNormalize):
        print("Using normalized environment for evaluation")

    # Evaluate agent
    print(f"\nEvaluating agent deterministically for {num_episodes} episodes...")

    performance = evaluate_agent(
        env=env,
        model=model,
        num_episodes=num_episodes,
        render=render,
        verbose=verbose,
    )

    # Get algorithm name from model path
    algorithm = "unknown"
    for alg in ["ppo", "a2c", "dqn", "sac"]:
        if alg in model_path.lower():
            algorithm = alg.upper()
            break

    # Save performance results with proper JSON serialization
    results_path = os.path.join(output_dir, f"{algorithm}_deterministic_results.json")

    def convert_to_serializable(obj):
        """Recursively convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
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

    # Save environment results
    base_env.save_results(
        os.path.join(output_dir, f"{algorithm}_deterministic_env_results.json"),
        controller_name=f"{algorithm} Agent (deterministic)",
    )

    # Visualize performance
    visualize_performance(performance, output_dir, agent_name=f"{algorithm} Agent")

    print(f"\nEvaluation completed. Results saved to {output_dir}")
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
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL agent on the dollhouse environment with control stability metric"
    )

    # Required arguments
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file for training SINDy model",
    )

    # Optional arguments
    parser.add_argument(
        "--env-params",
        type=str,
        default=None,
        help="Path to the saved environment parameters",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during evaluation"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Directory to save evaluation results"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed logs")

    args = parser.parse_args()

    main(
        model_path=args.model,
        data_file=args.data,
        env_params_path=args.env_params,
        num_episodes=args.episodes,
        render=args.render,
        output_dir=args.output,
        verbose=not args.quiet,
    )

# Example usage:
# For normalized model:
# python evaluate_rl_agent.py \
#   --model "results/ppo_20250523_normalized/logs/models/ppo_final_model" \
#   --data "../Data/dollhouse-data-2025-03-24.csv" \
#   --episodes 5

# For non-normalized model:
# python evaluate_rl_agent.py \
#   --model "results/ppo_20250523_regular/logs/models/ppo_final_model" \
#   --data "../Data/dollhouse-data-2025-03-24.csv" \
#   --episodes 5
