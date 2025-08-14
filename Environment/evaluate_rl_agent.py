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
    Now handles custom observations properly.

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
    env_params_copy.pop("observation_info", None)  # Remove observation info

    # Print observation configuration if available
    if "custom_observations" in env_params_copy:
        custom_obs = env_params_copy["custom_observations"]
        if custom_obs:
            print(f"Using custom observations: {custom_obs}")
        else:
            print("Using default observations")

    # Create environment with saved parameters
    try:
        env = DollhouseThermalEnv(**env_params_copy)

        # Print observation space information
        obs_info = env.get_observation_info()
        print(f"Observation space shape: {obs_info['observation_space_shape']}")
        print(f"Observations: {obs_info['observation_list']}")

    except Exception as e:
        print(f"Error creating environment with saved parameters: {e}")
        print("Falling back to environment without custom observation parameters...")

        # Remove custom observation parameters and try again
        fallback_params = env_params_copy.copy()
        fallback_params.pop("custom_observations", None)

        env = DollhouseThermalEnv(**fallback_params)
        print("Using default observations (fallback)")

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


def get_observation_values_for_logging(env, obs):
    """
    Extract meaningful observation values for logging, handling custom observations.

    Args:
        env: Environment (possibly wrapped)
        obs: Current observation vector

    Returns:
        dict: Dictionary with meaningful observation values
    """
    # Get the base environment
    if isinstance(env, VecNormalize):
        base_env = env.venv.envs[0]
    elif isinstance(env, DummyVecEnv):
        base_env = env.envs[0]
    else:
        base_env = env

    # Get observation info
    obs_info = base_env.get_observation_info()
    obs_list = obs_info["observation_list"]

    # Create dictionary mapping observation names to values
    obs_dict = {}
    for i, obs_name in enumerate(obs_list):
        if i < len(obs):
            obs_dict[obs_name] = obs[i]

    # Try to get temperatures from observations first, then fall back to environment state
    ground_temp = obs_dict.get("ground_temp", None)
    top_temp = obs_dict.get("top_temp", None)
    external_temp = obs_dict.get("external_temp", None)

    # If temperatures not directly in observations, get from environment state
    if ground_temp is None:
        ground_temp = base_env.ground_temp
    if top_temp is None:
        top_temp = base_env.top_temp
    if external_temp is None:
        current_step = min(
            base_env.current_step, len(base_env.external_temperatures) - 1
        )
        external_temp = base_env.external_temperatures[current_step]

    # For backward compatibility, ensure we have the basic values
    result = {
        "ground_temp": ground_temp,
        "top_temp": top_temp,
        "external_temp": external_temp,
        "heating_setpoint": obs_dict.get("heating_setpoint", base_env.heating_setpoint),
        "cooling_setpoint": obs_dict.get("cooling_setpoint", base_env.cooling_setpoint),
    }

    return result


def evaluate_agent(env, model, num_episodes=5, render=False, verbose=True):
    """
    Evaluate a trained agent on the environment using deterministic actions.
    Now handles custom observations properly.

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

            # Get observation values for logging
            if isinstance(env, VecNormalize):
                # Get unnormalized observation for display
                original_obs = (
                    env.get_original_obs()[0]
                    if hasattr(env, "get_original_obs")
                    else obs
                )
            else:
                original_obs = obs

            obs_values = get_observation_values_for_logging(env, original_obs)

            # Select action using model
            if is_vec_env:
                # For vectorized environments, we need to add batch dimension
                action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
                action = action[0]  # Remove batch dimension
            else:
                action, _ = model.predict(obs, deterministic=True)

            if verbose and len(temps) % 100 == 0:
                # Format temperature values safely
                ground_temp_str = (
                    f"{obs_values['ground_temp']:.1f}"
                    if obs_values["ground_temp"] != "N/A"
                    else "N/A"
                )
                top_temp_str = (
                    f"{obs_values['top_temp']:.1f}"
                    if obs_values["top_temp"] != "N/A"
                    else "N/A"
                )

                print(
                    f"Step {len(temps)}: Action: {action}, "
                    f"Temps: {ground_temp_str}/{top_temp_str}°C, "
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

            # Record data - extract meaningful values for logging
            if isinstance(env, VecNormalize):
                # Get unnormalized observation for recording
                original_obs = (
                    env.get_original_obs()[0]
                    if hasattr(env, "get_original_obs")
                    else obs
                )
            else:
                original_obs = obs

            obs_values = get_observation_values_for_logging(env, original_obs)

            # Extract temperature values (these should now always be numeric)
            ground_temp = obs_values["ground_temp"]
            top_temp = obs_values["top_temp"]
            external_temp = obs_values["external_temp"]

            temps.append([ground_temp, top_temp])
            ext_temps.append(external_temp)
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

    # Add observation information
    obs_info = base_env.get_observation_info()
    performance["observation_space_shape"] = obs_info["observation_space_shape"]
    performance["observations_used"] = obs_info["observation_list"]

    return performance


def visualize_performance(performance, output_dir, agent_name="RL Agent"):
    """
    Create visualizations of agent performance with control stability metric.
    Updated to handle custom observations.

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
    fig = plt.figure(figsize=(15, 20))

    # Temperature plot with dynamic setpoints
    plt.subplot(6, 1, 1)
    ground_temps = [temp[0] for temp in episode_temperatures[0]]
    top_temps = [temp[1] for temp in episode_temperatures[0]]
    plt.plot(ground_temps, label="Ground Floor Temperature", linewidth=2, color="blue")
    plt.plot(top_temps, label="Top Floor Temperature", linewidth=2, color="red")

    if has_dynamic_setpoints:
        heating_setpoints = [sp[0] for sp in episode_setpoints[0]]
        cooling_setpoints = [sp[1] for sp in episode_setpoints[0]]
        plt.plot(
            heating_setpoints, "r--", label="Heating Setpoint", linewidth=1.5, alpha=0.8
        )
        plt.plot(
            cooling_setpoints, "b--", label="Cooling Setpoint", linewidth=1.5, alpha=0.8
        )
    else:
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

    # Add normalization and observation info to title
    normalization_status = (
        " (Normalized)" if performance.get("was_normalized", False) else ""
    )
    obs_count = (
        performance.get("observation_space_shape", [0])[0]
        if performance.get("observation_space_shape")
        else "Unknown"
    )
    plt.title(
        f"{agent_name} - Temperatures (Episode 1){normalization_status} - {obs_count} observations",
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

    plt.title(f"{agent_name} - External Temperature (Episode 1)")
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
        f"{agent_name} - Actions (Episode 1) - Control Stability: {episode_control_stability:.3f}"
    )
    plt.ylabel("Action State")
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ["OFF/CLOSED", "ON/OPEN"])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Rewards plot
    plt.subplot(6, 1, 5)
    plt.plot(episode_rewards[0], label="Step Reward", linewidth=2, color="darkgreen")
    plt.title(f"{agent_name} - Rewards (Episode 1)")
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
    plt.title(f"{agent_name} - Comfort Zone Violations (Episode 1)")
    plt.xlabel("Timestep")
    plt.ylabel("Temperature Violation (°C)")
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
    plt.close()

    # Summary metrics plot
    plt.figure(figsize=(20, 8))
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
        bar = plt.bar(
            [agent_name], [value], color=color, edgecolor="black", linewidth=1
        )
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

    bar = plt.bar(
        [agent_name], [obs_count], color="lightblue", edgecolor="black", linewidth=1
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

    plt.suptitle(f"{agent_name} - Performance Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'{agent_name.lower().replace(" ", "_")}_summary.png'),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Control stability detailed analysis
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
    plt.title(f"{agent_name} - State Changes by Action (Episode 1)")
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
    plt.title(f"{agent_name} - Control Stability Distribution")
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
    plt.title(f"{agent_name} - Action Duty Cycles")
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

    # Correlation between rewards and control stability
    plt.subplot(2, 3, 5)
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
    plt.title(f"{agent_name} - Stability vs Reward")
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
    plt.title(f"{agent_name} - Energy Usage per Episode")
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

    plt.suptitle(f"{agent_name} - Control Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir, f'{agent_name.lower().replace(" ", "_")}_control_analysis.png'
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Temperature distribution analysis
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

    plt.title(f"{agent_name} - Ground Floor Temperature Distribution")
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

    plt.title(f"{agent_name} - Top Floor Temperature Distribution")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Temperature correlation plot
    plt.subplot(2, 3, 3)
    plt.scatter(all_ground_temps, all_top_temps, alpha=0.5, s=20, color="purple")
    plt.xlabel("Ground Floor Temperature (°C)")
    plt.ylabel("Top Floor Temperature (°C)")
    plt.title(f"{agent_name} - Floor Temperature Correlation")
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
    plt.title(f"{agent_name} - External vs Internal Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Comfort zone performance over time
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
    plt.bar(
        episodes,
        comfort_performance,
        color="lightgreen",
        edgecolor="black",
        linewidth=1,
    )
    plt.title(f"{agent_name} - Comfort Performance by Episode")
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

    # Reward components analysis (if available)
    plt.subplot(2, 3, 6)
    if len(episode_rewards) > 0:
        all_rewards = []
        for episode_reward_list in episode_rewards:
            all_rewards.extend(episode_reward_list)

        # Create reward distribution
        plt.hist(all_rewards, bins=30, alpha=0.7, edgecolor="black", color="gold")
        plt.title(f"{agent_name} - Step Reward Distribution")
        plt.xlabel("Step Reward")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # Add statistics
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        plt.axvline(
            x=mean_reward,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_reward:.3f}",
        )
        plt.text(
            0.05,
            0.95,
            f"Std: {std_reward:.3f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        plt.legend()

    plt.suptitle(f"{agent_name} - Temperature Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f'{agent_name.lower().replace(" ", "_")}_temperature_analysis.png',
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Comprehensive visualizations saved to {output_dir}")


def create_evaluation_report(performance, output_dir, agent_name="RL Agent"):
    """
    Create a comprehensive evaluation report in text format.

    Args:
        performance: Performance dictionary from evaluate_agent
        output_dir: Directory to save the report
        agent_name: Name of the agent
    """
    report_path = os.path.join(
        output_dir, f"{agent_name.lower().replace(' ', '_')}_evaluation_report.txt"
    )

    with open(report_path, "w") as f:
        f.write(f"{'='*80}\n")
        f.write(f"{agent_name.upper()} - COMPREHENSIVE EVALUATION REPORT\n")
        f.write(f"{'='*80}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Environment Configuration
        f.write("ENVIRONMENT CONFIGURATION\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"Observation Space Shape: {performance.get('observation_space_shape', 'N/A')}\n"
        )
        f.write(f"Observations Used: {performance.get('observations_used', 'N/A')}\n")
        f.write(
            f"Normalization Used: {'Yes' if performance.get('was_normalized', False) else 'No'}\n"
        )
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
    Updated to handle custom observations and provide comprehensive analysis.

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
    print(f"Results will be saved to: {output_dir}")

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
    print(f"Loading model from {model_path}")
    model = load_model(model_path)

    # Recreate environment (with normalization if applicable)
    print("Recreating environment...")
    env = recreate_environment(env_params_path, data_file, model_dir)

    # Get base environment for printing info
    if isinstance(env, VecNormalize):
        base_env = env.venv.envs[0]
    elif isinstance(env, DummyVecEnv):
        base_env = env.envs[0]
    else:
        base_env = env

    # Print environment configuration
    print(f"\nEnvironment Configuration:")
    print(f"Setpoint Pattern: {base_env.setpoint_pattern}")
    print(f"Base Heating Setpoint: {base_env.initial_heating_setpoint}°C")
    print(f"Base Cooling Setpoint: {base_env.initial_cooling_setpoint}°C")

    # Print observation configuration
    obs_info = base_env.get_observation_info()
    print(f"Observation Space Shape: {obs_info['observation_space_shape']}")
    print(f"Observations Used: {obs_info['observation_list']}")

    if isinstance(env, VecNormalize):
        print("Using normalized environment for evaluation")

    # Evaluate agent
    print(f"\nEvaluating agent deterministically for {num_episodes} episodes...")
    start_time = time.time()

    performance = evaluate_agent(
        env=env,
        model=model,
        num_episodes=num_episodes,
        render=render,
        verbose=verbose,
    )

    evaluation_time = time.time() - start_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")

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

    # Create comprehensive visualizations
    print("Generating visualizations...")
    visualize_performance(performance, output_dir, agent_name=f"{algorithm} Agent")

    # Create evaluation report
    print("Generating evaluation report...")
    create_evaluation_report(performance, output_dir, agent_name=f"{algorithm} Agent")

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Algorithm: {algorithm}")
    print(f"Episodes Evaluated: {num_episodes}")
    print(f"Evaluation Time: {evaluation_time:.2f} seconds")
    print(f"\nObservation Configuration:")
    print(f"  Observation Space Shape: {performance['observation_space_shape']}")
    print(f"  Observations Used: {performance['observations_used']}")
    print(f"  Normalization: {'Yes' if performance['was_normalized'] else 'No'}")
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
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL agent on the dollhouse environment with comprehensive analysis and custom observations support"
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


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

"""
# Basic evaluation with comprehensive analysis:
python evaluate_rl_agent.py \
  --model "results/ppo_20250523/logs/models/ppo_final_model" \
  --data "data/dollhouse-data.csv"

# Evaluate model trained with custom observations:
python evaluate_rl_agent.py \
  --model "results/ppo_compact_obs/logs/models/ppo_final_model" \
  --data "data/dollhouse-data.csv" \
  --episodes 10 \
  --output "detailed_evaluation"

# Evaluate normalized model with specific parameters:
python evaluate_rl_agent.py \
  --model "results/ppo_normalized/logs/models/ppo_final_model" \
  --data "data/dollhouse-data.csv" \
  --env-params "results/ppo_normalized/env_params.json" \
  --episodes 8

# Quick evaluation with minimal output:
python evaluate_rl_agent.py \
  --model "results/sac_minimal_obs/logs/models/sac_final_model" \
  --data "data/dollhouse-data.csv" \
  --episodes 3 \
  --quiet

# Evaluation with rendering (for visualization):
python evaluate_rl_agent.py \
  --model "results/ppo_extended_obs/logs/models/ppo_final_model" \
  --data "data/dollhouse-data.csv" \
  --episodes 2 \
  --render

# The script automatically:
# - Detects custom observation configuration
# - Applies normalization if used during training
# - Generates comprehensive visualizations and analysis
# - Creates detailed performance reports
# - Provides actionable insights about agent behavior
"""
