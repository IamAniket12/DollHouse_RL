import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import argparse
import json
import torch
import sys

# Import Stable Baselines
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append("../Environment")
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


def load_rl_model(model_path, vec_normalize_path=None, base_env=None):
    """
    Load a trained RL model and optional normalization parameters.

    Args:
        model_path: Path to the trained model
        vec_normalize_path: Path to VecNormalize parameters (optional)
        base_env: Base environment to wrap with VecNormalize (required if vec_normalize_path provided)

    Returns:
        tuple: (model, vec_env) where vec_env is None if no normalization or properly wrapped environment
    """
    # Determine algorithm from model path
    model_name = os.path.basename(model_path).lower()

    if "ppo" in model_name:
        model = PPO.load(model_path)
        print(f"Loaded PPO model from {model_path}")
    elif "a2c" in model_name:
        model = A2C.load(model_path)
        print(f"Loaded A2C model from {model_path}")
    elif "dqn" in model_name:
        model = DQN.load(model_path)
        print(f"Loaded DQN model from {model_path}")
    elif "sac" in model_name:
        model = SAC.load(model_path)
        print(f"Loaded SAC model from {model_path}")
    else:
        # Try PPO as default
        try:
            model = PPO.load(model_path)
            print(f"Loaded model as PPO from {model_path}")
        except:
            raise ValueError(
                f"Could not determine algorithm from model path: {model_path}"
            )

    # Load normalization if available
    vec_env = None
    if (
        vec_normalize_path
        and os.path.exists(vec_normalize_path)
        and base_env is not None
    ):
        print(f"Loading normalization parameters from {vec_normalize_path}")

        # Wrap base environment in DummyVecEnv first
        vec_env = DummyVecEnv([lambda: base_env])

        # Load and apply normalization
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)

        # Set to evaluation mode (don't update statistics)
        vec_env.training = False
        vec_env.norm_reward = False  # Don't normalize rewards during evaluation

        print(f"Applied normalization wrapper")
    elif vec_normalize_path and os.path.exists(vec_normalize_path):
        print(f"Warning: VecNormalize file found but no base environment provided")

    return model, vec_env


def run_episode_and_log(
    env,
    controller_type="rule_based",
    model_path=None,
    vec_normalize_path=None,
    hysteresis=0.5,
    output_dir="episode_logs",
    episode_name="episode_1",
    render=False,
):
    """
    Run a single episode and log all observations and actions to CSV.

    Args:
        env: The environment to run the episode in
        controller_type: "rule_based" or "rl_model"
        model_path: Path to trained RL model (if using RL controller)
        vec_normalize_path: Path to VecNormalize parameters (optional)
        hysteresis: Hysteresis for rule-based controller
        output_dir: Directory to save logs
        episode_name: Name for this episode's files
        render: Whether to render the environment

    Returns:
        dict: Episode metadata including total reward
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get original environment for accessing internal state
    orig_env = env.unwrapped if hasattr(env, "unwrapped") else env

    # Initialize controller
    if controller_type == "rule_based":
        controller = create_rule_based_controller(hysteresis=hysteresis)
        model = None
        vec_env = None
        print(f"Using rule-based controller with hysteresis={hysteresis}")

    elif controller_type == "rl_model":
        if model_path is None:
            raise ValueError("model_path must be provided for RL controller")

        # Load model and normalization (if available)
        model, vec_env = load_rl_model(model_path, vec_normalize_path, orig_env)
        controller = None
        print(f"Using RL model: {model_path}")

        # If we have normalization, we need to use the wrapped environment
        if vec_env is not None:
            print("Using normalized environment for RL model")
            env_for_episode = vec_env
        else:
            env_for_episode = env
    else:
        raise ValueError("controller_type must be 'rule_based' or 'rl_model'")

    # Use the appropriate environment for the episode
    if controller_type == "rl_model" and vec_env is not None:
        episode_env = vec_env
        is_vec_env = True
    else:
        episode_env = env
        is_vec_env = False

    # Reset environment
    if is_vec_env:
        obs = episode_env.reset()
        obs = obs[0] if len(obs.shape) > 1 else obs  # Handle vectorized obs
    else:
        obs, info = episode_env.reset()

    terminated = False
    truncated = False
    step = 0
    total_reward = 0

    # Storage for episode data
    episode_data = []

    while not terminated and not truncated:
        # Get original observation for logging (before normalization)
        if is_vec_env and hasattr(episode_env, "get_original_obs"):
            # For VecNormalize, get unnormalized observation
            original_obs = episode_env.get_original_obs()[0]
        else:
            original_obs = obs

        # Store current observation state
        current_obs = original_obs.copy()

        # Calculate hour of day for logging
        if len(current_obs) > 9:
            hour_of_day = current_obs[9]
        else:
            hour_of_day = (step * orig_env.time_step_seconds / 3600) % 24

        # Get action based on controller type
        if controller_type == "rule_based":
            action = controller(current_obs)

        else:  # RL model
            if is_vec_env:
                # For vectorized environments, we need to add batch dimension
                action, _states = model.predict(obs.reshape(1, -1), deterministic=True)
                action = action[0]  # Remove batch dimension
            else:
                action, _states = model.predict(obs, deterministic=True)

            # Ensure action is in correct format
            if isinstance(action, np.ndarray):
                action = action.flatten()

        # Take step in environment
        if is_vec_env:
            next_obs, reward, done, info = episode_env.step([action])
            next_obs = next_obs[0]
            reward = reward[0]
            done = done[0]
            info = info[0] if isinstance(info, list) else info

            # VecEnv uses 'done' instead of terminated/truncated
            terminated = done
            truncated = False
        else:
            next_obs, reward, terminated, truncated, info = episode_env.step(action)

        total_reward += reward

        # Log data for this step
        step_data = {
            # Step metadata
            "step": step,
            "hour_of_day": hour_of_day,
            "episode_start_time_offset": getattr(
                orig_env, "episode_start_time_offset", 0.0
            ),
            # Observations (state) - use original/unnormalized values
            "ground_temp": current_obs[0],
            "top_temp": current_obs[1],
            "external_temp": current_obs[2],
            "prev_ground_light": current_obs[3],
            "prev_ground_window": current_obs[4],
            "prev_top_light": current_obs[5],
            "prev_top_window": current_obs[6],
            "heating_setpoint": current_obs[7],
            "cooling_setpoint": current_obs[8],
            "time_step_in_episode": current_obs[10] if len(current_obs) > 10 else step,
            # Actions taken
            "action_ground_light": int(action[0]),
            "action_ground_window": int(action[1]),
            "action_top_light": int(action[2]),
            "action_top_window": int(action[3]),
            # Rewards and info
            "reward": reward,
            "ground_comfort_violation": info.get("ground_comfort_violation", 0),
            "top_comfort_violation": info.get("top_comfort_violation", 0),
            "energy_use": info.get("energy_use", action[0] + action[2]),
            # Derived features for analysis
            "avg_temp": (current_obs[0] + current_obs[1]) / 2,
            "temp_difference": current_obs[1] - current_obs[0],  # top - ground
            "avg_setpoint": (current_obs[7] + current_obs[8]) / 2,
            "setpoint_range": current_obs[8] - current_obs[7],
            "ground_temp_deviation": current_obs[0]
            - (current_obs[7] + current_obs[8]) / 2,
            "top_temp_deviation": current_obs[1]
            - (current_obs[7] + current_obs[8]) / 2,
            "external_temp_diff": current_obs[2]
            - (current_obs[7] + current_obs[8]) / 2,
            # Control patterns
            "lights_on": int(action[0]) + int(action[2]),
            "windows_open": int(action[1]) + int(action[3]),
            "ground_heating": int(action[0])
            * (1 - int(action[1])),  # light on, window closed
            "top_heating": int(action[2])
            * (1 - int(action[3])),  # light on, window closed
            "ground_cooling": (1 - int(action[0]))
            * int(action[1]),  # light off, window open
            "top_cooling": (1 - int(action[2]))
            * int(action[3]),  # light off, window open
        }

        episode_data.append(step_data)

        # Update for next iteration
        obs = next_obs
        step += 1

        if render and not is_vec_env:
            orig_env.render()

    # Convert to DataFrame
    df = pd.DataFrame(episode_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, f"{episode_name}_data.csv")
    df.to_csv(csv_path, index=False)

    # Create metadata
    metadata = {
        "episode_name": episode_name,
        "controller_type": controller_type,
        "model_path": model_path if model_path else None,
        "vec_normalize_path": vec_normalize_path if vec_normalize_path else None,
        "used_normalization": is_vec_env,
        "hysteresis": hysteresis if controller_type == "rule_based" else None,
        "total_steps": step,
        "total_reward": float(total_reward),
        "avg_reward_per_step": float(total_reward / step) if step > 0 else 0,
        "episode_length_hours": (step * orig_env.time_step_seconds) / 3600,
        "timestamp": datetime.now().isoformat(),
        # Environment configuration
        "env_config": {
            "episode_length": orig_env.episode_length,
            "time_step_seconds": orig_env.time_step_seconds,
            "initial_heating_setpoint": orig_env.initial_heating_setpoint,
            "initial_cooling_setpoint": orig_env.initial_cooling_setpoint,
            "external_temp_pattern": orig_env.external_temp_pattern,
            "setpoint_pattern": orig_env.setpoint_pattern,
            "reward_type": orig_env.reward_type,
            "energy_weight": orig_env.energy_weight,
            "comfort_weight": orig_env.comfort_weight,
            "random_start_time": orig_env.random_start_time,
            "use_reward_shaping": orig_env.use_reward_shaping,
        },
        # Performance summary
        "performance": {
            "avg_ground_temp": float(df["ground_temp"].mean()),
            "avg_top_temp": float(df["top_temp"].mean()),
            "avg_external_temp": float(df["external_temp"].mean()),
            "total_comfort_violations": float(
                df["ground_comfort_violation"].sum() + df["top_comfort_violation"].sum()
            ),
            "total_energy_use": float(df["energy_use"].sum()),
            "comfort_violation_rate": float(
                (
                    df["ground_comfort_violation"] + df["top_comfort_violation"] > 0
                ).mean()
            ),
            "avg_lights_on": float(df["lights_on"].mean()),
            "avg_windows_open": float(df["windows_open"].mean()),
        },
    }

    # Save metadata
    metadata_path = os.path.join(output_dir, f"{episode_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Episode completed!")
    print(f"  Total steps: {step}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward per step: {total_reward/step:.4f}")
    print(f"  Used normalization: {is_vec_env}")
    print(f"  Data saved to: {csv_path}")
    print(f"  Metadata saved to: {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Run episode with controller and log data"
    )

    # Required arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file for training SINDy model",
    )

    # Controller selection
    parser.add_argument(
        "--controller",
        type=str,
        choices=["rule_based", "rl_model"],
        default="rule_based",
        help="Type of controller to use",
    )

    # RL model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained RL model (required if using rl_model controller)",
    )
    parser.add_argument(
        "--vec-normalize-path",
        type=str,
        help="Path to VecNormalize parameters (optional, auto-detected if not provided)",
    )

    # Rule-based controller arguments
    parser.add_argument(
        "--hysteresis",
        type=float,
        default=0.5,
        help="Hysteresis parameter for rule-based controller",
    )

    # Environment arguments
    parser.add_argument(
        "--env-params", type=str, help="Path to saved environment parameters JSON file"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="episode_logs",
        help="Directory to save episode logs",
    )
    parser.add_argument(
        "--episode-name",
        type=str,
        default=None,
        help="Name for this episode's files (auto-generated if not provided)",
    )

    # Other options
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during episode"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.controller == "rl_model" and args.model_path is None:
        parser.error("--model-path is required when using rl_model controller")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Auto-generate episode name if not provided
    if args.episode_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.episode_name = f"{args.controller}_{timestamp}"

    # Auto-detect VecNormalize path if not provided
    if args.controller == "rl_model" and args.vec_normalize_path is None:
        # Try to find vec_normalize.pkl in the same directory as the model
        model_dir = os.path.dirname(args.model_path)
        potential_vec_path = os.path.join(model_dir, "vec_normalize.pkl")
        if os.path.exists(potential_vec_path):
            args.vec_normalize_path = potential_vec_path
            print(f"Auto-detected VecNormalize path: {potential_vec_path}")

    print(f"Starting episode logging with {args.controller} controller...")

    # Train SINDy model
    print(f"Training SINDy model on {args.data}...")
    start_time = time.time()
    sindy_model = train_sindy_model(file_path=args.data)
    training_time = time.time() - start_time
    print(f"SINDy model training completed in {training_time:.2f} seconds")

    # Create environment
    if args.env_params:
        # Load environment parameters from file
        with open(args.env_params, "r") as f:
            env_params = json.load(f)
        env_params["sindy_model"] = sindy_model
        env = DollhouseThermalEnv(**env_params)
        print(f"Environment created with parameters from {args.env_params}")
    else:
        # Use default parameters
        env_params = {
            "sindy_model": sindy_model,
            "episode_length": 2880,  # 24 hours
            "time_step_seconds": 30,
            "heating_setpoint": 26.0,
            "cooling_setpoint": 28.0,
            "external_temp_pattern": "sine",
            "setpoint_pattern": "schedule",
            "reward_type": "balanced",
            "energy_weight": 0.5,
            "comfort_weight": 1.0,
            "use_reward_shaping": True,
            "random_start_time": True,
            "random_seed": args.seed,
        }
        env = DollhouseThermalEnv(**env_params)
        print("Environment created with default parameters")

    # Run episode and log data
    metadata = run_episode_and_log(
        env=env,
        controller_type=args.controller,
        model_path=args.model_path,
        vec_normalize_path=args.vec_normalize_path,
        hysteresis=args.hysteresis,
        output_dir=args.output_dir,
        episode_name=args.episode_name,
        render=args.render,
    )

    print(f"\nEpisode logging completed successfully!")
    print(f"Output directory: {args.output_dir}")
    print(f"Episode name: {args.episode_name}")


if __name__ == "__main__":
    main()


# Example usage:
#
# Rule-based controller:
# python episode_logger.py --data "../Data/dollhouse-data-2025-03-24.csv" --controller rule_based --hysteresis 0.5
#
# RL model controller:
# python episode_logger.py --data "../Data/dollhouse-data-2025-03-24.csv" --controller rl_model --model-path "results/ppo_20250513_151705/logs/models/ppo_final_model.zip"
#
# With custom environment parameters:
# python episode_logger.py --data "../Data/dollhouse-data-2025-03-24.csv" --controller rl_model --model-path "path/to/model.zip" --env-params "path/to/env_params.json"
