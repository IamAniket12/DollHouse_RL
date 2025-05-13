import numpy as np
import os
import time
from datetime import datetime
import argparse
import json

# Import Stable Baselines
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# Import our modules
from train_sindy_model import train_sindy_model
from dollhouse_env import DollhouseThermalEnv


def train_rl_agent(
    env, algorithm="ppo", total_timesteps=5000000, seed=0, log_dir="logs"
):
    """
    Train an RL agent on the dollhouse environment.

    Args:
        env: The environment to train on
        algorithm: Algorithm to use ('ppo', 'a2c', 'dqn', or 'sac')
        total_timesteps: Total number of timesteps to train for
        seed: Random seed
        log_dir: Directory for tensorboard logs

    Returns:
        model: Trained model and model path
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(total_timesteps // 10, 1000),
        save_path=models_dir,
        name_prefix=algorithm,
    )

    # Create the algorithm
    if algorithm.lower() == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            learning_rate=3e-4,
        )
    elif algorithm.lower() == "a2c":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            learning_rate=7e-4,
        )
    elif algorithm.lower() == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            learning_rate=1e-4,
            buffer_size=50000,
        )
    elif algorithm.lower() == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            learning_rate=3e-4,
        )
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Choose from: 'ppo', 'a2c', 'dqn', 'sac'"
        )

    # Train the model
    print(f"\nTraining {algorithm.upper()} for {total_timesteps} timesteps...")
    start_time = time.time()

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Save the final model
    final_model_path = os.path.join(models_dir, f"{algorithm}_final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    return model, final_model_path


def setup_training(
    data_file,
    output_dir=None,
    algorithm="ppo",
    total_timesteps=5000000,
    reward_type="balanced",
    energy_weight=0.5,
    comfort_weight=1.0,
    seed=0,
):
    """
    Set up and train an RL agent.

    Args:
        data_file: Path to data file for training SINDy model
        output_dir: Directory to save results (default: auto-generated)
        algorithm: RL algorithm to use ('ppo', 'a2c', 'dqn', or 'sac')
        total_timesteps: Total timesteps for training
        reward_type: Type of reward function ('comfort', 'energy', or 'balanced')
        energy_weight: Weight for energy penalty in reward
        comfort_weight: Weight for comfort penalty in reward
        seed: Random seed for reproducibility

    Returns:
        tuple: (model, model_path)
    """
    # Set output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/{algorithm}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")

    # Train SINDy model
    print(f"Training SINDy model on {data_file}...")
    sindy_model = train_sindy_model(file_path=data_file)

    # Create environment
    env_params = {
        "sindy_model": sindy_model,
        "episode_length": 1000,  # 24 hours with 30-second timesteps
        "time_step_seconds": 30,
        "heating_setpoint": 26.0,
        "cooling_setpoint": 28.0,
        "external_temp_pattern": "fixed",
        "setpoint_pattern": "fixed",
        "reward_type": reward_type,
        "energy_weight": energy_weight,
        "comfort_weight": comfort_weight,
    }

    # Create base environment
    env = DollhouseThermalEnv(**env_params)

    # Save environment parameters
    with open(os.path.join(output_dir, "env_params.json"), "w") as f:
        # Convert non-serializable parameters to strings
        serializable_params = env_params.copy()
        serializable_params["sindy_model"] = "SINDy model object (not serializable)"
        json.dump(serializable_params, f, indent=4)

    # Train RL agent
    print(f"\nTraining {algorithm.upper()} agent...")
    model, model_path = train_rl_agent(
        env=env,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        seed=seed,
        log_dir=os.path.join(output_dir, "logs"),
    )

    print(f"\nTraining completed. Model saved to {model_path}")

    # Record training configuration
    config = {
        "algorithm": algorithm,
        "total_timesteps": total_timesteps,
        "reward_type": reward_type,
        "energy_weight": energy_weight,
        "comfort_weight": comfort_weight,
        "model_path": model_path,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
    }

    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    return model, model_path


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train RL agent for dollhouse thermal control"
    )

    # Required arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file for training SINDy model",
    )

    # Training parameters
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "a2c", "dqn", "sac"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--timesteps", type=int, default=5000000, help="Total timesteps for training"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )

    # Environment parameters
    parser.add_argument(
        "--reward",
        type=str,
        default="balanced",
        choices=["comfort", "energy", "balanced"],
        help="Reward function type",
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=0.1,
        help="Weight for energy penalty in reward",
    )
    parser.add_argument(
        "--comfort-weight",
        type=float,
        default=1.0,
        help="Weight for comfort penalty in reward",
    )

    # Output
    parser.add_argument(
        "--output", type=str, default=None, help="Directory to save results"
    )

    args = parser.parse_args()

    # Train RL agent
    model, model_path = setup_training(
        data_file=args.data,
        output_dir=args.output,
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        reward_type=args.reward,
        energy_weight=args.energy_weight,
        comfort_weight=args.comfort_weight,
        seed=args.seed,
    )

    print(f"Training complete!")
    print(f"Model saved to: {model_path}")
    print(f"Use this path when evaluating the model.")

# Example usage:
# python train_rl_agent.py --data "../Data/dollhouse-data-2025-03-24.csv" --algorithm ppo --timesteps 100000 --reward balanced
