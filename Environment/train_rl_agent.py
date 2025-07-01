import numpy as np
import os
import time
from datetime import datetime
import argparse
import json
import torch

# Import Stable Baselines
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

# Import WandB
import wandb
from wandb.integration.sb3 import WandbCallback

# Import our modules
from train_sindy_model import train_sindy_model
from dollhouse_env import DollhouseThermalEnv


class CustomWandbCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to WandB.
    """

    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_comfort_violations = []

    def _on_step(self) -> bool:
        # Log custom metrics from info dict if available
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    # Episode ended
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])

                    # Log episode metrics
                    wandb.log(
                        {
                            "episode/reward": info["episode"]["r"],
                            "episode/length": info["episode"]["l"],
                            "episode/reward_mean": (
                                np.mean(self.episode_rewards[-100:])
                                if len(self.episode_rewards) > 0
                                else 0
                            ),
                        },
                        step=self.num_timesteps,
                    )

                # Log environment-specific metrics
                if "ground_comfort_violation" in info:
                    wandb.log(
                        {
                            "env/ground_comfort_violation": info[
                                "ground_comfort_violation"
                            ],
                            "env/top_comfort_violation": info["top_comfort_violation"],
                            "env/energy_use": info["energy_use"],
                            "env/ground_temp": info["ground_temp"],
                            "env/top_temp": info["top_temp"],
                            "env/external_temp": info["external_temp"],
                        },
                        step=self.num_timesteps,
                    )

        return True


def check_gpu_availability():
    """
    Check if GPU is available and return the appropriate device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU available! Using {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU for training.")

    return device


def make_env(rank, seed, sindy_model, env_params, monitor_dir):
    """
    Create a function that returns a single environment instance.

    Args:
        rank: Unique identifier for the environment
        seed: Random seed
        sindy_model: Trained SINDy model
        env_params: Environment parameters
        monitor_dir: Directory for monitor logs

    Returns:
        A function that creates the environment
    """

    def _init():
        # Create a copy of env_params and update the seed
        local_params = env_params.copy()
        local_params["random_seed"] = seed + rank
        local_params["sindy_model"] = sindy_model

        # Create environment
        env = DollhouseThermalEnv(**local_params)

        # Wrap with Monitor for logging
        env = Monitor(env, os.path.join(monitor_dir, f"env_{rank}"))

        return env

    return _init


def create_vectorized_env(
    sindy_model,
    env_params,
    n_envs,
    seed,
    monitor_dir,
    vec_env_type="subproc",
    normalize=True,
):
    """
    Create vectorized environments for parallel training with optional normalization.

    Args:
        sindy_model: Trained SINDy model
        env_params: Environment parameters
        n_envs: Number of parallel environments
        seed: Base random seed
        monitor_dir: Directory for monitor logs
        vec_env_type: Type of vectorized environment ("dummy" or "subproc")
        normalize: Whether to apply observation and reward normalization

    Returns:
        Vectorized environment (possibly normalized)
    """
    os.makedirs(monitor_dir, exist_ok=True)

    # Create list of environment creation functions
    env_fns = [
        make_env(i, seed, sindy_model, env_params, monitor_dir) for i in range(n_envs)
    ]

    # Create vectorized environment
    if vec_env_type == "subproc":
        # SubprocVecEnv runs each environment in a separate process
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
        print(f"Created SubprocVecEnv with {n_envs} parallel environments")
    else:
        # DummyVecEnv runs environments sequentially in the main process
        vec_env = DummyVecEnv(env_fns)
        print(f"Created DummyVecEnv with {n_envs} environments")

    # Apply normalization if requested
    if normalize:
        # VecNormalize automatically normalizes observations and rewards
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,  # Normalize observations
            norm_reward=True,  # Normalize rewards
            clip_obs=10.0,  # Clip observations to [-10, 10] after normalization
            clip_reward=10.0,  # Clip rewards to [-10, 10]
            gamma=0.99,  # Discount factor for reward normalization
            epsilon=1e-8,  # Small value to avoid division by zero
        )
        print("Applied observation and reward normalization")

    return vec_env


def train_rl_agent(
    vec_env,
    n_envs,
    algorithm="ppo",
    total_timesteps=5000000,
    seed=0,
    log_dir="logs",
    wandb_project="dollhouse-thermal-control",
    wandb_entity=None,
    use_wandb=True,
):
    """
    Train an RL agent on the dollhouse environment with WandB logging and GPU support.

    Args:
        vec_env: Vectorized environment (possibly normalized)
        n_envs: Number of parallel environments
        algorithm: Algorithm to use ('ppo', 'a2c', 'dqn', or 'sac')
        total_timesteps: Total number of timesteps to train for
        seed: Random seed
        log_dir: Directory for tensorboard logs
        wandb_project: WandB project name
        wandb_entity: WandB entity (username or team)
        use_wandb: Whether to use WandB logging

    Returns:
        model: Trained model and model path
    """
    # Check for GPU
    device = check_gpu_availability()

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Initialize WandB if requested
    if use_wandb:
        run_name = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "algorithm": algorithm,
                "total_timesteps": total_timesteps,
                "seed": seed,
                "device": str(device),
                "n_envs": n_envs,
                "normalized": isinstance(vec_env, VecNormalize),
            },
            sync_tensorboard=True,  # Sync tensorboard metrics to wandb
        )

    # Set up callbacks
    callbacks = []
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=200000,  # Less frequent saves
        save_path=models_dir,
        name_prefix=algorithm,
    )
    callbacks.append(checkpoint_callback)

    # WandB callbacks
    if use_wandb:
        # Standard WandB callback for SB3
        wandb_callback = WandbCallback(
            gradient_save_freq=100000,
            model_save_path=f"{models_dir}/wandb",
            verbose=2,
        )
        callbacks.append(wandb_callback)

        # Custom callback for additional metrics
        custom_callback = CustomWandbCallback()
        callbacks.append(custom_callback)

    # Adjust batch sizes and steps for vectorized environments
    if algorithm.lower() == "ppo":
        # PPO works well with vectorized environments
        # n_steps * n_envs should be > batch_size
        n_steps = max(128, 1024 // n_envs)  # Keep total batch size constant
        batch_size = 128

        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device,
        )
        print(
            f"PPO configured with n_steps={n_steps}, batch_size={batch_size} for {n_envs} environments"
        )

    elif algorithm.lower() == "a2c":
        # A2C also benefits from vectorized environments
        n_steps = 256  # Adjust n_steps

        model = A2C(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            learning_rate=7e-4,
            n_steps=n_steps,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device,
        )
        print(f"A2C configured with n_steps={n_steps} for {n_envs} environments")

    elif algorithm.lower() == "dqn":
        # DQN doesn't benefit as much from vectorized environments
        # but can still use them
        model = DQN(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            device=device,
        )

    elif algorithm.lower() == "sac":
        # SAC can use vectorized environments
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            learning_rate=3e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_update_interval=1,
            target_entropy="auto",
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Choose from: 'ppo', 'a2c', 'dqn', 'sac'"
        )

    # Log model architecture to WandB
    if use_wandb:
        wandb.config.update(
            {
                "algorithm_params": {
                    "learning_rate": model.learning_rate,
                    "gamma": model.gamma,
                    "n_envs": n_envs,
                },
            }
        )

    # Train the model
    print(f"\nTraining {algorithm.upper()} for {total_timesteps} timesteps...")
    print(f"Using device: {device}")
    print(f"Training with {n_envs} parallel environments")
    if isinstance(vec_env, VecNormalize):
        print("With observation and reward normalization")
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,  # Show progress bar
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Average timesteps per second: {total_timesteps / training_time:.2f}")

    # Save the final model
    final_model_path = os.path.join(models_dir, f"{algorithm}_final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # If using normalization, save the normalization statistics
    if isinstance(vec_env, VecNormalize):
        vec_normalize_path = os.path.join(models_dir, "vec_normalize.pkl")
        vec_env.save(vec_normalize_path)
        print(f"Normalization statistics saved to {vec_normalize_path}")

    # Log final model to WandB
    if use_wandb:
        wandb.save(f"{final_model_path}.zip")
        if isinstance(vec_env, VecNormalize):
            wandb.save(vec_normalize_path)
        wandb.config.update(
            {
                "training_time_seconds": training_time,
                "final_model_path": final_model_path,
                "timesteps_per_second": total_timesteps / training_time,
            }
        )
        wandb.finish()

    return model, final_model_path


def setup_training(
    data_file,
    output_dir=None,
    algorithm="ppo",
    total_timesteps=5000000,
    reward_type="balanced",
    energy_weight=1.0,
    comfort_weight=1.0,
    seed=0,
    wandb_project="dollhouse-thermal-control",
    wandb_entity=None,
    use_wandb=True,
    n_envs=4,
    vec_env_type="subproc",
    normalize=True,
):
    """
    Set up and train an RL agent with WandB logging and GPU support.

    Args:
        data_file: Path to data file for training SINDy model
        output_dir: Directory to save results (default: auto-generated)
        algorithm: RL algorithm to use ('ppo', 'a2c', 'dqn', or 'sac')
        total_timesteps: Total timesteps for training
        reward_type: Type of reward function ('comfort', 'energy', or 'balanced')
        energy_weight: Weight for energy penalty in reward
        comfort_weight: Weight for comfort penalty in reward
        seed: Random seed for reproducibility
        wandb_project: WandB project name
        wandb_entity: WandB entity (username or team)
        use_wandb: Whether to use WandB logging
        n_envs: Number of parallel environments
        vec_env_type: Type of vectorized environment ("dummy" or "subproc")
        normalize: Whether to apply observation and reward normalization

    Returns:
        tuple: (model, model_path)
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Set output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/{algorithm}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")

    # Train SINDy model
    print(f"Training SINDy model on {data_file}...")
    sindy_model = train_sindy_model(file_path=data_file)

    # Create environment parameters
    env_params = {
        "episode_length": 2880,  # 24 hours
        "time_step_seconds": 30,
        "heating_setpoint": 26.0,
        "cooling_setpoint": 28.0,
        "external_temp_pattern": "sine",
        "setpoint_pattern": "schedule",
        "reward_type": reward_type,
        "energy_weight": energy_weight,
        "comfort_weight": comfort_weight,
        "use_reward_shaping": True,
        "random_start_time": True,
        "shaping_weight": 0.3,
    }

    # Create vectorized environment with optional normalization
    monitor_dir = os.path.join(output_dir, "monitor")
    vec_env = create_vectorized_env(
        sindy_model=sindy_model,
        env_params=env_params,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=monitor_dir,
        vec_env_type=vec_env_type,
        normalize=normalize,
    )

    # Save environment parameters
    with open(os.path.join(output_dir, "env_params.json"), "w") as f:
        # Convert non-serializable parameters to strings
        serializable_params = env_params.copy()
        serializable_params["sindy_model"] = "SINDy model object (not serializable)"
        serializable_params["n_envs"] = n_envs
        serializable_params["vec_env_type"] = vec_env_type
        serializable_params["normalized"] = normalize
        json.dump(serializable_params, f, indent=4)

    # Train RL agent
    print(
        f"\nTraining {algorithm.upper()} agent with {n_envs} parallel environments..."
    )
    if normalize:
        print("Using observation and reward normalization")

    model, model_path = train_rl_agent(
        vec_env=vec_env,
        n_envs=n_envs,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        seed=seed,
        log_dir=os.path.join(output_dir, "logs"),
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        use_wandb=use_wandb,
    )

    # Close the vectorized environment
    vec_env.close()

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
        "device": str(check_gpu_availability()),
        "n_envs": n_envs,
        "vec_env_type": vec_env_type,
        "normalized": normalize,
    }

    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    return model, model_path


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train RL agent for dollhouse thermal control with WandB logging and GPU support"
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
        "--timesteps", type=int, default=10000000, help="Total timesteps for training"
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
        default=1.0,
        help="Weight for energy penalty in reward",
    )
    parser.add_argument(
        "--comfort-weight",
        type=float,
        default=1.0,
        help="Weight for comfort penalty in reward",
    )

    # Vectorized environment parameters
    parser.add_argument(
        "--n-envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument(
        "--vec-env-type",
        type=str,
        default="subproc",
        choices=["dummy", "subproc"],
        help="Type of vectorized environment",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable observation and reward normalization",
    )

    # WandB parameters
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="dollhouse-thermal-control",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity (username or team)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
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
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        use_wandb=not args.no_wandb,
        n_envs=args.n_envs,
        vec_env_type=args.vec_env_type,
        normalize=not args.no_normalize,  # Use normalization by default
    )

    print(f"\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"Use this path when evaluating the model.")

    # Print GPU info if available
    if torch.cuda.is_available():
        print(f"\nGPU used: {torch.cuda.get_device_name(0)}")
        print(f"Final GPU memory usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# Example usage:
# With normalization (recommended):
# python train_rl_agent.py --data "../Data/dollhouse-data-2025-03-24.csv" --algorithm ppo --timesteps 10000000 --n-envs 4

# Without normalization (for comparison):
# python train_rl_agent.py --data "../Data/dollhouse-data-2025-03-24.csv" --algorithm ppo --timesteps 10000000 --n-envs 4 --no-normalize
