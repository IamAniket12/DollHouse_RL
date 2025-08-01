"""
Reinforcement Learning Agent Training Module.

This module provides functionality to train RL agents on the dollhouse thermal
environment using Stable Baselines3 with WandB logging and GPU acceleration.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import wandb
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from dollhouse_env import DollhouseThermalEnv
from train_sindy_model import train_sindy_model


class EnvironmentMetricsCallback(BaseCallback):
    """
    Custom callback for logging detailed environment metrics to WandB.

    This callback tracks episode-level metrics and environment-specific
    measurements during training for comprehensive monitoring.
    """

    def __init__(self, verbose: int = 0):
        """
        Initialize the metrics callback.

        Args:
            verbose: Verbosity level for logging
        """
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_comfort_violations = []

    def _on_step(self) -> bool:
        """
        Called after each environment step.

        Returns:
            True to continue training, False to stop
        """
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                self._log_episode_metrics(info["episode"])

            if "ground_comfort_violation" in info:
                self._log_environment_metrics(info)

        return True

    def _log_episode_metrics(self, episode_info: dict) -> None:
        """Log episode completion metrics."""
        self.episode_rewards.append(episode_info["r"])
        self.episode_lengths.append(episode_info["l"])

        wandb.log(
            {
                "episode/reward": episode_info["r"],
                "episode/length": episode_info["l"],
                "episode/reward_mean": (
                    np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                ),
            },
            step=self.num_timesteps,
        )

    def _log_environment_metrics(self, info: dict) -> None:
        """Log environment-specific metrics."""
        wandb.log(
            {
                "env/ground_comfort_violation": info["ground_comfort_violation"],
                "env/top_comfort_violation": info["top_comfort_violation"],
                "env/energy_use": info["energy_use"],
                "env/ground_temp": info["ground_temp"],
                "env/top_temp": info["top_temp"],
                "env/external_temp": info["external_temp"],
            },
            step=self.num_timesteps,
        )


def check_gpu_availability() -> torch.device:
    """
    Check GPU availability and return appropriate device.

    Returns:
        PyTorch device object (CUDA or CPU)
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


def make_env_factory(
    rank: int, seed: int, sindy_model, env_params: dict, monitor_dir: str
) -> Callable:
    """
    Create a factory function for environment instantiation.

    Args:
        rank: Unique identifier for the environment instance
        seed: Random seed for reproducibility
        sindy_model: Trained SINDy model for dynamics
        env_params: Environment configuration parameters
        monitor_dir: Directory for environment monitoring logs

    Returns:
        Factory function that creates a monitored environment instance
    """

    def _init():
        local_params = env_params.copy()
        local_params["random_seed"] = seed + rank
        local_params["sindy_model"] = sindy_model

        env = DollhouseThermalEnv(**local_params)
        env = Monitor(env, os.path.join(monitor_dir, f"env_{rank}"))

        return env

    return _init


def create_vectorized_environment(
    sindy_model,
    env_params: dict,
    n_envs: int,
    seed: int,
    monitor_dir: str,
    vec_env_type: str = "subproc",
    normalize: bool = True,
):
    """
    Create vectorized environments for parallel training.

    Args:
        sindy_model: Trained SINDy model for environment dynamics
        env_params: Environment configuration parameters
        n_envs: Number of parallel environments
        seed: Base random seed
        monitor_dir: Directory for monitoring logs
        vec_env_type: Type of vectorization ("dummy" or "subproc")
        normalize: Whether to apply observation and reward normalization

    Returns:
        Vectorized environment, optionally with normalization wrapper
    """
    os.makedirs(monitor_dir, exist_ok=True)

    env_factories = [
        make_env_factory(i, seed, sindy_model, env_params, monitor_dir)
        for i in range(n_envs)
    ]

    if vec_env_type == "subproc":
        vec_env = SubprocVecEnv(env_factories, start_method="spawn")
        print(f"Created SubprocVecEnv with {n_envs} parallel environments")
    else:
        vec_env = DummyVecEnv(env_factories)
        print(f"Created DummyVecEnv with {n_envs} environments")

    if normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            epsilon=1e-8,
        )
        print("Applied observation and reward normalization")

    return vec_env


def create_rl_model(
    algorithm: str, vec_env, n_envs: int, seed: int, log_dir: str, device: torch.device
):
    """
    Create an RL model with algorithm-specific configurations.

    Args:
        algorithm: RL algorithm name ("ppo", "a2c", "dqn", "sac")
        vec_env: Vectorized environment
        n_envs: Number of parallel environments
        seed: Random seed
        log_dir: Directory for tensorboard logs
        device: PyTorch device for training

    Returns:
        Configured RL model ready for training

    Raises:
        ValueError: If algorithm is not supported
    """
    common_params = {
        "env": vec_env,
        "verbose": 1,
        "tensorboard_log": log_dir,
        "seed": seed,
        "device": device,
    }

    if algorithm.lower() == "ppo":
        return _create_ppo_model(common_params, n_envs)
    elif algorithm.lower() == "a2c":
        return _create_a2c_model(common_params, n_envs)
    elif algorithm.lower() == "dqn":
        return _create_dqn_model(common_params)
    elif algorithm.lower() == "sac":
        return _create_sac_model(common_params)
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Choose from: 'ppo', 'a2c', 'dqn', 'sac'"
        )


def _create_ppo_model(common_params: dict, n_envs: int):
    """Create PPO model with optimized hyperparameters."""
    n_steps = max(128, 1024 // n_envs)
    batch_size = 128

    model = PPO(
        "MlpPolicy",
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
        **common_params,
    )

    print(
        f"PPO configured with n_steps={n_steps}, batch_size={batch_size} for {n_envs} environments"
    )
    return model


def _create_a2c_model(common_params: dict, n_envs: int):
    """Create A2C model with optimized hyperparameters."""
    n_steps = 256

    model = A2C(
        "MlpPolicy",
        learning_rate=7e-4,
        n_steps=n_steps,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        **common_params,
    )

    print(f"A2C configured with n_steps={n_steps} for {n_envs} environments")
    return model


def _create_dqn_model(common_params: dict):
    """Create DQN model with optimized hyperparameters."""
    model = DQN(
        "MlpPolicy",
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
        **common_params,
    )

    print("DQN configured with default hyperparameters")
    return model


def _create_sac_model(common_params: dict):
    """Create SAC model with optimized hyperparameters."""
    model = SAC(
        "MlpPolicy",
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
        **common_params,
    )

    print("SAC configured with default hyperparameters")
    return model


def setup_wandb_logging(
    wandb_project: str,
    wandb_entity: Optional[str],
    algorithm: str,
    total_timesteps: int,
    seed: int,
    device: torch.device,
    n_envs: int,
    normalize: bool,
) -> str:
    """
    Initialize WandB logging for experiment tracking.

    Args:
        wandb_project: WandB project name
        wandb_entity: WandB entity (username or team)
        algorithm: RL algorithm being used
        total_timesteps: Total training timesteps
        seed: Random seed
        device: Training device
        n_envs: Number of parallel environments
        normalize: Whether normalization is applied

    Returns:
        Run name for the experiment
    """
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
            "normalized": normalize,
        },
        sync_tensorboard=True,
    )

    return run_name


def create_training_callbacks(
    models_dir: str, algorithm: str, use_wandb: bool
) -> List[BaseCallback]:
    """
    Create training callbacks for monitoring and checkpointing.

    Args:
        models_dir: Directory to save model checkpoints
        algorithm: RL algorithm name
        use_wandb: Whether to use WandB logging

    Returns:
        List of configured callbacks
    """
    callbacks = []

    checkpoint_callback = CheckpointCallback(
        save_freq=200000,
        save_path=models_dir,
        name_prefix=algorithm,
    )
    callbacks.append(checkpoint_callback)

    if use_wandb:
        wandb_callback = WandbCallback(
            gradient_save_freq=100000,
            model_save_path=f"{models_dir}/wandb",
            verbose=2,
        )
        callbacks.append(wandb_callback)

        custom_callback = EnvironmentMetricsCallback()
        callbacks.append(custom_callback)

    return callbacks


def train_rl_agent(
    vec_env,
    n_envs: int,
    algorithm: str = "ppo",
    total_timesteps: int = 5000000,
    seed: int = 0,
    log_dir: str = "logs",
    wandb_project: str = "dollhouse-thermal-control",
    wandb_entity: Optional[str] = None,
    use_wandb: bool = True,
) -> Tuple[object, str]:
    """
    Train an RL agent with comprehensive logging and monitoring.

    Args:
        vec_env: Vectorized environment for training
        n_envs: Number of parallel environments
        algorithm: RL algorithm to use
        total_timesteps: Total timesteps for training
        seed: Random seed for reproducibility
        log_dir: Directory for tensorboard logs
        wandb_project: WandB project name
        wandb_entity: WandB entity
        use_wandb: Whether to use WandB logging

    Returns:
        Tuple of (trained_model, model_save_path)
    """
    device = check_gpu_availability()

    os.makedirs(log_dir, exist_ok=True)
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    if use_wandb:
        run_name = setup_wandb_logging(
            wandb_project,
            wandb_entity,
            algorithm,
            total_timesteps,
            seed,
            device,
            n_envs,
            isinstance(vec_env, VecNormalize),
        )

    callbacks = create_training_callbacks(models_dir, algorithm, use_wandb)
    model = create_rl_model(algorithm, vec_env, n_envs, seed, log_dir, device)

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

    print(f"\nTraining {algorithm.upper()} for {total_timesteps} timesteps...")
    print(f"Using device: {device}")
    print(f"Training with {n_envs} parallel environments")

    if isinstance(vec_env, VecNormalize):
        print("With observation and reward normalization")

    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Average timesteps per second: {total_timesteps / training_time:.2f}")

    final_model_path = os.path.join(models_dir, f"{algorithm}_final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    if isinstance(vec_env, VecNormalize):
        vec_normalize_path = os.path.join(models_dir, "vec_normalize.pkl")
        vec_env.save(vec_normalize_path)
        print(f"Normalization statistics saved to {vec_normalize_path}")

    if use_wandb:
        _finalize_wandb_logging(
            final_model_path, vec_env, training_time, total_timesteps
        )

    return model, final_model_path


def _finalize_wandb_logging(
    final_model_path: str, vec_env, training_time: float, total_timesteps: int
) -> None:
    """Finalize WandB logging with model artifacts and final metrics."""
    wandb.save(f"{final_model_path}.zip")

    if isinstance(vec_env, VecNormalize):
        vec_normalize_path = os.path.join(
            os.path.dirname(final_model_path), "vec_normalize.pkl"
        )
        wandb.save(vec_normalize_path)

    wandb.config.update(
        {
            "training_time_seconds": training_time,
            "final_model_path": final_model_path,
            "timesteps_per_second": total_timesteps / training_time,
        }
    )

    wandb.finish()


def setup_training_environment(
    data_file: str,
    reward_type: str = "balanced",
    energy_weight: float = 1.0,
    comfort_weight: float = 1.0,
    custom_observation_config: Optional[str] = None,
) -> Tuple[dict, object]:
    """
    Setup training environment configuration.

    Args:
        data_file: Path to training data file
        reward_type: Type of reward function
        energy_weight: Weight for energy penalty
        comfort_weight: Weight for comfort penalty
        custom_observation_config: Path to custom observation space JSON config

    Returns:
        Tuple of (environment_parameters, sindy_model)
    """
    print(f"Training SINDy model on {data_file}...")
    sindy_model = train_sindy_model(file_path=data_file)

    env_params = {
        "episode_length": 2880,
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
        "custom_observation_config": custom_observation_config,
    }

    return env_params, sindy_model


def setup_training(
    data_file: str,
    output_dir: Optional[str] = None,
    algorithm: str = "ppo",
    total_timesteps: int = 5000000,
    reward_type: str = "balanced",
    energy_weight: float = 1.0,
    comfort_weight: float = 1.0,
    seed: int = 0,
    wandb_project: str = "dollhouse-thermal-control",
    wandb_entity: Optional[str] = None,
    use_wandb: bool = True,
    n_envs: int = 4,
    vec_env_type: str = "subproc",
    normalize: bool = True,
    custom_observation_config: Optional[str] = None,
) -> Tuple[object, str]:
    """
    Complete training setup and execution pipeline.

    Args:
        data_file: Path to training data file
        output_dir: Directory to save results (auto-generated if None)
        algorithm: RL algorithm to use
        total_timesteps: Total timesteps for training
        reward_type: Type of reward function
        energy_weight: Weight for energy penalty
        comfort_weight: Weight for comfort penalty
        seed: Random seed for reproducibility
        wandb_project: WandB project name
        wandb_entity: WandB entity
        use_wandb: Whether to use WandB logging
        n_envs: Number of parallel environments
        vec_env_type: Type of vectorization
        normalize: Whether to apply normalization
        custom_observation_config: Path to custom observation space JSON config

    Returns:
        Tuple of (trained_model, model_save_path)
    """
    _set_random_seeds(seed)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/{algorithm}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")

    # Validate custom observation config
    if custom_observation_config:
        if not os.path.exists(custom_observation_config):
            raise FileNotFoundError(
                f"Custom observation config not found: {custom_observation_config}"
            )
        print(f"Using custom observation configuration: {custom_observation_config}")

    env_params, sindy_model = setup_training_environment(
        data_file, reward_type, energy_weight, comfort_weight, custom_observation_config
    )

    monitor_dir = os.path.join(output_dir, "monitor")
    vec_env = create_vectorized_environment(
        sindy_model, env_params, n_envs, seed, monitor_dir, vec_env_type, normalize
    )

    # Log observation space information
    base_env = vec_env.envs[0] if hasattr(vec_env, "envs") else vec_env.venv.envs[0]
    obs_info = base_env.get_observation_space_info()
    print(f"\nObservation Space Configuration:")
    print(f"  Variables: {obs_info['variables']}")
    print(f"  Shape: {obs_info['space_shape']}")

    _save_environment_config(
        output_dir,
        env_params,
        n_envs,
        vec_env_type,
        normalize,
        custom_observation_config,
    )

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

    vec_env.close()

    _save_training_config(
        output_dir,
        algorithm,
        total_timesteps,
        reward_type,
        energy_weight,
        comfort_weight,
        model_path,
        seed,
        n_envs,
        vec_env_type,
        normalize,
        custom_observation_config,
    )

    print(f"\nTraining completed. Model saved to {model_path}")

    return model, model_path


def _set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _save_environment_config(
    output_dir: str,
    env_params: dict,
    n_envs: int,
    vec_env_type: str,
    normalize: bool,
    custom_observation_config: Optional[str] = None,
) -> None:
    """Save environment configuration to JSON."""
    serializable_params = env_params.copy()
    serializable_params["sindy_model"] = "SINDy model object (not serializable)"
    serializable_params["n_envs"] = n_envs
    serializable_params["vec_env_type"] = vec_env_type
    serializable_params["normalized"] = normalize

    if custom_observation_config:
        serializable_params["custom_observation_config"] = custom_observation_config
        # Copy the observation config to the output directory for reference
        config_filename = os.path.basename(custom_observation_config)
        output_config_path = os.path.join(
            output_dir, f"observation_config_{config_filename}"
        )
        with open(custom_observation_config, "r") as src, open(
            output_config_path, "w"
        ) as dst:
            dst.write(src.read())
        print(f"Copied observation config to {output_config_path}")

    with open(os.path.join(output_dir, "env_params.json"), "w") as f:
        json.dump(serializable_params, f, indent=4)


def _save_training_config(
    output_dir: str,
    algorithm: str,
    total_timesteps: int,
    reward_type: str,
    energy_weight: float,
    comfort_weight: float,
    model_path: str,
    seed: int,
    n_envs: int,
    vec_env_type: str,
    normalize: bool,
    custom_observation_config: Optional[str] = None,
) -> None:
    """Save training configuration to JSON."""
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

    if custom_observation_config:
        config["custom_observation_config"] = custom_observation_config

    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train RL agent for dollhouse thermal control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file for training SINDy model",
    )
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
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="dollhouse-thermal-control",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="WandB entity (username or team)"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--output", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--custom-obs-config",
        type=str,
        default=None,
        help="Path to custom observation space configuration JSON file",
    )

    args = parser.parse_args()

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
        normalize=not args.no_normalize,
        custom_observation_config=args.custom_obs_config,
    )

    print(f"\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"Use this path when evaluating the model.")

    if torch.cuda.is_available():
        print(f"\nGPU used: {torch.cuda.get_device_name(0)}")
        print(f"Final GPU memory usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
