import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import argparse
import json
from normalized_observation_wrapper import NormalizedObservationWrapper

# Import Stable Baselines
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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


def evaluate_controller(
    env, controller, num_episodes=5, is_rl_agent=False, render=False
):
    """
    Evaluate a controller on the environment.

    Args:
        env: The environment to evaluate on
        controller: The controller (function for rule-based, model for RL)
        num_episodes: Number of episodes to evaluate
        is_rl_agent: Whether the controller is an RL agent
        render: Whether to render the environment

    Returns:
        dict: Evaluation results
    """
    total_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Select action using controller
            if is_rl_agent:
                action, _ = controller.predict(obs, deterministic=True)
            else:
                action = controller(obs)

            # Take action in environment
            obs, reward, done, info = env.step(action)

            episode_reward += reward

            if render:
                env.render()

        total_rewards.append(episode_reward)
        print(
            f"Episode {episode+1}/{num_episodes}: Total Reward = {episode_reward:.2f}"
        )

    # Get performance summary
    performance = env.get_performance_summary()

    print("\nController Evaluation Summary:")
    print(f"Average Total Reward: {performance['avg_total_reward']:.2f}")
    print(f"Ground Floor Comfort %: {performance['avg_ground_comfort_pct']:.2f}%")
    print(f"Top Floor Comfort %: {performance['avg_top_comfort_pct']:.2f}%")
    print(f"Average Light Hours: {performance['avg_light_hours']:.2f}")

    return performance


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
        model: Trained model
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

    return model


def run_comparison(
    data_file,
    output_dir=None,
    algorithm="ppo",
    total_timesteps=5000000,
    eval_episodes=5,
    render=True,
    compare_rule_based=True,
    reward_type="balanced",
    energy_weight=0.5,
    comfort_weight=1.0,
):
    """
    Run a full comparison between controllers.

    Args:
        data_file: Path to data file for training SINDy model
        output_dir: Directory to save results
        algorithm: RL algorithm to use
        total_timesteps: Total timesteps for training
        eval_episodes: Number of episodes for evaluation
        render: Whether to render during evaluation
        compare_rule_based: Whether to compare with rule-based controller
        reward_type: Type of reward function ('comfort', 'energy', or 'balanced')
        energy_weight: Weight for energy penalty in reward
        comfort_weight: Weight for comfort penalty in reward

    Returns:
        dict: Results
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

    # Create environment
    env = DollhouseThermalEnv(**env_params)
    # env = NormalizedObservationWrapper(env)

    # Save environment parameters
    with open(os.path.join(output_dir, "env_params.json"), "w") as f:
        # Convert non-serializable parameters to strings
        serializable_params = env_params.copy()
        serializable_params["sindy_model"] = "SINDy model object (not serializable)"
        json.dump(serializable_params, f, indent=4)

    results = {}

    # Evaluate rule-based controller if requested
    if compare_rule_based:
        print("\nEvaluating Rule-Based Controller...")
        rule_controller = create_rule_based_controller(hysteresis=0.5)
        rule_performance = evaluate_controller(
            env=env,
            controller=rule_controller,
            num_episodes=eval_episodes,
            is_rl_agent=False,
            render=render,
        )

        # Save rule-based results
        env.save_results(
            os.path.join(output_dir, "rule_based_results.json"),
            controller_name="Rule-Based Controller",
        )

        results["rule_based"] = rule_performance

    # Train RL agent
    print(f"\nTraining {algorithm.upper()} agent...")
    rl_model = train_rl_agent(
        env=env,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        log_dir=os.path.join(output_dir, "logs"),
    )

    # Evaluate RL agent
    print(f"\nEvaluating {algorithm.upper()} agent...")
    rl_performance = evaluate_controller(
        env=env,
        controller=rl_model,
        num_episodes=eval_episodes,
        is_rl_agent=True,
        render=render,
    )

    # Save RL agent results
    env.save_results(
        os.path.join(output_dir, f"{algorithm}_results.json"),
        controller_name=f"{algorithm.upper()} Agent",
    )

    results["rl_agent"] = rl_performance

    # Compare and visualize results if both controllers were evaluated
    if compare_rule_based:
        # Create comparison plot
        plt.figure(figsize=(12, 8))

        # Metrics to plot
        plot_metrics = [
            ("avg_total_reward", "Total Reward"),
            ("avg_ground_comfort_pct", "Ground Floor Comfort %"),
            ("avg_top_comfort_pct", "Top Floor Comfort %"),
            ("avg_light_hours", "Light Hours"),
        ]

        for i, (metric, label) in enumerate(plot_metrics):
            plt.subplot(2, 2, i + 1)

            rule_val = rule_performance.get(metric, 0)
            rl_val = rl_performance.get(metric, 0)

            bars = plt.bar(["Rule-Based", algorithm.upper()], [rule_val, rl_val])

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                )

            plt.title(label)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_plot.png"))
        plt.close()

        # Create a summary table
        print("\nPerformance Comparison:")
        print("-" * 80)
        print(f"{'Metric':<30} {'Rule-Based':<20} {algorithm.upper():<20}")
        print("-" * 80)

        metrics = [
            ("Total Reward", "avg_total_reward"),
            ("Ground Floor Comfort %", "avg_ground_comfort_pct"),
            ("Top Floor Comfort %", "avg_top_comfort_pct"),
            ("Average Light Hours", "avg_light_hours"),
        ]

        for label, key in metrics:
            rule_val = rule_performance.get(key, "N/A")
            rl_val = rl_performance.get(key, "N/A")

            if isinstance(rule_val, (int, float)) and isinstance(rl_val, (int, float)):
                print(f"{label:<30} {rule_val:<20.2f} {rl_val:<20.2f}")
            else:
                print(f"{label:<30} {rule_val:<20} {rl_val:<20}")

        print("-" * 80)

        # Save comparison summary
        comparison_summary = {
            "rule_based": rule_performance,
            "rl_agent": {"algorithm": algorithm, "performance": rl_performance},
            "parameters": {
                "data_file": data_file,
                "total_timesteps": total_timesteps,
                "eval_episodes": eval_episodes,
                "reward_type": reward_type,
                "energy_weight": energy_weight,
                "comfort_weight": comfort_weight,
            },
        }

        with open(os.path.join(output_dir, "comparison_summary.json"), "w") as f:
            json.dump(comparison_summary, f, indent=4)

    print(f"\nResults saved to {output_dir}")

    return results


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

    # Evaluation parameters
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes for evaluation"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Disable rendering during evaluation"
    )
    parser.add_argument(
        "--no-rule-based",
        action="store_true",
        help="Skip rule-based controller comparison",
    )

    # Reward parameters
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
        default=0.5,
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

    # Run comparison
    run_comparison(
        data_file=args.data,
        output_dir=args.output,
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        eval_episodes=args.episodes,
        render=not args.no_render,
        compare_rule_based=not args.no_rule_based,
        reward_type=args.reward,
        energy_weight=args.energy_weight,
        comfort_weight=args.comfort_weight,
    )
