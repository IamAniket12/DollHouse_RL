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


def recreate_environment(env_params_path, data_file=None):
    """
    Recreate the environment using saved parameters.

    Args:
        env_params_path: Path to the saved environment parameters
        data_file: Optional path to data file for SINDy model (overrides saved path)

    Returns:
        env: Recreated environment
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
    env_params["sindy_model"] = sindy_model

    # Create environment with saved parameters
    env = DollhouseThermalEnv(**env_params)

    return env


def evaluate_agent(
    env, model, num_episodes=5, deterministic=False, render=False, verbose=True
):
    """
    Evaluate a trained agent on the environment.

    Args:
        env: The environment to evaluate on
        model: The trained RL model
        num_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        render: Whether to render the environment
        verbose: Whether to print detailed logs

    Returns:
        dict: Evaluation results
    """
    total_rewards = []
    episode_temperatures = []
    episode_actions = []
    episode_rewards = []
    episode_external_temps = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        # Track temperatures and actions for this episode
        temps = []
        actions = []
        rewards = []
        ext_temps = []

        while not done:
            # Select action using model
            action, _ = model.predict(obs, deterministic=deterministic)

            if verbose:
                print(f"Action: {action}")

            # Take action in environment
            obs, reward, done, info = env.step(action)

            episode_reward += reward

            # Record data for analysis
            temps.append([obs[0], obs[1]])  # ground_temp, top_temp
            ext_temps.append(obs[2])  # external_temp
            actions.append(action)
            rewards.append(reward)

            if render:
                env.render()

        total_rewards.append(episode_reward)
        episode_temperatures.append(temps)
        episode_external_temps.append(ext_temps)
        episode_actions.append(actions)
        episode_rewards.append(rewards)

        if verbose:
            print(
                f"Episode {episode+1}/{num_episodes}: Total Reward = {episode_reward:.2f}"
            )

    # Get performance summary
    performance = env.get_performance_summary()

    if verbose:
        print("\nAgent Evaluation Summary:")
        print(f"Average Total Reward: {performance['avg_total_reward']:.2f}")
        print(f"Ground Floor Comfort %: {performance['avg_ground_comfort_pct']:.2f}%")
        print(f"Top Floor Comfort %: {performance['avg_top_comfort_pct']:.2f}%")
        print(f"Average Light Hours: {performance['avg_light_hours']:.2f}")

    # Add raw episode data to performance dict for detailed analysis
    performance["episode_data"] = {
        "temperatures": episode_temperatures,
        "external_temps": episode_external_temps,
        "actions": episode_actions,
        "rewards": episode_rewards,
        "total_rewards": total_rewards,
    }

    # Add environment parameters to performance dict
    performance["heating_setpoint"] = env.heating_setpoint
    performance["cooling_setpoint"] = env.cooling_setpoint
    performance["reward_type"] = env.reward_type
    performance["energy_weight"] = env.energy_weight
    performance["comfort_weight"] = env.comfort_weight

    return performance


def visualize_performance(performance, output_dir, agent_name="RL Agent"):
    """
    Create visualizations of agent performance.

    Args:
        performance: Performance dictionary from evaluate_agent
        output_dir: Directory to save visualizations
        agent_name: Name of the agent for plot titles
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract environment parameters
    heating_setpoint = performance.get("heating_setpoint", 20.0)
    cooling_setpoint = performance.get("cooling_setpoint", 24.0)

    # Extract episode data
    episode_temperatures = performance["episode_data"]["temperatures"]
    episode_external_temps = performance["episode_data"]["external_temps"]
    episode_actions = performance["episode_data"]["actions"]
    episode_rewards = performance["episode_data"]["rewards"]

    # Plot temperatures, actions, and rewards for the first episode
    plt.figure(figsize=(15, 12))

    # Temperature plot
    plt.subplot(4, 1, 1)
    ground_temps = [temp[0] for temp in episode_temperatures[0]]
    top_temps = [temp[1] for temp in episode_temperatures[0]]
    plt.plot(ground_temps, label="Ground Floor Temperature")
    plt.plot(top_temps, label="Top Floor Temperature")
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
    plt.title(f"{agent_name} - Temperatures (Episode 1)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # External temperature plot
    plt.subplot(4, 1, 2)
    ext_temps = episode_external_temps[0]
    plt.plot(ext_temps, label="External Temperature", color="purple")
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
    plt.title(f"{agent_name} - External Temperature (Episode 1)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Actions plot
    plt.subplot(4, 1, 3)
    actions = np.array(episode_actions[0])
    action_names = ["Ground Light", "Ground Window", "Top Light", "Top Window"]
    for i, name in enumerate(action_names):
        plt.plot(actions[:, i], label=name)
    plt.title(f"{agent_name} - Actions (Episode 1)")
    plt.ylabel("Action State (0/1)")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Rewards plot
    plt.subplot(4, 1, 4)
    plt.plot(episode_rewards[0], label="Step Reward")
    plt.title(f"{agent_name} - Rewards (Episode 1)")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir, f'{agent_name.lower().replace(" ", "_")}_episode_analysis.png'
        )
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
        plt.bar([agent_name], [performance.get(metric, 0)])
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
        os.path.join(output_dir, f'{agent_name.lower().replace(" ", "_")}_summary.png')
    )

    # Create temperature distribution plot (similar to your original script)
    plt.figure(figsize=(14, 6))

    # Combine all temperature data across episodes
    all_ground_temps = []
    all_top_temps = []

    for episode_temps in episode_temperatures:
        all_ground_temps.extend([temp[0] for temp in episode_temps])
        all_top_temps.extend([temp[1] for temp in episode_temps])

    # Ground floor temperature distribution
    plt.subplot(1, 2, 1)
    plt.hist(all_ground_temps, bins=30, alpha=0.7)
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
    plt.hist(all_top_temps, bins=30, alpha=0.7)
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
        )
    )

    print(f"Visualizations saved to {output_dir}")


def main(
    model_path,
    data_file,
    env_params_path=None,
    num_episodes=5,
    deterministic=False,
    render=False,
    output_dir=None,
    verbose=True,
):
    """
    Main function to evaluate a trained agent.

    Args:
        model_path: Path to the trained model
        data_file: Path to data file for training SINDy model
        env_params_path: Path to the saved environment parameters
        num_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        render: Whether to render the environment
        output_dir: Directory to save results
        verbose: Whether to print detailed logs
    """
    # Set output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"eval_results/{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Find env_params_path if not provided
    if env_params_path is None:
        # Try to find it in the same directory as the model
        model_dir = os.path.dirname(model_path)
        potential_path = os.path.join(model_dir, "env_params.json")
        if os.path.exists(potential_path):
            env_params_path = potential_path
            print(f"Found environment parameters at {env_params_path}")
        else:
            raise ValueError(
                "env_params_path not provided and not found in model directory"
            )

    # Load model
    model = load_model(model_path)

    # Recreate environment
    env = recreate_environment(env_params_path, data_file)

    # Evaluate agent
    print(
        f"\nEvaluating agent {'deterministically' if deterministic else 'stochastically'} for {num_episodes} episodes..."
    )

    stochastic_mode = "deterministic" if deterministic else "stochastic"
    performance = evaluate_agent(
        env=env,
        model=model,
        num_episodes=num_episodes,
        deterministic=deterministic,
        render=render,
        verbose=verbose,
    )

    # Get algorithm name from model path
    algorithm = "unknown"
    for alg in ["ppo", "a2c", "dqn", "sac"]:
        if alg in model_path.lower():
            algorithm = alg.upper()
            break

    # Save performance results
    results_path = os.path.join(
        output_dir, f"{algorithm}_{stochastic_mode}_results.json"
    )
    # Replace this part in your main function
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
                        [float(t) for t in temps] for temps in value["external_temps"]
                    ],
                    "actions": [
                        [[int(a) for a in action] for action in episode]
                        for episode in value["actions"]
                    ],
                    "rewards": [
                        [float(r) for r in rewards] for rewards in value["rewards"]
                    ],
                    "total_rewards": [float(r) for r in value["total_rewards"]],
                }
            elif isinstance(value, (np.integer, np.floating, np.ndarray)):
                serializable_perf[key] = (
                    value.item() if hasattr(value, "item") else value.tolist()
                )
            else:
                serializable_perf[key] = value

        json.dump(serializable_perf, f, indent=4)

    # Save environment results
    env.save_results(
        os.path.join(output_dir, f"{algorithm}_{stochastic_mode}_env_results.json"),
        controller_name=f"{algorithm} Agent ({stochastic_mode})",
    )

    # Visualize performance
    visualize_performance(
        performance, output_dir, agent_name=f"{algorithm} Agent ({stochastic_mode})"
    )

    print(f"\nEvaluation completed. Results saved to {output_dir}")

    return performance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL agent on the dollhouse environment"
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
        "--deterministic", action="store_true", help="Use deterministic actions"
    )
    parser.add_argument(
        "--stochastic", action="store_true", help="Use stochastic actions (default)"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during evaluation"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Directory to save evaluation results"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed logs")

    args = parser.parse_args()

    # Handle deterministic vs stochastic (prioritize explicit flags)
    deterministic = False
    if args.deterministic:
        deterministic = True
    if args.stochastic:
        deterministic = False

    main(
        model_path=args.model,
        data_file=args.data,
        env_params_path=args.env_params,
        num_episodes=args.episodes,
        deterministic=deterministic,
        render=args.render,
        output_dir=args.output,
        verbose=not args.quiet,
    )

# Example usage:
# python evaluate_rl_agent.py \
#   --model "results/ppo_20250513_151705/logs/models/ppo_final_model" \
#   --data "../Data/dollhouse-data-2025-03-24.csv" \
#   --env-params "results/ppo_20250513_151705/env_params.json" \
#   --stochastic \
#   --episodes 1
