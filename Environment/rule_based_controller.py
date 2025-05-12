import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import argparse

# Import the SINDy training function
from train_sindy_model import train_sindy_model
from dollhouse_env import DollhouseThermalEnv
from normalized_observation_wrapper import NormalizedObservationWrapper


def create_rule_based_controller(hysteresis=0.5, controller_type="simple"):
    """
    Create a rule-based controller with different strategies.

    Args:
        hysteresis: Temperature buffer to prevent oscillation
        controller_type: Type of controller strategy:
                         - "simple": Basic independent control for each floor
                         - "coordinated": Only opens windows when both floors need cooling
                         - "conservative": Prioritizes heating over cooling

    Returns:
        function: Rule-based controller function
    """

    def simple_controller(observation):
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
        print(f"Action taken: {action}")
        return action

    def coordinated_controller(observation):
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

        # Ground floor heating control
        if ground_temp < avg_setpoint - hysteresis:
            # Too cold - turn on light for heat
            action[0] = 1  # Turn ON ground light
        else:
            # Not cold - turn off light
            action[0] = 0  # Turn OFF ground light

        # Top floor heating control
        if top_temp < avg_setpoint - hysteresis:
            # Too cold - turn on light for heat
            action[2] = 1  # Turn ON top light
        else:
            # Not cold - turn off light
            action[2] = 0  # Turn OFF top light

        # Coordinated window control - only open windows if BOTH floors are above setpoint
        both_floors_warm = (
            ground_temp > avg_setpoint + hysteresis
            and top_temp > avg_setpoint + hysteresis
        )

        if both_floors_warm:
            # Both floors are too warm - open both windows
            action[1] = 1  # Open ground window
            action[3] = 1  # Open top window
        else:
            # At least one floor needs heating - close both windows
            action[1] = 0  # Close ground window
            action[3] = 0  # Close top window
        print(f"Action taken: {action}")
        return action

    def conservative_controller(observation):
        # Extract state variables
        ground_temp = observation[0]
        top_temp = observation[1]
        external_temp = observation[2]
        heating_setpoint = observation[7]
        cooling_setpoint = observation[8]

        # Initialize action
        action = np.zeros(4, dtype=int)

        # Ground floor control - prioritize heating over cooling
        if ground_temp < heating_setpoint - hysteresis:
            # Too cold - turn on light for heat, close window
            action[0] = 1  # Turn ON ground light
            action[1] = 0  # Close ground window
        elif ground_temp > cooling_setpoint + hysteresis:
            # Too hot - turn off light, open window
            action[0] = 0  # Turn OFF ground light
            action[1] = 1  # Open ground window
        else:
            # Within comfort band - use minimal energy
            action[0] = 0  # Turn OFF ground light
            action[1] = 0  # Close ground window

        # Top floor control - prioritize heating over cooling
        if top_temp < heating_setpoint - hysteresis:
            # Too cold - turn on light for heat, close window
            action[2] = 1  # Turn ON top light
            action[3] = 0  # Close top window
        elif top_temp > cooling_setpoint + hysteresis:
            # Too hot - turn off light, open window
            action[2] = 0  # Turn OFF top light
            action[3] = 1  # Open top window
        else:
            # Within comfort band - use minimal energy
            action[2] = 0  # Turn OFF top light
            action[3] = 0  # Close top window

        return action

    # Select the appropriate controller based on type
    if controller_type == "coordinated":
        return coordinated_controller
    elif controller_type == "conservative":
        return conservative_controller
    else:  # default to simple
        return simple_controller


def evaluate_rule_based(
    env, num_episodes=5, render=True, hysteresis=0.5, controller_type="simple"
):
    """
    Evaluate a rule-based controller on the environment.

    Args:
        env: The environment to evaluate on
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        hysteresis: Hysteresis parameter for the rule-based controller
        controller_type: Type of controller to use ("simple", "coordinated", or "conservative")

    Returns:
        dict: Evaluation results
    """
    # Create the rule-based controller
    controller = create_rule_based_controller(
        hysteresis=hysteresis, controller_type=controller_type
    )

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

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        # For tracking performance
        comfort_violations = 0

        while not done:
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

            # Take the action
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1

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

        print(
            f"\n{controller_type.capitalize()} Rule-Based Controller Evaluation Summary:"
        )
        print(f"Average Total Reward: {performance['avg_total_reward']:.2f}")
        print(f"Ground Floor Comfort %: {performance['avg_ground_comfort_pct']:.2f}%")
        print(f"Top Floor Comfort %: {performance['avg_top_comfort_pct']:.2f}%")
        print(f"Average Light Hours: {performance['avg_light_hours']:.2f}")

        # Save results
        if hasattr(orig_env, "save_results"):
            output_dir = "rule_based_results"
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                output_dir, f"{controller_type}_controller_results_{timestamp}.json"
            )

            orig_env.save_results(
                filepath,
                controller_name=f"{controller_type.capitalize()} Rule-Based Controller",
            )
            print(f"Results saved to {filepath}")
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


def run_rule_based_evaluation(
    data_file, output_dir=None, num_episodes=5, render=True, controller_type="simple"
):
    """
    Train a SINDy model and evaluate a rule-based controller.

    Args:
        data_file: Path to data file for training SINDy model
        output_dir: Directory to save results
        num_episodes: Number of episodes for evaluation
        render: Whether to render during evaluation
        controller_type: Type of controller to use ("simple", "coordinated", or "conservative")

    Returns:
        dict: Evaluation results
    """
    # Set output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{controller_type}_controller_results_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")

    # Train SINDy model
    print(f"Training SINDy model on {data_file}...")
    start_time = time.time()
    sindy_model = train_sindy_model(file_path=data_file)
    training_time = time.time() - start_time
    print(f"SINDy model training completed in {training_time:.2f} seconds")

    # Create environment
    env_params = {
        "sindy_model": sindy_model,
        "episode_length": 200,  # Adjust as needed
        "time_step_seconds": 30,
        "heating_setpoint": 22.0,
        "cooling_setpoint": 28.0,
        "external_temp_pattern": "sine",
        "setpoint_pattern": "fixed",
        "reward_type": "balanced",
        "energy_weight": 0.5,
        "comfort_weight": 1.0,
    }

    # Create environment
    env = DollhouseThermalEnv(**env_params)

    # Optionally normalize observations
    # env = NormalizedObservationWrapper(env)

    # Save environment parameters
    with open(os.path.join(output_dir, "env_params.json"), "w") as f:
        # Convert non-serializable parameters to strings
        serializable_params = env_params.copy()
        serializable_params["sindy_model"] = "SINDy model object (not serializable)"
        import json

        json.dump(serializable_params, f, indent=4)

    # Evaluate rule-based controller
    print(f"\nEvaluating {controller_type.capitalize()} Rule-Based Controller...")
    start_time = time.time()
    performance = evaluate_rule_based(
        env=env,
        num_episodes=num_episodes,
        render=render,
        hysteresis=0.5,
        controller_type=controller_type,
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
        plt.plot(
            time_hours, env.history["heating_setpoint"], "k--", label="Heating Setpoint"
        )
        plt.plot(
            time_hours, env.history["cooling_setpoint"], "k-.", label="Cooling Setpoint"
        )
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.title(f"{controller_type.capitalize()} Rule-Based Controller Performance")
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
            os.path.join(output_dir, f"{controller_type}_controller_performance.png")
        )
        print(
            f"Performance plot saved to {os.path.join(output_dir, f'{controller_type}_controller_performance.png')}"
        )

    return performance


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train SINDy model and evaluate rule-based controller"
    )

    # Required arguments
    parser.add_argument(
        "--data",
        type=str,
        default=[
            "../Data/dollhouse-data-2025-03-24.csv",
        ],
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
        "--controller",
        type=str,
        default="simple",
        choices=["simple", "coordinated", "conservative"],
        help="Type of controller to use",
    )

    args = parser.parse_args()

    # Run evaluation
    run_rule_based_evaluation(
        data_file=args.data,
        output_dir=args.output,
        num_episodes=args.episodes,
        render=not args.no_render,
        controller_type=args.controller,
    )
#  python rule_based_controller.py --data ../Data/dollhouse-data-2025-03-24.csv --episode 5
