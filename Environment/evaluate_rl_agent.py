"""
RL Agent Evaluation Module.

This module provides comprehensive evaluation capabilities for trained RL agents
on the dollhouse thermal environment with support for normalized environments.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dollhouse_env import DollhouseThermalEnv
from train_sindy_model import train_sindy_model


class RLAgentEvaluator:
    """
    Comprehensive evaluator for trained RL agents.

    Provides standardized evaluation, visualization, and analysis
    capabilities for RL agents on the thermal control task.
    """

    def __init__(self, env, model, model_info: Optional[Dict] = None):
        """
        Initialize RL agent evaluator.

        Args:
            env: Environment for evaluation (may be normalized)
            model: Trained RL model
            model_info: Optional metadata about the model
        """
        self.env = env
        self.model = model
        self.model_info = model_info or {}
        self.is_vec_env = isinstance(env, (DummyVecEnv, VecNormalize))
        self.base_env = self._get_base_environment()

    def _get_base_environment(self):
        """Extract the base environment from potential wrappers."""
        if isinstance(self.env, VecNormalize):
            return self.env.venv.envs[0]
        elif isinstance(self.env, DummyVecEnv):
            return self.env.envs[0]
        else:
            return self.env

    def evaluate(
        self,
        num_episodes: int = 5,
        render: bool = False,
        deterministic: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        Evaluate the RL agent comprehensively.

        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render during evaluation
            deterministic: Whether to use deterministic policy
            verbose: Whether to print detailed progress

        Returns:
            Dictionary containing comprehensive evaluation results
        """
        if verbose:
            print(f"Evaluating RL agent for {num_episodes} episodes...")
            print(
                f"Environment type: {'Normalized' if isinstance(self.env, VecNormalize) else 'Standard'}"
            )

        total_rewards = []
        episode_data = {
            "temperatures": [],
            "external_temps": [],
            "actions": [],
            "rewards": [],
            "setpoints": [],
        }

        for episode in range(num_episodes):
            episode_result = self._evaluate_single_episode(
                episode, num_episodes, render, deterministic, verbose
            )

            total_rewards.append(episode_result["total_reward"])
            for key in episode_data:
                episode_data[key].append(episode_result[key])

        performance = self._calculate_performance_metrics(total_rewards, episode_data)
        performance["model_info"] = self.model_info
        performance["evaluation_config"] = {
            "num_episodes": num_episodes,
            "deterministic": deterministic,
            "was_normalized": isinstance(self.env, VecNormalize),
        }

        if verbose:
            self._print_evaluation_summary(performance)

        return performance

    def _evaluate_single_episode(
        self,
        episode: int,
        num_episodes: int,
        render: bool,
        deterministic: bool,
        verbose: bool,
    ) -> Dict:
        """Evaluate a single episode and collect data."""
        # Reset environment
        if self.is_vec_env:
            obs = self.env.reset()
            obs = obs[0] if len(obs.shape) > 1 else obs
        else:
            obs, info = self.env.reset()

        terminated = False
        truncated = False
        episode_reward = 0

        temps = []
        ext_temps = []
        actions = []
        rewards = []
        setpoints = []

        while not terminated and not truncated:
            # Get current setpoints
            heating_sp = self.base_env.heating_setpoint
            cooling_sp = self.base_env.cooling_setpoint
            setpoints.append([heating_sp, cooling_sp])

            # Get action from model
            if self.is_vec_env:
                action, _ = self.model.predict(
                    obs.reshape(1, -1), deterministic=deterministic
                )
                action = action[0]
            else:
                action, _ = self.model.predict(obs, deterministic=deterministic)

            if verbose and len(temps) % 100 == 0:
                original_obs = self._get_original_observation(obs)
                print(
                    f"Episode {episode+1}, Step {len(temps)}: "
                    f"Action: {action}, Temps: {original_obs[0]:.1f}/{original_obs[1]:.1f}°C, "
                    f"Setpoints: {heating_sp:.1f}/{cooling_sp:.1f}°C"
                )

            # Take action
            if self.is_vec_env:
                obs, reward, done, info = self.env.step([action])
                obs = obs[0]
                reward = reward[0]
                done = done[0]
                info = info[0] if isinstance(info, list) else info
                terminated = done
                truncated = False
            else:
                obs, reward, terminated, truncated, info = self.env.step(action)

            episode_reward += reward

            # Record data using original observation values
            original_obs = self._get_original_observation(obs)
            temps.append([original_obs[0], original_obs[1]])
            ext_temps.append(original_obs[2])
            actions.append(action)
            rewards.append(reward)

            if render and not self.is_vec_env:
                self.base_env.render()

        if verbose:
            print(
                f"Episode {episode+1}/{num_episodes}: Total Reward = {episode_reward:.2f}"
            )

        return {
            "total_reward": episode_reward,
            "temperatures": temps,
            "external_temps": ext_temps,
            "actions": actions,
            "rewards": rewards,
            "setpoints": setpoints,
        }

    def _get_original_observation(self, obs: np.ndarray) -> np.ndarray:
        """Get original (unnormalized) observation values."""
        if isinstance(self.env, VecNormalize):
            # Get unnormalized observation for recording
            return (
                self.env.get_original_obs()[0]
                if hasattr(self.env, "get_original_obs")
                else obs
            )
        else:
            return obs

    def _calculate_performance_metrics(
        self, total_rewards: list, episode_data: Dict
    ) -> Dict:
        """Calculate comprehensive performance metrics."""
        # Get basic performance summary from environment
        if hasattr(self.base_env, "get_performance_summary"):
            performance = self.base_env.get_performance_summary()
        else:
            performance = {
                "avg_total_reward": np.mean(total_rewards),
                "std_total_reward": np.std(total_rewards),
            }

        # Add episode data
        performance["episode_data"] = episode_data

        # Set setpoint information
        if episode_data["setpoints"] and len(episode_data["setpoints"][0]) > 0:
            performance["heating_setpoint"] = episode_data["setpoints"][0][0][0]
            performance["cooling_setpoint"] = episode_data["setpoints"][0][0][1]
            performance["has_dynamic_setpoints"] = True
        else:
            performance["heating_setpoint"] = self.base_env.initial_heating_setpoint
            performance["cooling_setpoint"] = self.base_env.initial_cooling_setpoint
            performance["has_dynamic_setpoints"] = False

        # Add environment configuration
        performance["reward_type"] = self.base_env.reward_type
        performance["energy_weight"] = self.base_env.energy_weight
        performance["comfort_weight"] = self.base_env.comfort_weight

        return performance

    def _print_evaluation_summary(self, performance: Dict) -> None:
        """Print comprehensive evaluation summary."""
        print("\nRL Agent Evaluation Summary:")
        print(f"Average Total Reward: {performance['avg_total_reward']:.2f}")
        print(f"Ground Floor Comfort %: {performance['avg_ground_comfort_pct']:.2f}%")
        print(f"Top Floor Comfort %: {performance['avg_top_comfort_pct']:.2f}%")
        print(f"Average Light Hours: {performance['avg_light_hours']:.2f}")

        if performance.get("model_info"):
            print(f"\nModel Information:")
            for key, value in performance["model_info"].items():
                print(f"  {key}: {value}")


class ModelLoader:
    """Utility class for loading trained RL models and environments."""

    @staticmethod
    def load_model(model_path: str):
        """
        Load a trained RL model from file.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded RL model

        Raises:
            ValueError: If algorithm cannot be determined from path
        """
        algorithm = ModelLoader._determine_algorithm(model_path)

        algorithm_map = {"ppo": PPO, "a2c": A2C, "dqn": DQN, "sac": SAC}

        if algorithm not in algorithm_map:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        model = algorithm_map[algorithm].load(model_path)
        print(f"Successfully loaded {algorithm.upper()} model from {model_path}")

        return model, algorithm

    @staticmethod
    def _determine_algorithm(model_path: str) -> str:
        """Determine algorithm from model path."""
        path_lower = model_path.lower()

        for algorithm in ["ppo", "a2c", "dqn", "sac"]:
            if algorithm in path_lower:
                return algorithm

        raise ValueError(f"Cannot determine algorithm from model path: {model_path}")

    @staticmethod
    def recreate_environment(
        env_params_path: str,
        data_file: Optional[str] = None,
        model_dir: Optional[str] = None,
    ):
        """
        Recreate environment from saved parameters.

        Args:
            env_params_path: Path to saved environment parameters
            data_file: Optional data file path (overrides saved path)
            model_dir: Directory containing model and normalization files

        Returns:
            Recreated environment (possibly normalized)
        """
        with open(env_params_path, "r") as f:
            env_params = json.load(f)

        # Train or retrain SINDy model
        data_path = data_file if data_file else env_params.get("data_file")
        if not data_path:
            raise ValueError("No data file provided or found in environment parameters")

        print(f"Training SINDy model on {data_path}...")
        sindy_model = train_sindy_model(file_path=data_path)

        # Create environment
        env_params_copy = env_params.copy()
        env_params_copy["sindy_model"] = sindy_model

        # Remove training-specific parameters
        for key in ["n_envs", "vec_env_type", "normalized", "data_file"]:
            env_params_copy.pop(key, None)

        env = DollhouseThermalEnv(**env_params_copy)

        # Apply normalization if it was used during training
        normalized = env_params.get("normalized", False)

        if normalized and model_dir:
            vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")
            if os.path.exists(vec_normalize_path):
                print(f"Found normalization statistics at {vec_normalize_path}")
                env = DummyVecEnv([lambda: env])
                env = VecNormalize.load(vec_normalize_path, env)
                env.training = False
                env.norm_reward = False
                print("Applied normalization wrapper for evaluation")
            else:
                print(
                    f"Warning: Training used normalization but {vec_normalize_path} not found"
                )
                print("Proceeding without normalization - results may be suboptimal")
        elif normalized:
            print("Warning: Training used normalization but model_dir not provided")

        return env


class VisualizationEngine:
    """Engine for creating evaluation visualizations."""

    @staticmethod
    def create_evaluation_plots(
        performance: Dict, output_dir: str, agent_name: str = "RL Agent"
    ) -> None:
        """
        Create comprehensive evaluation visualizations.

        Args:
            performance: Performance dictionary from evaluation
            output_dir: Directory to save visualizations
            agent_name: Name of the agent for plot titles
        """
        os.makedirs(output_dir, exist_ok=True)

        VisualizationEngine._plot_episode_analysis(performance, output_dir, agent_name)
        VisualizationEngine._plot_summary_metrics(performance, output_dir, agent_name)
        VisualizationEngine._plot_temperature_distributions(
            performance, output_dir, agent_name
        )

    @staticmethod
    def _plot_episode_analysis(
        performance: Dict, output_dir: str, agent_name: str
    ) -> None:
        """Plot detailed episode analysis."""
        episode_data = performance["episode_data"]
        has_dynamic_setpoints = performance.get("has_dynamic_setpoints", False)

        n_subplots = 5 if has_dynamic_setpoints else 4
        fig, axes = plt.subplots(n_subplots, 1, figsize=(15, 4 * n_subplots))

        if n_subplots == 1:
            axes = [axes]

        # Temperature plot
        ground_temps = [temp[0] for temp in episode_data["temperatures"][0]]
        top_temps = [temp[1] for temp in episode_data["temperatures"][0]]

        axes[0].plot(ground_temps, label="Ground Floor Temperature", linewidth=2)
        axes[0].plot(top_temps, label="Top Floor Temperature", linewidth=2)

        if has_dynamic_setpoints:
            setpoints = episode_data["setpoints"][0]
            heating_setpoints = [sp[0] for sp in setpoints]
            cooling_setpoints = [sp[1] for sp in setpoints]
            axes[0].plot(
                heating_setpoints, "r--", label="Heating Setpoint", linewidth=1.5
            )
            axes[0].plot(
                cooling_setpoints, "b--", label="Cooling Setpoint", linewidth=1.5
            )
        else:
            heating_sp = performance.get("heating_setpoint", 20.0)
            cooling_sp = performance.get("cooling_setpoint", 24.0)
            axes[0].axhline(
                y=heating_sp,
                color="r",
                linestyle="--",
                label=f"Heating Setpoint ({heating_sp}°C)",
            )
            axes[0].axhline(
                y=cooling_sp,
                color="b",
                linestyle="--",
                label=f"Cooling Setpoint ({cooling_sp}°C)",
            )

        normalization_status = (
            " (Normalized)" if performance.get("was_normalized", False) else ""
        )
        axes[0].set_title(
            f"{agent_name} - Temperatures (Episode 1){normalization_status}"
        )
        axes[0].set_ylabel("Temperature (°C)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        subplot_idx = 1

        # Dynamic setpoints plot (if applicable)
        if has_dynamic_setpoints:
            setpoints = episode_data["setpoints"][0]
            heating_setpoints = [sp[0] for sp in setpoints]
            cooling_setpoints = [sp[1] for sp in setpoints]
            axes[subplot_idx].plot(
                heating_setpoints, "r-", label="Heating Setpoint", linewidth=2
            )
            axes[subplot_idx].plot(
                cooling_setpoints, "b-", label="Cooling Setpoint", linewidth=2
            )
            axes[subplot_idx].set_title(f"{agent_name} - Dynamic Setpoints (Episode 1)")
            axes[subplot_idx].set_ylabel("Temperature (°C)")
            axes[subplot_idx].legend()
            axes[subplot_idx].grid(True, alpha=0.3)
        subplot_idx += 1

        # Actions plot
        actions = np.array(episode_data["actions"][0])
        action_names = ["Ground Light", "Ground Window", "Top Light", "Top Window"]
        colors = ["red", "blue", "orange", "cyan"]

        for i, (name, color) in enumerate(zip(action_names, colors)):
            axes[subplot_idx].plot(actions[:, i], label=name, linewidth=2, color=color)

        axes[subplot_idx].set_title(f"{agent_name} - Actions (Episode 1)")
        axes[subplot_idx].set_ylabel("Action State (0/1)")
        axes[subplot_idx].set_ylim(-0.1, 1.1)
        axes[subplot_idx].legend()
        axes[subplot_idx].grid(True, alpha=0.3)
        subplot_idx += 1

        # Rewards plot
        axes[subplot_idx].plot(
            episode_data["rewards"][0], label="Step Reward", linewidth=2, color="green"
        )
        axes[subplot_idx].set_title(f"{agent_name} - Rewards (Episode 1)")
        axes[subplot_idx].set_xlabel("Timestep")
        axes[subplot_idx].set_ylabel("Reward")
        axes[subplot_idx].legend()
        axes[subplot_idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f'{agent_name.lower().replace(" ", "_")}_episode_analysis.png',
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    @staticmethod
    def _plot_summary_metrics(
        performance: Dict, output_dir: str, agent_name: str
    ) -> None:
        """Plot summary performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        metrics = [
            ("avg_total_reward", "Total Reward"),
            ("avg_ground_comfort_pct", "Ground Floor Comfort %"),
            ("avg_top_comfort_pct", "Top Floor Comfort %"),
            ("avg_light_hours", "Light Hours"),
        ]

        for i, (metric, label) in enumerate(metrics):
            row, col = i // 2, i % 2
            value = performance.get(metric, 0)

            bar = axes[row, col].bar([agent_name], [value])
            axes[row, col].set_title(label)
            axes[row, col].grid(True, alpha=0.3)

            # Add value label on bar
            axes[row, col].text(
                0, value + 0.1, f"{value:.2f}", ha="center", va="bottom"
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir, f'{agent_name.lower().replace(" ", "_")}_summary.png'
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    @staticmethod
    def _plot_temperature_distributions(
        performance: Dict, output_dir: str, agent_name: str
    ) -> None:
        """Plot temperature distribution analysis."""
        episode_data = performance["episode_data"]
        has_dynamic_setpoints = performance.get("has_dynamic_setpoints", False)

        # Combine all temperature data across episodes
        all_ground_temps = []
        all_top_temps = []
        all_heating_setpoints = []
        all_cooling_setpoints = []

        for episode_idx, episode_temps in enumerate(episode_data["temperatures"]):
            all_ground_temps.extend([temp[0] for temp in episode_temps])
            all_top_temps.extend([temp[1] for temp in episode_temps])

            if has_dynamic_setpoints and episode_idx < len(episode_data["setpoints"]):
                episode_setpoints = episode_data["setpoints"][episode_idx]
                all_heating_setpoints.extend([sp[0] for sp in episode_setpoints])
                all_cooling_setpoints.extend([sp[1] for sp in episode_setpoints])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Ground floor temperature distribution
        ax1.hist(all_ground_temps, bins=30, alpha=0.7, edgecolor="black")

        if has_dynamic_setpoints and all_heating_setpoints:
            min_heating = min(all_heating_setpoints)
            max_heating = max(all_heating_setpoints)
            min_cooling = min(all_cooling_setpoints)
            max_cooling = max(all_cooling_setpoints)

            ax1.axvspan(
                min_heating,
                max_heating,
                alpha=0.2,
                color="red",
                label=f"Heating Range ({min_heating:.1f}-{max_heating:.1f}°C)",
            )
            ax1.axvspan(
                min_cooling,
                max_cooling,
                alpha=0.2,
                color="blue",
                label=f"Cooling Range ({min_cooling:.1f}-{max_cooling:.1f}°C)",
            )
        else:
            heating_sp = performance.get("heating_setpoint", 20.0)
            cooling_sp = performance.get("cooling_setpoint", 24.0)
            ax1.axvline(
                x=heating_sp,
                color="r",
                linestyle="--",
                label=f"Heating Setpoint ({heating_sp}°C)",
            )
            ax1.axvline(
                x=cooling_sp,
                color="b",
                linestyle="--",
                label=f"Cooling Setpoint ({cooling_sp}°C)",
            )

        ax1.set_title(f"{agent_name} - Ground Floor Temperature Distribution")
        ax1.set_xlabel("Temperature (°C)")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Top floor temperature distribution
        ax2.hist(all_top_temps, bins=30, alpha=0.7, edgecolor="black")

        if has_dynamic_setpoints and all_heating_setpoints:
            ax2.axvspan(
                min_heating,
                max_heating,
                alpha=0.2,
                color="red",
                label=f"Heating Range ({min_heating:.1f}-{max_heating:.1f}°C)",
            )
            ax2.axvspan(
                min_cooling,
                max_cooling,
                alpha=0.2,
                color="blue",
                label=f"Cooling Range ({min_cooling:.1f}-{max_cooling:.1f}°C)",
            )
        else:
            heating_sp = performance.get("heating_setpoint", 20.0)
            cooling_sp = performance.get("cooling_setpoint", 24.0)
            ax2.axvline(
                x=heating_sp,
                color="r",
                linestyle="--",
                label=f"Heating Setpoint ({heating_sp}°C)",
            )
            ax2.axvline(
                x=cooling_sp,
                color="b",
                linestyle="--",
                label=f"Cooling Setpoint ({cooling_sp}°C)",
            )

        ax2.set_title(f"{agent_name} - Top Floor Temperature Distribution")
        ax2.set_xlabel("Temperature (°C)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f'{agent_name.lower().replace(" ", "_")}_temperature_distribution.png',
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def evaluate_rl_agent(
    model_path: str,
    data_file: str,
    env_params_path: Optional[str] = None,
    num_episodes: int = 5,
    render: bool = False,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    deterministic: bool = True,
) -> Dict:
    """
    Complete pipeline for evaluating a trained RL agent.

    Args:
        model_path: Path to the trained model
        data_file: Path to data file for SINDy model training
        env_params_path: Path to saved environment parameters
        num_episodes: Number of episodes to evaluate
        render: Whether to render during evaluation
        output_dir: Directory to save results
        verbose: Whether to print detailed logs
        deterministic: Whether to use deterministic policy

    Returns:
        Dictionary containing comprehensive evaluation results
    """
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"eval_results/{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, algorithm = ModelLoader.load_model(model_path)
    model_dir = os.path.dirname(model_path)

    # Find environment parameters if not provided
    if env_params_path is None:
        env_params_path = _find_env_params_path(model_dir)

    # Recreate environment
    env = ModelLoader.recreate_environment(env_params_path, data_file, model_dir)

    # Print configuration
    base_env = (
        env.venv.envs[0]
        if isinstance(env, VecNormalize)
        else (env.envs[0] if isinstance(env, DummyVecEnv) else env)
    )

    if verbose:
        print(f"\nEnvironment Configuration:")
        print(f"Setpoint Pattern: {base_env.setpoint_pattern}")
        print(f"Base Heating Setpoint: {base_env.initial_heating_setpoint}")
        print(f"Base Cooling Setpoint: {base_env.initial_cooling_setpoint}")
        print(
            f"Using {'normalized' if isinstance(env, VecNormalize) else 'standard'} environment"
        )

    # Create evaluator and run evaluation
    model_info = {
        "algorithm": algorithm.upper(),
        "model_path": model_path,
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    evaluator = RLAgentEvaluator(env, model, model_info)
    performance = evaluator.evaluate(
        num_episodes=num_episodes,
        render=render,
        deterministic=deterministic,
        verbose=verbose,
    )

    # Save results
    results_path = os.path.join(output_dir, f"{algorithm}_evaluation_results.json")
    _save_evaluation_results(performance, results_path)

    # Save environment results
    base_env.save_results(
        os.path.join(output_dir, f"{algorithm}_environment_results.json"),
        controller_name=f"{algorithm.upper()} Agent",
    )

    # Create visualizations
    VisualizationEngine.create_evaluation_plots(
        performance, output_dir, f"{algorithm.upper()} Agent"
    )

    if verbose:
        print(f"\nEvaluation completed. Results saved to {output_dir}")

    return performance


def _find_env_params_path(model_dir: str) -> str:
    """Find environment parameters file in model directory hierarchy."""
    potential_paths = [
        os.path.join(model_dir, "..", "..", "env_params.json"),
        os.path.join(model_dir, "..", "env_params.json"),
        os.path.join(os.path.dirname(model_dir), "env_params.json"),
    ]

    for path in potential_paths:
        if os.path.exists(path):
            print(f"Found environment parameters at {path}")
            return path

    raise FileNotFoundError("env_params.json not found in model directory hierarchy")


def _save_evaluation_results(performance: Dict, filepath: str) -> None:
    """Save evaluation results with proper JSON serialization."""
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
                "setpoints": (
                    [
                        [[float(sp) for sp in setpoint] for setpoint in episode]
                        for episode in value.get("setpoints", [])
                    ]
                    if "setpoints" in value
                    else []
                ),
            }
        elif isinstance(value, (np.integer, np.floating)):
            serializable_perf[key] = (
                float(value) if isinstance(value, np.floating) else int(value)
            )
        elif isinstance(value, np.ndarray):
            serializable_perf[key] = value.tolist()
        else:
            serializable_perf[key] = value

    with open(filepath, "w") as f:
        json.dump(serializable_perf, f, indent=4)

    print(f"Evaluation results saved to {filepath}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agent on dollhouse environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Path to saved environment parameters",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render environment during evaluation"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Directory to save evaluation results"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed logs")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy instead of deterministic",
    )

    args = parser.parse_args()

    # Run evaluation
    performance = evaluate_rl_agent(
        model_path=args.model,
        data_file=args.data,
        env_params_path=args.env_params,
        num_episodes=args.episodes,
        render=args.render,
        output_dir=args.output,
        verbose=not args.quiet,
        deterministic=not args.stochastic,
    )

    return performance


if __name__ == "__main__":
    main()
