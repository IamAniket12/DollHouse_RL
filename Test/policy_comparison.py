import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime


def create_rule_based_controller(heating_setpoint, cooling_setpoint, hysteresis=0.5):
    """
    Create a simple rule-based controller.
    """

    def controller(observation):
        ground_temp = observation[0]
        top_temp = observation[1]
        action = np.zeros(4, dtype=int)
        avg_setpoint = (heating_setpoint + cooling_setpoint) / 2

        if ground_temp < avg_setpoint - hysteresis:
            action[0], action[1] = 1, 0  # Light ON, Window OFF
        else:
            action[0], action[1] = 0, 1  # Light OFF, Window ON

        if top_temp < avg_setpoint - hysteresis:
            action[2], action[3] = 1, 0  # Light ON, Window OFF
        else:
            action[2], action[3] = 0, 1  # Light OFF, Window ON

        return action

    return controller


def create_observation(ground_temp, top_temp):
    """
    Create observation vector for the environment.
    """
    obs = np.zeros(11)
    obs[0], obs[1], obs[2] = ground_temp, top_temp, 22.0  # Temperatures
    obs[7], obs[8] = 26.0, 28.0  # Heating and cooling setpoints
    return obs.astype(np.float32)


def record_policy_actions(
    controller, temp_range, output_file, is_rl_model=False, deterministic=True
):
    """
    Record actions from a policy across a range of temperatures.
    """
    data = []
    for ground_temp in temp_range:
        for top_temp in temp_range:
            obs = create_observation(ground_temp, top_temp)
            if is_rl_model:
                action, _ = controller.predict(obs, deterministic=deterministic)
            else:
                action = controller(obs)
            # Round temperatures to 2 decimal places before saving
            data.append([round(ground_temp, 2), round(top_temp, 2)] + action.tolist())

    df = pd.DataFrame(
        data,
        columns=[
            "GroundTemp",
            "TopTemp",
            "GroundLight",
            "GroundWindow",
            "TopLight",
            "TopWindow",
        ],
    )
    df.to_csv(output_file, index=False)
    print(f"Actions recorded in {output_file}")


def visualize_policy(csv_files, labels, temp_range, output_dir, save_format="pdf"):
    """
    Visualize and compare policies from CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)

        # Temperature values should already be rounded in the CSV,
        # but ensure they are rounded to 2 decimal places just in case
        df["GroundTemp"] = df["GroundTemp"].round(2)
        df["TopTemp"] = df["TopTemp"].round(2)

        # Individual action heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        actions = ["GroundLight", "GroundWindow", "TopLight", "TopWindow"]
        titles = [
            "Ground Floor Light",
            "Ground Floor Window",
            "Top Floor Light",
            "Top Floor Window",
        ]

        for i, ax in enumerate(axes.flat):
            # Create pivot with rounded temperature values
            pivot = df.pivot(index="TopTemp", columns="GroundTemp", values=actions[i])

            # Create custom tick labels - show every nth label to reduce clutter
            n_ticks = 10  # Show approximately 10 ticks on each axis
            x_indices = np.linspace(0, len(pivot.columns) - 1, n_ticks, dtype=int)
            y_indices = np.linspace(0, len(pivot.index) - 1, n_ticks, dtype=int)

            x_labels = [f"{pivot.columns[i]:.1f}" for i in x_indices]
            y_labels = [f"{pivot.index[i]:.1f}" for i in y_indices]

            # Custom colormap for binary actions
            cmap = plt.cm.RdBu_r

            sns.heatmap(
                pivot,
                ax=ax,
                cmap=cmap,
                cbar=True,
                xticklabels=False,  # We'll set custom labels
                yticklabels=False,  # We'll set custom labels
                vmin=0,
                vmax=1,
                cbar_kws={"ticks": [0, 1], "label": "Action State"},
            )

            # Custom colorbar labels
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([0, 1])
            colorbar.set_ticklabels(["OFF", "ON"])

            # Set custom tick positions and labels
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels)
            ax.set_yticks(y_indices)
            ax.set_yticklabels(y_labels)

            ax.set_title(f"{titles[i]} - {label}")
            ax.set_xlabel("Ground Floor Temperature (°C)")
            ax.set_ylabel("Top Floor Temperature (°C)")

        plt.tight_layout()
        output_file = os.path.join(
            output_dir, f"{label}_individual_actions.{save_format}"
        )
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"Individual actions visualization saved to {output_file}")

        # Combined actions visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Ground floor combined (Light=1, Window=0 -> 'Heating'; Light=0, Window=1 -> 'Cooling')
        df["GroundAction"] = (
            df["GroundLight"] * 2 + df["GroundWindow"]
        )  # 2=Light only, 1=Window only, 0=neither, 3=both
        ground_pivot = df.pivot(
            index="TopTemp", columns="GroundTemp", values="GroundAction"
        )

        # Top floor combined
        df["TopAction"] = (
            df["TopLight"] * 2 + df["TopWindow"]
        )  # 2=Light only, 1=Window only, 0=neither, 3=both
        top_pivot = df.pivot(index="TopTemp", columns="GroundTemp", values="TopAction")

        # Create custom colormap for combined actions
        colors = [
            "white",
            "lightblue",
            "lightcoral",
            "purple",
        ]  # 0=none, 1=window, 2=light, 3=both
        from matplotlib.colors import ListedColormap

        cmap_combined = ListedColormap(colors)

        for i, (pivot, floor_name) in enumerate(
            [(ground_pivot, "Ground"), (top_pivot, "Top")]
        ):
            ax = axes[i]

            # Create custom tick labels
            x_indices = np.linspace(0, len(pivot.columns) - 1, n_ticks, dtype=int)
            y_indices = np.linspace(0, len(pivot.index) - 1, n_ticks, dtype=int)

            x_labels = [f"{pivot.columns[j]:.1f}" for j in x_indices]
            y_labels = [f"{pivot.index[j]:.1f}" for j in y_indices]

            im = sns.heatmap(
                pivot,
                ax=ax,
                cmap=cmap_combined,
                cbar=True,
                xticklabels=False,
                yticklabels=False,
                vmin=0,
                vmax=3,
                cbar_kws={"ticks": [0, 1, 2, 3], "label": "Action Combination"},
            )

            # Custom colorbar labels
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks(
                [0.375, 1.125, 1.875, 2.625]
            )  # Center the labels in each color segment
            colorbar.set_ticklabels(["None", "Window Only", "Light Only", "Both"])

            # Set custom tick positions and labels
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels)
            ax.set_yticks(y_indices)
            ax.set_yticklabels(y_labels)

            ax.set_title(f"{floor_name} Floor Combined Actions - {label}")
            ax.set_xlabel("Ground Floor Temperature (°C)")
            ax.set_ylabel("Top Floor Temperature (°C)")

        plt.tight_layout()
        output_file = os.path.join(
            output_dir, f"{label}_combined_actions.{save_format}"
        )
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"Combined actions visualization saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Record and visualize policy actions.")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained PPO model"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png"],
        help="Visualization format",
    )
    parser.add_argument(
        "--resolution", type=int, default=50, help="Temperature grid resolution"
    )
    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"policy_comparison_{timestamp}"
    else:
        output_dir = args.output

    # Ensure the output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created: {output_dir}")
    except Exception as e:
        print(f"Failed to create output directory: {output_dir}")
        print(f"Error: {e}")
        return

    temp_range = np.linspace(20.0, 30.0, args.resolution)

    # Rule-based controller
    rule_based = create_rule_based_controller(26.0, 28.0)
    rule_based_file = os.path.join(output_dir, "rule_based_actions.csv")
    record_policy_actions(rule_based, temp_range, rule_based_file)

    # PPO model
    print(f"Loading PPO model from {args.model}...")
    ppo_model = PPO.load(args.model)

    ppo_stochastic_file = os.path.join(output_dir, "ppo_stochastic_actions.csv")
    record_policy_actions(
        ppo_model,
        temp_range,
        ppo_stochastic_file,
        is_rl_model=True,
        deterministic=False,
    )

    ppo_deterministic_file = os.path.join(output_dir, "ppo_deterministic_actions.csv")
    record_policy_actions(
        ppo_model,
        temp_range,
        ppo_deterministic_file,
        is_rl_model=True,
        deterministic=True,
    )

    # Visualize policies
    visualize_policy(
        [rule_based_file, ppo_stochastic_file, ppo_deterministic_file],
        ["Rule-Based", "PPO Stochastic", "PPO Deterministic"],
        temp_range,
        output_dir,
        args.format,
    )


if __name__ == "__main__":
    main()
# python policy_comparison.py --model "../Environment/results/ppo_20250513_151705/logs/models/ppo_final_model" --format pdf --resolution 100
