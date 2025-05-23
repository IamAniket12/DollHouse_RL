import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import argparse
from datetime import datetime


def create_rule_based_controller(heating_setpoint, cooling_setpoint, hysteresis=0.5):
    """
    Create a simple rule-based controller.

    Args:
        heating_setpoint: Heating setpoint temperature
        cooling_setpoint: Cooling setpoint temperature
        hysteresis: Temperature buffer to prevent oscillation

    Returns:
        function: Rule-based controller function
    """

    def controller(observation):
        # Extract state variables
        ground_temp = observation[0]
        top_temp = observation[1]

        # Initialize action: [ground_light, ground_window, top_light, top_window]
        action = np.zeros(4, dtype=int)

        # Average setpoint
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


def create_observation(ground_temp, top_temp, current_action=None):
    """
    Create observation vector matching the training environment format.

    Observation format:
    [ground_temp, top_temp, external_temp, ground_light, ground_window, top_light, top_window,
     heating_setpoint, cooling_setpoint, hour_of_day, time_step]

    Args:
        ground_temp: Ground floor temperature
        top_temp: Top floor temperature
        current_action: Previous action (if available)

    Returns:
        np.array: Observation vector of size 11
    """
    obs = np.zeros(11)

    # Set temperatures
    obs[0] = ground_temp  # Ground floor temperature
    obs[1] = top_temp  # Top floor temperature
    obs[2] = 22.0  # External temperature (fixed for policy visualization)

    # Set previous action states (or default to 0 if not available)
    if current_action is not None:
        obs[3] = current_action[0]  # Previous ground light state
        obs[4] = current_action[1]  # Previous ground window state
        obs[5] = current_action[2]  # Previous top light state
        obs[6] = current_action[3]  # Previous top window state
    else:
        obs[3] = 0.0  # Ground light state
        obs[4] = 0.0  # Ground window state
        obs[5] = 0.0  # Top light state
        obs[6] = 0.0  # Top window state

    # Set setpoints and time
    obs[7] = 26.0  # Heating setpoint
    obs[8] = 28.0  # Cooling setpoint
    obs[9] = 12.0  # Hour of day (fixed for policy visualization)
    obs[10] = 500.0  # Time step (fixed for policy visualization)

    return obs.astype(np.float32)


def calculate_action_combination(action):
    """
    Calculate action combination code for a single floor.

    Args:
        action: [light_state, window_state]

    Returns:
        int: Action combination code (0-3)
    """
    light, window = action[0], action[1]

    if light == 0 and window == 0:
        return 0  # Both OFF
    elif light == 1 and window == 0:
        return 1  # Light ON, Window OFF
    elif light == 0 and window == 1:
        return 2  # Light OFF, Window ON
    else:  # light == 1 and window == 1
        return 3  # Both ON


def generate_policy_grid(controller, temp_range, is_rl_model=False, deterministic=True):
    """
    Generate a policy grid for a controller across a range of temperatures.

    Args:
        controller: Controller function or model
        temp_range: Range of temperatures to evaluate
        is_rl_model: Whether the controller is an RL model
        deterministic: Whether to use deterministic actions (for RL model)

    Returns:
        Ground actions grid and Top actions grid (2D arrays of action combinations)
    """
    n = len(temp_range)
    ground_actions = np.zeros((n, n))
    top_actions = np.zeros((n, n))

    # For each temperature combination
    for i, top_temp in enumerate(temp_range):
        for j, ground_temp in enumerate(temp_range):
            # Create observation
            if is_rl_model:
                obs = create_observation(ground_temp, top_temp)
            else:
                # For rule-based controller, use simple observation
                obs = np.zeros(11)
                obs[0] = ground_temp
                obs[1] = top_temp
                obs[2] = 25.0  # External temp
                obs[7] = 26.0  # Heating setpoint
                obs[8] = 28.0  # Cooling setpoint

            # Get actions
            if is_rl_model:
                action, _ = controller.predict(obs, deterministic=deterministic)
            else:
                action = controller(obs)

            # Calculate action combinations for each floor
            ground_actions[i, j] = calculate_action_combination([action[0], action[1]])
            top_actions[i, j] = calculate_action_combination([action[2], action[3]])

    return ground_actions, top_actions


def visualize_policies(
    rule_based_ground,
    rule_based_top,
    ppo_stochastic_ground,
    ppo_stochastic_top,
    ppo_deterministic_ground,
    ppo_deterministic_top,
    temp_range,
    output_dir,
    save_format="pdf",
):
    """
    Visualize and compare the controller policies with raw action values.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define colors for raw actions
    colors = [
        "#440154",  # Light OFF, Window OFF
        "#31688e",  # Light ON, Window OFF
        "#35b779",  # Light OFF, Window ON
        "#fde725",  # Light ON, Window ON
    ]
    cmap = ListedColormap(colors)

    # Create meshgrid for plotting
    X, Y = np.meshgrid(temp_range, temp_range)

    # Action labels
    action_labels = [
        "Light OFF, Window OFF",
        "Light ON, Window OFF",
        "Light OFF, Window ON",
        "Light ON, Window ON",
    ]

    # Create legend elements
    legend_elements = [
        Patch(facecolor=colors[i], label=action_labels[i]) for i in range(len(colors))
    ]

    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )

    # Function to create a single policy plot
    def create_policy_plot(ground_policy, top_policy, title_prefix):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Ground floor plot
        im1 = ax1.contourf(
            X,
            Y,
            ground_policy,
            cmap=cmap,
            alpha=0.9,
            levels=np.arange(0, len(colors) + 1),
        )
        cbar1 = plt.colorbar(
            im1, ax=ax1, ticks=np.arange(len(colors)), fraction=0.046, pad=0.04
        )
        cbar1.set_label("Raw Action", rotation=270, labelpad=15)

        ax1.set_title(f"{title_prefix} - Ground Floor Policy")
        ax1.set_xlabel("Ground Floor Temperature (°C)")
        ax1.set_ylabel("Top Floor Temperature (°C)")
        ax1.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 1.0)
        )
        ax1.grid(True, alpha=0.3)

        # Top floor plot
        im2 = ax2.contourf(
            X, Y, top_policy, cmap=cmap, alpha=0.9, levels=np.arange(0, len(colors) + 1)
        )
        cbar2 = plt.colorbar(
            im2, ax=ax2, ticks=np.arange(len(colors)), fraction=0.046, pad=0.04
        )
        cbar2.set_label("Raw Action", rotation=270, labelpad=15)

        ax2.set_title(f"{title_prefix} - Top Floor Policy")
        ax2.set_xlabel("Ground Floor Temperature (°C)")
        ax2.set_ylabel("Top Floor Temperature (°C)")
        ax2.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 1.0)
        )
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # Ensure raw actions are correctly interpreted
    def adjust_actions(policy_grid):
        """
        Adjust the raw actions to align with the rule-based controller's logic.
        """
        # Example: Flip or modify actions if needed to match expected behavior
        return policy_grid  # Modify this if necessary

    # Adjust policies if needed
    rule_based_ground = adjust_actions(rule_based_ground)
    rule_based_top = adjust_actions(rule_based_top)
    ppo_stochastic_ground = adjust_actions(ppo_stochastic_ground)
    ppo_stochastic_top = adjust_actions(ppo_stochastic_top)
    ppo_deterministic_ground = adjust_actions(ppo_deterministic_ground)
    ppo_deterministic_top = adjust_actions(ppo_deterministic_top)

    # Generate plots for each controller
    controllers = [
        (rule_based_ground, rule_based_top, "Rule-Based Controller", "rule_based"),
        (
            ppo_stochastic_ground,
            ppo_stochastic_top,
            "PPO Controller (Stochastic)",
            "ppo_stochastic",
        ),
        (
            ppo_deterministic_ground,
            ppo_deterministic_top,
            "PPO Controller (Deterministic)",
            "ppo_deterministic",
        ),
    ]

    for ground_policy, top_policy, title, filename in controllers:
        fig = create_policy_plot(ground_policy, top_policy, title)
        fig.savefig(
            os.path.join(output_dir, f"{filename}_policy_map.{save_format}"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)
        print(f"Saved {title} policy map")

    print(f"All policy visualizations saved to {output_dir}")


def main():
    """
    Main function to run the policy comparison.
    """
    parser = argparse.ArgumentParser(
        description="Compare rule-based and PPO controller policies"
    )

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
        choices=["pdf", "png", "svg", "jpg"],
        help="Format to save visualizations",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=50,
        help="Resolution of the temperature grid (number of points)",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"policy_comparison_{timestamp}"
    else:
        output_dir = args.output

    # Generate temperature range
    temp_range = np.linspace(20.0, 30.0, args.resolution)

    # Create rule-based controller
    heating_setpoint = 26.0
    cooling_setpoint = 28.0
    rule_based = create_rule_based_controller(heating_setpoint, cooling_setpoint)

    # Load PPO model
    print(f"Loading PPO model from {args.model}...")
    ppo_model = PPO.load(args.model)

    # Generate policy grids
    print("Generating rule-based controller policy...")
    rule_based_ground, rule_based_top = generate_policy_grid(rule_based, temp_range)

    print("Generating PPO controller policy (stochastic)...")
    ppo_stochastic_ground, ppo_stochastic_top = generate_policy_grid(
        ppo_model, temp_range, is_rl_model=True, deterministic=False
    )

    print("Generating PPO controller policy (deterministic)...")
    ppo_deterministic_ground, ppo_deterministic_top = generate_policy_grid(
        ppo_model, temp_range, is_rl_model=True, deterministic=True
    )

    # Visualize policies
    print("Creating policy visualizations...")
    visualize_policies(
        rule_based_ground,
        rule_based_top,
        ppo_stochastic_ground,
        ppo_stochastic_top,
        ppo_deterministic_ground,
        ppo_deterministic_top,
        temp_range,
        output_dir,
        args.format,
    )

    print(f"Policy comparison completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
# python temp.py --model "../Environment/results/ppo_20250513_151705/logs/models/ppo_final_model" --format pdf --resolution 100
