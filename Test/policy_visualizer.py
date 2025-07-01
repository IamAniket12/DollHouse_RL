import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import argparse
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Interactive plots will be skipped.")


class PolicyVisualizer:
    """
    Class for creating interpretable 2D and 3D policy visualizations from logged episode data.
    """

    def __init__(self, csv_path, metadata_path=None, output_dir="policy_plots"):
        """
        Initialize the visualizer with episode data.

        Args:
            csv_path: Path to the CSV file with episode data
            metadata_path: Path to the metadata JSON file (optional)
            output_dir: Directory to save generated plots
        """
        self.csv_path = csv_path
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        print(f"Loading data from {csv_path}")
        self.data = pd.read_csv(csv_path)

        # Load metadata if available
        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

        # Extract controller info for plot titles
        self.controller_name = self._get_controller_name()

        # Validate required columns
        self._validate_data()

        print(f"Loaded episode data: {len(self.data)} steps")
        print(f"Controller: {self.controller_name}")
        if self.metadata:
            print(f"Total reward: {self.metadata.get('total_reward', 'N/A')}")

    def _get_controller_name(self):
        """Extract controller name for plot titles."""
        if self.metadata:
            controller_type = self.metadata.get("controller_type", "Unknown")
            if controller_type == "rule_based":
                hysteresis = self.metadata.get("hysteresis", "N/A")
                return f"Rule-based (hysteresis={hysteresis})"
            elif controller_type == "rl_model":
                model_path = self.metadata.get("model_path", "")
                model_name = (
                    os.path.basename(model_path).replace(".zip", "")
                    if model_path
                    else "RL Model"
                )
                return f"{model_name}"
            else:
                return controller_type
        else:
            # Try to infer from filename
            filename = os.path.basename(self.csv_path)
            if "rule_based" in filename.lower():
                return "Rule-based Controller"
            elif any(alg in filename.lower() for alg in ["ppo", "a2c", "dqn", "sac"]):
                return "RL Controller"
            else:
                return "Unknown Controller"

    def _validate_data(self):
        """Validate that required columns exist in the data."""
        required_cols = [
            "ground_temp",
            "top_temp",
            "external_temp",
            "action_ground_light",
            "action_ground_window",
            "action_top_light",
            "action_top_window",
            "reward",
        ]

        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add derived columns if not present
        if "lights_on" not in self.data.columns:
            self.data["lights_on"] = (
                self.data["action_ground_light"] + self.data["action_top_light"]
            )

        if "windows_open" not in self.data.columns:
            self.data["windows_open"] = (
                self.data["action_ground_window"] + self.data["action_top_window"]
            )

        if "avg_temp" not in self.data.columns:
            self.data["avg_temp"] = (
                self.data["ground_temp"] + self.data["top_temp"]
            ) / 2

        if "temp_difference" not in self.data.columns:
            self.data["temp_difference"] = (
                self.data["top_temp"] - self.data["ground_temp"]
            )

        # Add setpoint-related columns if setpoints exist
        if (
            "heating_setpoint" in self.data.columns
            and "cooling_setpoint" in self.data.columns
        ):
            if "avg_setpoint" not in self.data.columns:
                self.data["avg_setpoint"] = (
                    self.data["heating_setpoint"] + self.data["cooling_setpoint"]
                ) / 2

            if "ground_temp_deviation" not in self.data.columns:
                self.data["ground_temp_deviation"] = (
                    self.data["ground_temp"] - self.data["avg_setpoint"]
                )

            if "top_temp_deviation" not in self.data.columns:
                self.data["top_temp_deviation"] = (
                    self.data["top_temp"] - self.data["avg_setpoint"]
                )

        print("Data validation completed successfully")

    def create_2d_policy_plots(self):
        """Create comprehensive 2D policy visualization plots."""
        print("Creating 2D policy plots...")

        # Set up the plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Temperature vs Actions Heat Map
        self._plot_temperature_action_heatmap()

        # 2. Action Patterns Over Time
        self._plot_action_patterns_over_time()

        # 3. Decision Boundaries
        self._plot_decision_boundaries()

        # 4. State-Action Correlation Matrix
        self._plot_state_action_correlations()

        # 5. Policy Response to Temperature Deviations (if setpoints available)
        if "ground_temp_deviation" in self.data.columns:
            self._plot_temperature_deviation_response()

        # 6. Multi-dimensional State-Action View
        self._plot_multidimensional_policy()

        print(f"2D plots saved to {self.output_dir}")

    def _plot_temperature_action_heatmap(self):
        """Plot heatmaps showing action probabilities for different temperature conditions."""
        print("Creating temperature-action heatmaps...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Action Policy Heatmaps - {self.controller_name}", fontsize=16)

        # Define temperature bins
        n_bins = 12
        ground_bins = np.linspace(
            self.data["ground_temp"].min(), self.data["ground_temp"].max(), n_bins
        )
        top_bins = np.linspace(
            self.data["top_temp"].min(), self.data["top_temp"].max(), n_bins
        )

        # Create binned temperature data
        self.data["ground_temp_bin"] = pd.cut(self.data["ground_temp"], ground_bins)
        self.data["top_temp_bin"] = pd.cut(self.data["top_temp"], top_bins)

        # Actions to plot
        actions = [
            ("action_ground_light", "Ground Light", "Reds"),
            ("action_ground_window", "Ground Window", "Blues"),
            ("action_top_light", "Top Light", "Oranges"),
            ("action_top_window", "Top Window", "Greens"),
        ]

        for idx, (action_col, action_name, cmap) in enumerate(actions):
            ax = axes[idx // 2, idx % 2]

            try:
                # Create pivot table for heatmap
                pivot_data = (
                    self.data.groupby(["ground_temp_bin", "top_temp_bin"])[action_col]
                    .mean()
                    .unstack()
                )

                # Create heatmap
                im = ax.imshow(
                    pivot_data.values, cmap=cmap, aspect="auto", origin="lower"
                )

                # Set labels
                ax.set_title(f"{action_name} Policy")
                ax.set_xlabel("Top Floor Temperature (°C)")
                ax.set_ylabel("Ground Floor Temperature (°C)")

                # Improve tick labels
                n_ticks = 5
                x_tick_positions = np.linspace(
                    0, len(pivot_data.columns) - 1, n_ticks
                ).astype(int)
                y_tick_positions = np.linspace(
                    0, len(pivot_data.index) - 1, n_ticks
                ).astype(int)

                x_labels = [
                    f"{pivot_data.columns[i].mid:.1f}" for i in x_tick_positions
                ]
                y_labels = [f"{pivot_data.index[i].mid:.1f}" for i in y_tick_positions]

                ax.set_xticks(x_tick_positions)
                ax.set_xticklabels(x_labels)
                ax.set_yticks(y_tick_positions)
                ax.set_yticklabels(y_labels)

                # Add colorbar
                plt.colorbar(im, ax=ax, label="Action Probability")

            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"Error creating heatmap\n{str(e)}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{action_name} Policy (Error)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "2d_temperature_action_heatmap.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_action_patterns_over_time(self):
        """Plot action patterns over time with temperature context."""
        print("Creating action patterns over time...")

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle(f"Action Patterns Over Time - {self.controller_name}", fontsize=16)

        # Create time axis (try different time representations)
        if "hour_of_day" in self.data.columns:
            time_axis = self.data["hour_of_day"]
            time_label = "Hour of Day"
        elif "step" in self.data.columns:
            time_axis = (
                self.data["step"] * 30 / 3600
            )  # Convert to hours assuming 30-second steps
            time_label = "Time (hours)"
        else:
            time_axis = np.arange(len(self.data)) * 30 / 3600
            time_label = "Time (hours)"

        # Temperature plot
        axes[0].plot(
            time_axis,
            self.data["ground_temp"],
            label="Ground Floor",
            linewidth=2,
            alpha=0.8,
        )
        axes[0].plot(
            time_axis, self.data["top_temp"], label="Top Floor", linewidth=2, alpha=0.8
        )
        axes[0].plot(
            time_axis,
            self.data["external_temp"],
            label="External",
            linewidth=2,
            alpha=0.6,
        )

        # Add setpoint bands if available
        if (
            "heating_setpoint" in self.data.columns
            and "cooling_setpoint" in self.data.columns
        ):
            axes[0].fill_between(
                time_axis,
                self.data["heating_setpoint"],
                self.data["cooling_setpoint"],
                alpha=0.2,
                color="gray",
                label="Comfort Zone",
            )

        axes[0].set_ylabel("Temperature (°C)")
        axes[0].set_title("Temperature Evolution")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Action plot - Ground floor
        axes[1].fill_between(
            time_axis,
            0,
            self.data["action_ground_light"],
            alpha=0.7,
            label="Ground Light",
            step="pre",
        )
        axes[1].fill_between(
            time_axis,
            1,
            1 + self.data["action_ground_window"],
            alpha=0.7,
            label="Ground Window",
            step="pre",
        )
        axes[1].set_ylabel("Ground Floor Actions")
        axes[1].set_ylim(-0.1, 2.1)
        axes[1].set_yticks([0.5, 1.5])
        axes[1].set_yticklabels(["Light", "Window"])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Action plot - Top floor
        axes[2].fill_between(
            time_axis,
            0,
            self.data["action_top_light"],
            alpha=0.7,
            label="Top Light",
            step="pre",
        )
        axes[2].fill_between(
            time_axis,
            1,
            1 + self.data["action_top_window"],
            alpha=0.7,
            label="Top Window",
            step="pre",
        )
        axes[2].set_ylabel("Top Floor Actions")
        axes[2].set_xlabel(time_label)
        axes[2].set_ylim(-0.1, 2.1)
        axes[2].set_yticks([0.5, 1.5])
        axes[2].set_yticklabels(["Light", "Window"])
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "2d_action_patterns_time.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_decision_boundaries(self):
        """Plot decision boundaries in temperature space."""
        print("Creating decision boundary plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Decision Boundaries - {self.controller_name}", fontsize=16)

        actions = [
            ("action_ground_light", "Ground Light Decision"),
            ("action_ground_window", "Ground Window Decision"),
            ("action_top_light", "Top Light Decision"),
            ("action_top_window", "Top Window Decision"),
        ]

        for idx, (action_col, action_name) in enumerate(actions):
            ax = axes[idx // 2, idx % 2]

            try:
                # Simple scatter plot approach (more reliable)
                action_0 = self.data[self.data[action_col] == 0]
                action_1 = self.data[self.data[action_col] == 1]

                if len(action_0) > 0:
                    ax.scatter(
                        action_0["ground_temp"],
                        action_0["top_temp"],
                        c="blue",
                        s=15,
                        alpha=0.4,
                        label="Action=0 (OFF/CLOSED)",
                    )

                if len(action_1) > 0:
                    ax.scatter(
                        action_1["ground_temp"],
                        action_1["top_temp"],
                        c="red",
                        s=15,
                        alpha=0.4,
                        label="Action=1 (ON/OPEN)",
                    )

                ax.set_xlabel("Ground Floor Temperature (°C)")
                ax.set_ylabel("Top Floor Temperature (°C)")
                ax.set_title(action_name)
                ax.grid(True, alpha=0.3)
                ax.legend()

            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"Error creating plot\n{str(e)}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{action_name} (Error)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "2d_decision_boundaries.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_state_action_correlations(self):
        """Plot correlation matrix between states and actions."""
        print("Creating state-action correlation matrix...")

        # Select relevant columns for correlation analysis
        state_cols = [
            "ground_temp",
            "top_temp",
            "external_temp",
            "avg_temp",
            "temp_difference",
        ]
        action_cols = [
            "action_ground_light",
            "action_ground_window",
            "action_top_light",
            "action_top_window",
        ]

        # Add setpoint columns if available
        if "heating_setpoint" in self.data.columns:
            state_cols.extend(["heating_setpoint", "cooling_setpoint", "avg_setpoint"])

        if "ground_temp_deviation" in self.data.columns:
            state_cols.extend(["ground_temp_deviation", "top_temp_deviation"])

        # Create correlation matrix
        all_cols = [col for col in state_cols + action_cols if col in self.data.columns]
        corr_matrix = self.data[all_cols].corr()

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"label": "Correlation"},
        )
        plt.title(f"State-Action Correlation Matrix - {self.controller_name}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "2d_state_action_correlations.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_temperature_deviation_response(self):
        """Plot how the policy responds to temperature deviations from setpoints."""
        print("Creating temperature deviation response plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(
            f"Temperature Deviation Response - {self.controller_name}", fontsize=16
        )

        # Calculate deviations and create bins
        deviation_range = max(
            abs(self.data["ground_temp_deviation"].min()),
            abs(self.data["ground_temp_deviation"].max()),
            abs(self.data["top_temp_deviation"].min()),
            abs(self.data["top_temp_deviation"].max()),
        )

        deviation_bins = np.linspace(-deviation_range, deviation_range, 15)

        # Ground floor and top floor responses
        for idx, (temp_col, temp_name) in enumerate(
            [("ground_temp_deviation", "Ground"), ("top_temp_deviation", "Top")]
        ):
            # Bin deviations
            temp_data = self.data.copy()
            temp_data["deviation_bin"] = pd.cut(temp_data[temp_col], deviation_bins)

            # Light response
            ax = axes[idx, 0]
            try:
                light_response = (
                    temp_data.groupby("deviation_bin")[
                        f"action_{temp_name.lower()}_light"
                    ]
                    .agg(["mean", "count"])
                    .reset_index()
                )
                light_response = light_response[
                    light_response["count"] >= 3
                ]  # Filter out bins with too few samples
                light_response["deviation_mid"] = light_response["deviation_bin"].apply(
                    lambda x: x.mid
                )

                ax.plot(
                    light_response["deviation_mid"],
                    light_response["mean"],
                    marker="o",
                    linewidth=2,
                    markersize=6,
                )
                ax.set_xlabel(
                    f"{temp_name} Floor Temperature Deviation from Setpoint (°C)"
                )
                ax.set_ylabel("Light Action Probability")
                ax.set_title(f"{temp_name} Floor Light Response")
                ax.grid(True, alpha=0.3)
                ax.axvline(
                    x=0, color="red", linestyle="--", alpha=0.7, label="Setpoint"
                )
                ax.legend()
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"Error: {str(e)}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            # Window response
            ax = axes[idx, 1]
            try:
                window_response = (
                    temp_data.groupby("deviation_bin")[
                        f"action_{temp_name.lower()}_window"
                    ]
                    .agg(["mean", "count"])
                    .reset_index()
                )
                window_response = window_response[window_response["count"] >= 3]
                window_response["deviation_mid"] = window_response[
                    "deviation_bin"
                ].apply(lambda x: x.mid)

                ax.plot(
                    window_response["deviation_mid"],
                    window_response["mean"],
                    marker="s",
                    linewidth=2,
                    markersize=6,
                )
                ax.set_xlabel(
                    f"{temp_name} Floor Temperature Deviation from Setpoint (°C)"
                )
                ax.set_ylabel("Window Action Probability")
                ax.set_title(f"{temp_name} Floor Window Response")
                ax.grid(True, alpha=0.3)
                ax.axvline(
                    x=0, color="red", linestyle="--", alpha=0.7, label="Setpoint"
                )
                ax.legend()
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"Error: {str(e)}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "2d_temperature_deviation_response.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_multidimensional_policy(self):
        """Plot policy in multiple state dimensions simultaneously."""
        print("Creating multidimensional policy plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Multi-dimensional Policy View - {self.controller_name}", fontsize=16
        )

        # Define state pairs for analysis
        state_pairs = [
            ("ground_temp", "external_temp", "Ground vs External Temp"),
            ("top_temp", "external_temp", "Top vs External Temp"),
            ("temp_difference", "avg_temp", "Temp Difference vs Avg Temp"),
            ("external_temp", "avg_temp", "External vs Avg Temp"),
            ("ground_temp", "top_temp", "Ground vs Top Temp"),
            ("lights_on", "windows_open", "Lights vs Windows"),
        ]

        # Add hour-based plot if available
        if "hour_of_day" in self.data.columns:
            state_pairs[4] = ("hour_of_day", "avg_temp", "Hour of Day vs Avg Temp")

        for idx, (x_col, y_col, title) in enumerate(state_pairs):
            if idx >= 6:  # Only plot first 6
                break

            ax = axes[idx // 3, idx % 3]

            # Check if columns exist
            if x_col not in self.data.columns or y_col not in self.data.columns:
                ax.text(
                    0.5,
                    0.5,
                    f"Data not available\n{x_col}, {y_col}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(title)
                continue

            try:
                # Create combined action for coloring (lights_on + windows_open)
                if "combined_actions" not in self.data.columns:
                    self.data["combined_actions"] = (
                        self.data["lights_on"] + self.data["windows_open"]
                    )

                # Scatter plot with action coloring
                scatter = ax.scatter(
                    self.data[x_col],
                    self.data[y_col],
                    c=self.data["combined_actions"],
                    cmap="viridis",
                    s=20,
                    alpha=0.6,
                )

                ax.set_xlabel(x_col.replace("_", " ").title())
                ax.set_ylabel(y_col.replace("_", " ").title())
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label("Total Actions")

            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"Error: {str(e)}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{title} (Error)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "2d_multidimensional_policy.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def create_3d_policy_plots(self):
        """Create 3D policy visualization plots."""
        print("Creating 3D policy plots...")

        # 1. 3D Action Surface Plot
        self._plot_3d_action_surface()

        # 2. 3D Trajectory Plot
        self._plot_3d_trajectory()

        # 3. Interactive 3D Plotly Visualizations (if available)
        if PLOTLY_AVAILABLE:
            self._create_interactive_3d_plots()
        else:
            print("Skipping interactive 3D plots (Plotly not available)")

        print(f"3D plots saved to {self.output_dir}")

    def _plot_3d_action_surface(self):
        """Create 3D surface plots showing action policies."""
        print("Creating 3D action surface plots...")

        # Create 2x2 subplot for different actions
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"3D Action Policy Surfaces - {self.controller_name}", fontsize=16)

        actions = [
            ("action_ground_light", "Ground Light"),
            ("action_ground_window", "Ground Window"),
            ("action_top_light", "Top Light"),
            ("action_top_window", "Top Window"),
        ]

        # Sample data for performance (use every nth point)
        sample_step = max(1, len(self.data) // 1000)
        sample_data = self.data[::sample_step].copy()

        for idx, (action_col, action_name) in enumerate(actions):
            ax = fig.add_subplot(2, 2, idx + 1, projection="3d")

            try:
                # Create 3D scatter plot
                colors = sample_data[action_col]
                scatter = ax.scatter(
                    sample_data["ground_temp"],
                    sample_data["top_temp"],
                    sample_data["external_temp"],
                    c=colors,
                    cmap="RdYlBu_r",
                    s=30,
                    alpha=0.6,
                )

                ax.set_xlabel("Ground Temperature (°C)")
                ax.set_ylabel("Top Temperature (°C)")
                ax.set_zlabel("External Temperature (°C)")
                ax.set_title(action_name)

                # Add colorbar
                plt.colorbar(scatter, ax=ax, shrink=0.8, label="Action")

            except Exception as e:
                # Fallback: create empty plot with error message
                ax.text(0.5, 0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes)
                ax.set_title(f"{action_name} (Error)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "3d_action_surface.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_3d_trajectory(self):
        """Plot 3D trajectory of state evolution with actions."""
        print("Creating 3D trajectory plot...")

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Sample trajectory points for clarity (use every nth point but keep order)
        sample_size = min(500, len(self.data))
        if sample_size < len(self.data):
            indices = np.linspace(0, len(self.data) - 1, sample_size, dtype=int)
            sample_data = self.data.iloc[indices].copy()
        else:
            sample_data = self.data.copy()

        try:
            # Color trajectory by time
            colors = np.linspace(0, 1, len(sample_data))

            # Plot trajectory
            scatter = ax.scatter(
                sample_data["ground_temp"],
                sample_data["top_temp"],
                sample_data["external_temp"],
                c=colors,
                cmap="plasma",
                s=40,
                alpha=0.7,
            )

            # Add trajectory line
            ax.plot(
                sample_data["ground_temp"],
                sample_data["top_temp"],
                sample_data["external_temp"],
                alpha=0.3,
                linewidth=1,
                color="gray",
            )

            # Mark start and end points
            ax.scatter(
                sample_data["ground_temp"].iloc[0],
                sample_data["top_temp"].iloc[0],
                sample_data["external_temp"].iloc[0],
                c="green",
                s=100,
                marker="o",
                label="Start",
            )
            ax.scatter(
                sample_data["ground_temp"].iloc[-1],
                sample_data["top_temp"].iloc[-1],
                sample_data["external_temp"].iloc[-1],
                c="red",
                s=100,
                marker="s",
                label="End",
            )

            ax.set_xlabel("Ground Temperature (°C)")
            ax.set_ylabel("Top Temperature (°C)")
            ax.set_zlabel("External Temperature (°C)")
            ax.set_title(f"3D State Trajectory - {self.controller_name}")
            ax.legend()

            # Add colorbar for time
            plt.colorbar(scatter, ax=ax, shrink=0.8, label="Normalized Time")

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                0.5,
                f"Error creating 3D trajectory: {str(e)}",
                transform=ax.transAxes,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "3d_trajectory.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_interactive_3d_plots(self):
        """Create interactive 3D plots using Plotly."""
        print("Creating interactive 3D plots...")

        try:
            # 1. Interactive 3D Action Policy
            self._create_interactive_action_policy()

            # 2. Interactive 3D Trajectory
            self._create_interactive_trajectory()

            # 3. Interactive Multi-Action View
            self._create_interactive_multi_action()

        except Exception as e:
            print(f"Error creating interactive plots: {e}")

    def _create_interactive_action_policy(self):
        """Create interactive 3D action policy visualization."""
        print("Creating interactive action policy...")

        # Sample data for performance
        sample_size = min(1000, len(self.data))
        sample_indices = np.random.choice(len(self.data), sample_size, replace=False)
        sample_data = self.data.iloc[sample_indices].copy()

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scatter3d"}, {"type": "scatter3d"}],
                [{"type": "scatter3d"}, {"type": "scatter3d"}],
            ],
            subplot_titles=["Ground Light", "Ground Window", "Top Light", "Top Window"],
        )

        actions = [
            ("action_ground_light", 1, 1),
            ("action_ground_window", 1, 2),
            ("action_top_light", 2, 1),
            ("action_top_window", 2, 2),
        ]

        for action_col, row, col in actions:
            fig.add_trace(
                go.Scatter3d(
                    x=sample_data["ground_temp"],
                    y=sample_data["top_temp"],
                    z=sample_data["external_temp"],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=sample_data[action_col],
                        colorscale="RdYlBu",
                        opacity=0.7,
                    ),
                    text=[
                        f"Step: {i}<br>Action: {a}"
                        for i, a in zip(sample_data.index, sample_data[action_col])
                    ],
                    hovertemplate="Ground: %{x:.1f}°C<br>Top: %{y:.1f}°C<br>External: %{z:.1f}°C<br>%{text}",
                    name=action_col.replace("action_", "").replace("_", " ").title(),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        fig.update_layout(
            title=f"Interactive 3D Action Policy - {self.controller_name}", height=800
        )

        # Update scene properties for all subplots
        for i in range(1, 5):
            fig.update_layout(
                **{
                    f"scene{i}": dict(
                        xaxis_title="Ground Temperature (°C)",
                        yaxis_title="Top Temperature (°C)",
                        zaxis_title="External Temperature (°C)",
                    )
                }
            )

        # Save as HTML
        fig.write_html(
            os.path.join(self.output_dir, "interactive_3d_action_policy.html")
        )

    def _create_interactive_trajectory(self):
        """Create interactive 3D trajectory visualization."""
        print("Creating interactive trajectory...")

        # Sample for performance
        sample_size = min(800, len(self.data))
        indices = np.linspace(0, len(self.data) - 1, sample_size, dtype=int)
        sample_data = self.data.iloc[indices].copy()

        # Create combined action for coloring
        combined_actions = sample_data["lights_on"] + sample_data["windows_open"]

        fig = go.Figure()

        # Add trajectory
        fig.add_trace(
            go.Scatter3d(
                x=sample_data["ground_temp"],
                y=sample_data["top_temp"],
                z=sample_data["external_temp"],
                mode="markers+lines",
                marker=dict(
                    size=5,
                    color=combined_actions,
                    colorscale="Viridis",
                    colorbar=dict(title="Total Actions"),
                    opacity=0.8,
                ),
                line=dict(width=2, color="rgba(0,0,0,0.3)"),
                text=[
                    f"Step: {i}<br>Lights: {l}<br>Windows: {w}<br>Reward: {r:.3f}"
                    for i, l, w, r in zip(
                        sample_data.index,
                        sample_data["lights_on"],
                        sample_data["windows_open"],
                        sample_data["reward"],
                    )
                ],
                hovertemplate="Ground: %{x:.1f}°C<br>Top: %{y:.1f}°C<br>External: %{z:.1f}°C<br>%{text}",
                name="State Trajectory",
            )
        )

        fig.update_layout(
            title=f"Interactive 3D State Trajectory - {self.controller_name}",
            scene=dict(
                xaxis_title="Ground Temperature (°C)",
                yaxis_title="Top Temperature (°C)",
                zaxis_title="External Temperature (°C)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            height=700,
        )

        fig.write_html(os.path.join(self.output_dir, "interactive_3d_trajectory.html"))

    def _create_interactive_multi_action(self):
        """Create interactive multi-action view."""
        print("Creating interactive multi-action view...")

        # Sample data
        sample_size = min(1000, len(self.data))
        sample_indices = np.random.choice(len(self.data), sample_size, replace=False)
        sample_data = self.data.iloc[sample_indices].copy()

        fig = go.Figure()

        # Create different traces for different action combinations
        action_combinations = [
            (0, 0, "No Actions", "blue"),
            (1, 0, "One Light", "orange"),
            (0, 1, "One Window", "green"),
            (2, 0, "Both Lights", "red"),
            (0, 2, "Both Windows", "purple"),
            (1, 1, "Mixed Actions", "brown"),
        ]

        for lights, windows, name, color in action_combinations:
            mask = (sample_data["lights_on"] == lights) & (
                sample_data["windows_open"] == windows
            )
            if mask.sum() > 0:
                subset = sample_data[mask]

                fig.add_trace(
                    go.Scatter3d(
                        x=subset["ground_temp"],
                        y=subset["top_temp"],
                        z=subset["external_temp"],
                        mode="markers",
                        marker=dict(size=4, color=color, opacity=0.7),
                        text=[
                            f"Step: {i}<br>Reward: {r:.3f}"
                            for i, r in zip(subset.index, subset["reward"])
                        ],
                        hovertemplate="Ground: %{x:.1f}°C<br>Top: %{y:.1f}°C<br>External: %{z:.1f}°C<br>%{text}",
                        name=name,
                    )
                )

        fig.update_layout(
            title=f"Interactive 3D Action Combinations - {self.controller_name}",
            scene=dict(
                xaxis_title="Ground Temperature (°C)",
                yaxis_title="Top Temperature (°C)",
                zaxis_title="External Temperature (°C)",
            ),
            height=700,
        )

        fig.write_html(
            os.path.join(self.output_dir, "interactive_3d_multi_action.html")
        )

    def create_summary_report(self):
        """Create a comprehensive summary report with key insights."""
        print("Creating summary report...")

        # Calculate key metrics
        total_steps = len(self.data)
        total_reward = (
            self.metadata.get("total_reward", self.data["reward"].sum())
            if self.metadata
            else self.data["reward"].sum()
        )

        # Action statistics
        action_stats = {
            "ground_light_usage": self.data["action_ground_light"].mean(),
            "ground_window_usage": self.data["action_ground_window"].mean(),
            "top_light_usage": self.data["action_top_light"].mean(),
            "top_window_usage": self.data["action_top_window"].mean(),
            "avg_lights_on": self.data["lights_on"].mean(),
            "avg_windows_open": self.data["windows_open"].mean(),
        }

        # Temperature statistics
        temp_stats = {
            "avg_ground_temp": self.data["ground_temp"].mean(),
            "avg_top_temp": self.data["top_temp"].mean(),
            "avg_external_temp": self.data["external_temp"].mean(),
            "ground_temp_std": self.data["ground_temp"].std(),
            "top_temp_std": self.data["top_temp"].std(),
            "temp_range_ground": self.data["ground_temp"].max()
            - self.data["ground_temp"].min(),
            "temp_range_top": self.data["top_temp"].max() - self.data["top_temp"].min(),
        }

        # Comfort statistics (if available)
        comfort_stats = {}
        if "ground_comfort_violation" in self.data.columns:
            comfort_stats = {
                "total_comfort_violations": (
                    self.data["ground_comfort_violation"]
                    + self.data["top_comfort_violation"]
                ).sum(),
                "comfort_violation_rate": (
                    (
                        self.data["ground_comfort_violation"]
                        + self.data["top_comfort_violation"]
                    )
                    > 0
                ).mean(),
                "avg_comfort_violation": (
                    self.data["ground_comfort_violation"]
                    + self.data["top_comfort_violation"]
                ).mean(),
            }

        # Create summary plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Policy Analysis Summary - {self.controller_name}", fontsize=16)

        # Action usage bar plot
        actions = ["Ground\nLight", "Ground\nWindow", "Top\nLight", "Top\nWindow"]
        usage = [
            action_stats["ground_light_usage"],
            action_stats["ground_window_usage"],
            action_stats["top_light_usage"],
            action_stats["top_window_usage"],
        ]

        bars = axes[0, 0].bar(actions, usage, color=["orange", "blue", "red", "green"])
        axes[0, 0].set_title("Average Action Usage")
        axes[0, 0].set_ylabel("Usage Rate (0-1)")
        axes[0, 0].set_ylim(0, 1)
        # Add value labels on bars
        for bar, val in zip(bars, usage):
            height = bar.get_height()
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
            )

        # Temperature distribution
        axes[0, 1].hist(
            self.data["ground_temp"], alpha=0.7, label="Ground", bins=20, density=True
        )
        axes[0, 1].hist(
            self.data["top_temp"], alpha=0.7, label="Top", bins=20, density=True
        )
        axes[0, 1].set_title("Temperature Distribution")
        axes[0, 1].set_xlabel("Temperature (°C)")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Reward over time
        # Use moving average for smoother plot
        window_size = max(1, len(self.data) // 50)
        smooth_reward = (
            self.data["reward"].rolling(window=window_size, center=True).mean()
        )
        axes[0, 2].plot(smooth_reward, alpha=0.8)
        axes[0, 2].set_title("Reward Over Time (Smoothed)")
        axes[0, 2].set_xlabel("Step")
        axes[0, 2].set_ylabel("Reward")
        axes[0, 2].grid(True, alpha=0.3)

        # Action correlation heatmap
        action_cols = [
            "action_ground_light",
            "action_ground_window",
            "action_top_light",
            "action_top_window",
        ]
        action_corr = self.data[action_cols].corr()
        im = axes[1, 0].imshow(action_corr, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1, 0].set_xticks(range(len(action_cols)))
        axes[1, 0].set_yticks(range(len(action_cols)))
        short_labels = ["G.Light", "G.Window", "T.Light", "T.Window"]
        axes[1, 0].set_xticklabels(short_labels, rotation=45)
        axes[1, 0].set_yticklabels(short_labels)
        axes[1, 0].set_title("Action Correlation")

        # Add correlation values to heatmap
        for i in range(len(action_cols)):
            for j in range(len(action_cols)):
                axes[1, 0].text(
                    j,
                    i,
                    f"{action_corr.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(action_corr.iloc[i, j]) > 0.5 else "black",
                )

        plt.colorbar(im, ax=axes[1, 0], label="Correlation")

        # Energy vs Comfort scatter (if data available)
        if "energy_use" in self.data.columns and comfort_stats:
            scatter = axes[1, 1].scatter(
                self.data["energy_use"],
                self.data["ground_comfort_violation"]
                + self.data["top_comfort_violation"],
                alpha=0.5,
                s=10,
            )
            axes[1, 1].set_xlabel("Energy Use (Lights On)")
            axes[1, 1].set_ylabel("Total Comfort Violations")
            axes[1, 1].set_title("Energy vs Comfort Trade-off")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Temperature vs Action scatter as alternative
            axes[1, 1].scatter(
                self.data["avg_temp"], self.data["lights_on"], alpha=0.5, s=10
            )
            axes[1, 1].set_xlabel("Average Temperature (°C)")
            axes[1, 1].set_ylabel("Lights On")
            axes[1, 1].set_title("Temperature vs Lighting")
            axes[1, 1].grid(True, alpha=0.3)

        # Performance summary text
        axes[1, 2].axis("off")
        summary_text = f"""Controller Performance Summary

Controller: {self.controller_name}
Total Steps: {total_steps:,}
Total Reward: {total_reward:.2f}
Avg Reward/Step: {total_reward/total_steps:.4f}

Action Usage:
• Avg Lights On: {action_stats['avg_lights_on']:.2f}
• Avg Windows Open: {action_stats['avg_windows_open']:.2f}

Temperature Control:
• Avg Ground: {temp_stats['avg_ground_temp']:.1f}°C
• Avg Top: {temp_stats['avg_top_temp']:.1f}°C
• Ground Variability: {temp_stats['ground_temp_std']:.1f}°C
• Top Variability: {temp_stats['top_temp_std']:.1f}°C"""

        if comfort_stats:
            summary_text += f"""

Comfort Performance:
• Violation Rate: {comfort_stats['comfort_violation_rate']:.1%}
• Total Violations: {comfort_stats['total_comfort_violations']:.1f}"""

        axes[1, 2].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[1, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "summary_report.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save summary statistics to JSON
        summary_data = {
            "controller_name": self.controller_name,
            "total_steps": int(total_steps),
            "total_reward": float(total_reward),
            "action_statistics": {k: float(v) for k, v in action_stats.items()},
            "temperature_statistics": {k: float(v) for k, v in temp_stats.items()},
            "comfort_statistics": (
                {k: float(v) for k, v in comfort_stats.items()} if comfort_stats else {}
            ),
            "metadata": self.metadata,
        }

        with open(os.path.join(self.output_dir, "summary_statistics.json"), "w") as f:
            json.dump(summary_data, f, indent=4)

        print(f"Summary report saved to {self.output_dir}")
        return summary_data

    def generate_all_visualizations(self):
        """Generate all visualizations and reports."""
        print(
            f"Generating comprehensive policy visualizations for {self.controller_name}..."
        )

        try:
            # Generate all visualization types
            self.create_2d_policy_plots()
            self.create_3d_policy_plots()
            summary_data = self.create_summary_report()

            # Create index HTML file for easy viewing
            self._create_index_html()

            print(f"\nAll visualizations completed successfully!")
            print(f"View results in: {self.output_dir}")
            print(f"Open 'index.html' for a comprehensive view of all plots")

            return summary_data

        except Exception as e:
            print(f"Error during visualization generation: {e}")
            return None

    def _create_index_html(self):
        """Create an index HTML file linking to all generated visualizations."""
        print("Creating index HTML file...")

        # Check which files exist
        plot_files = {
            "2d_temperature_action_heatmap.png": "Temperature-Action Heatmaps",
            "2d_action_patterns_time.png": "Action Patterns Over Time",
            "2d_decision_boundaries.png": "Decision Boundaries",
            "2d_state_action_correlations.png": "State-Action Correlations",
            "2d_temperature_deviation_response.png": "Temperature Deviation Response",
            "2d_multidimensional_policy.png": "Multi-dimensional Policy View",
            "3d_action_surface.png": "3D Action Surfaces",
            "3d_trajectory.png": "3D State Trajectory",
            "summary_report.png": "Summary Report",
        }

        interactive_files = {
            "interactive_3d_action_policy.html": "Interactive Action Policy",
            "interactive_3d_trajectory.html": "Interactive Trajectory",
            "interactive_3d_multi_action.html": "Interactive Multi-Action View",
        }

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Policy Analysis - {self.controller_name}</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 40px; 
            background-color: #f5f5f5; 
            line-height: 1.6;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background-color: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 0 15px rgba(0,0,0,0.1); 
        }}
        h1 {{ 
            color: #2c3e50; 
            text-align: center; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        h2 {{ 
            color: #3498db; 
            margin-top: 40px; 
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #2c3e50;
            margin: 15px 0 10px 0;
        }}
        .stats {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stats strong {{ color: #fff; }}
        .plot-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); 
            gap: 25px; 
            margin: 25px 0; 
        }}
        .plot-item {{ 
            text-align: center; 
            background-color: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .plot-item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }}
        .plot-item img {{ 
            max-width: 100%; 
            height: auto; 
            border-radius: 5px; 
            border: 2px solid #ecf0f1;
        }}
        .plot-item p {{ 
            color: #7f8c8d; 
            font-size: 14px; 
            margin: 10px 0;
        }}
        .interactive-section {{
            background-color: #ecf0f1;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .interactive-link {{ 
            display: inline-block; 
            margin: 10px 15px 10px 0; 
            padding: 12px 25px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            text-decoration: none; 
            border-radius: 25px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .interactive-link:hover {{ 
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }}
        .metadata {{ 
            font-family: 'Courier New', monospace; 
            background-color: #2c3e50; 
            color: #ecf0f1;
            padding: 20px; 
            border-radius: 8px; 
            white-space: pre-wrap;
            overflow-x: auto;
            font-size: 12px;
        }}
        .file-list {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
        }}
        .file-list ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .file-list a {{
            color: #007bff;
            text-decoration: none;
        }}
        .file-list a:hover {{
            text-decoration: underline;
        }}
        .timestamp {{
            text-align: center;
            color: #6c757d;
            font-style: italic;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Policy Analysis Dashboard</h1>
        
        <div class="stats">
            <h2 style="color: white; margin-top: 0; border: none; padding: 0;">📊 {self.controller_name}</h2>
            <strong>Quick Stats:</strong><br>
            📈 Total Steps: {len(self.data):,}<br>
            🏆 Total Reward: {self.metadata.get('total_reward', 'N/A') if self.metadata else 'N/A'}<br>
            ⏱️ Episode Length: {self.metadata.get('episode_length_hours', 'N/A') if self.metadata else 'N/A'} hours<br>
            🕒 Analysis Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>

        <h2>📈 2D Policy Visualizations</h2>
        <div class="plot-grid">"""

        # Add 2D plots
        for filename, description in plot_files.items():
            if filename.startswith("2d_") and os.path.exists(
                os.path.join(self.output_dir, filename)
            ):
                html_content += f"""
            <div class="plot-item">
                <h3>{description}</h3>
                <img src="{filename}" alt="{description}">
                <p>{self._get_plot_description(filename)}</p>
            </div>"""

        html_content += """
        </div>

        <h2>🌐 3D Policy Visualizations</h2>
        <div class="plot-grid">"""

        # Add 3D plots
        for filename, description in plot_files.items():
            if filename.startswith("3d_") and os.path.exists(
                os.path.join(self.output_dir, filename)
            ):
                html_content += f"""
            <div class="plot-item">
                <h3>{description}</h3>
                <img src="{filename}" alt="{description}">
                <p>{self._get_plot_description(filename)}</p>
            </div>"""

        html_content += """
        </div>"""

        # Add interactive section if files exist
        interactive_exist = any(
            os.path.exists(os.path.join(self.output_dir, f))
            for f in interactive_files.keys()
        )
        if interactive_exist:
            html_content += """
        <h2>🎮 Interactive 3D Visualizations</h2>
        <div class="interactive-section">
            <p>🖱️ Click the links below to open interactive 3D visualizations (rotate, zoom, hover for details):</p>"""

            for filename, description in interactive_files.items():
                if os.path.exists(os.path.join(self.output_dir, filename)):
                    html_content += f"""
            <a href="{filename}" class="interactive-link">🎯 {description}</a>"""

            html_content += """
        </div>"""

        # Add summary report
        if os.path.exists(os.path.join(self.output_dir, "summary_report.png")):
            html_content += """
        <h2>📋 Performance Summary</h2>
        <div class="plot-item">
            <img src="summary_report.png" alt="Summary Report">
            <p>Comprehensive performance analysis with key metrics and statistics</p>
        </div>"""

        # Add data files section
        html_content += """
        <h2>📁 Data Files</h2>
        <div class="file-list">
            <p><strong>📄 Available Data Files:</strong></p>
            <ul>"""

        if os.path.exists(self.csv_path):
            html_content += f"""
                <li>📊 <a href="{os.path.basename(self.csv_path)}">Episode Data CSV</a> - Raw episode data with observations and actions</li>"""

        if os.path.exists(os.path.join(self.output_dir, "summary_statistics.json")):
            html_content += f"""
                <li>📈 <a href="summary_statistics.json">Summary Statistics JSON</a> - Computed performance metrics</li>"""

        if self.metadata_path and os.path.exists(self.metadata_path):
            html_content += f"""
                <li>🔧 <a href="{os.path.basename(self.metadata_path)}">Episode Metadata JSON</a> - Controller configuration and episode details</li>"""

        html_content += """
            </ul>
        </div>"""

        # Add metadata section if available
        if self.metadata:
            html_content += f"""
        <h2>⚙️ Episode Configuration</h2>
        <div class="metadata">{json.dumps(self.metadata, indent=2)}</div>"""

        html_content += f"""
        <div class="timestamp">
            🕒 Generated on {pd.Timestamp.now().strftime('%Y-%m-%d at %H:%M:%S')} | 
            🎯 Controller: {self.controller_name} | 
            📊 Total Data Points: {len(self.data):,}
        </div>
    </div>
</body>
</html>"""

        # Write the HTML file
        with open(
            os.path.join(self.output_dir, "index.html"), "w", encoding="utf-8"
        ) as f:
            f.write(html_content)

    def _get_plot_description(self, filename):
        """Get descriptive text for each plot type."""
        descriptions = {
            "2d_temperature_action_heatmap.png": "Shows action probabilities across different temperature conditions. Warmer colors indicate higher action probability.",
            "2d_action_patterns_time.png": "Temporal evolution of actions with temperature context. Shows how actions change over time in response to temperature variations.",
            "2d_decision_boundaries.png": "Policy decision boundaries in temperature space. Red points show when actions are taken, blue when not.",
            "2d_state_action_correlations.png": "Correlation matrix between environmental states and controller actions. Reveals which states most influence decisions.",
            "2d_temperature_deviation_response.png": "How the policy responds to temperature deviations from setpoints. Shows controller sensitivity to comfort zone violations.",
            "2d_multidimensional_policy.png": "Policy behavior across multiple state dimensions simultaneously. Each plot shows different state relationships.",
            "3d_action_surface.png": "Action policies visualized in 3D temperature space (ground, top, external). Color indicates action probability.",
            "3d_trajectory.png": "System state evolution through 3D space over time. Line shows trajectory, colors indicate temporal progression.",
            "summary_report.png": "Comprehensive performance summary including action usage, temperature control, and key metrics.",
        }
        return descriptions.get(
            filename,
            "Detailed visualization of policy behavior and performance metrics.",
        )

    def compare_controllers(csv_paths, labels=None, output_dir="controller_comparison"):
        """
        Compare multiple controllers by creating side-by-side visualizations.

        Args:
            csv_paths: List of CSV file paths for different controllers
            labels: List of labels for each controller (optional)
            output_dir: Directory to save comparison plots
        """
        if labels is None:
            labels = [f"Controller {i+1}" for i in range(len(csv_paths))]

        os.makedirs(output_dir, exist_ok=True)

        # Load all datasets
        datasets = []
        for csv_path in csv_paths:
            data = pd.read_csv(csv_path)
            datasets.append(data)

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Controller Comparison", fontsize=16)

        # Reward comparison
        for i, (data, label) in enumerate(zip(datasets, labels)):
            axes[0, 0].plot(data["reward"].rolling(50).mean(), label=label, alpha=0.8)
        axes[0, 0].set_title("Reward Over Time (Smoothed)")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Action usage comparison
        action_usage = []
        for data in datasets:
            usage = {
                "Lights": data["lights_on"].mean(),
                "Windows": data["windows_open"].mean(),
            }
            action_usage.append(usage)

        x = np.arange(len(labels))
        width = 0.35

        lights_usage = [usage["Lights"] for usage in action_usage]
        windows_usage = [usage["Windows"] for usage in action_usage]

        axes[0, 1].bar(x - width / 2, lights_usage, width, label="Lights", alpha=0.8)
        axes[0, 1].bar(x + width / 2, windows_usage, width, label="Windows", alpha=0.8)
        axes[0, 1].set_title("Average Action Usage")
        axes[0, 1].set_ylabel("Usage Rate")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(labels)
        axes[0, 1].legend()

        # Temperature control comparison
        for i, (data, label) in enumerate(zip(datasets, labels)):
            axes[1, 0].hist(
                data["avg_temp"], bins=30, alpha=0.7, label=label, density=True
            )
        axes[1, 0].set_title("Temperature Distribution")
        axes[1, 0].set_xlabel("Average Temperature (°C)")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].legend()

        # Performance metrics comparison
        metrics = ["Total Reward", "Avg Lights", "Avg Windows", "Temp Std"]
        controller_metrics = []

        for data in datasets:
            controller_metrics.append(
                [
                    data["reward"].sum(),
                    data["lights_on"].mean(),
                    data["windows_open"].mean(),
                    data["avg_temp"].std(),
                ]
            )

        # Normalize metrics for radar chart effect
        controller_metrics = np.array(controller_metrics)
        normalized_metrics = (controller_metrics - controller_metrics.min(axis=0)) / (
            controller_metrics.max(axis=0) - controller_metrics.min(axis=0) + 1e-8
        )

        x = np.arange(len(metrics))
        for i, (norm_metrics, label) in enumerate(zip(normalized_metrics, labels)):
            axes[1, 1].plot(
                x, norm_metrics, "o-", label=label, linewidth=2, markersize=6
            )

        axes[1, 1].set_title("Normalized Performance Metrics")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics, rotation=45)
        axes[1, 1].set_ylabel("Normalized Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "controller_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Controller comparison saved to {output_dir}")


def main():
    """Main function to run the policy visualizer."""
    parser = argparse.ArgumentParser(
        description="Generate interpretable policy visualizations from episode data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualizations
  python policy_visualizer.py --csv-path "episode_logs/rule_based_episode_data.csv"
  
  # Generate only 2D plots
  python policy_visualizer.py --csv-path "episode_data.csv" --plot-type 2d
  
  # Custom output directory with metadata
  python policy_visualizer.py --csv-path "data.csv" --metadata-path "metadata.json" --output-dir "analysis"
        """,
    )

    # Required arguments
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to the CSV file with episode data",
    )

    # Optional arguments
    parser.add_argument(
        "--metadata-path",
        type=str,
        help="Path to the metadata JSON file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="policy_plots",
        help="Directory to save generated plots (default: policy_plots)",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["2d", "3d", "summary", "all"],
        default="all",
        help="Type of plots to generate (default: all)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive 3D plots (useful if Plotly not available)",
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.csv_path):
        print(f"❌ Error: CSV file not found: {args.csv_path}")
        return 1

    # Auto-detect metadata path if not provided
    if args.metadata_path is None:
        csv_dir = os.path.dirname(args.csv_path)
        csv_basename = os.path.basename(args.csv_path).replace("_data.csv", "")
        potential_metadata_path = os.path.join(csv_dir, f"{csv_basename}_metadata.json")
        if os.path.exists(potential_metadata_path):
            args.metadata_path = potential_metadata_path
            print(f"✅ Auto-detected metadata file: {potential_metadata_path}")

    if args.metadata_path and not os.path.exists(args.metadata_path):
        print(f"⚠️  Warning: Metadata file not found: {args.metadata_path}")
        args.metadata_path = None

    # Disable interactive plots if requested
    if args.no_interactive:
        global PLOTLY_AVAILABLE
        PLOTLY_AVAILABLE = False
        print("🚫 Interactive plots disabled")

    print(f"🎯 Creating policy visualizations from {args.csv_path}")
    print(f"📁 Output directory: {args.output_dir}")

    try:
        # Create visualizer
        visualizer = PolicyVisualizer(
            csv_path=args.csv_path,
            metadata_path=args.metadata_path,
            output_dir=args.output_dir,
        )

        # Generate requested visualizations
        if args.plot_type == "2d":
            print("📈 Generating 2D visualizations...")
            visualizer.create_2d_policy_plots()
        elif args.plot_type == "3d":
            print("🌐 Generating 3D visualizations...")
            visualizer.create_3d_policy_plots()
        elif args.plot_type == "summary":
            print("📋 Generating summary report...")
            visualizer.create_summary_report()
        else:  # "all"
            print("🎨 Generating all visualizations...")
            visualizer.generate_all_visualizations()

        print(f"\n✅ Visualization complete!")
        print(f"📁 Results saved to: {args.output_dir}")

        if args.plot_type == "all":
            index_path = os.path.join(args.output_dir, "index.html")
            print(f"🌐 Open {index_path} to view all visualizations")

        return 0

    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
# python policy_visualizer.py --csv-path "episode_logs/rule_based_20250627_114432_data.csv"
