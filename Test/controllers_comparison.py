import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def compare_controllers(
    csv_paths, labels=None, output_dir="controller_comparison", metadata_paths=None
):
    """
    Compare multiple controllers by creating side-by-side visualizations.

    Args:
        csv_paths: List of CSV file paths for different controllers
        labels: List of labels for each controller (optional, auto-generated if None)
        output_dir: Directory to save comparison plots
        metadata_paths: List of metadata JSON file paths (optional)

    Returns:
        dict: Comparison results and statistics
    """
    print(f"🔄 Comparing {len(csv_paths)} controllers...")

    # Validate inputs
    if not csv_paths:
        raise ValueError("At least one CSV path must be provided")

    # Auto-generate labels if not provided
    if labels is None:
        labels = []
        for i, csv_path in enumerate(csv_paths):
            # Try to extract controller type from filename
            filename = os.path.basename(csv_path).lower()
            if "rule_based" in filename:
                labels.append(f"Rule-based")
            elif "ppo" in filename:
                labels.append(f"PPO")
            elif "a2c" in filename:
                labels.append(f"A2C")
            elif "dqn" in filename:
                labels.append(f"DQN")
            elif "sac" in filename:
                labels.append(f"SAC")
            else:
                labels.append(f"Controller {i+1}")

    # Ensure we have the right number of labels
    if len(labels) != len(csv_paths):
        labels = [f"Controller {i+1}" for i in range(len(csv_paths))]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load all datasets and metadata
    datasets = []
    metadata_list = []

    for i, csv_path in enumerate(csv_paths):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"📁 Loading {labels[i]}: {csv_path}")
        data = pd.read_csv(csv_path)

        # Add derived columns if missing
        if "lights_on" not in data.columns:
            data["lights_on"] = data["action_ground_light"] + data["action_top_light"]
        if "windows_open" not in data.columns:
            data["windows_open"] = (
                data["action_ground_window"] + data["action_top_window"]
            )
        if "avg_temp" not in data.columns:
            data["avg_temp"] = (data["ground_temp"] + data["top_temp"]) / 2

        datasets.append(data)

        # Load metadata if available
        metadata = None
        if metadata_paths and i < len(metadata_paths) and metadata_paths[i]:
            if os.path.exists(metadata_paths[i]):
                with open(metadata_paths[i], "r") as f:
                    metadata = json.load(f)

        metadata_list.append(metadata)

    print("📊 Creating comparison visualizations...")

    # Create comprehensive comparison plots
    results = {}

    # 1. Main comparison dashboard
    results["main_comparison"] = create_main_comparison(datasets, labels, output_dir)

    # 2. Detailed performance analysis
    results["performance_analysis"] = create_performance_analysis(
        datasets, labels, metadata_list, output_dir
    )

    # 3. Action pattern comparison
    results["action_patterns"] = create_action_pattern_comparison(
        datasets, labels, output_dir
    )

    # 4. Temperature control comparison
    results["temperature_control"] = create_temperature_comparison(
        datasets, labels, output_dir
    )

    # 5. Statistical summary
    results["statistical_summary"] = create_statistical_summary(
        datasets, labels, metadata_list, output_dir
    )

    # 6. Create HTML dashboard
    create_comparison_dashboard(results, labels, output_dir)

    print(f"✅ Controller comparison completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(
        f"🌐 Open {os.path.join(output_dir, 'comparison_dashboard.html')} to view results"
    )

    return results


def create_main_comparison(datasets, labels, output_dir):
    """Create the main comparison dashboard plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Controller Performance Comparison", fontsize=16, fontweight="bold")

    # 1. Reward comparison over time
    max_len = min(len(data) for data in datasets)  # Use shortest episode length
    window_size = max(1, max_len // 100)  # Adaptive smoothing

    for i, (data, label) in enumerate(zip(datasets, labels)):
        # Truncate to common length and smooth
        truncated_data = data.iloc[:max_len]
        smoothed_reward = (
            truncated_data["reward"].rolling(window=window_size, center=True).mean()
        )
        axes[0, 0].plot(smoothed_reward, label=label, linewidth=2, alpha=0.8)

    axes[0, 0].set_title("Reward Over Time (Smoothed)", fontweight="bold")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Action usage comparison
    action_categories = ["Lights", "Windows"]
    action_data = {category: [] for category in action_categories}

    for data in datasets:
        action_data["Lights"].append(data["lights_on"].mean())
        action_data["Windows"].append(data["windows_open"].mean())

    x = np.arange(len(labels))
    width = 0.35

    bars1 = axes[0, 1].bar(
        x - width / 2,
        action_data["Lights"],
        width,
        label="Lights",
        alpha=0.8,
        color="orange",
    )
    bars2 = axes[0, 1].bar(
        x + width / 2,
        action_data["Windows"],
        width,
        label="Windows",
        alpha=0.8,
        color="blue",
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axes[0, 1].set_title("Average Action Usage", fontweight="bold")
    axes[0, 1].set_ylabel("Usage Rate")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(
        labels, rotation=45 if len(max(labels, key=len)) > 8 else 0
    )
    axes[0, 1].legend()
    axes[0, 1].set_ylim(
        0, max(max(action_data["Lights"]), max(action_data["Windows"])) * 1.2
    )

    # 3. Temperature distribution comparison
    for i, (data, label) in enumerate(zip(datasets, labels)):
        axes[1, 0].hist(data["avg_temp"], bins=30, alpha=0.6, label=label, density=True)

    axes[1, 0].set_title("Temperature Distribution", fontweight="bold")
    axes[1, 0].set_xlabel("Average Temperature (°C)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Performance metrics radar chart
    metrics = ["Total Reward", "Reward Stability", "Energy Efficiency", "Temp Control"]
    controller_metrics = []

    for data in datasets:
        total_reward = data["reward"].sum()
        reward_stability = -data["reward"].std()  # Negative because lower std is better
        energy_efficiency = -data[
            "lights_on"
        ].mean()  # Negative because lower usage is more efficient
        temp_control = -(data["avg_temp"].std())  # Negative because lower std is better

        controller_metrics.append(
            [total_reward, reward_stability, energy_efficiency, temp_control]
        )

    # Normalize metrics to 0-1 scale
    controller_metrics = np.array(controller_metrics)
    normalized_metrics = np.zeros_like(controller_metrics)

    for i in range(controller_metrics.shape[1]):
        col = controller_metrics[:, i]
        if col.max() != col.min():
            normalized_metrics[:, i] = (col - col.min()) / (col.max() - col.min())
        else:
            normalized_metrics[:, i] = 0.5  # If all values are the same

    x = np.arange(len(metrics))
    for i, (norm_metrics, label) in enumerate(zip(normalized_metrics, labels)):
        axes[1, 1].plot(
            x, norm_metrics, "o-", label=label, linewidth=2, markersize=6, alpha=0.8
        )

    axes[1, 1].set_title("Normalized Performance Metrics", fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, rotation=45, ha="right")
    axes[1, 1].set_ylabel("Normalized Score (0-1)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "main_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    return {
        "action_usage": {
            label: {"lights": lights, "windows": windows}
            for label, lights, windows in zip(
                labels, action_data["Lights"], action_data["Windows"]
            )
        },
        "normalized_metrics": {
            label: metrics.tolist()
            for label, metrics in zip(labels, normalized_metrics)
        },
    }


def create_performance_analysis(datasets, labels, metadata_list, output_dir):
    """Create detailed performance analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Detailed Performance Analysis", fontsize=16, fontweight="bold")

    # Calculate performance metrics
    performance_data = []
    for i, (data, label) in enumerate(zip(datasets, labels)):
        metadata = metadata_list[i] if i < len(metadata_list) else None

        metrics = {
            "label": label,
            "total_reward": data["reward"].sum(),
            "avg_reward": data["reward"].mean(),
            "reward_std": data["reward"].std(),
            "episode_length": len(data),
            "total_energy": data["lights_on"].sum(),
            "avg_energy": data["lights_on"].mean(),
            "temp_stability": data["avg_temp"].std(),
            "avg_temp": data["avg_temp"].mean(),
        }

        # Add comfort metrics if available
        if "ground_comfort_violation" in data.columns:
            metrics["comfort_violations"] = (
                data["ground_comfort_violation"] + data["top_comfort_violation"]
            ).sum()
            metrics["comfort_rate"] = (
                (data["ground_comfort_violation"] + data["top_comfort_violation"]) > 0
            ).mean()

        # Add metadata info if available
        if metadata:
            metrics["controller_type"] = metadata.get("controller_type", "unknown")
            metrics["total_reward_meta"] = metadata.get(
                "total_reward", metrics["total_reward"]
            )

        performance_data.append(metrics)

    # 1. Total reward comparison
    total_rewards = [p["total_reward"] for p in performance_data]
    bars = axes[0, 0].bar(
        labels,
        total_rewards,
        alpha=0.8,
        color=plt.cm.Set3(np.linspace(0, 1, len(labels))),
    )
    axes[0, 0].set_title("Total Episode Reward", fontweight="bold")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, value in zip(bars, total_rewards):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(total_rewards) * 0.01,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Reward stability (coefficient of variation)
    cv_rewards = [
        p["reward_std"] / abs(p["avg_reward"]) if p["avg_reward"] != 0 else 0
        for p in performance_data
    ]
    bars = axes[0, 1].bar(
        labels, cv_rewards, alpha=0.8, color=plt.cm.Set2(np.linspace(0, 1, len(labels)))
    )
    axes[0, 1].set_title("Reward Stability (Lower = More Stable)", fontweight="bold")
    axes[0, 1].set_ylabel("Coefficient of Variation")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # 3. Energy efficiency
    energy_per_reward = [
        p["total_energy"] / max(p["total_reward"], 1) for p in performance_data
    ]
    bars = axes[0, 2].bar(
        labels,
        energy_per_reward,
        alpha=0.8,
        color=plt.cm.Set1(np.linspace(0, 1, len(labels))),
    )
    axes[0, 2].set_title(
        "Energy per Unit Reward (Lower = More Efficient)", fontweight="bold"
    )
    axes[0, 2].set_ylabel("Energy / Reward")
    axes[0, 2].tick_params(axis="x", rotation=45)

    # 4. Temperature control quality
    temp_stabilities = [p["temp_stability"] for p in performance_data]
    bars = axes[1, 0].bar(
        labels,
        temp_stabilities,
        alpha=0.8,
        color=plt.cm.Pastel1(np.linspace(0, 1, len(labels))),
    )
    axes[1, 0].set_title(
        "Temperature Variability (Lower = Better Control)", fontweight="bold"
    )
    axes[1, 0].set_ylabel("Temperature Std Dev (°C)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 5. Comfort violations (if available)
    if any("comfort_violations" in p for p in performance_data):
        comfort_violations = [p.get("comfort_violations", 0) for p in performance_data]
        bars = axes[1, 1].bar(
            labels,
            comfort_violations,
            alpha=0.8,
            color=plt.cm.Set3(np.linspace(0, 1, len(labels))),
        )
        axes[1, 1].set_title("Total Comfort Violations", fontweight="bold")
        axes[1, 1].set_ylabel("Violation Count")
        axes[1, 1].tick_params(axis="x", rotation=45)
    else:
        # Alternative: Show average temperature
        avg_temps = [p["avg_temp"] for p in performance_data]
        bars = axes[1, 1].bar(
            labels,
            avg_temps,
            alpha=0.8,
            color=plt.cm.Pastel2(np.linspace(0, 1, len(labels))),
        )
        axes[1, 1].set_title("Average Temperature", fontweight="bold")
        axes[1, 1].set_ylabel("Temperature (°C)")
        axes[1, 1].tick_params(axis="x", rotation=45)

    # 6. Summary table
    axes[1, 2].axis("off")

    # Create summary table data
    table_data = []
    headers = ["Controller", "Total Reward", "Avg Energy", "Temp Std"]

    for p in performance_data:
        table_data.append(
            [
                (
                    p["label"][:12] + "..." if len(p["label"]) > 12 else p["label"]
                ),  # Truncate long names
                f"{p['total_reward']:.1f}",
                f"{p['avg_energy']:.2f}",
                f"{p['temp_stability']:.2f}",
            ]
        )

    # Create table
    table = axes[1, 2].table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.3, 1, 0.6],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    axes[1, 2].set_title("Performance Summary", fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "performance_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return performance_data


def create_action_pattern_comparison(datasets, labels, output_dir):
    """Create action pattern comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Action Pattern Comparison", fontsize=16, fontweight="bold")

    # 1. Action correlation heatmap
    action_cols = [
        "action_ground_light",
        "action_ground_window",
        "action_top_light",
        "action_top_window",
    ]

    for i, (data, label) in enumerate(zip(datasets, labels)):
        ax = axes[i // 2, i % 2] if len(datasets) <= 4 else axes[0, i % 2]

        if i >= 4:  # If more than 4 controllers, only show first 4
            break

        try:
            corr_matrix = data[action_cols].corr()
            im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

            # Add correlation values
            for row in range(len(action_cols)):
                for col in range(len(action_cols)):
                    ax.text(
                        col,
                        row,
                        f"{corr_matrix.iloc[row, col]:.2f}",
                        ha="center",
                        va="center",
                        color=(
                            "white"
                            if abs(corr_matrix.iloc[row, col]) > 0.5
                            else "black"
                        ),
                    )

            ax.set_xticks(range(len(action_cols)))
            ax.set_yticks(range(len(action_cols)))
            short_labels = ["G.Light", "G.Window", "T.Light", "T.Window"]
            ax.set_xticklabels(short_labels, rotation=45)
            ax.set_yticklabels(short_labels)
            ax.set_title(f"{label} - Action Correlations")

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{label} - Error")

    # Hide unused subplots
    for i in range(len(datasets), 4):
        axes[i // 2, i % 2].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "action_patterns.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Create action usage over time comparison
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Action Usage Over Time", fontsize=16, fontweight="bold")

    # Truncate all datasets to the same length for fair comparison
    min_length = min(len(data) for data in datasets)
    window_size = max(1, min_length // 50)

    # Lights usage over time
    for data, label in zip(datasets, labels):
        truncated_data = data.iloc[:min_length]
        smoothed_lights = (
            truncated_data["lights_on"].rolling(window=window_size, center=True).mean()
        )
        axes[0].plot(smoothed_lights, label=label, linewidth=2, alpha=0.8)

    axes[0].set_title("Lights Usage Over Time (Smoothed)")
    axes[0].set_ylabel("Average Lights On")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Windows usage over time
    for data, label in zip(datasets, labels):
        truncated_data = data.iloc[:min_length]
        smoothed_windows = (
            truncated_data["windows_open"]
            .rolling(window=window_size, center=True)
            .mean()
        )
        axes[1].plot(smoothed_windows, label=label, linewidth=2, alpha=0.8)

    axes[1].set_title("Windows Usage Over Time (Smoothed)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Average Windows Open")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "action_usage_time.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    return {"action_correlations_calculated": True}


def create_temperature_comparison(datasets, labels, output_dir):
    """Create temperature control comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Temperature Control Comparison", fontsize=16, fontweight="bold")

    # 1. Temperature distributions
    for data, label in zip(datasets, labels):
        axes[0, 0].hist(
            data["ground_temp"],
            bins=30,
            alpha=0.6,
            label=f"{label} - Ground",
            density=True,
        )
        axes[0, 1].hist(
            data["top_temp"], bins=30, alpha=0.6, label=f"{label} - Top", density=True
        )

    axes[0, 0].set_title("Ground Floor Temperature Distribution")
    axes[0, 0].set_xlabel("Temperature (°C)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Top Floor Temperature Distribution")
    axes[0, 1].set_xlabel("Temperature (°C)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 2. Temperature control statistics
    temp_stats = []
    for data, label in zip(datasets, labels):
        stats = {
            "Controller": label,
            "Ground Mean": data["ground_temp"].mean(),
            "Ground Std": data["ground_temp"].std(),
            "Top Mean": data["top_temp"].mean(),
            "Top Std": data["top_temp"].std(),
            "Temp Difference": (data["top_temp"] - data["ground_temp"]).mean(),
        }
        temp_stats.append(stats)

    # Temperature means comparison
    ground_means = [s["Ground Mean"] for s in temp_stats]
    top_means = [s["Top Mean"] for s in temp_stats]

    x = np.arange(len(labels))
    width = 0.35

    axes[1, 0].bar(x - width / 2, ground_means, width, label="Ground Floor", alpha=0.8)
    axes[1, 0].bar(x + width / 2, top_means, width, label="Top Floor", alpha=0.8)
    axes[1, 0].set_title("Average Temperatures")
    axes[1, 0].set_ylabel("Temperature (°C)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=45)
    axes[1, 0].legend()

    # Temperature variability comparison
    ground_stds = [s["Ground Std"] for s in temp_stats]
    top_stds = [s["Top Std"] for s in temp_stats]

    axes[1, 1].bar(x - width / 2, ground_stds, width, label="Ground Floor", alpha=0.8)
    axes[1, 1].bar(x + width / 2, top_stds, width, label="Top Floor", alpha=0.8)
    axes[1, 1].set_title("Temperature Variability (Lower = Better Control)")
    axes[1, 1].set_ylabel("Standard Deviation (°C)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=45)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "temperature_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return temp_stats


def create_statistical_summary(datasets, labels, metadata_list, output_dir):
    """Create statistical summary and save to JSON."""
    summary = {
        "comparison_timestamp": datetime.now().isoformat(),
        "controllers": labels,
        "summary_statistics": {},
    }

    for i, (data, label) in enumerate(zip(datasets, labels)):
        metadata = metadata_list[i] if i < len(metadata_list) else None

        controller_stats = {
            "episode_length": len(data),
            "total_reward": float(data["reward"].sum()),
            "average_reward": float(data["reward"].mean()),
            "reward_std": float(data["reward"].std()),
            "total_energy_usage": float(data["lights_on"].sum()),
            "average_energy_usage": float(data["lights_on"].mean()),
            "average_ground_temp": float(data["ground_temp"].mean()),
            "average_top_temp": float(data["top_temp"].mean()),
            "ground_temp_std": float(data["ground_temp"].std()),
            "top_temp_std": float(data["top_temp"].std()),
            "average_temp_difference": float(
                (data["top_temp"] - data["ground_temp"]).mean()
            ),
            "windows_usage": float(data["windows_open"].mean()),
        }

        # Add comfort metrics if available
        if "ground_comfort_violation" in data.columns:
            controller_stats["total_comfort_violations"] = float(
                (data["ground_comfort_violation"] + data["top_comfort_violation"]).sum()
            )
            controller_stats["comfort_violation_rate"] = float(
                (
                    (data["ground_comfort_violation"] + data["top_comfort_violation"])
                    > 0
                ).mean()
            )

        # Add metadata if available
        if metadata:
            controller_stats["metadata"] = metadata

        summary["summary_statistics"][label] = controller_stats

    # Save summary to JSON
    with open(os.path.join(output_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(
        f"📊 Statistical summary saved to {os.path.join(output_dir, 'comparison_summary.json')}"
    )

    return summary


def create_comparison_dashboard(results, labels, output_dir):
    """Create an HTML dashboard for the controller comparison."""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Controller Comparison Dashboard</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 40px; 
            background-color: #f5f5f5; 
            line-height: 1.6;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background-color: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 0 15px rgba(0,0,0,0.1); 
        }}
        h1 {{ 
            color: #2c3e50; 
            text-align: center; 
            border-bottom: 3px solid #e74c3c; 
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        h2 {{ 
            color: #e74c3c; 
            margin-top: 40px; 
            border-left: 4px solid #e74c3c;
            padding-left: 15px;
        }}
        .comparison-info {{ 
            background: linear-gradient(135deg, #ff7f7f 0%, #ff9999 100%);
            color: white;
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .plot-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
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
        .plot-item h3 {{
            color: #2c3e50;
            margin: 15px 0 10px 0;
        }}
        .plot-item p {{ 
            color: #7f8c8d; 
            font-size: 14px; 
            margin: 10px 0;
        }}
        .controller-list {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .controller-list ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .file-list {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
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
        <h1>🥊 Controller Comparison Dashboard</h1>
        
        <div class="comparison-info">
            <h2 style="color: white; margin-top: 0; border: none; padding: 0;">📊 Comparison Overview</h2>
            <strong>Controllers Analyzed:</strong> {len(labels)}<br>
            <strong>Controllers:</strong> {', '.join(labels)}<br>
            <strong>Analysis Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>

        <div class="controller-list">
            <h3>🎯 Controllers in This Comparison:</h3>
            <ul>"""

    for i, label in enumerate(labels, 1):
        html_content += f"""
                <li><strong>Controller {i}:</strong> {label}</li>"""

    html_content += f"""
            </ul>
        </div>

        <h2>📈 Comparison Visualizations</h2>
        <div class="plot-grid">
            <div class="plot-item">
                <h3>Main Performance Comparison</h3>
                <img src="main_comparison.png" alt="Main Comparison">
                <p>Overall performance comparison including rewards, action usage, temperature control, and normalized metrics</p>
            </div>
            
            <div class="plot-item">
                <h3>Detailed Performance Analysis</h3>
                <img src="performance_analysis.png" alt="Performance Analysis">
                <p>In-depth analysis of reward, energy efficiency, temperature control quality, and comfort violations</p>
            </div>
            
            <div class="plot-item">
                <h3>Action Pattern Analysis</h3>
                <img src="action_patterns.png" alt="Action Patterns">
                <p>Action correlation matrices showing how different actions are coordinated by each controller</p>
            </div>
            
            <div class="plot-item">
                <h3>Action Usage Over Time</h3>
                <img src="action_usage_time.png" alt="Action Usage Time">
                <p>Evolution of lighting and window control actions throughout the episode</p>
            </div>
            
            <div class="plot-item">
                <h3>Temperature Control Comparison</h3>
                <img src="temperature_comparison.png" alt="Temperature Comparison">
                <p>Temperature distributions, average values, and variability comparison between controllers</p>
            </div>
        </div>

        <h2>📁 Data Files</h2>
        <div class="file-list">
            <p><strong>📄 Generated Files:</strong></p>
            <ul>
                <li>📊 <a href="comparison_summary.json">comparison_summary.json</a> - Complete statistical summary in JSON format</li>
                <li>📈 <a href="main_comparison.png">main_comparison.png</a> - Main comparison dashboard</li>
                <li>🔍 <a href="performance_analysis.png">performance_analysis.png</a> - Detailed performance analysis</li>
                <li>🎯 <a href="action_patterns.png">action_patterns.png</a> - Action correlation patterns</li>
                <li>📊 <a href="action_usage_time.png">action_usage_time.png</a> - Action usage over time</li>
                <li>🌡️ <a href="temperature_comparison.png">temperature_comparison.png</a> - Temperature control analysis</li>
            </ul>
        </div>

        <div class="timestamp">
            🕒 Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} | 
            🥊 Controllers: {len(labels)} | 
            📊 Comprehensive Performance Comparison
        </div>
    </div>
</body>
</html>"""

    # Write the HTML file
    with open(
        os.path.join(output_dir, "comparison_dashboard.html"), "w", encoding="utf-8"
    ) as f:
        f.write(html_content)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Compare multiple controller performances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two controllers
  python controller_comparison.py --csv-paths "rule_based_data.csv" "ppo_data.csv" --labels "Rule-based" "PPO"
  
  # Compare multiple controllers with auto-detected labels
  python controller_comparison.py --csv-paths "rule_based_data.csv" "ppo_data.csv" "a2c_data.csv"
  
  # Compare with custom output directory
  python controller_comparison.py --csv-paths "controller1.csv" "controller2.csv" --output-dir "my_comparison"
  
  # Include metadata files
  python controller_comparison.py --csv-paths "data1.csv" "data2.csv" --metadata-paths "meta1.json" "meta2.json"
        """,
    )

    # Required arguments
    parser.add_argument(
        "--csv-paths",
        nargs="+",
        required=True,
        help="Paths to CSV files with episode data for each controller",
    )

    # Optional arguments
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Labels for each controller (auto-detected if not provided)",
    )
    parser.add_argument(
        "--metadata-paths",
        nargs="*",
        help="Paths to metadata JSON files (optional, should match order of csv-paths)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="controller_comparison",
        help="Directory to save comparison results (default: controller_comparison)",
    )

    args = parser.parse_args()

    # Validate arguments
    if len(args.csv_paths) < 2:
        print("❌ Error: At least 2 CSV files required for comparison")
        return 1

    if args.labels and len(args.labels) != len(args.csv_paths):
        print("❌ Error: Number of labels must match number of CSV paths")
        return 1

    if args.metadata_paths and len(args.metadata_paths) != len(args.csv_paths):
        print("❌ Error: Number of metadata paths must match number of CSV paths")
        return 1

    # Validate that all CSV files exist
    for csv_path in args.csv_paths:
        if not os.path.exists(csv_path):
            print(f"❌ Error: CSV file not found: {csv_path}")
            return 1

    print(f"🥊 Starting controller comparison...")
    print(f"📁 Controllers: {len(args.csv_paths)}")
    print(f"💾 Output directory: {args.output_dir}")

    try:
        # Run comparison
        results = compare_controllers(
            csv_paths=args.csv_paths,
            labels=args.labels,
            output_dir=args.output_dir,
            metadata_paths=args.metadata_paths,
        )

        print(f"\n✅ Comparison completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback

        traceback.print_exc()
        return 1


# Standalone usage examples
def example_usage():
    """Show example usage patterns."""
    print(
        """
🥊 Controller Comparison Tool - Usage Examples

1. **Basic Comparison (Python code):**
```python
from controller_comparison import compare_controllers

# Compare rule-based vs RL controllers
results = compare_controllers(
    csv_paths=[
        "episode_logs/rule_based_episode_data.csv",
        "episode_logs/ppo_episode_data.csv"
    ],
    labels=["Rule-based", "PPO"],
    output_dir="rule_vs_ppo_comparison"
)
```

2. **Multiple Controller Comparison:**
```python
# Compare multiple RL algorithms
results = compare_controllers(
    csv_paths=[
        "logs/rule_based_data.csv",
        "logs/ppo_data.csv", 
        "logs/a2c_data.csv",
        "logs/sac_data.csv"
    ],
    labels=["Rule-based", "PPO", "A2C", "SAC"],
    output_dir="multi_controller_comparison"
)
```

3. **Command Line Usage:**
```bash
# Compare two controllers
python controller_comparison.py --csv-paths "rule_based_data.csv" "ppo_data.csv" --labels "Rule-based" "PPO"

# Compare multiple with auto-detection
python controller_comparison.py --csv-paths "rule_based_data.csv" "ppo_data.csv" "a2c_data.csv"

# Include metadata
python controller_comparison.py --csv-paths "data1.csv" "data2.csv" --metadata-paths "meta1.json" "meta2.json"
```

4. **With Metadata:**
```python
# Include metadata for richer analysis
results = compare_controllers(
    csv_paths=["controller1_data.csv", "controller2_data.csv"],
    labels=["Controller 1", "Controller 2"],
    metadata_paths=["controller1_metadata.json", "controller2_metadata.json"],
    output_dir="detailed_comparison"
)
```

**📊 What You Get:**
- Main comparison dashboard with key metrics
- Detailed performance analysis
- Action pattern comparisons  
- Temperature control analysis
- Statistical summary (JSON)
- Beautiful HTML dashboard
- All plots as high-quality PNG files

**🎯 Key Metrics Compared:**
- Total and average rewards
- Energy efficiency (lights usage)
- Temperature control quality
- Action correlation patterns
- Comfort violations (if available)
- Reward stability
- Temperature variability

**📁 Output Files:**
- `comparison_dashboard.html` - Main interactive dashboard
- `main_comparison.png` - Overview comparison
- `performance_analysis.png` - Detailed metrics
- `action_patterns.png` - Action correlations
- `temperature_comparison.png` - Temperature analysis
- `comparison_summary.json` - Raw statistics
    """
    )


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        # No arguments provided, show examples
        example_usage()
    else:
        exit(main())
