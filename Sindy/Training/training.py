"""
SINDy Hyperparameter Optimization Script - Updated with Separate Residual Plots

This script performs hyperparameter tuning for Sparse Identification of Nonlinear Dynamics (SINDy) models.
It evaluates different combinations of feature libraries and optimizers to find the best model for
the given data.

Usage:
    python sindy_hyperopt.py --train train_data.csv --test test_data.csv [options]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os
import json
import time
from datetime import datetime
import warnings
from joblib import Parallel, delayed

# pySINDy imports
from pysindy import SINDy
from pysindy.feature_library import (
    PolynomialLibrary,
    FourierLibrary,
    GeneralizedLibrary,
)
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import STLSQ, SR3, SSR, FROLS

# Define global variable for results directory
results_dir = None


def load_and_prepare_data(file_path, add_features=True, warmup_period_minutes=1):
    """
    Load and prepare data for SINDy model training and testing

    Parameters:
    file_path (str): Path to the CSV data file
    add_features (bool): Whether to add physics-informed features
    warmup_period_minutes (float): Warmup period to exclude from analysis

    Returns:
    X (np.array): State variables (temperatures)
    u (np.array): Input variables
    scaler_X (StandardScaler): Scaler used for state variables (None in this case)
    warmup_indices (np.array): Indices representing the warmup period
    """
    # Load the data
    data = pd.read_csv(file_path)

    # Calculate time differences in seconds
    if "Timestamp" in data.columns:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"])
        data["time_diff_seconds"] = data["Timestamp"].diff().dt.total_seconds()

        if len(data) > 1:
            median_diff = data["time_diff_seconds"].median()
            data["time_diff_seconds"] = data["time_diff_seconds"].fillna(median_diff)
        else:
            data["time_diff_seconds"] = data["time_diff_seconds"].fillna(30)
    else:
        data["time_diff_seconds"] = 30

    # Calculate cumulative time in seconds from the start
    data["cumulative_time_seconds"] = data["time_diff_seconds"].cumsum()

    # Identify warmup period
    warmup_seconds = warmup_period_minutes * 60
    warmup_mask = data["cumulative_time_seconds"] <= warmup_seconds
    warmup_indices = np.where(warmup_mask)[0]

    print(f"Loaded {len(data)} data points from {file_path}")
    print(
        f"Warmup period: {warmup_period_minutes} minutes ({sum(warmup_mask)} data points)"
    )

    # Map categorical values to numerical
    data["Ground Floor Light"] = data["Ground Floor Light"].map({"ON": 1, "OFF": 0})
    data["Ground Floor Window"] = data["Ground Floor Window"].map(
        {"OPEN": 1, "CLOSED": 0}
    )
    data["Top Floor Light"] = data["Top Floor Light"].map({"ON": 1, "OFF": 0})
    data["Top Floor Window"] = data["Top Floor Window"].map({"OPEN": 1, "CLOSED": 0})

    # Create physics-informed features
    if add_features:
        # Temperature differences (heat transfer drivers)
        data["Floor_Temp_Diff"] = (
            data["Top Floor Temperature (°C)"] - data["Ground Floor Temperature (°C)"]
        )
        data["Ground_Ext_Temp_Diff"] = (
            data["Ground Floor Temperature (°C)"] - data["External Temperature (°C)"]
        )
        data["Top_Ext_Temp_Diff"] = (
            data["Top Floor Temperature (°C)"] - data["External Temperature (°C)"]
        )

        # Window effects (accelerated heat transfer when open)
        data["Ground_Window_Ext_Effect"] = (
            data["Ground Floor Window"] * data["Ground_Ext_Temp_Diff"]
        )
        data["Top_Window_Ext_Effect"] = (
            data["Top Floor Window"] * data["Top_Ext_Temp_Diff"]
        )

        # Lag features for thermal inertia
        data["Ground_Temp_Lag1"] = data["Ground Floor Temperature (°C)"].shift(1)
        data["Top_Temp_Lag1"] = data["Top Floor Temperature (°C)"].shift(1)
        data["Ext_Temp_Lag1"] = data["External Temperature (°C)"].shift(1)

        # Temperature rate of change
        data["Ground_Temp_Rate"] = (
            data["Ground Floor Temperature (°C)"].diff() / data["time_diff_seconds"]
        )
        data["Top_Temp_Rate"] = (
            data["Top Floor Temperature (°C)"].diff() / data["time_diff_seconds"]
        )

        # Fill NaNs with appropriate values
        data = data.ffill().fillna(0)

    # Extract state variables to predict (X)
    X = data[["Ground Floor Temperature (°C)", "Top Floor Temperature (°C)"]].values

    # Construct input variable list
    if add_features:
        u_columns = [
            "Ground Floor Light",
            "Ground Floor Window",
            "Top Floor Light",
            "Top Floor Window",
            "External Temperature (°C)",
            "time_diff_seconds",
            "Floor_Temp_Diff",
            "Ground_Ext_Temp_Diff",
            "Top_Ext_Temp_Diff",
            "Ground_Window_Ext_Effect",
            "Top_Window_Ext_Effect",
            "Ground_Temp_Lag1",
            "Top_Temp_Lag1",
            "Ext_Temp_Lag1",
            "Ground_Temp_Rate",
            "Top_Temp_Rate",
        ]
    else:
        u_columns = [
            "Ground Floor Light",
            "Ground Floor Window",
            "Top Floor Light",
            "Top Floor Window",
            "External Temperature (°C)",
            "time_diff_seconds",
        ]

    u = data[u_columns].values

    return X, u, None, warmup_indices


def get_library_configurations():
    """
    Define an expanded set of library configurations for SINDy model search
    with more combinations of polynomial and Fourier terms.

    Returns:
    library_configs (list): List of library configuration dictionaries
    """
    library_configs = [
        # Basic libraries
        {
            "name": "linear",
            "description": "Linear terms only (degree 1 polynomial)",
            "library": PolynomialLibrary(degree=1, include_bias=True),
            "complexity": 1,
        },
        {
            "name": "polynomial_2",
            "description": "Polynomial with degree 2",
            "library": PolynomialLibrary(degree=2, include_bias=True),
            "complexity": 2,
        },
        {
            "name": "polynomial_3",
            "description": "Polynomial with degree 3",
            "library": PolynomialLibrary(degree=3, include_bias=True),
            "complexity": 3,
        },
        {
            "name": "fourier_1",
            "description": "Fourier terms with 1 frequency",
            "library": FourierLibrary(n_frequencies=1),
            "complexity": 1,
        },
        {
            "name": "fourier_2",
            "description": "Fourier terms with 2 frequencies",
            "library": FourierLibrary(n_frequencies=2),
            "complexity": 2,
        },
        {
            "name": "fourier_3",
            "description": "Fourier terms with 3 frequencies",
            "library": FourierLibrary(n_frequencies=3),
            "complexity": 3,
        },
        # Combinations of polynomial and Fourier terms
        {
            "name": "linear_fourier_1",
            "description": "Linear and Fourier terms with 1 frequency",
            "library": GeneralizedLibrary(
                [
                    PolynomialLibrary(degree=1, include_bias=True),
                    FourierLibrary(n_frequencies=1),
                ]
            ),
            "complexity": 2,
        },
        {
            "name": "linear_fourier_2",
            "description": "Linear and Fourier terms with 2 frequencies",
            "library": GeneralizedLibrary(
                [
                    PolynomialLibrary(degree=1, include_bias=True),
                    FourierLibrary(n_frequencies=2),
                ]
            ),
            "complexity": 3,
        },
        {
            "name": "linear_fourier_3",
            "description": "Linear and Fourier terms with 3 frequencies",
            "library": GeneralizedLibrary(
                [
                    PolynomialLibrary(degree=1, include_bias=True),
                    FourierLibrary(n_frequencies=3),
                ]
            ),
            "complexity": 4,
        },
        {
            "name": "poly2_fourier_1",
            "description": "Polynomial degree 2 with Fourier terms (1 frequency)",
            "library": GeneralizedLibrary(
                [
                    PolynomialLibrary(degree=2, include_bias=True),
                    FourierLibrary(n_frequencies=1),
                ]
            ),
            "complexity": 3,
        },
        {
            "name": "poly2_fourier_2",
            "description": "Polynomial degree 2 with Fourier terms (2 frequencies)",
            "library": GeneralizedLibrary(
                [
                    PolynomialLibrary(degree=2, include_bias=True),
                    FourierLibrary(n_frequencies=2),
                ]
            ),
            "complexity": 4,
        },
        {
            "name": "poly3_fourier_1",
            "description": "Polynomial degree 3 with Fourier terms (1 frequency)",
            "library": GeneralizedLibrary(
                [
                    PolynomialLibrary(degree=3, include_bias=True),
                    FourierLibrary(n_frequencies=1),
                ]
            ),
            "complexity": 4,
        },
        # More specialized configurations
        {
            "name": "poly_no_bias",
            "description": "Polynomial with degree 2 (no bias term)",
            "library": PolynomialLibrary(degree=2, include_bias=False),
            "complexity": 2,
        },
        {
            "name": "poly_3_no_bias",
            "description": "Polynomial with degree 3 (no bias term)",
            "library": PolynomialLibrary(degree=3, include_bias=False),
            "complexity": 3,
        },
        {
            "name": "linear_no_bias_fourier_2",
            "description": "Linear (no bias) and Fourier terms with 2 frequencies",
            "library": GeneralizedLibrary(
                [
                    PolynomialLibrary(degree=1, include_bias=False),
                    FourierLibrary(n_frequencies=2),
                ]
            ),
            "complexity": 3,
        },
    ]

    return library_configs


def get_optimizer_configurations():
    """
    Define optimizer configurations for SINDy model search based on the
    official pySINDy documentation.

    Returns:
    optimizer_configs (list): List of optimizer configuration dictionaries
    """
    optimizer_configs = []

    # STLSQ optimizer
    # Sequentially thresholded least squares algorithm
    # Main parameters: threshold, alpha
    thresholds = [0.0, 0.001, 0.01, 0.05, 0.1]
    alphas = [0.0, 0.001, 0.01, 0.05, 0.1]

    for threshold in thresholds:
        for alpha in alphas:
            optimizer_configs.append(
                {
                    "name": "STLSQ",
                    "parameters": {"threshold": threshold, "alpha": alpha},
                }
            )

    # SR3 optimizer
    # Sparse relaxed regularized regression
    # Main parameters: threshold, nu, thresholder
    thresholds = [0.01, 0.05, 0.1]
    nus = [0.1, 1.0, 10.0]
    thresholders = ["L0", "L1"]

    for threshold in thresholds:
        for nu in nus:
            for thresholder in thresholders:
                optimizer_configs.append(
                    {
                        "name": "SR3",
                        "parameters": {
                            "threshold": threshold,
                            "nu": nu,
                            "thresholder": thresholder,
                        },
                    }
                )

    # SSR optimizer
    # Stepwise sparse regression greedy algorithm
    # Main parameters: alpha
    alphas = [0.0, 0.01, 0.05, 0.1]

    for alpha in alphas:
        optimizer_configs.append({"name": "SSR", "parameters": {"alpha": alpha}})

    # FROLS optimizer
    # Forward Regression Orthogonal Least-Squares
    # Main parameters: max_iter, alpha
    max_iters = [5, 10, 15]
    alphas = [0.01, 0.05, 0.1]

    for max_iter in max_iters:
        for alpha in alphas:
            optimizer_configs.append(
                {"name": "FROLS", "parameters": {"max_iter": max_iter, "alpha": alpha}}
            )

    return optimizer_configs


def create_sindy_model(feature_library, optimizer_config):
    """
    Create a SINDy model with specified library and optimizer

    Parameters:
    feature_library: The feature library to use
    optimizer_config (dict): Optimizer configuration

    Returns:
    model (SINDy): The configured SINDy model
    """
    optimizer_type = optimizer_config["name"]
    params = optimizer_config["parameters"]

    # Create the appropriate optimizer
    if optimizer_type == "STLSQ":
        optimizer = STLSQ(**params)
    elif optimizer_type == "SR3":
        optimizer = SR3(**params)
    elif optimizer_type == "SSR":
        optimizer = SSR(**params)
    elif optimizer_type == "FROLS":
        optimizer = FROLS(**params)
    else:
        # Default to STLSQ
        optimizer = STLSQ(threshold=0.01, alpha=0.1)
        print(f"Warning: Unknown optimizer type {optimizer_type}, using default STLSQ")

    # Create SINDy model with finite difference
    model = SINDy(
        discrete_time=True,
        feature_library=feature_library,
        differentiation_method=FiniteDifference(),
        optimizer=optimizer,
    )

    return model


def single_step_prediction(model, X, u=None):
    """
    Perform single-step prediction with a SINDy model

    Parameters:
    model: Trained SINDy model
    X (np.array): Input states
    u (np.array): Input control variables

    Returns:
    X_pred (np.array): Predicted states
    """
    X_pred = model.predict(X, u=u)
    return X_pred


def multi_step_prediction(model, X_init, u, steps):
    """
    Perform multi-step prediction with a SINDy model

    Parameters:
    model: Trained SINDy model
    X_init: Initial state
    u: Input variables
    steps: Number of steps to predict

    Returns:
    X_pred: Predicted states
    """
    X_pred = np.zeros((steps, X_init.shape[1]))
    X_pred[0] = X_init[0]

    for i in range(1, steps):
        # Check for NaN or inf values to avoid propagating instability
        if np.any(np.isnan(X_pred[i - 1])) or np.any(np.isinf(X_pred[i - 1])):
            print(f"Warning: NaN or inf values at step {i}")
            # Fill remaining steps with NaN
            X_pred[i:] = np.nan
            break

        # Make prediction
        try:
            prediction = model.predict(
                X_pred[i - 1].reshape(1, -1), u=u[i - 1].reshape(1, -1)
            )[0]

            # Check for unreasonable values
            if np.any(np.abs(prediction) > 100):
                print(f"Warning: Prediction values too large at step {i}")
                # Fill remaining steps with NaN
                X_pred[i:] = np.nan
                break

            X_pred[i] = prediction
        except Exception as e:
            print(f"Error at step {i}: {str(e)}")
            # Fill remaining steps with NaN
            X_pred[i:] = np.nan
            break

    return X_pred


def filter_warmup_period(X, u, warmup_indices):
    """
    Filter out the warmup period from data arrays

    Parameters:
    X, u: Data arrays
    warmup_indices: Indices to filter out

    Returns:
    X_filtered, u_filtered: Filtered arrays
    """
    # Create a mask of all indices (True)
    mask = np.ones(len(X), dtype=bool)

    # Set the warmup period indices to False
    mask[warmup_indices] = False

    # Filter the arrays using the mask
    X_filtered = X[mask]
    u_filtered = u[mask]

    return X_filtered, u_filtered


def calculate_rmse(actual, predicted, variable_idx=None):
    """
    Calculate RMSE between actual and predicted values

    Parameters:
    actual: Actual values
    predicted: Predicted values
    variable_idx: Optional index for specific variable

    Returns:
    rmse: RMSE value(s)
    """
    # Ensure arrays have the same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]

    # Calculate RMSE for specific variable or all variables
    if variable_idx is not None:
        rmse = np.sqrt(
            mean_squared_error(actual[:, variable_idx], predicted[:, variable_idx])
        )
        return float(rmse)
    else:
        # Calculate RMSE for each variable
        rmse_dict = {}
        for i in range(actual.shape[1]):
            rmse_dict[f"var_{i}"] = float(
                np.sqrt(mean_squared_error(actual[:, i], predicted[:, i]))
            )
        return rmse_dict


def plot_residuals_separate(test_datasets, best_result, results_dir):
    """
    Plot residuals separately for each test dataset for the best configuration

    Parameters:
    test_datasets: List of test datasets with their names
    best_result: Best model evaluation result containing plot data
    results_dir: Directory to save plots
    """
    # Create a figure for residual plots
    n_tests = len(test_datasets)

    # Calculate subplot grid dimensions
    cols = min(3, n_tests)  # Maximum 3 columns
    rows = (n_tests + cols - 1) // cols  # Ceiling division

    # Create separate plots for Ground Floor and Top Floor residuals
    for variable_idx, variable_name in enumerate(["Ground Floor", "Top Floor"]):
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        fig.suptitle(f"{variable_name} Residuals - Best Configuration", fontsize=16)

        # Handle single subplot case
        if n_tests == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if hasattr(axes, "__len__") else [axes]
        else:
            axes = axes.flatten()

        for i, test_result in enumerate(best_result["test_results"]):
            if "plot_data" in test_result and test_result["multi_step"]["success"]:
                # Get data
                test_name = test_result["test_name"]
                plot_data = test_result["plot_data"]

                X_test = np.array(plot_data["X_test"])
                X_pred_single = np.array(plot_data["X_pred_single"])
                X_pred_multi = np.array(plot_data["X_pred_multi"])

                # Calculate residuals
                single_residuals = (
                    X_test[:, variable_idx] - X_pred_single[:, variable_idx]
                )
                multi_residuals = (
                    X_test[:, variable_idx] - X_pred_multi[:, variable_idx]
                )

                # Plot on subplot
                ax = axes[i] if i < len(axes) else None
                if ax is not None:
                    time_steps = np.arange(len(X_test))
                    ax.plot(
                        time_steps,
                        single_residuals,
                        "b--",
                        label="Single-step",
                        alpha=0.7,
                    )
                    ax.plot(
                        time_steps,
                        multi_residuals,
                        "r-.",
                        label="Multi-step",
                        alpha=0.7,
                    )
                    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)

                    # Calculate RMSE for title
                    rmse_single = np.sqrt(np.mean(single_residuals**2))
                    rmse_multi = np.sqrt(np.mean(multi_residuals**2))

                    ax.set_title(
                        f"{test_name}\nSingle RMSE: {rmse_single:.4f}°C, Multi RMSE: {rmse_multi:.4f}°C",
                        fontsize=10,
                    )
                    ax.set_xlabel("Time Steps")
                    ax.set_ylabel("Residual (°C)")
                    ax.grid(True, alpha=0.3)
                    ax.legend()

        # Hide empty subplots
        for i in range(n_tests, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save figure
        filename = f'residuals_{variable_name.lower().replace(" ", "_")}_separate.png'
        plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()


def plot_predictions(test_name, plot_data, results_dir):
    """
    Plot single-step and multi-step predictions along with residuals

    Parameters:
    test_name: Name of the test dataset
    plot_data: Dictionary containing prediction data
    results_dir: Directory to save plots
    """
    # Convert data from list to numpy arrays
    X_test = np.array(plot_data["X_test"])
    X_pred_single = np.array(plot_data["X_pred_single"])
    X_pred_multi = np.array(plot_data["X_pred_multi"])

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 3, figure=fig)

    # Main title
    fig.suptitle(f"SINDy Model Predictions - {test_name}", fontsize=16)

    # Ground Floor Temperature - Predictions
    ax1 = fig.add_subplot(gs[0, 0])
    time_steps = np.arange(len(X_test))
    ax1.plot(time_steps, X_test[:, 0], "k-", label="Actual")
    ax1.plot(time_steps, X_pred_single[:, 0], "b--", label="Single-step")
    ax1.plot(time_steps, X_pred_multi[:, 0], "r-.", label="Multi-step")
    ax1.set_title("Ground Floor Temperature")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top Floor Temperature - Predictions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_steps, X_test[:, 1], "k-", label="Actual")
    ax2.plot(time_steps, X_pred_single[:, 1], "b--", label="Single-step")
    ax2.plot(time_steps, X_pred_multi[:, 1], "r-.", label="Multi-step")
    ax2.set_title("Top Floor Temperature")
    ax2.set_ylabel("Temperature (°C)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Combined scatter plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(X_test[:, 0], X_test[:, 1], c="k", s=15, alpha=0.5, label="Actual")
    ax3.scatter(
        X_pred_multi[:, 0],
        X_pred_multi[:, 1],
        c="r",
        s=15,
        alpha=0.5,
        label="Multi-step",
    )
    ax3.set_xlabel("Ground Floor Temperature (°C)")
    ax3.set_ylabel("Top Floor Temperature (°C)")
    ax3.set_title("Temperature Relationship")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Ground Floor - Residuals
    ax4 = fig.add_subplot(gs[1, 0])
    single_residuals_ground = X_test[:, 0] - X_pred_single[:, 0]
    multi_residuals_ground = X_test[:, 0] - X_pred_multi[:, 0]

    ax4.plot(time_steps, single_residuals_ground, "b--", label="Single-step")
    ax4.plot(time_steps, multi_residuals_ground, "r-.", label="Multi-step")
    ax4.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax4.set_title("Ground Floor Residuals")
    ax4.set_xlabel("Time Steps")
    ax4.set_ylabel("Residual (°C)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Top Floor - Residuals
    ax5 = fig.add_subplot(gs[1, 1])
    single_residuals_top = X_test[:, 1] - X_pred_single[:, 1]
    multi_residuals_top = X_test[:, 1] - X_pred_multi[:, 1]

    ax5.plot(time_steps, single_residuals_top, "b--", label="Single-step")
    ax5.plot(time_steps, multi_residuals_top, "r-.", label="Multi-step")
    ax5.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax5.set_title("Top Floor Residuals")
    ax5.set_xlabel("Time Steps")
    ax5.set_ylabel("Residual (°C)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Residual distribution
    ax6 = fig.add_subplot(gs[1, 2])
    # Combine ground and top floor residuals
    all_residuals_single = np.concatenate(
        [single_residuals_ground, single_residuals_top]
    )
    all_residuals_multi = np.concatenate([multi_residuals_ground, multi_residuals_top])

    # Plot histograms
    ax6.hist(
        all_residuals_single, bins=30, alpha=0.5, color="blue", label="Single-step"
    )
    ax6.hist(all_residuals_multi, bins=30, alpha=0.5, color="red", label="Multi-step")
    ax6.set_title("Residual Distribution")
    ax6.set_xlabel("Residual (°C)")
    ax6.set_ylabel("Frequency")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Calculate RMSE values for annotation
    rmse_ground_single = np.sqrt(np.mean(single_residuals_ground**2))
    rmse_top_single = np.sqrt(np.mean(single_residuals_top**2))
    rmse_ground_multi = np.sqrt(np.mean(multi_residuals_ground**2))
    rmse_top_multi = np.sqrt(np.mean(multi_residuals_top**2))

    # Add RMSE annotation
    stats_text = (
        f"RMSE Values:\n"
        f"Single-step: Ground={rmse_ground_single:.4f}°C, Top={rmse_top_single:.4f}°C\n"
        f"Multi-step: Ground={rmse_ground_multi:.4f}°C, Top={rmse_top_multi:.4f}°C"
    )
    fig.text(
        0.5, 0.01, stats_text, ha="center", bbox=dict(facecolor="white", alpha=0.5)
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure
    plt.savefig(
        os.path.join(results_dir, f'predictions_{test_name.replace(".", "_")}.png'),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def evaluate_model_config(lib_config, opt_config, X_train, u_train, test_datasets):
    """
    Evaluate a specific model configuration on multiple test datasets

    Parameters:
    lib_config: Library configuration
    opt_config: Optimizer configuration
    X_train, u_train: Training data
    test_datasets: List of test datasets

    Returns:
    result: Evaluation results dictionary containing both single-step and multi-step metrics
    """
    # Extract configuration details
    library_name = lib_config["name"]
    library_complexity = lib_config["complexity"]
    feature_library = lib_config["library"]

    # Create model
    model = create_sindy_model(feature_library, opt_config)

    # Format optimizer parameters for display
    opt_params_str = ", ".join(
        [f"{k}={v}" for k, v in opt_config["parameters"].items()]
    )

    # Initialize result dictionary
    result = {
        "library_name": library_name,
        "library_complexity": library_complexity,
        "optimizer_name": opt_config["name"],
        "optimizer_params": opt_config["parameters"],
        "optimizer_params_str": opt_params_str,
        "trained": False,
        "test_results": [],
        # Multi-step prediction metrics
        "avg_rmse_ground": None,  # Average ground floor RMSE for multi-step prediction
        "avg_rmse_top": None,  # Average top floor RMSE for multi-step prediction
        "multi_step_success_rate": 0.0,  # Fraction of test datasets with successful multi-step prediction
        # Single-step prediction metrics
        "avg_single_rmse_ground": None,  # Average ground floor RMSE for single-step prediction
        "avg_single_rmse_top": None,  # Average top floor RMSE for single-step prediction
        "single_step_success_rate": 0.0,  # Fraction of test datasets with successful single-step prediction
        "error": None,
    }

    try:
        # Train the model
        model.fit(X_train, u=u_train, multiple_trajectories=False)
        result["trained"] = True

        # Evaluate on each test dataset
        successful_multi_tests = 0
        successful_single_tests = 0
        total_multi_rmse_ground = 0
        total_multi_rmse_top = 0
        total_single_rmse_ground = 0
        total_single_rmse_top = 0

        for test_data in test_datasets:
            test_name = test_data["name"]
            X_test = test_data["X"]
            u_test = test_data["u"]

            test_result = {"test_name": test_name, "single_step": {}, "multi_step": {}}

            # Single-step prediction
            try:
                X_pred_single = single_step_prediction(model, X_test, u=u_test)
                rmse_ground_single = calculate_rmse(
                    X_test, X_pred_single, variable_idx=0
                )
                rmse_top_single = calculate_rmse(X_test, X_pred_single, variable_idx=1)

                test_result["single_step"] = {
                    "success": True,
                    "rmse_ground": rmse_ground_single,
                    "rmse_top": rmse_top_single,
                }

                # Count successful single-step predictions
                successful_single_tests += 1
                total_single_rmse_ground += rmse_ground_single
                total_single_rmse_top += rmse_top_single

            except Exception as e:
                test_result["single_step"] = {"success": False, "error": str(e)}

            # Multi-step prediction
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X_pred_multi = multi_step_prediction(
                        model, X_test[:1], u_test, len(X_test)
                    )

                # Check if prediction contains NaN values
                if np.any(np.isnan(X_pred_multi)):
                    raise ValueError("Prediction contains NaN values")

                rmse_ground_multi = calculate_rmse(X_test, X_pred_multi, variable_idx=0)
                rmse_top_multi = calculate_rmse(X_test, X_pred_multi, variable_idx=1)

                test_result["multi_step"] = {
                    "success": True,
                    "rmse_ground": rmse_ground_multi,
                    "rmse_top": rmse_top_multi,
                }

                # Count successful multi-step predictions
                successful_multi_tests += 1
                total_multi_rmse_ground += rmse_ground_multi
                total_multi_rmse_top += rmse_top_multi

                # Save prediction data for plotting for ALL test datasets
                test_result["plot_data"] = {
                    "X_test": X_test.tolist(),
                    "X_pred_single": X_pred_single.tolist(),
                    "X_pred_multi": X_pred_multi.tolist(),
                }

            except Exception as e:
                test_result["multi_step"] = {"success": False, "error": str(e)}

            result["test_results"].append(test_result)

        # Calculate averages for multi-step if any tests were successful
        if successful_multi_tests > 0:
            result["avg_rmse_ground"] = total_multi_rmse_ground / successful_multi_tests
            result["avg_rmse_top"] = total_multi_rmse_top / successful_multi_tests
            result["multi_step_success_rate"] = successful_multi_tests / len(
                test_datasets
            )

        # Calculate averages for single-step if any tests were successful
        if successful_single_tests > 0:
            result["avg_single_rmse_ground"] = (
                total_single_rmse_ground / successful_single_tests
            )
            result["avg_single_rmse_top"] = (
                total_single_rmse_top / successful_single_tests
            )
            result["single_step_success_rate"] = successful_single_tests / len(
                test_datasets
            )

    except Exception as e:
        result["error"] = f"Model training failed: {str(e)}"

    return result


def run_hyperparameter_optimization(
    train_data_files,
    test_data_files,
    add_features=True,
    warmup_period_minutes=1,
    n_jobs=-1,
):
    """
    Run hyperparameter optimization for SINDy models using multiple training and test files

    Parameters:
    train_data_files: List of training data file paths
    test_data_files: List of test data file paths
    add_features: Whether to add physics-informed features
    warmup_period_minutes: Warmup period to exclude
    n_jobs: Number of parallel jobs

    Returns:
    all_results: All evaluation results
    best_config: Best configuration found based on multi-step prediction performance
    """
    # Load training data from multiple files
    print("Loading training data from multiple files...")
    X_list = []
    u_list = []

    for i, file_path in enumerate(train_data_files):
        print(f"Processing training file {i+1}/{len(train_data_files)}: {file_path}")
        try:
            X_train, u_train, _, warmup_indices = load_and_prepare_data(
                file_path,
                add_features=add_features,
                warmup_period_minutes=warmup_period_minutes,
            )

            # Filter out warmup period
            X_train_filtered, u_train_filtered = filter_warmup_period(
                X_train, u_train, warmup_indices
            )

            # Add to lists
            X_list.append(X_train_filtered)
            u_list.append(u_train_filtered)

            print(
                f"  Added {len(X_train_filtered)} data points after filtering warmup period"
            )
        except Exception as e:
            print(f"Error loading training file {file_path}: {e}")

    # Combine training data from all files
    if len(X_list) > 1:
        X_train = np.vstack(X_list)
        u_train = np.vstack(u_list)
        print(
            f"Combined {len(train_data_files)} training files for a total of {len(X_train)} data points"
        )
    elif len(X_list) == 1:
        X_train = X_list[0]
        u_train = u_list[0]
        print(f"Using single training file with {len(X_train)} data points")
    else:
        raise ValueError(
            "No valid training data could be loaded from the specified files"
        )

    print(f"Final training data shape: X={X_train.shape}, u={u_train.shape}")

    # Load test datasets from multiple files
    print("\nLoading test datasets from multiple files...")
    test_datasets = []

    for i, file_path in enumerate(test_data_files):
        print(f"Processing test file {i+1}/{len(test_data_files)}: {file_path}")
        try:
            X_test, u_test, _, test_warmup_indices = load_and_prepare_data(
                file_path,
                add_features=add_features,
                warmup_period_minutes=warmup_period_minutes,
            )
            X_test_filtered, u_test_filtered = filter_warmup_period(
                X_test, u_test, test_warmup_indices
            )

            # Create a test dataset object
            test_datasets.append(
                {
                    "name": os.path.basename(file_path),
                    "X": X_test_filtered,
                    "u": u_test_filtered,
                }
            )

            print(
                f"  {os.path.basename(file_path)}: {len(X_test_filtered)} data points after filtering warmup period"
            )
        except Exception as e:
            print(f"Error loading test file {file_path}: {e}")

    if not test_datasets:
        raise ValueError("No valid test data could be loaded from the specified files")

    print(f"Loaded {len(test_datasets)} test datasets")

    # Get configurations to test
    library_configs = get_library_configurations()
    optimizer_configs = get_optimizer_configurations()

    total_configs = len(library_configs) * len(optimizer_configs)
    print(
        f"\nTesting {total_configs} configurations ({len(library_configs)} libraries × {len(optimizer_configs)} optimizers) on {len(test_datasets)} test datasets"
    )

    # Create list of all configurations to test
    all_configs = []
    for lib_config in library_configs:
        for opt_config in optimizer_configs:
            all_configs.append((lib_config, opt_config))

    # Define function for parallel processing
    def evaluate_config(lib_config, opt_config):
        opt_name = opt_config["name"]
        opt_details = ", ".join(
            [f"{k}={v}" for k, v in opt_config["parameters"].items()]
        )
        print(f"Evaluating: {lib_config['name']} with {opt_name} ({opt_details})")
        return evaluate_model_config(
            lib_config, opt_config, X_train, u_train, test_datasets
        )

    # Run evaluations in parallel
    print(f"Starting parallel evaluation with {n_jobs} jobs...")
    start_time = time.time()

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_config)(lib_config, opt_config)
        for lib_config, opt_config in all_configs
    )

    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.2f} seconds")

    # Filter successful results (only consider configurations where multi-step prediction succeeded)
    successful_results = [r for r in results if r.get("multi_step_success_rate", 0) > 0]

    if successful_results:
        # Find best configuration based on multi-step prediction performance
        # 1. First priority: Multi-step prediction success rate (higher is better)
        # 2. Second priority: Multi-step prediction RMSE (lower is better)
        def score_result(r):
            success_rate = r.get("multi_step_success_rate", 0)
            avg_multi_rmse = (
                r.get("avg_rmse_ground", float("inf"))
                + r.get("avg_rmse_top", float("inf"))
            ) / 2
            return (
                -success_rate,
                avg_multi_rmse,
            )  # Sort by (-success_rate, rmse) to prioritize high success rates and low RMSE

        best_result = min(successful_results, key=score_result)

        # Create best configuration dictionary with details focused on multi-step performance
        best_config = {
            "library_name": best_result["library_name"],
            "library_complexity": best_result["library_complexity"],
            "optimizer_name": best_result["optimizer_name"],
            "optimizer_params": best_result["optimizer_params"],
            "optimizer_params_str": best_result.get("optimizer_params_str", ""),
            "avg_ground_floor_rmse": best_result["avg_rmse_ground"],  # multi-step RMSE
            "avg_top_floor_rmse": best_result["avg_rmse_top"],  # multi-step RMSE
            "multi_step_success_rate": best_result["multi_step_success_rate"],
            "test_results": best_result["test_results"],
        }

        # Generate plots for the best configuration for each test dataset
        print(f"Generating visualizations for the best configuration...")
        for test_result in best_result["test_results"]:
            if "plot_data" in test_result and test_result["multi_step"]["success"]:
                print(f"  Plotting results for {test_result['test_name']}...")
                plot_predictions(
                    test_result["test_name"], test_result["plot_data"], results_dir
                )

        # Generate separate residual plots for all test datasets
        print(f"Generating separate residual plots for the best configuration...")
        plot_residuals_separate(test_datasets, best_result, results_dir)
    else:
        best_config = None
        print(
            "No successful configurations found that could perform multi-step prediction"
        )

    return results, best_config


def main():
    """
    Main function for SINDy hyperparameter optimization with support for multiple training and test files
    """
    import argparse

    # Declare the global variable at the start of the function
    global results_dir

    parser = argparse.ArgumentParser(description="SINDy Hyperparameter Optimization")
    parser.add_argument(
        "--train",
        type=str,
        nargs="+",
        default=[
            "../../Data/dollhouse-data-2025-03-24.csv",
        ],
        help="Path(s) to training data CSV file(s) - multiple files can be specified",
    )
    parser.add_argument(
        "--test",
        type=str,
        nargs="+",
        default=[
            "../../Data/dollhouse-data-2025-02-28.csv",
            "../../Data/dollhouse-data-2025-03-24.csv",
            "../../Data/dollhouse-data-2025-04-03.csv",
            "../../Data/dollhouse-data-2025-02-27.csv",
            "../../Data/dollhouse-data-2025-02-19.csv",
        ],
        help="Path(s) to test data CSV file(s) - multiple files can be specified",
    )
    parser.add_argument(
        "--warmup", type=float, default=1.0, help="Warmup period in minutes to exclude"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores)",
    )
    parser.add_argument(
        "--outdir", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--features",
        action="store_true",
        default=True,
        help="Add physics-informed features",
    )

    args = parser.parse_args()

    # Create output directory
    if args.outdir:
        results_dir = args.outdir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/sindy_hyperopt_{timestamp}"

    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(results_dir)}")

    # Run hyperparameter optimization
    print(f"Starting hyperparameter optimization")
    print(f"  Using physics-informed features: {args.features}")
    print(f"  Training files: {len(args.train)} files")
    for i, file in enumerate(args.train):
        print(f"    {i+1}. {file}")
    print(f"  Testing files: {len(args.test)} files")
    for i, file in enumerate(args.test):
        print(f"    {i+1}. {file}")

    all_results, best_config = run_hyperparameter_optimization(
        args.train,
        args.test,
        add_features=args.features,
        warmup_period_minutes=args.warmup,
        n_jobs=args.jobs,
    )

    # Save results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "train_files": args.train,
        "test_files": args.test,
        "warmup_period_minutes": args.warmup,
        "physics_features": args.features,
        "results": all_results,
        "best_config": best_config,
    }

    results_file = os.path.join(results_dir, "sindy_hyperopt_results.json")
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to {results_file}")

    # Create summary file with just the best configuration
    if best_config:
        summary_file = os.path.join(results_dir, "best_config_summary.json")
        with open(summary_file, "w") as f:
            json.dump(best_config, f, indent=2)
        print(f"Best configuration summary saved to {summary_file}")

    # Print summary of successful configurations
    successful_results = [
        r for r in all_results if r.get("multi_step_success_rate", 0) > 0
    ]
    print(
        f"\nSuccessful configurations: {len(successful_results)} out of {len(all_results)}"
    )

    if successful_results:
        # Sort by success rate and RMSE - same criteria used for best configuration selection
        def score_result(r):
            # First priority: Multi-step prediction success rate (higher is better)
            success_rate = r.get("multi_step_success_rate", 0)
            # Second priority: Average multi-step RMSE (lower is better)
            avg_rmse = (
                r.get("avg_rmse_ground", float("inf"))
                + r.get("avg_rmse_top", float("inf"))
            ) / 2
            return (-success_rate, avg_rmse)

        sorted_results = sorted(successful_results, key=score_result)

        # Find which result is the best one returned from optimization
        best_result = None
        if best_config:
            # Match the best_config to its original result
            for result in successful_results:
                if (
                    result["library_name"] == best_config["library_name"]
                    and result["optimizer_name"] == best_config["optimizer_name"]
                    and result["optimizer_params"] == best_config["optimizer_params"]
                ):
                    best_result = result
                    break

        # Print top 10 configurations
        print(
            "\nTop configurations (sorted by multi-step success rate, then by multi-step RMSE):"
        )
        print("-" * 130)
        print(
            f"{'Library':<15} {'Optimizer':<10} {'Parameters':<40} {'Success Rate':<15} {'Ground RMSE':<15} {'Top RMSE':<15} {'Note'}"
        )
        print("-" * 130)

        for i, result in enumerate(sorted_results[:10]):
            opt_params_str = result.get("optimizer_params_str", "")
            if len(opt_params_str) > 38:
                opt_params_str = opt_params_str[:35] + "..."

            # Add a note if this is the best configuration
            note = "BEST" if best_result and result == best_result else ""

            print(
                f"{result['library_name']:<15} {result['optimizer_name']:<10} {opt_params_str:<40} "
                f"{result.get('multi_step_success_rate', 0):<15.2f} "
                f"{result.get('avg_rmse_ground', float('inf')):<15.4f} {result.get('avg_rmse_top', float('inf')):<15.4f} {note}"
            )

        # Print best configuration details
        if best_config:
            print("\nBest configuration (based on multi-step prediction performance):")
            print(
                f"  Library: {best_config['library_name']} (complexity: {best_config['library_complexity']})"
            )
            print(f"  Optimizer: {best_config['optimizer_name']}")
            print(f"  Parameters: {best_config['optimizer_params_str']}")
            print(
                f"  Multi-step Avg Ground Floor RMSE: {best_config['avg_ground_floor_rmse']:.4f}°C"
            )
            print(
                f"  Multi-step Avg Top Floor RMSE: {best_config['avg_top_floor_rmse']:.4f}°C"
            )
            print(
                f"  Multi-step Success Rate: {best_config['multi_step_success_rate']:.2f} "
                f"({int(best_config['multi_step_success_rate'] * len(args.test))}/{len(args.test)} test files)"
            )

            # Print per-test results for best configuration
            print("\n  Performance on individual test datasets:")
            for test_result in best_config["test_results"]:
                test_name = test_result["test_name"]

                # Print multi-step results if successful
                if test_result["multi_step"]["success"]:
                    print(
                        f"    {test_name}: Multi-step Ground RMSE = {test_result['multi_step']['rmse_ground']:.4f}°C, "
                        f"Top RMSE = {test_result['multi_step']['rmse_top']:.4f}°C"
                    )

                    # If single-step is also successful, print for comparison
                    if test_result["single_step"]["success"]:
                        print(
                            f"                Single-step Ground RMSE = {test_result['single_step']['rmse_ground']:.4f}°C, "
                            f"Top RMSE = {test_result['single_step']['rmse_top']:.4f}°C"
                        )
                else:
                    print(
                        f"    {test_name}: Multi-step failed - {test_result['multi_step'].get('error', 'Unknown error')}"
                    )
                    if test_result["single_step"]["success"]:
                        print(
                            f"                Single-step only: Ground RMSE = {test_result['single_step']['rmse_ground']:.4f}°C, "
                            f"Top RMSE = {test_result['single_step']['rmse_top']:.4f}°C"
                        )

            # Plot location reminder
            print(f"\nPrediction plots saved to {results_dir}")
            print(f"Separate residual plots saved to {results_dir}")
    else:
        print(
            "No successful configurations found. Try different hyperparameter ranges or adjust model configurations."
        )


if __name__ == "__main__":
    main()
