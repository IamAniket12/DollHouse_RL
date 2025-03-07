import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from pysindy import (
    SINDy,
    CustomLibrary,
    PolynomialLibrary,
    FourierLibrary,
    IdentityLibrary,
    STLSQ,
    GeneralizedLibrary,
)
from pysindy.differentiation import SmoothedFiniteDifference
import argparse
import re
import os
import json
from tqdm import tqdm
import time
import scipy.stats as stats
import statsmodels.api as sm
from pysindy.optimizers import SR3, FROLS, SSR
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(file_path, normalize=True):
    """
    Load and prepare data for SINDy model, including time differences in seconds

    Parameters:
    -----------
    file_path : str
        Path to the CSV file with temperature data
    normalize : bool
        Whether to normalize the state variables

    Returns:
    --------
    tuple
        (X, u, data_processed)
    """
    # Load the data
    data = pd.read_csv(file_path)

    # Calculate time differences in seconds
    if "Timestamp" in data.columns:
        # Convert to datetime format
        data["Timestamp"] = pd.to_datetime(data["Timestamp"])

        # Calculate time difference in seconds between consecutive readings
        data["time_diff_seconds"] = data["Timestamp"].diff().dt.total_seconds()

        # Fill first row NaN with median time difference or 30 seconds (default sampling)
        if len(data) > 1:
            median_diff = data["time_diff_seconds"].median()
            data["time_diff_seconds"] = data["time_diff_seconds"].fillna(median_diff)
        else:
            data["time_diff_seconds"] = data["time_diff_seconds"].fillna(
                30
            )  # Assuming 30-second default
    else:
        # If no Timestamp column, assume constant 30-second intervals
        data["time_diff_seconds"] = 30

    # Map categorical values to numerical
    data["Ground Floor Light"] = data["Ground Floor Light"].map({"ON": 1, "OFF": 0})
    data["Ground Floor Window"] = data["Ground Floor Window"].map(
        {"OPEN": 1, "CLOSED": 0}
    )
    data["Top Floor Light"] = data["Top Floor Light"].map({"ON": 1, "OFF": 0})
    data["Top Floor Window"] = data["Top Floor Window"].map({"OPEN": 1, "CLOSED": 0})

    # Extract features (X) and input (u) variables
    X = data[["Ground Floor Temperature (°C)", "Top Floor Temperature (°C)"]].values

    # Include time difference in the input variables
    u = data[
        [
            "Ground Floor Light",
            "Ground Floor Window",
            "Top Floor Light",
            "Top Floor Window",
            "External Temperature (°C)",
            "time_diff_seconds",  # Include time difference in seconds as input
        ]
    ].values

    # Normalize state variables if requested
    if normalize:

        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)

        # Store the scaler parameters for reference
        data["temp_mean_ground"] = scaler_X.mean_[0]
        data["temp_mean_top"] = scaler_X.mean_[1]
        data["temp_std_ground"] = scaler_X.scale_[0]
        data["temp_std_top"] = scaler_X.scale_[1]

    return X, u


def create_sindy_model(
    threshold=0.001,
    alpha=0.1,
    poly_order=1,
    library_type="polynomial",
    optimizer_type="STLSQ",
):
    """
    Create a SINDy model using PySINDy's built-in libraries

    Parameters:
    -----------
    threshold : float
        Threshold for sparse regression
    alpha : float
        Regularization parameter
    poly_order : int
        Order of polynomial features (1 = linear, 2 = quadratic, etc.)
    library_type : str
        Type of feature library to use (polynomial, fourier, identity)

    Returns:
    --------
    SINDy model
    """
    print("CREATING SINDY MODEL")
    # Define the optimizer based on the optimizer_type
    if optimizer_type == "STLSQ":
        # Sequentially thresholded least squares
        optimizer = STLSQ(
            threshold=threshold,
            alpha=alpha,
        )

    elif optimizer_type == "SR3":
        # Sparse relaxed regularized regression

        optimizer = SR3(thresholder="l1", nu=1, tol=1e-6, max_iter=10000)

    elif optimizer_type == "SSR":
        # Stepwise Sparse Least Squares Estimator

        optimizer = SSR(normalize_columns=True, kappa=alpha)

    elif optimizer_type == "FROLS":
        # SINDy with physics-informed constraints
        optimizer = FROLS(normalize_columns=True, kappa=alpha)

    else:
        # Default to STLSQ if an invalid optimizer is specified
        print(f"Warning: Unknown optimizer type '{optimizer_type}'. Using STLSQ.")
        optimizer = STLSQ(threshold=threshold, alpha=alpha)

    # Define differentiation method
    der = SmoothedFiniteDifference()

    # Choose the appropriate library based on type
    if library_type == "polynomial":
        # Use PolynomialLibrary for polynomial terms

        lib = PolynomialLibrary(degree=poly_order)

    elif library_type == "fourier":
        # Use FourierLibrary for sine and cosine terms

        lib = FourierLibrary(n_frequencies=poly_order)

    elif library_type == "identity":
        # Use IdentityLibrary for simple linear relationships

        lib = IdentityLibrary()

    elif library_type == "combined":
        # Use a combination of libraries

        polynomial_library = PolynomialLibrary(degree=poly_order)
        fourier_library = FourierLibrary(n_frequencies=min(3, poly_order))
        lib = GeneralizedLibrary([polynomial_library, fourier_library])

    else:
        # Default to polynomial library if an invalid type is specified
        print(
            f"Warning: Unknown library type '{library_type}'. Using polynomial library."
        )

        lib = PolynomialLibrary(degree=poly_order)

    # Create SINDy model
    model = SINDy(
        discrete_time=True,
        feature_library=lib,
        differentiation_method=der,
        optimizer=optimizer,
    )

    return model


def train_sindy_model_multiple_trajectories(model, X_list, u_list, t=None):
    """
    Train a SINDy model with multiple trajectories

    Parameters:
    -----------
    model : SINDy model
        Initialized SINDy model
    X_list : list of array-like
        List of state trajectories from different days/experiments
    u_list : list of array-like
        List of input trajectories corresponding to each state trajectory
    t : float or array-like, optional
        Time step(s)

    Returns:
    --------
    Trained SINDy model
    """
    # Fit the model with multiple trajectories
    model.fit(X_list, u=u_list, multiple_trajectories=True)

    return model


def single_step_prediction(model, X, u=None):
    """
    Perform single-step prediction with the trained model
    """
    # Make predictions
    X_pred = model.predict(X, u=u)
    return X_pred


def multi_step_prediction(model, X_init, u, steps, dt=1):
    """
    Perform multi-step prediction with the trained model

    Parameters:
    -----------
    model : SINDy model
        Trained SINDy model
    X_init : array-like
        Initial state
    u : array-like
        Input variables
    steps : int
        Number of steps to predict ahead
    dt : float
        Time step size

    Returns:
    --------
    array-like
        Predicted states for each step
    """
    # Initialize array to store predictions
    X_pred = np.zeros((steps, X_init.shape[1]))

    # Set initial state
    X_pred[0] = X_init[0]

    # Perform multi-step prediction
    for i in range(1, steps):
        # Use the previous prediction and corresponding input
        X_pred[i] = model.predict(X_pred[i - 1].reshape(1, -1), u=u[i].reshape(1, -1))[
            0
        ]

    return X_pred


def plot_results(
    actual,
    predicted_single,
    predicted_multi=None,
    variable_idx=1,
    variable_name="Top Floor Temperature (°C)",
    results_dir=None,
    mode="both",
    title_suffix="",
):
    """
    Plot actual vs predicted values
    """
    plt.figure(figsize=(12, 6))

    # Create x-axis indices
    x = np.arange(len(actual))

    # Extract the variable of interest
    actual_var = actual[:, variable_idx]
    pred_single_var = predicted_single[:, variable_idx]

    # Plot actual values
    plt.plot(x, actual_var, "k-", label="Actual", linewidth=2)

    # Plot single-step predictions
    if mode in ["single", "both"]:
        plt.plot(
            x, pred_single_var, "b-", label="Single-step Prediction", linewidth=1.5
        )

    # Plot multi-step predictions if provided
    if predicted_multi is not None and mode in ["multi", "both"]:
        pred_multi_var = predicted_multi[:, variable_idx]
        plt.plot(
            x[: len(pred_multi_var)],
            pred_multi_var,
            "r--",
            label="Multi-step Prediction",
            linewidth=1.5,
        )

    # Add labels and title
    plt.xlabel("Time Step")
    plt.ylabel(variable_name)
    plt.title(f"{variable_name} Prediction {title_suffix}")
    plt.legend()
    plt.grid(True)

    # Add MSE to the plot
    if mode in ["single", "both"]:
        mse_single = mean_squared_error(actual_var, pred_single_var)
        rmse_single = np.sqrt(mse_single)
        plt.text(
            0.05,
            0.95,
            f"Single-step RMSE: {rmse_single:.4f}°C",
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    if predicted_multi is not None and mode in ["multi", "both"]:
        # Calculate MSE only for the available multi-step predictions
        pred_multi_var = predicted_multi[:, variable_idx]
        actual_var_subset = actual_var[: len(pred_multi_var)]
        if len(pred_multi_var) > 0:
            mse_multi = mean_squared_error(actual_var_subset, pred_multi_var)
            rmse_multi = np.sqrt(mse_multi)
            plt.text(
                0.05,
                0.90,
                f"Multi-step RMSE: {rmse_multi:.4f}°C",
                transform=plt.gca().transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8),
            )

    plt.tight_layout()

    # Save the plot if results directory is provided
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        floor_type = "ground_floor" if variable_idx == 0 else "top_floor"
        file_name = (
            f"{floor_type}_{mode}_prediction{title_suffix.replace(' ', '_')}.png"
        )
        file_path = os.path.join(results_dir, file_name)
        plt.savefig(file_path, dpi=300)
        print(f"Plot saved to {file_path}")


def plot_residuals_comparison(
    actual,
    pred_single,
    pred_multi=None,
    variable_idx=0,
    variable_name="Temperature",
    results_dir=None,
    title_suffix="",
):
    """
    Create a comparison of residuals between single-step and multi-step predictions
    """
    # Extract the variable of interest
    actual_var = actual[:, variable_idx]
    pred_single_var = pred_single[:, variable_idx]

    # Calculate residuals
    residuals_single = actual_var - pred_single_var

    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Residual Analysis for {variable_name} {title_suffix}", fontsize=16)

    # 1. Time Series Plot (actual vs predicted)
    axes[0, 0].plot(actual_var, "b-", label="Actual", linewidth=2)
    axes[0, 0].plot(pred_single_var, "r--", label="Predicted", linewidth=1.5)
    axes[0, 0].set_title("Time Series: Actual vs Predicted")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Temperature (°C)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Residuals over time
    axes[0, 1].plot(residuals_single, "g-", linewidth=1.5)
    axes[0, 1].axhline(y=0, color="r", linestyle="-", linewidth=1)
    axes[0, 1].set_title("Residuals over Time")
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Residual (°C)")
    axes[0, 1].grid(True)

    # 3. Histogram of residuals
    axes[1, 0].hist(residuals_single, bins=20, alpha=0.7, color="skyblue")
    axes[1, 0].axvline(x=0, color="r", linestyle="-", linewidth=1)
    # Add normal distribution curve for comparison
    import scipy.stats as stats

    x = np.linspace(min(residuals_single), max(residuals_single), 100)
    axes[1, 0].plot(
        x,
        stats.norm.pdf(x, np.mean(residuals_single), np.std(residuals_single))
        * len(residuals_single)
        * (max(residuals_single) - min(residuals_single))
        / 20,
        "r-",
        linewidth=1.5,
        label="Normal Dist.",
    )
    axes[1, 0].set_title("Histogram of Residuals")
    axes[1, 0].set_xlabel("Residual (°C)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()

    # 4. Scatterplot of Actual vs Predicted
    axes[1, 1].scatter(actual_var, pred_single_var, c="blue", alpha=0.5)

    # Add a perfect prediction line
    min_val = min(np.min(actual_var), np.min(pred_single_var))
    max_val = max(np.max(actual_var), np.max(pred_single_var))
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)

    axes[1, 1].set_title("Actual vs Predicted")
    axes[1, 1].set_xlabel("Actual Temperature (°C)")
    axes[1, 1].set_ylabel("Predicted Temperature (°C)")
    axes[1, 1].grid(True)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_var, pred_single_var))

    # Add statistics to the plot
    stats_text = (
        f"Mean Error: {np.mean(residuals_single):.4f}°C\n"
        f"Std Dev: {np.std(residuals_single):.4f}°C\n"
        f"Min: {np.min(residuals_single):.4f}°C\n"
        f"Max: {np.max(residuals_single):.4f}°C\n"
        f"RMSE: {rmse:.4f}°C"
    )

    fig.text(
        0.5,
        0.01,
        stats_text,
        ha="center",
        bbox=dict(facecolor="white", alpha=0.9),
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot if results directory is provided
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        floor_type = "ground_floor" if variable_idx == 0 else "top_floor"
        file_name = f"{floor_type}_residuals{title_suffix.replace(' ', '_')}.png"
        file_path = os.path.join(results_dir, file_name)
        plt.savefig(file_path, dpi=300)
        print(f"Residual plot saved to {file_path}")


def tune_hyperparameters(X, u, param_grid, dt=1):
    """
    Tune hyperparameters for SINDy model, considering both single-step and multi-step predictions
    for both ground and top floor temperatures.

    Parameters:
    -----------
    X : array-like
        State variables
    u : array-like
        Input variables
    param_grid : dict
        Dictionary of parameter grids to search
    dt : float
        Time step

    Returns:
    --------
    dict
        Best parameters, best model, and results
    """
    # Generate all parameter combinations
    params_list = list(ParameterGrid(param_grid))
    print(f"Starting hyperparameter search with {len(params_list)} combinations...")

    # Initialize best model tracking
    best_mse = float("inf")
    best_params = None
    best_model = None
    all_results = []

    # Loop through all parameter combinations with progress bar
    for params in tqdm(params_list, desc="Tuning hyperparameters"):
        # Create model with current parameters
        model = create_sindy_model(
            threshold=params["threshold"],
            alpha=params["alpha"],
            poly_order=params["poly_order"],
            library_type=(
                params["library_type"] if "library_type" in params else "polynomial"
            ),
            optimizer_type=(
                params["optimizer_type"] if "optimizer_type" in params else "STLSQ"
            ),
        )

        try:
            # Train the model
            print("Fit model with multiple trajectories")
            X = [np.array(x) for x in X]  # Convert each trajectory to a NumPy array
            u = [np.array(u_) for u_ in u]  # Convert control inputs
            # print(f"Shape of x_input: {X.shape}, Shape of u_input: {u.shape}")

            model.fit(X, u=u, multiple_trajectories=True)
            print("Model fit successful")
            # model.fit(X, u=u, multiple_trajectories=True)

            # --------------- SINGLE-STEP PREDICTION ---------------
            # Make single-step predictions
            X_pred_single = model.predict(X, u=u, multiple_trajectories=True)
            print("single step prediction successful")
            # Calculate single-step MSE for each floor
            single_mse_ground = np.mean(
                [
                    np.sqrt(mean_squared_error(X[i][:, 0], X_pred_single[i][:, 0]))
                    for i in range(len(X))
                ]
            )
            single_mse_top = np.mean(
                [
                    np.sqrt(mean_squared_error(X[i][:, 1], X_pred_single[i][:, 1]))
                    for i in range(len(X))
                ]
            )
            print("single step mse calculated")
            print("single step mse calculated", single_mse_ground, single_mse_top)
            # --------------- MULTI-STEP PREDICTION ---------------
            # Make multi-step predictions
            # Fix for the multi-step prediction part of the code
            # --------------- MULTI-STEP PREDICTION ---------------
            # Make multi-step predictions
            X_pred_multi = [np.zeros_like(X[i]) for i in range(len(X))]

            print("performing multi step prediction")
            # Perform multi-step prediction for each trajectory separately
            for traj_idx in range(len(X)):
                steps = len(X[traj_idx])  # Get trajectory length
                X_pred_multi[traj_idx][0] = X[traj_idx][
                    0
                ]  # Initialize with first state

                for i in range(1, steps):
                    # Get current state (the previous prediction)
                    current_state = X_pred_multi[traj_idx][i - 1].reshape(1, -1)

                    # Get current control input
                    if u[traj_idx].ndim > 1:
                        current_input = u[traj_idx][i - 1].reshape(1, -1)
                    else:
                        current_input = np.array([[u[traj_idx][i - 1]]])

                    # Make single-step prediction
                    next_state = model.predict(
                        current_state, u=current_input, multiple_trajectories=False
                    )

                    # Store the prediction
                    X_pred_multi[traj_idx][i] = next_state

            print("multi step prediction successful")

            # Calculate multi-step MSE for each floor
            multi_mse_ground = np.mean(
                [
                    np.sqrt(mean_squared_error(X[i][:, 0], X_pred_multi[i][:, 0]))
                    for i in range(len(X))
                ]
            )
            multi_mse_top = np.mean(
                [
                    np.sqrt(mean_squared_error(X[i][:, 1], X_pred_multi[i][:, 1]))
                    for i in range(len(X))
                ]
            )
            print("multi step mse calculated", multi_mse_ground, multi_mse_top)
            # --------------- AVERAGE MSE CALCULATION ---------------
            # Calculate the average MSE across all prediction types and floors
            mse_values = [
                single_mse_ground,
                single_mse_top,
                multi_mse_ground,
                multi_mse_top,
            ]
            avg_mse = np.mean(mse_values)

            # Record this result
            result = {
                "params": params,
                "single_mse_ground": single_mse_ground,
                "single_mse_top": single_mse_top,
                "multi_mse_ground": multi_mse_ground,
                "multi_mse_top": multi_mse_top,
                "avg_mse": avg_mse,
                "complexity": len(model.coefficients()),
            }
            all_results.append(result)

            # Check if this is the best model so far
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_params = params
                best_model = model

                print(f"New best model: AVG MSE={best_mse:.6f}")
                print(
                    f"  Single-step Ground: {single_mse_ground:.6f}, Top: {single_mse_top:.6f}"
                )
                print(
                    f"  Multi-step Ground: {multi_mse_ground:.6f}, Top: {multi_mse_top:.6f}"
                )
                print(f"  Params: {best_params}")

        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            continue

    return {
        "best_params": best_params,
        "best_model": best_model,
        "best_mse": best_mse,
        "all_results": all_results,
    }


def main():
    # Define file paths and directories
    training_data_path_1 = "../../Data/dollhouse-data-2025-02-28.csv"  # First day data
    training_data_path_2 = (
        "../../Data/dollhouse-data-2025-02-19_testing.csv"  # Second day data
    )
    results_dir = "results/multiple_trajectories"  # Directory for results

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Load both datasets
    X_train_1, u_train_1 = load_and_prepare_data(training_data_path_1)
    X_train_2, u_train_2 = load_and_prepare_data(training_data_path_2)

    print(f"Loaded first dataset: {X_train_1.shape} states, {u_train_1.shape} inputs")
    print(f"Loaded second dataset: {X_train_2.shape} states, {u_train_2.shape} inputs")

    # Create lists for multiple trajectories
    X_list = [np.array(X_train_1), np.array(X_train_2)]
    u_list = [np.array(u_train_1), np.array(u_train_2)]

    param_grid = {
        "threshold": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        "alpha": [0.001, 0.001, 0.01, 0.1, 1.0],
        "poly_order": [1, 2, 3],  # Linear and quadratic terms
        "library_type": ["polynomial", "fourier", "identity"],
        "optimizer_type": ["STLSQ", "SR3", "SSR", "FROLS"],
    }

    # Perform hyperparameter tuning
    tuning_results = tune_hyperparameters(
        X_list,
        u_list,
        param_grid,
    )

    # Get best model and parameters
    best_model = tuning_results["best_model"]
    best_params = tuning_results["best_params"]
    best_mse = tuning_results["best_mse"]

    print(f"\nBest hyperparameters found: {best_params}")
    print(f"Best validation MSE: {best_mse:.6f}")

    # Create serializable results
    serializable_results = {
        "best_params": best_params,
        "best_mse": float(best_mse),
        "all_results": [
            {
                "params": result["params"],
                "single_mse_ground": float(result["single_mse_ground"]),
                "single_mse_top": float(result["single_mse_top"]),
                "multi_mse_ground": float(result["multi_mse_ground"]),
                "multi_mse_top": float(result["multi_mse_top"]),
                "avg_mse": float(result["avg_mse"]),
                "complexity": int(result["complexity"]),
            }
            for result in tuning_results["all_results"]
        ],
    }

    # Save results to current directory without checking args.results_dir
    output_filename = "tuning_results.json"
    with open(output_filename, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Tuning results saved to {output_filename}")

    print(f"\nBest hyperparameters found: {best_params}")
    print(f"Best validation MSE: {best_mse:.6f}")

    # Train the model with multiple trajectories
    # model = train_sindy_model_multiple_trajectories(model, X_list, u_list)
    model = best_model
    # Print the discovered equations
    print("\nDiscovered SINDy Equations (from multiple trajectories):")
    try:
        print(model.print())
    except:
        print("Could not print equations directly, showing coefficients instead:")
        coefficients = model.coefficients()
        for i, coef in enumerate(coefficients):
            print(f"Equation {i} coefficients:", coef)

    # Evaluate on each dataset separately
    # [rest of your evaluation code]
    # Define variable names for plots
    variable_names = ["Ground Floor Temperature (°C)", "Top Floor Temperature (°C)"]

    # Make predictions on both original and new data
    # Single-step predictions
    X_pred_train_single = single_step_prediction(model, X_train_1, u=u_train_1)
    X_pred_new_single = single_step_prediction(model, X_train_2, u=u_train_2)

    # Multi-step predictions with error handling
    steps_train = len(X_train_1)
    try:
        print("Attempting multi-step prediction on training data...")
        X_pred_train_multi = multi_step_prediction(
            model, X_train_1[:1], u_train_1, steps_train
        )
        train_multi_success = True
        print("Multi-step prediction on training data successful.")
    except Exception as e:
        print(f"Multi-step prediction failed on training data: {e}")
        print("Continuing with single-step predictions only for training data.")
        X_pred_train_multi = None
        train_multi_success = False

    steps_new = len(X_train_2)
    try:
        print("Attempting multi-step prediction on new data...")
        X_pred_new_multi = multi_step_prediction(
            model, X_train_2[:1], u_train_2, steps_new
        )
        new_multi_success = True
        print("Multi-step prediction on new data successful.")
    except Exception as e:
        print(f"Multi-step prediction failed on new data: {e}")
        print("Continuing with single-step predictions only for new data.")
        X_pred_new_multi = None
        new_multi_success = False

    # Calculate and print RMSE for each prediction
    print("\nModel Performance (RMSE):")
    print("Original Training Data:")
    for idx, name in enumerate(variable_names):
        train_single_rmse = np.sqrt(
            mean_squared_error(X_train_1[:, idx], X_pred_train_single[:, idx])
        )
        print(f"  {name}: Single-step RMSE = {train_single_rmse:.4f}°C", end="")

        if train_multi_success:
            train_multi_rmse = np.sqrt(
                mean_squared_error(
                    X_train_1[: len(X_pred_train_multi), idx],
                    X_pred_train_multi[:, idx],
                )
            )
            print(f", Multi-step RMSE = {train_multi_rmse:.4f}°C")
        else:
            print(", Multi-step RMSE = N/A (prediction failed)")

    print("\nNew Data:")
    for idx, name in enumerate(variable_names):
        new_single_rmse = np.sqrt(
            mean_squared_error(X_train_2[:, idx], X_pred_new_single[:, idx])
        )
        print(f"  {name}: Single-step RMSE = {new_single_rmse:.4f}°C", end="")

        if new_multi_success:
            new_multi_rmse = np.sqrt(
                mean_squared_error(
                    X_train_2[: len(X_pred_new_multi), idx], X_pred_new_multi[:, idx]
                )
            )
            print(f", Multi-step RMSE = {new_multi_rmse:.4f}°C")
        else:
            print(", Multi-step RMSE = N/A (prediction failed)")

    # Plot results for both floors on training data
    for idx, name in enumerate(variable_names):
        # Predictions on training data
        train_mode = "both" if train_multi_success else "single"
        plot_results(
            X_train_1,
            X_pred_train_single,
            X_pred_train_multi if train_multi_success else None,
            variable_idx=idx,
            variable_name=name,
            results_dir=results_dir,
            mode=train_mode,
            title_suffix="(Training Data)",
        )

        # Residual analysis on training data
        plot_residuals_comparison(
            X_train_1,
            X_pred_train_single,
            variable_idx=idx,
            variable_name=name,
            results_dir=results_dir,
            title_suffix="(Training Data)",
        )

    # Plot results for both floors on new data
    for idx, name in enumerate(variable_names):
        # Predictions on new data
        new_mode = "both" if new_multi_success else "single"
        plot_results(
            X_train_2,
            X_pred_new_single,
            X_pred_new_multi if new_multi_success else None,
            variable_idx=idx,
            variable_name=name,
            results_dir=results_dir,
            mode=new_mode,
            title_suffix="(New Data)",
        )

        # Residual analysis on new data
        plot_residuals_comparison(
            X_train_2,
            X_pred_new_single,
            variable_idx=idx,
            variable_name=name,
            results_dir=results_dir,
            title_suffix="(New Data)",
        )

    print(f"\nAll results saved to {results_dir}")


if __name__ == "__main__":
    main()
