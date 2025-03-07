import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics import mean_squared_error
from pysindy import SINDy, PolynomialLibrary, FourierLibrary, CustomLibrary
from pysindy.optimizers import STLSQ, SR3, FROLS, SSR
from pysindy.differentiation import SmoothedFiniteDifference
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
    normalize_columns=True,
    max_iter=10000,
):
    lib = PolynomialLibrary(degree=poly_order)
    der = SmoothedFiniteDifference()
    optimizer = STLSQ(threshold=0.01, alpha=0.1)

    # Create SINDy model
    model = SINDy(
        feature_library=lib,
        optimizer=optimizer,
        discrete_time=True,
        differentiation_method=der,
    )

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


def main():
    # Define file paths and directories
    training_data_path_1 = "../../Data/dollhouse-data-2025-02-28.csv"  # First day data
    training_data_path_2 = (
        "../../Data/dollhouse-data-2025-02-19_testing.csv"  # Second day data
    )
    new_data_path = (
        "../../Data/dollhouse-data-2025-02-27_testing.csv"  # New data to evaluate
    )
    results_dir = "results/predictions"  # Directory for results

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Load both training datasets
    X_train_1, u_train_1 = load_and_prepare_data(training_data_path_1)
    X_train_2, u_train_2 = load_and_prepare_data(training_data_path_2)

    print(f"Loaded first dataset: {X_train_1.shape} states, {u_train_1.shape} inputs")
    print(f"Loaded second dataset: {X_train_2.shape} states, {u_train_2.shape} inputs")

    # Create lists for multiple trajectories
    X_list = [np.array(X_train_1), np.array(X_train_2)]
    u_list = [np.array(u_train_1), np.array(u_train_2)]

    # Load new data for testing
    X_new, u_new = load_and_prepare_data(new_data_path)
    print(f"Loaded new data: {X_new.shape} states, {u_new.shape} inputs")

    # Define model parameters
    params = {
        "threshold": 0.001,
        "alpha": 0.001,
        "poly_order": 1,
        "library_type": "polynomial",
        "optimizer_type": "STLSQ",
    }

    # Create model with parameters
    model = create_sindy_model(
        threshold=params["threshold"],
        alpha=params["alpha"],
        poly_order=params["poly_order"],
        library_type=params["library_type"],
        optimizer_type=params["optimizer_type"],
    )

    # Fit the model on the multiple trajectories
    model.fit(X_list, u=u_list, multiple_trajectories=True)

    # Print the discovered equations
    print("\nDiscovered SINDy Equations (from multiple trajectories):")
    try:
        print(model.print())
    except:
        print("Could not print equations directly, showing coefficients instead:")
        coefficients = model.coefficients()
        for i, coef in enumerate(coefficients):
            print(f"Equation {i} coefficients:", coef)

    # Define variable names for plots
    variable_names = ["Ground Floor Temperature (°C)", "Top Floor Temperature (°C)"]

    # Make predictions on the new test data
    # Single-step prediction
    X_pred_new_single = single_step_prediction(model, X_new, u=u_new)

    # Multi-step prediction
    try:
        print("Attempting multi-step prediction on new data...")
        steps_new = len(X_new)
        X_pred_new_multi = multi_step_prediction(model, X_new[:1], u_new, steps_new)
        multi_success = True
        print("Multi-step prediction successful!")
    except Exception as e:
        print(f"Multi-step prediction failed: {e}")
        X_pred_new_multi = None
        multi_success = False

    # Calculate and print RMSE for test data
    print("\nTest Data Prediction Performance (RMSE):")
    for idx, name in enumerate(variable_names):
        # Single-step RMSE
        single_rmse = np.sqrt(
            mean_squared_error(X_new[:, idx], X_pred_new_single[:, idx])
        )
        print(f"  {name}: Single-step RMSE = {single_rmse:.4f}°C", end="")

        # Multi-step RMSE (if available)
        if multi_success:
            multi_rmse = np.sqrt(
                mean_squared_error(
                    X_new[: len(X_pred_new_multi), idx], X_pred_new_multi[:, idx]
                )
            )
            print(f", Multi-step RMSE = {multi_rmse:.4f}°C")
        else:
            print(", Multi-step RMSE = N/A (prediction failed)")

    # Plot results for both floors on test data
    for idx, name in enumerate(variable_names):
        # Generate mode based on multi-step success
        mode = "both" if multi_success else "single"

        # Create prediction plot
        plot_results(
            X_new,
            X_pred_new_single,
            X_pred_new_multi if multi_success else None,
            variable_idx=idx,
            variable_name=name,
            results_dir=results_dir,
            mode=mode,
            title_suffix="(Test Data)",
        )

        # Create residual analysis plot
        plot_residuals_comparison(
            X_new,
            X_pred_new_single,
            variable_idx=idx,
            variable_name=name,
            results_dir=results_dir,
            title_suffix="(Test Data)",
        )

    print(f"\nAll prediction results saved to {results_dir}")


if __name__ == "__main__":
    main()
