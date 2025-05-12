import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse

# Import SINDy packages
from pysindy import SINDy, PolynomialLibrary, FourierLibrary, GeneralizedLibrary, STLSQ
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import STLSQ, SR3, SSR, FROLS


def load_and_prepare_data(file_path, warmup_period_minutes=1):
    """
    Load and prepare data for SINDy model training.

    Args:
        file_path: Path to CSV data file
        warmup_period_minutes: Warmup period to exclude from analysis

    Returns:
        X (np.array): State variables (temperatures)
        u (np.array): Input variables
        scaler_X (StandardScaler): Scaler used for state variables (or None)
        warmup_indices (np.array): Indices representing the warmup period
    """
    print(f"Loading data from {file_path}")

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
    data["Top_Window_Ext_Effect"] = data["Top Floor Window"] * data["Top_Ext_Temp_Diff"]

    # Lag features for thermal inertia
    data["Ground_Temp_Lag1"] = data["Ground Floor Temperature (°C)"].shift(1)
    data["Top_Temp_Lag1"] = data["Top Floor Temperature (°C)"].shift(1)
    data["Ext_Temp_Lag1"] = data["External Temperature (°C)"].shift(1)
    data["Ground_Temp_Lag2"] = data["Ground Floor Temperature (°C)"].shift(2)
    data["Top_Temp_Lag2"] = data["Top Floor Temperature (°C)"].shift(2)
    data["Ext_Temp_Lag2"] = data["External Temperature (°C)"].shift(2)

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
        "Ground_Temp_Lag2",
        "Top_Temp_Lag2",
        "Ext_Temp_Lag2",
        "Ground_Temp_Rate",
        "Top_Temp_Rate",
    ]

    u = data[u_columns].values

    # Get indices for warmup period
    warmup_indices = np.where(warmup_mask)[0]

    # No normalization since you found better results without it
    return X, u, None, warmup_indices


def filter_warmup_period(X, u, warmup_indices):
    """
    Filter out the warmup period from data arrays

    Args:
        X: State variables
        u: Input variables
        warmup_indices: Indices representing the warmup period

    Returns:
        X_filtered: Filtered state variables
        u_filtered: Filtered input variables
    """
    # Create a mask of all indices (True)
    mask = np.ones(len(X), dtype=bool)

    # Set the warmup period indices to False
    mask[warmup_indices] = False

    # Filter the arrays using the mask
    X_filtered = X[mask]
    u_filtered = u[mask]

    return X_filtered, u_filtered


def train_sindy_model(file_path, output_dir=None, threshold=0.1, alpha=0.1, degree=2):
    """
    Train a SINDy model on the given data file.

    Args:
        file_path: Path to the data file to train on
        output_dir: Directory to save the model (optional)
        threshold: Threshold parameter for STLSQ
        alpha: Alpha parameter for STLSQ
        degree: Degree for polynomial library

    Returns:
        SINDy model: The trained model
    """
    if file_path is None:
        raise ValueError("A data file path must be provided to train the SINDy model")

    print(
        f"Training SINDy model with parameters: threshold={threshold}, alpha={alpha}, polynomial degree={degree}"
    )

    # Create a SINDy model with the specified parameters
    feature_library = PolynomialLibrary(degree=1, include_bias=True)
    # optimizer = STLSQ(threshold=threshold, alpha=alpha)
    optimizer = SR3(threshold=0.1, nu=0.1, thresholder="L0")
    # Define differentiation method
    der = FiniteDifference()

    # Create SINDy model (discrete time because we're using time series data)
    model = SINDy(
        discrete_time=True,
        feature_library=feature_library,
        differentiation_method=der,
        optimizer=optimizer,
    )

    # Load and prepare the data
    X, u, _, warmup_indices = load_and_prepare_data(file_path)

    # Filter out warmup period
    X_filtered, u_filtered = filter_warmup_period(X, u, warmup_indices)

    print(f"Training SINDy model on {len(X_filtered)} data points...")

    # Train the model
    model.fit(X_filtered, u=u_filtered)

    print("SINDy model training complete.")

    # Print model equations if possible
    try:
        print("\nDiscovered model equations:")
        print(model.print())
    except Exception as e:
        print(f"Could not print equations directly: {e}")
        print("Coefficients:")
        coefficients = model.coefficients()
        for i, coef in enumerate(coefficients):
            print(f"Equation {i+1}: {coef}")

    # Save the model if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_info = {
            "threshold": threshold,
            "alpha": alpha,
            "degree": degree,
            "timestamp": timestamp,
            "data_file": file_path,
        }

        # Save model info as JSON
        import json

        with open(
            os.path.join(output_dir, f"sindy_model_info_{timestamp}.json"), "w"
        ) as f:
            json.dump(model_info, f, indent=4)

        print(f"Model info saved to {output_dir}")

    return model


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a SINDy model")
    parser.add_argument(
        "--data",
        type=str,
        default=[
            "../../Data/dollhouse-data-2025-03-24.csv",
        ],
        help="Path to data file for training SINDy model",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Directory to save model info"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="Threshold parameter for STLSQ"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Alpha parameter for STLSQ"
    )
    parser.add_argument(
        "--degree", type=int, default=2, help="Degree for polynomial library"
    )

    args = parser.parse_args()

    # Train the model
    model = train_sindy_model(
        file_path=args.data,
        output_dir=args.output,
        threshold=args.threshold,
        alpha=args.alpha,
        degree=args.degree,
    )
