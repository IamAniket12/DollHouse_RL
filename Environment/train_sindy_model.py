"""
SINDy Model Training Module.

This module provides functionality to train Sparse Identification of Nonlinear Dynamics (SINDy)
models on dollhouse thermal data for system identification and dynamics modeling.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pysindy import SINDy, PolynomialLibrary
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import SR3, STLSQ


def load_and_prepare_data(
    file_path: str, warmup_period_minutes: float = 1
) -> Tuple[np.ndarray, np.ndarray, Optional[object], np.ndarray]:
    """
    Load and prepare thermal data for SINDy model training.

    This function loads CSV data, creates physics-informed features, and prepares
    the data for system identification using SINDy.

    Args:
        file_path: Path to CSV data file containing thermal measurements
        warmup_period_minutes: Duration in minutes to exclude from analysis as warmup

    Returns:
        Tuple containing:
        - X: State variables (temperatures) as numpy array
        - u: Input variables (controls + features) as numpy array
        - scaler_X: Scaler object (None in current implementation)
        - warmup_indices: Indices representing the warmup period

    Raises:
        FileNotFoundError: If the specified data file cannot be found
        KeyError: If required columns are missing from the data
    """
    print(f"Loading data from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = pd.read_csv(file_path)
    data = _process_timestamps(data)
    warmup_mask = _identify_warmup_period(data, warmup_period_minutes)

    print(f"Loaded {len(data)} data points from {file_path}")
    print(
        f"Warmup period: {warmup_period_minutes} minutes ({sum(warmup_mask)} data points)"
    )

    data = _map_categorical_values(data)
    data = _create_physics_features(data)
    data = _create_lag_features(data)
    data = _create_rate_features(data)
    data = data.ffill().fillna(0)

    X = _extract_state_variables(data)
    u = _extract_input_variables(data)
    warmup_indices = np.where(warmup_mask)[0]

    return X, u, None, warmup_indices


def _process_timestamps(data: pd.DataFrame) -> pd.DataFrame:
    """Process timestamp information and calculate time differences."""
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

    data["cumulative_time_seconds"] = data["time_diff_seconds"].cumsum()
    return data


def _identify_warmup_period(
    data: pd.DataFrame, warmup_period_minutes: float
) -> pd.Series:
    """Identify data points within the warmup period."""
    warmup_seconds = warmup_period_minutes * 60
    return data["cumulative_time_seconds"] <= warmup_seconds


def _map_categorical_values(data: pd.DataFrame) -> pd.DataFrame:
    """Map categorical control values to numerical representations."""
    categorical_mappings = {
        "Ground Floor Light": {"ON": 1, "OFF": 0},
        "Ground Floor Window": {"OPEN": 1, "CLOSED": 0},
        "Top Floor Light": {"ON": 1, "OFF": 0},
        "Top Floor Window": {"OPEN": 1, "CLOSED": 0},
    }

    for column, mapping in categorical_mappings.items():
        if column in data.columns:
            data[column] = data[column].map(mapping)

    return data


def _create_physics_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create physics-informed features for thermal dynamics."""
    required_columns = [
        "Top Floor Temperature (°C)",
        "Ground Floor Temperature (°C)",
        "External Temperature (°C)",
    ]

    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"Required column '{col}' not found in data")

    data["Floor_Temp_Diff"] = (
        data["Top Floor Temperature (°C)"] - data["Ground Floor Temperature (°C)"]
    )
    data["Ground_Ext_Temp_Diff"] = (
        data["Ground Floor Temperature (°C)"] - data["External Temperature (°C)"]
    )
    data["Top_Ext_Temp_Diff"] = (
        data["Top Floor Temperature (°C)"] - data["External Temperature (°C)"]
    )

    if "Ground Floor Window" in data.columns:
        data["Ground_Window_Ext_Effect"] = (
            data["Ground Floor Window"] * data["Ground_Ext_Temp_Diff"]
        )
    if "Top Floor Window" in data.columns:
        data["Top_Window_Ext_Effect"] = (
            data["Top Floor Window"] * data["Top_Ext_Temp_Diff"]
        )

    return data


def _create_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create lagged temperature features for thermal inertia modeling."""
    lag_columns = [
        ("Ground Floor Temperature (°C)", "Ground_Temp_Lag1", "Ground_Temp_Lag2"),
        ("Top Floor Temperature (°C)", "Top_Temp_Lag1", "Top_Temp_Lag2"),
        ("External Temperature (°C)", "Ext_Temp_Lag1", "Ext_Temp_Lag2"),
    ]

    for source_col, lag1_col, lag2_col in lag_columns:
        if source_col in data.columns:
            data[lag1_col] = data[source_col].shift(1)
            data[lag2_col] = data[source_col].shift(2)

    return data


def _create_rate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create temperature rate of change features."""
    rate_columns = [
        ("Ground Floor Temperature (°C)", "Ground_Temp_Rate"),
        ("Top Floor Temperature (°C)", "Top_Temp_Rate"),
    ]

    for temp_col, rate_col in rate_columns:
        if temp_col in data.columns:
            data[rate_col] = data[temp_col].diff() / data["time_diff_seconds"]

    return data


def _extract_state_variables(data: pd.DataFrame) -> np.ndarray:
    """Extract state variables for prediction."""
    state_columns = ["Ground Floor Temperature (°C)", "Top Floor Temperature (°C)"]

    for col in state_columns:
        if col not in data.columns:
            raise KeyError(f"Required state column '{col}' not found in data")

    return data[state_columns].values


def _extract_input_variables(data: pd.DataFrame) -> np.ndarray:
    """Extract input variables and features for the SINDy model."""
    input_columns = [
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

    available_columns = [col for col in input_columns if col in data.columns]
    missing_columns = [col for col in input_columns if col not in data.columns]

    if missing_columns:
        print(f"Warning: Missing input columns: {missing_columns}")

    return data[available_columns].values


def filter_warmup_period(
    X: np.ndarray, u: np.ndarray, warmup_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove warmup period data from training arrays.

    Args:
        X: State variables array
        u: Input variables array
        warmup_indices: Indices representing the warmup period

    Returns:
        Tuple of filtered (X, u) arrays with warmup period removed
    """
    mask = np.ones(len(X), dtype=bool)
    mask[warmup_indices] = False

    return X[mask], u[mask]


def train_sindy_model(
    file_path: str,
    output_dir: Optional[str] = None,
    threshold: float = 0.1,
    alpha: float = 0.1,
    degree: int = 2,
    optimizer_type: str = "sr3",
) -> SINDy:
    """
    Train a SINDy model for thermal dynamics identification.

    This function creates and trains a SINDy model on the provided thermal data,
    using sparse regression to identify the underlying dynamical system.

    Args:
        file_path: Path to the CSV data file for training
        output_dir: Directory to save model information (optional)
        threshold: Sparsity threshold for the optimizer
        alpha: Regularization parameter for STLSQ optimizer
        degree: Polynomial degree for the feature library
        optimizer_type: Type of optimizer ("sr3" or "stlsq")

    Returns:
        Trained SINDy model ready for prediction

    Raises:
        ValueError: If file_path is None or optimizer_type is invalid
        FileNotFoundError: If the data file cannot be found
    """
    if file_path is None:
        raise ValueError("A data file path must be provided to train the SINDy model")

    print(f"Training SINDy model with parameters:")
    print(f"  threshold={threshold}, alpha={alpha}")
    print(f"  polynomial degree={degree}, optimizer={optimizer_type}")

    model = _create_sindy_model(threshold, alpha, degree, optimizer_type)
    X, u, _, warmup_indices = load_and_prepare_data(file_path)
    X_filtered, u_filtered = filter_warmup_period(X, u, warmup_indices)

    print(f"Training SINDy model on {len(X_filtered)} data points...")

    model.fit(X_filtered, u=u_filtered)
    print("SINDy model training complete.")

    _display_model_equations(model)

    if output_dir:
        _save_model_info(
            output_dir, threshold, alpha, degree, optimizer_type, file_path
        )

    return model


def _create_sindy_model(
    threshold: float, alpha: float, degree: int, optimizer_type: str
) -> SINDy:
    """Create and configure a SINDy model."""
    feature_library = PolynomialLibrary(degree=1, include_bias=True)

    if optimizer_type.lower() == "sr3":
        optimizer = SR3(threshold=threshold, nu=0.1, thresholder="L0")
    elif optimizer_type.lower() == "stlsq":
        optimizer = STLSQ(threshold=threshold, alpha=alpha)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    differentiation_method = FiniteDifference()

    return SINDy(
        discrete_time=True,
        feature_library=feature_library,
        differentiation_method=differentiation_method,
        optimizer=optimizer,
    )


def _display_model_equations(model: SINDy) -> None:
    """Display the discovered model equations."""
    print("\nDiscovered model equations:")
    model_str = model.print()
    if model_str:
        print(model_str)
    else:
        print("Coefficients:")
        coefficients = model.coefficients()
        for i, coef in enumerate(coefficients):
            print(f"Equation {i+1}: {coef}")


def _save_model_info(
    output_dir: str,
    threshold: float,
    alpha: float,
    degree: int,
    optimizer_type: str,
    file_path: str,
) -> None:
    """Save model training information to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_info = {
        "threshold": threshold,
        "alpha": alpha,
        "degree": degree,
        "optimizer_type": optimizer_type,
        "timestamp": timestamp,
        "data_file": file_path,
    }

    info_path = os.path.join(output_dir, f"sindy_model_info_{timestamp}.json")
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=4)

    print(f"Model info saved to {output_dir}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train a SINDy model for thermal dynamics identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        default="../../Data/dollhouse-data-2025-03-24.csv",
        help="Path to data file for training SINDy model",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Directory to save model info"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Sparsity threshold for the optimizer",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Regularization parameter for STLSQ optimizer",
    )
    parser.add_argument(
        "--degree", type=int, default=2, help="Polynomial degree for feature library"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sr3",
        choices=["sr3", "stlsq"],
        help="Optimization algorithm to use",
    )

    args = parser.parse_args()

    model = train_sindy_model(
        file_path=args.data,
        output_dir=args.output,
        threshold=args.threshold,
        alpha=args.alpha,
        degree=args.degree,
        optimizer_type=args.optimizer,
    )

    return model


if __name__ == "__main__":
    main()
