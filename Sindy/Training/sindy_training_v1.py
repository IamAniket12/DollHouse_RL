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


# Functions to add to your existing script
def load_and_prepare_data(file_path):
    """
    Load and prepare data for SINDy model training

    Parameters:
    -----------
    file_path : str
        Path to the CSV file with temperature data
    test_size : float
        Fraction of data to use for testing

    Returns:
    --------
    tuple
        (X, u, X_train, X_test, u_train, u_test, scaler_X, scaler_u)
    """
    # Load the data
    data = pd.read_csv(file_path)

    # Map categorical values to numerical
    data["Ground Floor Light"] = data["Ground Floor Light"].map({"ON": 1, "OFF": 0})
    data["Ground Floor Window"] = data["Ground Floor Window"].map(
        {"OPEN": 1, "CLOSED": 0}
    )
    data["Top Floor Light"] = data["Top Floor Light"].map({"ON": 1, "OFF": 0})
    data["Top Floor Window"] = data["Top Floor Window"].map({"OPEN": 1, "CLOSED": 0})

    # Extract features (X) and input (u) variables
    X = data[["Ground Floor Temperature (°C)", "Top Floor Temperature (°C)"]].values
    u = data[
        [
            "Ground Floor Light",
            "Ground Floor Window",
            "Top Floor Light",
            "Top Floor Window",
            "External Temperature (°C)",
        ]
    ].values

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
        optimizer = STLSQ(threshold=threshold, alpha=alpha)

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


def train_sindy_model(model, X, u=None, t=None):
    """
    Train a SINDy model

    Parameters:
    -----------
    model : SINDy model
        Initialized SINDy model
    X : array-like
        State variables
    u : array-like, optional
        Input variables
    t : float or array-like, optional
        Time step(s)

    Returns:
    --------
    Trained SINDy model
    """
    # Fit the model
    model.fit(X, u=u)

    return model


def interpret_sindy_equations_improved(model, variable_names, input_names=None):
    """
    Improved function to interpret SINDy equations in a human-readable format

    Parameters:
    -----------
    model : SINDy model
        Trained SINDy model
    variable_names : list of str
        Names of the state variables
    input_names : list of str, optional
        Names of the input variables

    Returns:
    --------
    list of str
        Human-readable equations
    """
    # Get the coefficients from the model
    coefficients = model.coefficients()

    model.print()

    # These are the known library terms based on how we defined our model
    # For our simple model with 2 variables (x0, x1) and 5 inputs (u0-u4)
    # The expected features are in this order:
    # Constant(1) for each variable: 1
    # Linear terms for state variables: x0, x1
    # Linear terms for input variables: u0, u1, u2, u3, u4

    feature_mapping = {
        0: "1",  # Constant term
        1: variable_names[0],  # x0
        2: variable_names[1],  # x1
        3: input_names[0] if input_names else "u0",  # u0
        4: input_names[1] if input_names else "u1",  # u1
        5: input_names[2] if input_names else "u2",  # u2
        6: input_names[3] if input_names else "u3",  # u3
        7: input_names[4] if input_names else "u4",  # u4
    }

    # Initialize list for interpreted equations
    interpreted_equations = []

    # Process each dimension (state variable)
    for i, coef_vector in enumerate(coefficients):
        if i >= len(variable_names):
            break

        # Get the variable name for this equation
        var_name = variable_names[i]

        # Start building the equation string
        equation = f"{var_name}[k+1] = "

        # Add terms with non-zero coefficients
        terms = []

        # The order of coefficients should match our expected feature mapping
        for j, coef in enumerate(coef_vector):
            if abs(coef) > 1e-10:  # Only include non-zero terms
                # Get the feature name based on its index
                feature_idx = j % len(feature_mapping)
                feature_name = feature_mapping.get(feature_idx, f"term{j}")

                # Format the term
                if coef == 1.0:
                    terms.append(f"{feature_name}")
                elif coef == -1.0:
                    terms.append(f"-{feature_name}")
                else:
                    terms.append(f"{coef:.4f} {feature_name}")

        # Join terms with + signs (handling negatives)
        equation_terms = ""
        for term in terms:
            if term.startswith("-"):
                equation_terms += f" {term}"
            elif equation_terms:
                equation_terms += f" + {term}"
            else:
                equation_terms += term

        equation += equation_terms if equation_terms else "0"
        interpreted_equations.append(equation)

    return interpreted_equations


def plot_hyperparameter_results(tuning_results, results_dir=None):
    """
    Plot hyperparameter tuning results

    Parameters:
    -----------
    tuning_results : dict
        Results from hyperparameter tuning
    results_dir : str, optional
        Directory to save results
    """
    all_results = tuning_results["all_results"]
    best_params = tuning_results["best_params"]

    # Extract parameters and MSE values
    thresholds = []
    alphas = []
    poly_orders = []
    mse_values = []
    complexity = []

    for result in all_results:
        thresholds.append(result["params"]["threshold"])
        alphas.append(result["params"]["alpha"])
        poly_orders.append(result["params"]["poly_order"])
        mse_values.append(result["avg_mse"])
        # Check if complexity exists in the result
        if "complexity" in result:
            complexity.append(result["complexity"])

    # Create a DataFrame for easier analysis
    results_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "alpha": alphas,
            "poly_order": poly_orders,
            "mse": mse_values,
        }
    )

    # Add complexity if available
    if complexity:
        results_df["complexity"] = complexity

    # Plot threshold vs MSE
    plt.figure(figsize=(10, 6))

    # Group by threshold and poly_order, take mean MSE
    threshold_groups = (
        results_df.groupby(["threshold", "poly_order"])["mse"].mean().reset_index()
    )

    # Plot for each poly_order
    for poly_order in threshold_groups["poly_order"].unique():
        data = threshold_groups[threshold_groups["poly_order"] == poly_order]
        plt.plot(data["threshold"], data["mse"], "o-", label=f"Poly Order {poly_order}")

    plt.xscale("log")
    plt.xlabel("Threshold")
    plt.ylabel("Average MSE")
    plt.title("Threshold vs MSE by Polynomial Order")
    plt.grid(True)
    plt.legend()

    # Highlight best threshold
    best_threshold_idx = threshold_groups[
        (threshold_groups["threshold"] == best_params["threshold"])
        & (threshold_groups["poly_order"] == best_params["poly_order"])
    ].index

    if len(best_threshold_idx) > 0:
        best_point = threshold_groups.iloc[best_threshold_idx[0]]
        plt.scatter(
            [best_point["threshold"]],
            [best_point["mse"]],
            c="red",
            s=100,
            marker="*",
            label="Best",
        )

    plt.legend()

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, "threshold_vs_mse.png"), dpi=300)

    plt.tight_layout()

    # Plot alpha vs MSE
    plt.figure(figsize=(10, 6))

    # Group by alpha and poly_order, take mean MSE
    alpha_groups = (
        results_df.groupby(["alpha", "poly_order"])["mse"].mean().reset_index()
    )

    # Plot for each poly_order
    for poly_order in alpha_groups["poly_order"].unique():
        data = alpha_groups[alpha_groups["poly_order"] == poly_order]
        plt.plot(data["alpha"], data["mse"], "o-", label=f"Poly Order {poly_order}")

    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Average MSE")
    plt.title("Alpha vs MSE by Polynomial Order")
    plt.grid(True)
    plt.legend()

    # Highlight best alpha
    best_alpha_idx = alpha_groups[
        (alpha_groups["alpha"] == best_params["alpha"])
        & (alpha_groups["poly_order"] == best_params["poly_order"])
    ].index

    if len(best_alpha_idx) > 0:
        best_point = alpha_groups.iloc[best_alpha_idx[0]]
        plt.scatter(
            [best_point["alpha"]],
            [best_point["mse"]],
            c="red",
            s=100,
            marker="*",
            label="Best",
        )

    plt.legend()

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, "alpha_vs_mse.png"), dpi=300)

    plt.tight_layout()
    # plt.show()

    # Plot complexity vs MSE (only if complexity is available)
    if "complexity" in results_df.columns:
        plt.figure(figsize=(10, 6))

        # Create a scatter plot with colormap based on polynomial order
        scatter = plt.scatter(
            results_df["complexity"],
            results_df["mse"],
            c=results_df["poly_order"],
            cmap="viridis",
            alpha=0.7,
        )

        # Now we can add the colorbar since we have a mappable object (scatter)
        plt.colorbar(scatter, label="Polynomial Order")

        plt.xlabel("Model Complexity (Number of Terms)")
        plt.ylabel("MSE")
        plt.title("Model Complexity vs MSE")
        plt.grid(True)

        # Highlight the best point
        best_idx = results_df[
            (results_df["threshold"] == best_params["threshold"])
            & (results_df["alpha"] == best_params["alpha"])
            & (results_df["poly_order"] == best_params["poly_order"])
        ].index

        if len(best_idx) > 0:
            best_point = results_df.iloc[best_idx[0]]
            plt.scatter(
                [best_point["complexity"]],
                [best_point["mse"]],
                c="red",
                s=100,
                marker="*",
                label="Best Model",
            )

        plt.legend()

        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(os.path.join(results_dir, "complexity_vs_mse.png"), dpi=300)

        plt.tight_layout()
        # plt.show()
    else:
        print("Complexity data not available for plotting")


def single_step_prediction(model, X, u=None):
    """
    Perform single-step prediction with the trained model

    Parameters:
    -----------
    model : SINDy model
        Trained SINDy model
    X : array-like
        State variables
    u : array-like, optional
        Input variables

    Returns:
    --------
    array-like
        Predicted values
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
            model.fit(X, u=u, t=dt)

            # --------------- SINGLE-STEP PREDICTION ---------------
            # Make single-step predictions
            X_pred_single = model.predict(X, u=u)

            # Calculate single-step MSE for each floor
            single_mse_ground = np.sqrt(
                mean_squared_error(X[:, 0], X_pred_single[:, 0])
            )
            single_mse_top = np.sqrt(mean_squared_error(X[:, 1], X_pred_single[:, 1]))

            # --------------- MULTI-STEP PREDICTION ---------------
            # Make multi-step predictions
            steps = len(X)
            X_pred_multi = np.zeros((steps, X.shape[1]))
            X_pred_multi[0] = X[0]  # Start with the first actual state

            # Perform multi-step prediction
            for i in range(1, steps):
                X_pred_multi[i] = model.predict(
                    X_pred_multi[i - 1].reshape(1, -1), u=u[i].reshape(1, -1)
                )[0]

            # Calculate multi-step MSE for each floor (using only the predicted steps)
            multi_mse_ground = np.sqrt(
                mean_squared_error(X[1:, 0], X_pred_multi[1:, 0])
            )
            multi_mse_top = np.sqrt(mean_squared_error(X[1:, 1], X_pred_multi[1:, 1]))

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


def plot_results(
    actual,
    predicted_single,
    predicted_multi=None,
    variable_idx=1,
    variable_name="Top Floor Temperature (°C)",
    results_dir=None,
    mode="both",
    params=None,
):
    """
    Plot actual vs predicted values for a specific variable and save to results directory

    Parameters:
    -----------
    actual : array-like
        Actual state variables
    predicted_single : array-like
        Single-step predictions
    predicted_multi : array-like, optional
        Multi-step predictions
    variable_idx : int
        Index of the variable to plot
    variable_name : str
        Name of the variable to plot
    results_dir : str, optional
        Directory to save results
    mode : str
        Prediction mode ('single', 'multi', or 'both')
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
    plt.title(f"{variable_name} Prediction")
    plt.legend()
    plt.grid(True)

    # Add MSE to the plot
    if mode in ["single", "both"]:
        mse_single = np.sqrt(mean_squared_error(actual_var, pred_single_var))
        plt.text(
            0.05,
            0.95,
            f"Single-step RMSE: {mse_single:.4f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    if predicted_multi is not None and mode in ["multi", "both"]:
        # Calculate MSE only for the available multi-step predictions
        pred_multi_var = predicted_multi[:, variable_idx]
        actual_var_subset = actual_var[: len(pred_multi_var)]
        if len(pred_multi_var) > 0:
            mse_multi = np.sqrt(mean_squared_error(actual_var_subset, pred_multi_var))
            plt.text(
                0.05,
                0.90,
                f"Multi-step RMSE: {mse_multi:.4f}",
                transform=plt.gca().transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8),
            )

    plt.tight_layout()

    # Save the plot if results directory is provided
    if results_dir:
        # Create the results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Create filename based on variable name and prediction mode
        var_string = "ground_floor" if variable_idx == 0 else "top_floor"
        plt.savefig(
            os.path.join(results_dir, f"{var_string}_{mode}_prediction.png"), dpi=300
        )
        print(
            f"Plot saved to {os.path.join(results_dir, f'{var_string}_{mode}_prediction.png')}"
        )


def plot_hyperparameter_results(tuning_results, results_dir=None):
    """
    Plot hyperparameter tuning results

    Parameters:
    -----------
    tuning_results : dict
        Results from hyperparameter tuning
    results_dir : str, optional
        Directory to save results
    """
    all_results = tuning_results["all_results"]
    best_params = tuning_results["best_params"]

    # Extract parameters and MSE values
    thresholds = []
    alphas = []
    poly_orders = []
    mse_values = []
    complexity = []

    for result in all_results:
        thresholds.append(result["params"]["threshold"])
        alphas.append(result["params"]["alpha"])
        poly_orders.append(result["params"]["poly_order"])
        mse_values.append(result["avg_mse"])
        # Check if complexity exists in the result
        if "complexity" in result:
            complexity.append(result["complexity"])
    library_types = []
    for result in all_results:
        # Add to your existing lists
        library_types.append(result["params"]["library_type"])
    # Create a DataFrame for easier analysis
    results_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "alpha": alphas,
            "poly_order": poly_orders,
            "mse": mse_values,
            "library_type": library_types,
        }
    )

    # Add complexity if available
    if complexity:
        results_df["complexity"] = complexity

    # Plot threshold vs MSE
    plt.figure(figsize=(10, 6))

    # Group by threshold and poly_order, take mean MSE
    threshold_groups = (
        results_df.groupby(["threshold", "poly_order"])["mse"].mean().reset_index()
    )

    # Plot for each poly_order
    for poly_order in threshold_groups["poly_order"].unique():
        data = threshold_groups[threshold_groups["poly_order"] == poly_order]
        plt.plot(data["threshold"], data["mse"], "o-", label=f"Poly Order {poly_order}")

    plt.xscale("log")
    plt.xlabel("Threshold")
    plt.ylabel("Average MSE")
    plt.title("Threshold vs MSE by Polynomial Order")
    plt.grid(True)
    plt.legend()

    # Highlight best threshold
    best_threshold_idx = threshold_groups[
        (threshold_groups["threshold"] == best_params["threshold"])
        & (threshold_groups["poly_order"] == best_params["poly_order"])
    ].index

    if len(best_threshold_idx) > 0:
        best_point = threshold_groups.iloc[best_threshold_idx[0]]
        plt.scatter(
            [best_point["threshold"]],
            [best_point["mse"]],
            c="red",
            s=100,
            marker="*",
            label="Best",
        )

    plt.legend()

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, "threshold_vs_mse.png"), dpi=300)

    plt.tight_layout()
    # plt.show()

    # Plot alpha vs MSE
    plt.figure(figsize=(10, 6))

    # Group by alpha and poly_order, take mean MSE
    alpha_groups = (
        results_df.groupby(["alpha", "poly_order"])["mse"].mean().reset_index()
    )

    # Plot for each poly_order
    for poly_order in alpha_groups["poly_order"].unique():
        data = alpha_groups[alpha_groups["poly_order"] == poly_order]
        plt.plot(data["alpha"], data["mse"], "o-", label=f"Poly Order {poly_order}")

    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Average MSE")
    plt.title("Alpha vs MSE by Polynomial Order")
    plt.grid(True)
    plt.legend()

    # Highlight best alpha
    best_alpha_idx = alpha_groups[
        (alpha_groups["alpha"] == best_params["alpha"])
        & (alpha_groups["poly_order"] == best_params["poly_order"])
    ].index

    if len(best_alpha_idx) > 0:
        best_point = alpha_groups.iloc[best_alpha_idx[0]]
        plt.scatter(
            [best_point["alpha"]],
            [best_point["mse"]],
            c="red",
            s=100,
            marker="*",
            label="Best",
        )

    plt.legend()

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, "alpha_vs_mse.png"), dpi=300)

    plt.tight_layout()
    # plt.show()

    # Plot complexity vs MSE (only if complexity is available)
    if "complexity" in results_df.columns:
        plt.figure(figsize=(10, 6))

        # Create a scatter plot with colormap based on polynomial order
        scatter = plt.scatter(
            results_df["complexity"],
            results_df["mse"],
            c=results_df["poly_order"],
            cmap="viridis",
            alpha=0.7,
        )

        # Now we can add the colorbar since we have a mappable object (scatter)
        plt.colorbar(scatter, label="Polynomial Order")

        plt.xlabel("Model Complexity (Number of Terms)")
        plt.ylabel("MSE")
        plt.title("Model Complexity vs MSE")
        plt.grid(True)

        # Highlight the best point
        best_idx = results_df[
            (results_df["threshold"] == best_params["threshold"])
            & (results_df["alpha"] == best_params["alpha"])
            & (results_df["poly_order"] == best_params["poly_order"])
        ].index

        if len(best_idx) > 0:
            best_point = results_df.iloc[best_idx[0]]
            plt.scatter(
                [best_point["complexity"]],
                [best_point["mse"]],
                c="red",
                s=100,
                marker="*",
                label="Best Model",
            )

        plt.legend()

        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(os.path.join(results_dir, "complexity_vs_mse.png"), dpi=300)

        plt.tight_layout()
        # plt.show()
    else:
        print("Complexity data not available for plotting")


def calculate_rmse(actual, predicted, variable_idx=1):
    """
    Calculate Mean Squared Error for a specific variable

    Parameters:
    -----------
    actual : array-like
        Actual state variables
    predicted : array-like
        Predicted state variables
    variable_idx : int
        Index of the variable to calculate MSE for

    Returns:
    --------
    float
        MSE value
    """
    return np.sqrt(
        mean_squared_error(actual[:, variable_idx], predicted[:, variable_idx])
    )


def plot_residuals(
    actual,
    predicted,
    variable_idx=0,
    variable_name="Temperature",
    results_dir=None,
    mode="single",
):
    """
    Create comprehensive residual plots to evaluate model performance

    Parameters:
    -----------
    actual : array-like
        Actual state variables
    predicted : array-like
        Predicted state variables
    variable_idx : int
        Index of the variable to analyze
    variable_name : str
        Name of the variable for plot titles
    results_dir : str, optional
        Directory to save results
    mode : str
        Prediction mode ('single' or 'multi')
    """
    # Extract the variable of interest
    actual_var = actual[:, variable_idx]
    pred_var = predicted[:, variable_idx]

    # Calculate residuals
    residuals = actual_var - pred_var
    rmse = np.sqrt(mean_squared_error(actual_var, pred_var))

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Residual Analysis for {variable_name} ({mode}-step prediction)\nRMSE: {rmse:.4f}°C",
        fontsize=16,
    )

    # 1. Time Series Plot (actual vs predicted)
    axes[0, 0].plot(actual_var, "b-", label="Actual", linewidth=2)
    axes[0, 0].plot(pred_var, "r--", label="Predicted", linewidth=1.5)
    axes[0, 0].set_title("Time Series: Actual vs Predicted")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Temperature (°C)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Residuals over time
    axes[0, 1].plot(residuals, "g-", linewidth=1.5)
    axes[0, 1].axhline(y=0, color="r", linestyle="-", linewidth=1)
    axes[0, 1].set_title("Residuals over Time")
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Residual (°C)")
    axes[0, 1].grid(True)

    # 3. Histogram of residuals
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, color="skyblue")
    axes[1, 0].axvline(x=0, color="r", linestyle="-", linewidth=1)
    # Add normal distribution curve for comparison
    import scipy.stats as stats

    x = np.linspace(min(residuals), max(residuals), 100)
    axes[1, 0].plot(
        x,
        stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
        * len(residuals)
        * (max(residuals) - min(residuals))
        / 20,
        "r-",
        linewidth=1.5,
        label="Normal Dist.",
    )
    axes[1, 0].set_title("Histogram of Residuals")
    axes[1, 0].set_xlabel("Residual (°C)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()

    # 4. Q-Q plot (to check normality of residuals)
    import statsmodels.api as sm

    sm.qqplot(residuals, line="45", ax=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot (Normality Check)")

    # Add statistics to the plot
    stats_text = (
        f"Mean: {np.mean(residuals):.4f}°C\n"
        f"Std Dev: {np.std(residuals):.4f}°C\n"
        f"Min: {np.min(residuals):.4f}°C\n"
        f"Max: {np.max(residuals):.4f}°C\n"
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
        plt.savefig(
            os.path.join(results_dir, f"{floor_type}_{mode}_residuals.png"), dpi=300
        )
        print(
            f"Residual plot saved to {os.path.join(results_dir, f'{floor_type}_{mode}_residuals.png')}"
        )

    # plt.show()


def plot_residuals_comparison(
    actual,
    pred_single,
    pred_multi,
    variable_idx=0,
    variable_name="Temperature",
    results_dir=None,
):
    """
    Create a comparison of residuals between single-step and multi-step predictions

    Parameters:
    -----------
    actual : array-like
        Actual state variables
    pred_single : array-like
        Single-step predicted state variables
    pred_multi : array-like
        Multi-step predicted state variables
    variable_idx : int
        Index of the variable to analyze
    variable_name : str
        Name of the variable for plot titles
    results_dir : str, optional
        Directory to save results
    """
    # Extract the variable of interest
    actual_var = actual[:, variable_idx]
    pred_single_var = pred_single[:, variable_idx]
    pred_multi_var = pred_multi[:, variable_idx][
        : len(actual_var)
    ]  # Ensure same length

    # Calculate residuals
    residuals_single = actual_var - pred_single_var
    residuals_multi = actual_var[: len(pred_multi_var)] - pred_multi_var

    # Calculate RMSE
    rmse_single = np.sqrt(mean_squared_error(actual_var, pred_single_var))
    rmse_multi = np.sqrt(
        mean_squared_error(actual_var[: len(pred_multi_var)], pred_multi_var)
    )

    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Residual Comparison for {variable_name}\nSingle-step RMSE: {rmse_single:.4f}°C, Multi-step RMSE: {rmse_multi:.4f}°C",
        fontsize=16,
    )

    # 1. Side-by-side time series plot
    axes[0, 0].plot(actual_var, "k-", label="Actual", linewidth=2)
    axes[0, 0].plot(pred_single_var, "b--", label="Single-step", linewidth=1.5)
    axes[0, 0].plot(pred_multi_var, "r-.", label="Multi-step", linewidth=1.5)
    axes[0, 0].set_title("Time Series: Actual vs Predicted")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Temperature (°C)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Residuals over time
    axes[0, 1].plot(
        residuals_single, "b-", label="Single-step", linewidth=1.5, alpha=0.7
    )
    axes[0, 1].plot(residuals_multi, "r-", label="Multi-step", linewidth=1.5, alpha=0.7)
    axes[0, 1].axhline(y=0, color="k", linestyle="-", linewidth=1)
    axes[0, 1].set_title("Residuals over Time")
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Residual (°C)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. Histogram of residuals
    axes[1, 0].hist(
        residuals_single, bins=20, alpha=0.5, color="blue", label="Single-step"
    )
    axes[1, 0].hist(
        residuals_multi, bins=20, alpha=0.5, color="red", label="Multi-step"
    )
    axes[1, 0].axvline(x=0, color="k", linestyle="-", linewidth=1)
    axes[1, 0].set_title("Histogram of Residuals")
    axes[1, 0].set_xlabel("Residual (°C)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()

    # 4. Scatterplot of Actual vs Predicted
    axes[1, 1].scatter(
        actual_var, pred_single_var, c="blue", alpha=0.5, label="Single-step"
    )
    axes[1, 1].scatter(
        actual_var[: len(pred_multi_var)],
        pred_multi_var,
        c="red",
        alpha=0.5,
        label="Multi-step",
    )

    # Add a perfect prediction line
    min_val = min(np.min(actual_var), np.min(pred_single_var), np.min(pred_multi_var))
    max_val = max(np.max(actual_var), np.max(pred_single_var), np.max(pred_multi_var))
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)

    axes[1, 1].set_title("Actual vs Predicted")
    axes[1, 1].set_xlabel("Actual Temperature (°C)")
    axes[1, 1].set_ylabel("Predicted Temperature (°C)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Add statistics to the plot
    stats_text = (
        f"Single-step: Mean Error = {np.mean(residuals_single):.4f}°C, Std Dev = {np.std(residuals_single):.4f}°C\n"
        f"Multi-step: Mean Error = {np.mean(residuals_multi):.4f}°C, Std Dev = {np.std(residuals_multi):.4f}°C\n"
        f"Single-step RMSE: {rmse_single:.4f}°C, Multi-step RMSE: {rmse_multi:.4f}°C"
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
        plt.savefig(
            os.path.join(results_dir, f"{floor_type}_residuals_comparison.png"), dpi=300
        )
        print(
            f"Residual comparison plot saved to {os.path.join(results_dir, f'{floor_type}_residuals_comparison.png')}"
        )

    # plt.show()


def plot_prediction_mse_comparison(tuning_results, results_dir=None):
    """
    Plot comparison of single-step and multi-step MSE for both floors

    Parameters:
    -----------
    tuning_results : dict
        Results from hyperparameter tuning
    results_dir : str, optional
        Directory to save results
    """
    all_results = tuning_results["all_results"]
    best_params = tuning_results["best_params"]

    # Sort results by average MSE
    sorted_results = sorted(all_results, key=lambda x: x["avg_mse"])

    # Take the top 20 models
    top_results = sorted_results[:20]

    # Extract data for plotting
    labels = [f"Model {i+1}" for i in range(len(top_results))]
    single_ground = [result["single_mse_ground"] for result in top_results]
    single_top = [result["single_mse_top"] for result in top_results]
    multi_ground = [result["multi_mse_ground"] for result in top_results]
    multi_top = [result["multi_mse_top"] for result in top_results]

    # Set up the plot
    plt.figure(figsize=(14, 8))
    bar_width = 0.2
    index = np.arange(len(labels))

    # Plot bars for each metric
    plt.bar(
        index,
        single_ground,
        bar_width,
        label="Single-step Ground Floor",
        color="blue",
        alpha=0.7,
    )
    plt.bar(
        index + bar_width,
        single_top,
        bar_width,
        label="Single-step Top Floor",
        color="green",
        alpha=0.7,
    )
    plt.bar(
        index + 2 * bar_width,
        multi_ground,
        bar_width,
        label="Multi-step Ground Floor",
        color="orange",
        alpha=0.7,
    )
    plt.bar(
        index + 3 * bar_width,
        multi_top,
        bar_width,
        label="Multi-step Top Floor",
        color="red",
        alpha=0.7,
    )

    # Add best model indicator
    best_idx = None
    for i, result in enumerate(top_results):
        if (
            result["params"]["threshold"] == best_params["threshold"]
            and result["params"]["alpha"] == best_params["alpha"]
            and result["params"]["poly_order"] == best_params["poly_order"]
        ):
            best_idx = i
            break

    if best_idx is not None:
        plt.scatter(
            [best_idx + 1.5 * bar_width],
            [top_results[best_idx]["avg_mse"]],
            s=200,
            c="black",
            marker="*",
            label="Best Model",
        )

    # Customize the plot
    plt.xlabel("Model Ranking")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE Comparison Across Top 20 Models")
    plt.xticks(index + 1.5 * bar_width, labels, rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save plot if results directory is provided
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, "mse_comparison.png"), dpi=300)

    # plt.show()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Train and evaluate SINDy model for temperature prediction"
    )
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to the CSV data file"
    )
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="single",
        choices=["single", "multi", "both"],
        help="Prediction mode: single-step, multi-step, or both",
    )
    parser.add_argument(
        "--floor",
        type=str,
        default="top",
        choices=["ground", "top", "both"],
        help="Floor to predict: ground floor, top floor, or both",
    )
    # Add new arguments for tuning
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Fraction of training data to use for validation (only used with --tune)",
    )
    parser.add_argument(
        "--poly_order",
        type=int,
        default=1,
        help="Order of polynomial features (if not tuning)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0001,
        help="Threshold for sparse regression",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Regularization parameter for sparse regression",
    )
    # Add new argument for library type
    parser.add_argument(
        "--library_type",
        type=str,
        default="polynomial",
        choices=["polynomial", "fourier", "identity"],
        help="Type of feature library to use",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="STLSQ",
        choices=["STLSQ", "SR3", "FROLS", "SSLE", "SINDyPI"],
        help="Type of optimizer to use for SINDy",
    )
    parser.add_argument(
        "--dt", type=float, default=1, help="Time step for multi-step prediction"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results", help="Directory to save results"
    )

    args = parser.parse_args()

    # Create results directory if it doesn't exist
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)

    # Load and prepare data
    X, u = load_and_prepare_data(args.file_path)

    # Define variable and input names for interpretation
    variable_names = ["Ground Floor Temperature (°C)", "Top Floor Temperature (°C)"]
    input_names = [
        "Ground Floor Light",
        "Ground Floor Window",
        "Top Floor Light",
        "Top Floor Window",
        "External Temperature (°C)",
    ]

    # Check if we should do hyperparameter tuning
    if args.tune:
        print("Performing hyperparameter tuning...")

        # Define parameter grid
        param_grid = {
            "threshold": [0.00001, 0.0001, 0.001, 0.01, 0.1],
            "alpha": [0.001, 0.001, 0.01, 0.1, 1.0],
            "poly_order": [1, 2, 3],  # Linear and quadratic terms
            "library_type": ["polynomial", "fourier", "identity"],
            "optimizer_type": ["STLSQ", "SR3", "SSR", "FROLS"],
        }

        # Perform hyperparameter tuning
        tuning_results = tune_hyperparameters(
            X,
            u,
            param_grid,
        )

        # Get best model and parameters
        best_model = tuning_results["best_model"]
        best_params = tuning_results["best_params"]
        best_mse = tuning_results["best_mse"]

        print(f"\nBest hyperparameters found: {best_params}")
        print(f"Best validation MSE: {best_mse:.6f}")

        # Save tuning results

        if args.results_dir:
            with open(os.path.join(args.results_dir, "tuning_results.json"), "w") as f:
                # Convert numpy arrays and other non-serializable objects to lists
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
                json.dump(serializable_results, f, indent=2)

            print(
                f"Tuning results saved to {os.path.join(args.results_dir, 'tuning_results.json')}"
            )

        # Plot hyperparameter tuning results
        plot_hyperparameter_results(tuning_results, args.results_dir)
        plot_prediction_mse_comparison(tuning_results, args.results_dir)
        # Use the best model for further evaluation
        model = best_model

        # If the best model is None (all combinations failed), create a new model with best parameters
        if model is None:
            print(
                "All combinations failed. Using best parameters to create a new model."
            )
            model = create_sindy_model(
                threshold=best_params["threshold"],
                alpha=best_params["alpha"],
                poly_order=best_params["poly_order"],
            )
            model = train_sindy_model(model, X, u=u, t=args.dt)

        # Store the best parameters for plotting
        params = best_params

    else:
        # Use the provided parameters
        print(
            f"Using provided parameters: threshold={args.threshold}, alpha={args.alpha}, poly_order={args.poly_order}"
        )
        model = create_sindy_model(args.threshold, args.alpha, args.poly_order)
        model = train_sindy_model(model, X, u=u, t=args.dt)

        # Store the provided parameters for plotting
        params = {
            "threshold": args.threshold,
            "alpha": args.alpha,
            "poly_order": args.poly_order,
        }

    # Print model parameters
    print(
        f"\nSINDy Model Parameters: threshold={params['threshold']}, alpha={params['alpha']}, poly_order={params['poly_order']}"
    )

    # Print model coefficients
    print("\nSINDy Model Coefficients:")
    coefficients = model.coefficients()

    for i, coef in enumerate(coefficients):
        print(f"Equation {i} coefficients:", coef)

    # Print interpreted equations
    print("\nSINDy Model Interpreted Equations:")
    try:
        # Try the improved interpretation function
        interpreted_equations = interpret_sindy_equations_improved(
            model, variable_names, input_names
        )
        for eq in interpreted_equations:
            print(eq)
    except Exception as e:
        print(f"Error interpreting equations with improved method: {e}")
        print("Continuing with predictions...")

    # Save the parameters and model equations to a text file
    if args.results_dir:
        param_info = "Best " if args.tune else "Used "
        with open(os.path.join(args.results_dir, "sindy_model_info.txt"), "w") as f:
            f.write(f"SINDy Model Information:\n\n")
            f.write(f"{param_info}Parameters:\n")
            f.write(f"  threshold: {params['threshold']}\n")
            f.write(f"  alpha: {params['alpha']}\n")
            f.write(f"  poly_order: {params['poly_order']}\n\n")

            f.write("Model Equations:\n")
            for i, eq in enumerate(interpreted_equations):
                f.write(f"{variable_names[i]} equation:\n{eq}\n\n")

            f.write("\nCoefficients:\n")
            for i, coef in enumerate(coefficients):
                f.write(f"Equation {i}: {coef}\n")

        print(
            f"Model information saved to {os.path.join(args.results_dir, 'sindy_model_info.txt')}"
        )

    # Convert test data back to original scale for evaluation
    X_test_original = X

    # Determine which floor(s) to analyze
    if args.floor == "ground":
        floor_indices = [0]
    elif args.floor == "top":
        floor_indices = [1]
    else:  # 'both'
        floor_indices = [0, 1]

    # Make predictions based on mode
    if args.prediction_mode in ["single", "both"]:
        # Single-step prediction
        X_pred_single = single_step_prediction(model, X, u=u)

        # Calculate and print MSE for each selected floor
        for idx in floor_indices:
            single_mse = calculate_rmse(X_test_original, X_pred_single, idx)
            print(
                f"\nSingle-step prediction RMSE for {variable_names[idx]}: {single_mse:.4f}"
            )
    else:
        # If we're only doing multi-step, we still need a placeholder for the plot function
        X_pred_single = X_test_original

    if args.prediction_mode in ["multi", "both"]:
        # Multi-step prediction - start with the first test point and predict forward
        steps = len(X)
        X_pred_multi_scaled = multi_step_prediction(model, X[:1], u, steps, dt=args.dt)

        # Convert predictions back to original scale
        X_pred_multi = X_pred_multi_scaled

        # Calculate and print MSE for each selected floor
        for idx in floor_indices:
            multi_mse = calculate_rmse(
                X_test_original[: len(X_pred_multi)], X_pred_multi, idx
            )
            print(
                f"Multi-step prediction RMSE for {variable_names[idx]}: {multi_mse:.4f}"
            )
    else:
        X_pred_multi = None
    # Add these imports at the top if needed

    # Add residual plots
    print("\nGenerating residual plots for model evaluation...")

    # Import required libraries if not already imported
    import statsmodels.api as sm

    # Create individual residual plots for each prediction type and floor
    for idx in floor_indices:
        variable_name = variable_names[idx]

        if args.prediction_mode in ["single", "both"]:
            # Single-step residual plot
            plot_residuals(
                X_test_original,
                X_pred_single,
                variable_idx=idx,
                variable_name=variable_name,
                results_dir=args.results_dir,
                mode="single",
            )

        if args.prediction_mode in ["multi", "both"]:
            # Multi-step residual plot
            plot_residuals(
                X_test_original[: len(X_pred_multi)],
                X_pred_multi,
                variable_idx=idx,
                variable_name=variable_name,
                results_dir=args.results_dir,
                mode="multi",
            )

        if args.prediction_mode == "both":
            # Comparison plot for both prediction types
            plot_residuals_comparison(
                X_test_original,
                X_pred_single,
                X_pred_multi,
                variable_idx=idx,
                variable_name=variable_name,
                results_dir=args.results_dir,
            )

    # Print RMSE values for clearer interpretation
    print("\nModel Performance (RMSE):")
    for idx in floor_indices:
        if args.prediction_mode in ["single", "both"]:
            single_rmse = calculate_rmse(X_test_original, X_pred_single, idx)
            print(f"Single-step {variable_names[idx]} RMSE: {single_rmse:.4f}°C")

        if args.prediction_mode in ["multi", "both"]:
            multi_rmse = calculate_rmse(
                X_test_original[: len(X_pred_multi)], X_pred_multi, idx
            )

            print(f"Multi-step {variable_names[idx]} RMSE: {multi_rmse:.4f}°C")

    # Plot results for each selected floor
    for idx in floor_indices:
        plot_results(
            X_test_original,
            X_pred_single,
            X_pred_multi if args.prediction_mode in ["multi", "both"] else None,
            variable_idx=idx,
            variable_name=variable_names[idx],
            results_dir=args.results_dir,
            mode=args.prediction_mode,
            params=params,  # Include parameters in the plot
        )


if __name__ == "__main__":
    main()
