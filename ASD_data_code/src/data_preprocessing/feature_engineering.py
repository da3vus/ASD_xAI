
# from sklearn.decomposition import PCA  # Uncomment if PCA is needed
import pandas as pd
import logging

def create_features(X, y, apply_pca=False, n_components=50):

    logging.info("=== Starting Feature Engineering Process ===")

    # Select only numeric columns
    logging.info("Selecting numeric features...")
    X_numeric = X.select_dtypes(include=[float, int])
    non_numeric_cols = X.select_dtypes(exclude=[float, int]).columns
    logging.info(f"Numeric features selected: {X_numeric.shape[1]} columns")
    logging.info(f"Non-numeric columns excluded: {non_numeric_cols.tolist()}")

    logging.info("=== Feature Engineering Process Completed===")
    return X_numeric, y
    
    
if __name__ == "__main__":
    print("=== Running Feature Engineering Script ===")

    # Sample dataset for testing
    data = {
        "Feature1": [1, 2, 3, 4, 5],
        "Feature2": [5, 4, 3, 2, 1],
        "Feature3": [1.0, None, 3.0, 4.0, 5.0],
        "NonNumeric": ["A", "B", "C", "D", "E"],
        "Target": [0, 1, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    X = df.drop(columns=["Target"])
    y = df["Target"]

    print("\n=== Initial Data ===")
    print(f"Features Shape: {X.shape}")
    print(f"Target Shape: {y.shape}")
    print("First 5 rows of features:")
    print(X.head())
    print("First 5 rows of target:")
    print(y.head())

    # Apply feature engineering without PCA
    X_processed, y_processed = create_features(X, y, apply_pca=False)

    print("\n=== Processed Data Without PCA ===")
    print(f"Processed Features Shape: {X_processed.shape}")
    print("First 5 rows of processed features:")
    print(X_processed.head())
    print("Processed Target:")
    print(y_processed.head())


