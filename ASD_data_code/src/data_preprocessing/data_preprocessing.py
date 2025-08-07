

import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def standardize_column_names(df, standard_names):
    """Renames columns based on a dictionary of standard names."""
    df.rename(columns=standard_names, inplace=True)

def merge_data(df1, df2, df3, df4, on, how='outer'):
    """Merges four dataframes on specified columns using the specified method."""
    return df1.merge(df2, on=on, how=how).merge(df3, on=on, how=how).merge(df4, on=on, how=how)

def preprocess_features(X):
    """Preprocess numeric features: fill missing values and scale."""
    # Separate numeric and non-numeric columns
    numeric_cols = X.select_dtypes(include=['number']).columns

    return X[numeric_cols] 


def diagnose_merge(phenotypic_data, anat_qap, dti_qap, functional_qap):
    """Diagnoses potential issues in merging by checking key overlaps."""
    datasets = {"phenotypic_data": phenotypic_data, "anat_qap": anat_qap, "dti_qap": dti_qap, "functional_qap": functional_qap}
    for name, df in datasets.items():
        logging.info(f"{name}: Unique SUB_IDs = {len(df['SUB_ID'].unique())}, Unique SITE_IDs = {len(df['SITE_ID'].unique())}")

    for name, df in datasets.items():
        for other_name, other_df in datasets.items():
            if name != other_name:
                unmatched = df[~df['SUB_ID'].isin(other_df['SUB_ID'])]
                logging.info(f"SUB_IDs in {name} not in {other_name}: {len(unmatched)}")

def preprocess_data(phenotypic_path, anat_qap_path, dti_qap_path, functional_qap_path, target_label='DX_GROUP'):
    """
    Preprocesses the data from the given paths and returns features (X) and target (y).
    """
    logging.info("Starting data preprocessing...")

    # Load datasets
    try:
        logging.info(f"Loading Phenotypic Data from: {phenotypic_path}")
        phenotypic_data = pd.read_csv(phenotypic_path, encoding='latin1')

        logging.info(f"Loading Anat QAP Data from: {anat_qap_path}")
        anat_qap = pd.read_csv(anat_qap_path, encoding='latin1')

        logging.info(f"Loading DTI QAP Data from: {dti_qap_path}")
        dti_qap = pd.read_csv(dti_qap_path, encoding='latin1')

        logging.info(f"Loading Functional QAP Data from: {functional_qap_path}")
        functional_qap = pd.read_csv(functional_qap_path, encoding='latin1')
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None, None

    # Standardize column names
    column_mapping = {'Sub_ID': 'SUB_ID', 'Site_ID': 'SITE_ID'}
    standardize_column_names(phenotypic_data, column_mapping)
    standardize_column_names(anat_qap, column_mapping)
    standardize_column_names(dti_qap, column_mapping)
    standardize_column_names(functional_qap, column_mapping)

    # Diagnose merge
    diagnose_merge(phenotypic_data, anat_qap, dti_qap, functional_qap)

    # Merge data
    merged_data = merge_data(
        phenotypic_data, anat_qap, dti_qap, functional_qap,
        on=['SUB_ID', 'SITE_ID'], how='outer'
    )
    logging.info(f"Merged Data Shape (outer join): {merged_data.shape}")

    # Log missing data
    missing_counts = merged_data.isnull().sum()
    logging.info(f"Missing values after merge:\n{missing_counts}")

    # Check for target variable
    if target_label not in merged_data.columns:
        logging.error(f"Target label '{target_label}' not found in the merged data.")
        return None, None
    else:
        logging.info(f"Target label '{target_label}' is present.")

    # Log target label distribution
    target_counts = merged_data[target_label].value_counts()
    logging.info(f"Target label distribution:\n{target_counts}")

    # Prepare features and target
    merged_data = merged_data.dropna(subset=[target_label])  # Ensure target label has no NaNs
    X = merged_data.drop(columns=[target_label])
    y = merged_data[target_label]

    # Preprocess features
    try:
        X_preprocessed = preprocess_features(X)
    except Exception as e:
        logging.error(f"Error during feature preprocessing: {e}")
        return None, None

    logging.info("Data preprocessing completed successfully.")
    return X_preprocessed, y

if __name__ == '__main__':
    # Update paths to absolute paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    phenotypic_path = os.path.join(base_path, '../../data/ABIDEII_Composite_Phenotypic.csv')
    anat_qap_path = os.path.join(base_path, '../../data/ABIDEII_MRI_Quality_Metrics/anat_qap.csv')
    dti_qap_path = os.path.join(base_path, '../../data/ABIDEII_MRI_Quality_Metrics/dti_qap.csv')
    functional_qap_path = os.path.join(base_path, '../../data/ABIDEII_MRI_Quality_Metrics/functional_qap.csv')

    X, y = preprocess_data(phenotypic_path, anat_qap_path, dti_qap_path, functional_qap_path)

    if X is not None and y is not None:
        logging.info(f"Preprocessed features shape: {X.shape}")
        logging.info(f"Target variable shape: {y.shape}")
    else:
        logging.error("Data preprocessing failed.")
