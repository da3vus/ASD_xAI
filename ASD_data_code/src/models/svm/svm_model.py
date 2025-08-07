
import sys
import logging
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from fairlearn.postprocessing import ThresholdOptimizer
import shap
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, MetricFrame, selection_rate


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.data_preprocessing.data_preprocessing import preprocess_data

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#added class to drop missing columns, which will then be implemented in model pipeline
class DropAllMissingColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # Identify columns with all NaN values
        self.cols_to_drop_ = X.columns[X.isnull().all()].tolist()
        return self
    
    def transform(self, X):
        # Drop identified columns
        return X.drop(columns=self.cols_to_drop_)

def load_data():
    """Load and preprocess data."""
    logging.info("Starting data preprocessing...")

    # Correct path to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

    # Define the correct data paths
    phenotypic_data_path = os.path.join(root_dir, "data/ABIDEII_Composite_Phenotypic.csv")
    anat_qap_path = os.path.join(root_dir, "data/ABIDEII_MRI_Quality_Metrics/anat_qap.csv")
    dti_qap_path = os.path.join(root_dir, "data/ABIDEII_MRI_Quality_Metrics/dti_qap.csv")
    functional_qap_path = os.path.join(root_dir, "data/ABIDEII_MRI_Quality_Metrics/functional_qap.csv")

    # Preprocess the data using the correct paths
    X, y = preprocess_data(
        phenotypic_path=phenotypic_data_path,
        anat_qap_path=anat_qap_path,
        dti_qap_path=dti_qap_path,
        functional_qap_path=functional_qap_path
    )

    if X is None or y is None:
        logging.error("Data preprocessing failed.")
        raise FileNotFoundError("One or more data files are missing or could not be processed.")

    logging.info("Data successfully loaded and preprocessed.")
    return X, y


def train_svm(X_train, X_test, y_train, y_test, cv_folds=10):
    """Train the SVM Model."""
    logging.info(f"Training SVM model with {cv_folds}-fold Cross-Validation...")
    print(f"\n=== Training SVM Model with {cv_folds}-fold Cross-Validation ===")

    # Ensure all features are numeric
    feature_names = X_train.select_dtypes(include=['number']).columns  # Save feature names
    print(f"Original Training Data Shape: {X_train.shape}")
    print(f"Original Test Data Shape: {X_test.shape}")
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])
    print(f"Numeric Training Data Shape: {X_train.shape}")
    print(f"Numeric Test Data Shape: {X_test.shape}")

    # Map target labels to {0, 1} if necessary
    y_train_mapped = y_train.map({1: 0, 2: 1})
    y_test_mapped = y_test.map({1: 0, 2: 1})

    #added pipeline includes imputation, scaling, and variance threshold.
    # SimpleImputer(strategy="mean") = mean, SimpleImputer(strategy="median") = median, KNNImputer(n_neighbors=5) = KNN
    pipe = Pipeline([
    ('drop_missing_cols', DropAllMissingColumns()),  # Remove all-NaN columns
    ('imputer', KNNImputer(n_neighbors=5)),  # Handle NaNs. 
    ('variance_threshold_constant', VarianceThreshold(threshold=0)), #remove 0 variance
    ('variance_threshold', VarianceThreshold(threshold=0.01)), #remove low variance feature
    ('scaler', StandardScaler()), # Scale features
    ('svm', SVC(probability=True, random_state=42))
    ])
    
    # Grid Search Parameters
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': [0.01, 0.1, 1],
        'svm__kernel': ['rbf']
    }

    # Grid Search
    grid_search = GridSearchCV(pipe, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train_mapped)

    # Best Model
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_}")
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    logging.info(f"Best Cross-Validation Score: {grid_search.best_score_}")

    # Predictions
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    #Fairness analysis
    analyze_fairness(best_model, X_test, y_test)

    # Metrics
    accuracy = accuracy_score(y_test_mapped, y_pred)
    auc = roc_auc_score(y_test_mapped, y_proba)
    precision = precision_score(y_test_mapped, y_pred)
    recall = recall_score(y_test_mapped, y_pred)
    f1 = f1_score(y_test_mapped, y_pred)
    print(f"\nTest Accuracy: {accuracy}")
    print(f"AUC-ROC: {auc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}\n")
    logging.info(f"Test Accuracy: {accuracy}")
    logging.info(f"AUC-ROC: {auc}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_mapped, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"SVM (AUC = {auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_test_mapped, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # Save Model
    joblib.dump(best_model, f"svm_model_{cv_folds}_folds.pkl")
    logging.info(f"Model saved as 'svm_model_{cv_folds}_folds.pkl'.")

    return best_model, X_train, X_test


def analyze_fairness(model, X_test, y_test):
    #map labels (1 is male and 2 is female)
    y_test_mapped = y_test.map({1: 0, 2: 1})
    sensitive_feature = X_test['SEX'].astype(str)

    #baseline predictions
    y_pred = model.predict(X_test)

    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate}
    # metricframe before mitigation
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test_mapped,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )

    print("\n=== Fairness Analysis Before Bias Mitigation ===")
    print(metric_frame.by_group)

    dpd = demographic_parity_difference(y_test_mapped, y_pred, sensitive_features=sensitive_feature)
    eod = equalized_odds_difference(y_test_mapped, y_pred, sensitive_features=sensitive_feature)
    print(f"Demographic Parity Difference: {dpd}")
    print(f"Equalized Odds Difference: {eod}")

    #bias mitigation for equalized odds
    mitigator = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        predict_method="predict_proba",
        prefit=True
    )
    mitigator.fit(X_test, y_test_mapped, sensitive_features=sensitive_feature)
    y_pred_mitigated = mitigator.predict(X_test, sensitive_features=sensitive_feature)

    # mitigated MetricFrame
    mitigated_metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test_mapped,
        y_pred=y_pred_mitigated,
        sensitive_features=sensitive_feature
    )

    print("\n=== Fairness Analysis After Bias Mitigation ===")
    print(mitigated_metric_frame.by_group)

    dpd_mitigated = demographic_parity_difference(y_test_mapped, y_pred_mitigated, sensitive_features=sensitive_feature)
    eod_mitigated = equalized_odds_difference(y_test_mapped, y_pred_mitigated, sensitive_features=sensitive_feature)
    print(f"Mitigated Demographic Parity Difference: {dpd_mitigated}")
    print(f"Mitigated Equalized Odds Difference: {eod_mitigated}")
               
if __name__ == "__main__":
    try:
        # Load data
        logging.info("Loading and preprocessing data...")
        X, y = load_data()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training Data Shape: {X_train.shape}")
        print(f"Test Data Shape: {X_test.shape}")
        
        print(f"\n=== Training with 10-fold Cross-Validation ===")
        best_model, X_train_transformed, X_test_transformed = train_svm(X_train, X_test, y_train, y_test, cv_folds=10)


    except Exception as e:
        logging.error(f"An error occurred: {e}")


