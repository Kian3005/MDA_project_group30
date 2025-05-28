import pandas as pd
import numpy as np
from scipy.stats import iqr
import logging
import joblib # For saving/loading models and preprocessing objects

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical 
import mlflow

log = logging.getLogger(__name__)

def preprocess_and_clean_data(
    preprocessed_data_final: pd.DataFrame,
    params: dict # For outlier threshold parameters
) -> pd.DataFrame:
    """
    Converts 'sustainability' to boolean, removes outliers from 'ecMaxContribution'
    using the IQR method, and applies log transformation to the target variable.

    Args:
        preprocessed_data_final: The input DataFrame after initial preprocessing.
        params: Dictionary containing parameters for outlier removal (e.g., iqr_multiplier).

    Returns:
        A DataFrame with 'sustainability' as boolean, outliers removed,
        and 'log_ecMaxContribution' created.
    """
    df = preprocessed_data_final.copy()
    log.info("Starting data preprocessing for model training.")

    # Convert sustainability to boolean
    df["sustainability"] = df["sustainability"].astype(bool)
    log.info("Converted 'sustainability' column to boolean.")

    # Perform IQR method to remove outliers in ecMaxContribution
    iqr_multiplier = params.get('iqr_multiplier', 2.0) # Default to 2.0 if not in params
    
    if "ecMaxContribution" not in df.columns:
        log.warning("Column 'ecMaxContribution' not found for outlier removal. Skipping.")
        df["log_ecMaxContribution"] = np.log(df["ecMaxContribution"]) # Still attempt log transform if column exists
        return df

    q1 = np.quantile(df["ecMaxContribution"], 0.25)
    q3 = np.quantile(df["ecMaxContribution"], 0.75)
    data_iqr = q3 - q1

    lower_threshold = q1 - iqr_multiplier * data_iqr
    upper_threshold = q3 + iqr_multiplier * data_iqr

    log.info(f"IQR Outlier Removal (Multiplier: {iqr_multiplier}):")
    log.info(f"Lower threshold: {lower_threshold:.2f}, Upper threshold: {upper_threshold:.2f}")

    # Identify outliers
    outliers_df = df[(df["ecMaxContribution"] < lower_threshold) | (df["ecMaxContribution"] > upper_threshold)]
    df_no_outliers = df[
        (df["ecMaxContribution"] >= lower_threshold) & (df["ecMaxContribution"] <= upper_threshold)
    ].copy() # Ensure a copy to avoid SettingWithCopyWarning

    rows_removed = len(df) - len(df_no_outliers)
    log.info(f"Removed {rows_removed} rows as outliers from 'ecMaxContribution'. Remaining rows: {len(df_no_outliers)}")

    # Perform log transformation on the target variable
    # Add a small constant to avoid log(0) if there are zeros.
    # It's better to understand the data's distribution; for contributions, 0 might mean no contribution.
    # Adjust this constant if needed based on your data characteristics.
    df_no_outliers["log_ecMaxContribution"] = np.log1p(df_no_outliers["ecMaxContribution"]) # np.log1p for log(1+x)
    log.info("Applied log1p transformation to 'ecMaxContribution' creating 'log_ecMaxContribution'.")

    log.info("Data preprocessing for model training complete.")
    return df_no_outliers


def train_model(
    df_no_outliers: pd.DataFrame,
    parameters: dict # Combined parameters for model and bayesian search
) -> (Pipeline, dict):
    """
    Trains a RandomForestRegressor model using Bayesian Optimization for hyperparameter tuning.
    The function also preprocesses features using StandardScaler and OneHotEncoder.

    Args:
        df_no_outliers: DataFrame with cleaned data and log-transformed target.
        parameters: Dictionary containing model training and hyperparameter tuning parameters.

    Returns:
        A tuple containing:
        - The trained sklearn Pipeline object (including preprocessor and best regressor).
        - A dictionary of evaluation metrics (MSE, RMSE, MAE, R2).
    """
    log.info("Starting model training and hyperparameter tuning.")

    # Define features and target
    target_column = "log_ecMaxContribution"

    all_feature_columns = [col for col in df_no_outliers.columns if col not in ["id", "ecMaxContribution", target_column]]

    topic_columns = [col for col in all_feature_columns if col.startswith('topic_')]
    impact_columns = [col for col in all_feature_columns if col.startswith('impact_')]
    continents_columns = [col for col in all_feature_columns if col.startswith('continent_')]

    categorical_features = ['legalBasis', 'fundingScheme', 'scientific_domain', 'sustainability', 'problem_type']
    numerical_features = ['project_length_days', 'number_of_organizations', 'proportion_of_small_and_medium_orgs']

    categorical_features = [f for f in categorical_features if f in df_no_outliers.columns]
    numerical_features = [f for f in numerical_features if f in df_no_outliers.columns]

    numerical_features.extend(topic_columns)
    numerical_features.extend(impact_columns)
    numerical_features.extend(continents_columns)

    # Ensure no duplicates in numerical features if extend was called multiple times
    numerical_features = list(set(numerical_features))
    categorical_features = list(set(categorical_features)) # Also for categorical

    X = df_no_outliers.drop(columns=["ecMaxContribution", target_column])
    y = df_no_outliers[target_column]

    log.info(f"Features used: {list(X.columns)}")
    log.info(f"Target variable: {target_column}")

    # Data Preprocessing using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Define the Random Forest Regressor
    rf_model_params = parameters.get('random_forest_regressor', {})
    rf_model = RandomForestRegressor(random_state=rf_model_params.get('random_state', 42))

    # Define the search space for RandomForestRegressor from parameters
    search_spaces_rf = {}
    raw_bayes_search_space = parameters.get('bayes_search_space', {})

    for key, value in raw_bayes_search_space.items():
        if key == 'max_features' and isinstance(value[0], float):
            search_spaces_rf[key] = Real(value[0], value[1], prior='uniform')
        elif key == 'bootstrap' and isinstance(value, list) and all(isinstance(v, bool) for v in value): # Handle 'bootstrap' specifically as categorical
            search_spaces_rf[key] = Categorical(value)
        elif isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) for v in value):
            search_spaces_rf[key] = Integer(value[0], value[1])
        else:
            # Fallback for unexpected types or structures.
            # This will catch any hyperparameter definitions that don't fit the expected patterns.
            raise ValueError(f"Unsupported hyperparameter type or format for '{key}': {value}. "
                             "Expected a list of two ints for Integer, two floats for Real (only for max_features), "
                             "or a list of booleans for Categorical.")


    # Set up BayesSearchCV for the Random Forest model
    bayes_search_params = parameters.get('bayes_search_params', {})
    bayes_search_rf = BayesSearchCV(
        estimator=rf_model,
        search_spaces=search_spaces_rf,
        n_iter=bayes_search_params.get('n_iter', 40),
        scoring=bayes_search_params.get('scoring', 'neg_mean_squared_error'),
        cv=bayes_search_params.get('cv', 5),
        n_jobs=bayes_search_params.get('n_jobs', -1),
        random_state=bayes_search_params.get('random_state', 42),
        verbose=bayes_search_params.get('verbose', 0)
    )

    # Create a pipeline to combine preprocessing and Bayes Search for the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor_bayes_search', bayes_search_rf)])

    # Split data into training and testing sets
    test_size = parameters.get('test_size', 0.2)
    random_state = parameters.get('train_test_split_random_state', 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    log.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    # Train the pipeline (which includes fitting BayesSearchCV)
    log.info("Starting Bayesian Hyperparameter Tuning for RandomForestRegressor...")
    pipeline.fit(X_train, y_train)
    log.info("Bayesian Hyperparameter Tuning Complete.")

    # Access the best estimator and its parameters
    best_rf_estimator = pipeline.named_steps['regressor_bayes_search'].best_estimator_
    best_rf_params = pipeline.named_steps['regressor_bayes_search'].best_params_

    log.info(f"Best parameters found by Bayes Search: {best_rf_params}")
    log.info(f"Bayes Search Best CV score (neg_MSE): {pipeline.named_steps['regressor_bayes_search'].best_score_:.2f}")
    log.info(f"Bayes Search Best CV score (RMSE): {np.sqrt(-pipeline.named_steps['regressor_bayes_search'].best_score_):.2f}")

    # Model Evaluation on the test set using the best estimator
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log) # Inverse transform to original scale
    y_test_original_scale = np.expm1(y_test) # Inverse transform original y_test for correct metrics

    mse = mean_squared_error(y_test_original_scale, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original_scale, y_pred)
    r2 = r2_score(y_test_original_scale, y_pred)

    log.info(f'Test Set Metrics (on original scale, using best model from tuning):')
    log.info(f'Mean Squared Error: {mse:.2f}')
    log.info(f'Root Mean Squared Error: {rmse:.2f}')
    log.info(f'Mean Absolute Error: {mae:.2f}')
    log.info(f'R-squared: {r2:.2f}')

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'best_params': best_rf_params,
        'best_cv_neg_mse': pipeline.named_steps['regressor_bayes_search'].best_score_,
        'best_cv_rmse': np.sqrt(-pipeline.named_steps['regressor_bayes_search'].best_score_)
    }

    return pipeline, metrics

def save_model_artifacts(
    trained_pipeline: Pipeline,
    metrics: dict,
    preprocessed_data_final: pd.DataFrame # This DataFrame is after all preprocessing and consolidation
) -> (Pipeline, dict, list, list, list, list):
    """
    Saves the trained pipeline and various artifacts needed for deployment (UI lists).

    Args:
        trained_pipeline: The trained sklearn Pipeline object.
        metrics: Dictionary of evaluation metrics.
        preprocessed_data_final: The input DataFrame to help extract final feature names
                                 and generate UI lists based on its final state.

    Returns:
        The trained pipeline, metrics, and lists for UI (topics, impacts, continents, feature columns).
        These are returned for Kedro's catalog to save them.
    """
    log.info("Saving model and related artifacts.")

    # Log metrics to MLflow
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(metric_name, metric_value)
        elif isinstance(metric_value, dict): # For best_params or similar nested dicts
            mlflow.log_params(metric_value)

    # Access the preprocessor from the trained pipeline
    preprocessor = trained_pipeline.named_steps['preprocessor']

    final_processed_feature_names = preprocessor.get_feature_names_out()
    log.info(f"Extracted final processed feature names using preprocessor.get_feature_names_out(): {len(final_processed_feature_names)} features.")

    # If you need to log the feature names for debugging or future use
    mlflow.log_text(str(list(final_processed_feature_names)), "final_processed_feature_names_for_model.txt")


    # Prepare UI lists based on the *actual columns present in the final preprocessed data*
    # This is crucial because `consolidate_infrequent_topics` modifies these.

    # Filter out columns that are not directly 'topic_', 'impact_', 'continent_' prefixes
    # This helps in accurately getting the columns that were originally part of these categories.
    topic_columns_final = sorted([col for col in preprocessed_data_final.columns if col.startswith('topic_')])
    impact_columns_final = sorted([col for col in preprocessed_data_final.columns if col.startswith('impact_')])
    continents_columns_final = sorted([col for col in preprocessed_data_final.columns if col.startswith('continent_')])

    # Prepare UI lists - still based on the *final* columns from `preprocessed_data_final`
    raw_topics_for_ui = [col.replace('topic_', '') for col in topic_columns_final]
    topics_without_other = [topic for topic in raw_topics_for_ui if topic.lower() != 'other']
    other_topic_exists = 'topic_other' in topic_columns_final
    sorted_topics = sorted(topics_without_other)
    final_ui_topics = sorted_topics + ['other'] if other_topic_exists else sorted_topics

    final_ui_impacts = sorted([col.replace('impact_', '') for col in impact_columns_final])
    final_ui_continents = sorted([col.replace('continent_', '') for col in continents_columns_final])

    # Log the UI lists as artifacts if needed, or simply return them via Kedro's catalog
    mlflow.log_text(str(final_ui_topics), "ui_topics.txt")
    mlflow.log_text(str(final_ui_impacts), "ui_impacts.txt")
    mlflow.log_text(str(final_ui_continents), "ui_continents.txt")

    # Save the entire pipeline (preprocessor + best regressor)
    # Using joblib for saving the scikit-learn pipeline
    model_path = "model.pkl"
    joblib.dump(trained_pipeline, model_path)
    mlflow.log_artifact(model_path)
    log.info(f"Model saved to {model_path} and logged to MLflow.")

    log.info("Model and artifacts prepared for saving.")
    # Return everything for Kedro's catalog to save
    return trained_pipeline, metrics, final_ui_topics, final_ui_impacts, final_ui_continents, list(final_processed_feature_names) # Ensure it's a list for consistency
