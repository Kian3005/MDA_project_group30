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
from skopt.space import Real, Integer

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
    # Ensure these columns are available based on your data_preprocessing pipeline outputs
    target_column = "log_ecMaxContribution" # Use the log-transformed target for training

    # Dynamic identification of topic, impact, and continent columns
    # We need to ensure that the columns coming from the previous pipeline are correctly identified.
    # This assumes a consistent naming convention.
    all_feature_columns = [col for col in df_no_outliers.columns if col not in ["id", "ecMaxContribution", target_column]]

    # These lists should be passed as parameters if their exact names aren't fixed.
    # For robustness, we dynamically identify them or rely on them being present after concat.
    topic_columns = [col for col in all_feature_columns if col.startswith('topic_')]
    impact_columns = [col for col in all_feature_columns if col.startswith('impact_')]
    continents_columns = [col for col in all_feature_columns if col.startswith('continent_')]

    # Assuming other numerical and categorical features are stable
    categorical_features = ['legalBasis', 'fundingScheme', 'scientific_domain', 'sustainability', 'problem_type']
    numerical_features = ['project_length_days', 'number_of_organizations', 'proportion_of_small_and_medium_orgs']

    # Filter out columns that might not exist in the dataframe after all merges/drops
    categorical_features = [f for f in categorical_features if f in df_no_outliers.columns]
    numerical_features = [f for f in numerical_features if f in df_no_outliers.columns]

    # Add dynamic sparse features to numerical features
    numerical_features.extend(topic_columns)
    numerical_features.extend(impact_columns)
    numerical_features.extend(continents_columns)


    X = df_no_outliers.drop(columns=["id", "ecMaxContribution", target_column]) # 'id' is typically not a feature
    y = df_no_outliers[target_column]

    log.info(f"Features used: {list(X.columns)}")
    log.info(f"Target variable: {target_column}")

    # Data Preprocessing using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns (like original 'main_topics' if not dropped)
    )

    # Define the Random Forest Regressor
    rf_model_params = parameters.get('random_forest_regressor', {})
    rf_model = RandomForestRegressor(random_state=rf_model_params.get('random_state', 42))

    # Define the search space for RandomForestRegressor from parameters
    search_spaces_rf = parameters.get('bayes_search_space', {})
    # Convert list/tuple ranges from YAML to skopt space objects
    for key, value in search_spaces_rf.items():
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (int, float)):
            if key == 'max_features' and isinstance(value[0], float):
                 search_spaces_rf[key] = Real(value[0], value[1], prior='uniform')
            else:
                search_spaces_rf[key] = Integer(value[0], value[1])


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
    preprocessed_data_final: pd.DataFrame # Use the input data to extract final column names
) -> (Pipeline, dict, list, list, list, list):
    """
    Saves the trained pipeline and various artifacts needed for deployment (UI lists).

    Args:
        trained_pipeline: The trained sklearn Pipeline object.
        metrics: Dictionary of evaluation metrics.
        preprocessed_data_final: The input DataFrame to help extract feature names.

    Returns:
        The trained pipeline, metrics, and lists for UI (topics, impacts, continents, feature columns).
        These are returned for Kedro's catalog to save them.
    """
    log.info("Saving model and related artifacts.")

    # Extract dynamic feature names from the trained preprocessor
    # This is more robust as it reflects the actual features used in the trained pipeline
    preprocessor = trained_pipeline.named_steps['preprocessor']
    ohe_categories = preprocessor.named_transformers_['cat'].categories_
    # Ensure the order of categorical features matches the order used by OneHotEncoder
    categorical_features_used = [
        col for col in ['legalBasis', 'fundingScheme', 'scientific_domain', 'sustainability', 'problem_type']
        if col in preprocessed_data_final.columns
    ]
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features_used)

    # Re-extract numerical features (including topic, impact, continent) from the input dataframe
    # This relies on the convention established in `data_preprocessing_nodes`
    numerical_features_used = [
        'project_length_days', 'number_of_organizations', 'proportion_of_small_and_medium_orgs'
    ]
    numerical_features_used = [f for f in numerical_features_used if f in preprocessed_data_final.columns] # filter for safety

    topic_columns = sorted([col for col in preprocessed_data_final.columns if col.startswith('topic_')])
    impact_columns = sorted([col for col in preprocessed_data_final.columns if col.startswith('impact_')])
    continents_columns = sorted([col for col in preprocessed_data_final.columns if col.startswith('continent_')])

    # Combine all feature names in the order they'd appear after preprocessing
    final_feature_columns_order = list(numerical_features_used) + list(topic_columns) + list(impact_columns) + list(continents_columns) + list(ohe_feature_names)

    # Prepare UI lists
    # Remove 'topic_', 'impact_', 'continent_' prefixes and sort
    # Ensure 'other' topic is handled specifically if it exists
    raw_topics_for_ui = [col.replace('topic_', '') for col in topic_columns]
    topics_without_other = [topic for topic in raw_topics_for_ui if topic.lower() != 'other']
    other_topic_exists = 'topic_other' in topic_columns
    sorted_topics = sorted(topics_without_other)
    final_ui_topics = sorted_topics + ['other'] if other_topic_exists else sorted_topics


    final_ui_impacts = sorted([col.replace('impact_', '') for col in impact_columns])
    final_ui_continents = sorted([col.replace('continent_', '') for col in continents_columns])

    log.info("Model and artifacts prepared for saving.")
    # Return everything for Kedro's catalog to save
    return trained_pipeline, metrics, final_ui_topics, final_ui_impacts, final_ui_continents, final_feature_columns_order