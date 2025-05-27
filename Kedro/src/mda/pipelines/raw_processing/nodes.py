print("HELLO FROM RAW_PROCESSING NODES.PY!")
import pandas as pd
import numpy as np
import datetime
import logging # Import logging for proper Kedro logging
import re # Import regex module for general character cleaning

log = logging.getLogger(__name__) # Initialize Kedro's logger


# --- Helper Functions (if any are truly generic and reusable across multiple nodes) ---
# Your helper functions like preprocess_topics_list, parse_impact, parse_continent_data
# would go here if they are directly called by the nodes below.

# --- Kedro Nodes ---

def merge_and_initial_clean(dfp: pd.DataFrame, dfo: pd.DataFrame, dfoutput: pd.DataFrame) -> pd.DataFrame:
    """
    Merges project, organization, and output data, performs initial column drops,
    and type conversions.

    Args:
        dfp: Project DataFrame.
        dfo: Organization DataFrame.
        dfoutput: Project Output DataFrame.

    Returns:
        A merged and initially cleaned DataFrame.
    """
    log.info("Starting data merging and initial cleaning...")

    # One to one merge dfp and df_full_output
    newdf = dfp.merge(dfoutput, on="id")

    # One to many merge newdf and dfo
    fulldf = newdf.merge(dfo, left_on="id", right_on="projectID", validate="one_to_many")

    # Define columns to drop
    cols_to_drop_initial = [
        "active", "nature", "acronym", "title", "totalCost_x", "topics", "objective_x", "rcn_x",
        "grantDoi", "contentUpdateDate_x", "projectID", "projectAcronym", "organisationID",
        "vatNumber", "name", "shortName", "street", "postCode", "city", "geolocation",
        "organizationURL", "contactForm", "contentUpdateDate_y", "rcn_y", "totalCost_y",
        "ecContribution", "netEcContribution", "ecSignatureDate", "masterCall", "subCall",
        "frameworkProgramme", "status"
    ]
    # Filter out columns that might not exist to avoid errors
    cols_to_drop_initial = [col for col in cols_to_drop_initial if col in fulldf.columns]
    fulldf = fulldf.drop(columns=cols_to_drop_initial, errors='ignore')

    # Give right structure to funding variable
    if "ecMaxContribution" in fulldf.columns:
        fulldf["ecMaxContribution"] = fulldf["ecMaxContribution"].astype(str).str.replace(',', '.', regex=False).astype(float)

    # Give 0/1 to SME, assuming missing SME means not SME
    if "SME" in fulldf.columns:
        fulldf["SME"] = fulldf["SME"].astype(bool).astype(int)

    # Convert start and end date to datetime
    if "startDate" in fulldf.columns:
        fulldf["startDate"] = pd.to_datetime(fulldf["startDate"], errors='coerce')
    if "endDate" in fulldf.columns:
        fulldf["endDate"] = pd.to_datetime(fulldf["endDate"], errors='coerce')

    # --- START FIX FOR UNICODEENCODEERROR ---
    # Based on diagnosis, '\u2004' is in 'objective_y'
    if "objective_y" in fulldf.columns:
        log.info("Cleaning problematic character '\\u2004' from 'objective_y' column...")
        # Convert to string type first to ensure .str accessor works
        # Replace '\u2004' (Four-Per-Em Space) with a regular space
        fulldf["objective_y"] = fulldf["objective_y"].astype(str).str.replace('\u2004', ' ', regex=False)
        log.info("Finished cleaning 'objective_y' column.")
    # --- END FIX FOR UNICODEENCODEERROR ---

    # --- NEW ADDITION: General cleaning for all non-ASCII characters ---
    log.info("Cleaning all string columns for non-ASCII characters that might cause encoding issues...")
    for col in fulldf.select_dtypes(include=['object', 'string']).columns:
        # Check if the column actually contains string data, and not just mixed types
        if fulldf[col].apply(lambda x: isinstance(x, str)).any():
            # This regex matches any character that is NOT a printable ASCII character (0x20 to 0x7E)
            # or common whitespace characters like tabs, newlines, and carriage returns.
            # It replaces them with a space. You could also replace with '' to remove completely.
            fulldf[col] = fulldf[col].astype(str).apply(lambda x: re.sub(r'[^\x20-\x7E\t\n\r]+', ' ', x))
    log.info("Finished cleaning all string columns for non-ASCII characters.")
    # --- END NEW ADDITION ---

    log.info("Data merging and initial cleaning complete.")
    return fulldf

def handle_missing_values_and_adjustments(fulldf_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Handles specific missing values and applies manual adjustments
    identified during initial data exploration.

    Args:
        fulldf_merged: The DataFrame after initial merging and cleaning.

    Returns:
        The DataFrame with missing values handled and manual adjustments applied.
    """
    log.info("Starting missing value handling and manual adjustments...")

    # Exclude problematic IDs
    ids_to_exclude = [101149703, 101101973, 101041246]
    fulldf_C = fulldf_merged[~fulldf_merged["id"].isin(ids_to_exclude)].copy()

    # Handling 'fundingScheme' - dropping rows with NaN for this column
    if "fundingScheme" in fulldf_C.columns:
        initial_rows = len(fulldf_C)
        fulldf_C = fulldf_C[fulldf_C["fundingScheme"].notna()].copy()
        rows_dropped = initial_rows - len(fulldf_C)
        if rows_dropped > 0:
            log.warning(f"Dropped {rows_dropped} rows due to missing 'fundingScheme'.")

    # Fill NaNs for specific columns if they exist
    for col in ["problem_type", "expected_impact", "scientific_domain", "main_topics"]:
        if col in fulldf_C.columns:
            if fulldf_C[col].dtype.name == 'category':
                fulldf_C[col] = fulldf_C[col].astype('object')
            # Use specific fills as per your original nodes.py logic
            if col == "problem_type":
                fulldf_C[col] = fulldf_C[col].fillna("no problem").str.lower().str.strip()
            elif col == "expected_impact":
                fulldf_C[col] = fulldf_C[col].fillna("no impact")
            else: # For scientific_domain and main_topics, fill with "unknown"
                fulldf_C[col] = fulldf_C[col].fillna("unknown")
            fulldf_C[col] = fulldf_C[col].astype("category") # Convert back to category

    # Sustainability handling - fill NaN with 0 then ensure int type
    if "sustainability" in fulldf_C.columns:
        fulldf_C["sustainability"] = pd.to_numeric(fulldf_C["sustainability"], errors='coerce').fillna(0).astype(int)

    log.info("Missing value handling and manual adjustments complete.")
    return fulldf_C

def feature_engineering(fulldf_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering steps.

    Args:
        fulldf_cleaned: The cleaned DataFrame.

    Returns:
        The DataFrame with new features.
    """
    log.info("Starting feature engineering...")

    fulldf_fe = fulldf_cleaned.copy()

    if "startDate" in fulldf_fe.columns and "endDate" in fulldf_fe.columns:
        fulldf_fe["startDate"] = pd.to_datetime(fulldf_fe["startDate"], errors='coerce')
        fulldf_fe["endDate"] = pd.to_datetime(fulldf_fe["endDate"], errors='coerce')
        fulldf_fe["project_length_days"] = (fulldf_fe["endDate"] - fulldf_fe["startDate"]).dt.days
        fulldf_fe["project_length_days"] = fulldf_fe["project_length_days"].fillna(0).astype(int)

    if "id" in fulldf_fe.columns:
        fulldf_fe['number_of_organizations'] = fulldf_fe.groupby('id')['id'].transform('count')

    if "SME" in fulldf_fe.columns and "number_of_organizations" in fulldf_fe.columns:
        fulldf_fe['number_of_small_and_medium_orgs'] = fulldf_fe.groupby('id')['SME'].transform('sum')
        fulldf_fe['proportion_of_small_and_medium_orgs'] = np.where(
            fulldf_fe['number_of_organizations'] > 0,
            fulldf_fe['number_of_small_and_medium_orgs'] / fulldf_fe['number_of_organizations'],
            0
        )
        fulldf_fe['proportion_of_small_and_medium_orgs'] = fulldf_fe['proportion_of_small_and_medium_orgs'].fillna(0)

    # Drop columns that are not needed anymore
    cols_to_drop_fe = ["SME", "nutsCode", "activityType", "endOfParticipation", "order", "role", "number_of_small_and_medium_orgs"]
    cols_to_drop_fe = [col for col in cols_to_drop_fe if col in fulldf_fe.columns]
    fulldf_fe = fulldf_fe.drop(columns=cols_to_drop_fe, errors='ignore')

    log.info("Feature engineering complete.")
    return fulldf_fe