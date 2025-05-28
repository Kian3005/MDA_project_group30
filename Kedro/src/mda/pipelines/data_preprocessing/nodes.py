import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval
from collections import defaultdict
import joblib
import datetime
from sentence_transformers import SentenceTransformer, util
import logging

log = logging.getLogger(__name__)

print("HELLO FROM DATA_PREPROCESSING!")


# --- Helper Functions (keep them here as they are used by multiple nodes) ---

def preprocess_topics_list(topic_list):
    if isinstance(topic_list, str):
        try:
            evaluated_list = literal_eval(topic_list)
            if isinstance(evaluated_list, list):
                return [str(i).lower() for i in evaluated_list]
            else:
                return []
        except (ValueError, SyntaxError):
            return []
    elif isinstance(topic_list, list):
        return [str(i).lower() for i in topic_list]
    return []

def parse_impact(x):
    if isinstance(x, str):
        x = x.strip()
        if x.startswith('[') and x.endswith(']'):
            try:
                return [str(i).lower() for i in literal_eval(x)]
            except (ValueError, SyntaxError):
                return [x.lower()] if x else []
        elif x == 'Confidential':
            return []
        else:
            return [x.lower()] if x else []
    elif isinstance(x, list):
        return [str(i).lower() for i in x]
    return []

def parse_continent_data(x):
    if isinstance(x, str):
        x = x.strip()
        if x.startswith('[') and x.endswith(']'):
            try:
                return [str(i).lower() for i in literal_eval(x)]
            except (ValueError, SyntaxError):
                return [x.lower()] if x else []
        else:
            return [x.lower()] if x else []
    elif isinstance(x, list):
        return [str(i).lower() for i in x]
    elif pd.isna(x):
        return []
    return []


# --- MODIFIED NODE: Preprocess Main Topics (uses the generated merging library) ---

def preprocess_main_topics(fulldf: pd.DataFrame, df_merge_library: pd.DataFrame) -> (pd.DataFrame, MultiLabelBinarizer):
    """
    Preprocesses 'main_topics' column, applies MultiLabelBinarizer,
    and merges topic columns based on LLM-driven rules (from df_merge_library),
    handling memory efficiently.

    Args:
        fulldf (pd.DataFrame): The input DataFrame containing 'main_topics'.
        df_merge_library (pd.DataFrame): DataFrame containing LLM-driven merge rules.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame with processed and merged topic columns.
            - MultiLabelBinarizer: The fitted MLB object for main topics.
    """
    # Use the already processed 'main_topics' if available, otherwise process it
    if 'main_topics_processed' not in fulldf.columns:
        fulldf['main_topics_processed'] = fulldf['main_topics'].astype('object').apply(preprocess_topics_list)

    mlb_topics = MultiLabelBinarizer(sparse_output=True)
    topic_labels_matrix = mlb_topics.fit_transform(fulldf['main_topics_processed'])

    cleaned_mlb_column_names_all = [
        f'topic_{label.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower()}'
        for label in mlb_topics.classes_
    ]

    temp_binary_sparse_df = pd.DataFrame.sparse.from_spmatrix(
        topic_labels_matrix,
        columns=cleaned_mlb_column_names_all,
        index=fulldf.index
    )

    log.info("--- Merging exactly duplicate column names from MLB output ---")
    unique_name_to_original_cols = defaultdict(list)
    for col_original, col_cleaned in zip(temp_binary_sparse_df.columns, cleaned_mlb_column_names_all):
        unique_name_to_original_cols[col_cleaned].append(col_original)

    final_mlb_sparse_series = []
    for cleaned_name, original_cols in unique_name_to_original_cols.items():
        if len(original_cols) > 1:
            merged_series = (temp_binary_sparse_df[original_cols].sum(axis=1) > 0).astype('Sparse[int8]')
            merged_series.name = cleaned_name
            final_mlb_sparse_series.append(merged_series)
        else:
            single_series = temp_binary_sparse_df[original_cols[0]].astype('Sparse[int8]')
            single_series.name = cleaned_name
            final_mlb_sparse_series.append(single_series)

    merged_mlb_output_sparse_df = pd.concat(final_mlb_sparse_series, axis=1, copy=False)
    log.info(f"Total topic columns (after MLB and exact merge): {len(merged_mlb_output_sparse_df.columns.tolist())}")

    log.info("--- Applying LLM-driven Merges (on sparse data) - Vectorized ---")
    canonical_to_source_map = defaultdict(list)
    processed_source_cols = set()
    canonical_cols_in_map = set()

    current_topic_cols_in_sparse_df = merged_mlb_output_sparse_df.columns.tolist()
    current_topic_cols_set = set(current_topic_cols_in_sparse_df)

    df_merge_library['Canonical_Name_Cleaned'] = df_merge_library['Canonical_Name'].apply(
        lambda x: f'topic_{x.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower()}'
    )

    for index, row in df_merge_library.iterrows():
        canonical_name_cleaned = row['Canonical_Name_Cleaned']
        # Collect all source topics from the row and clean their names
        group_topics_from_llm = [
            f'topic_{topic.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower()}'
            for topic in row.drop(['Canonical_Name', 'Canonical_Name_Cleaned'], errors='ignore').dropna().tolist()
        ]

        # Only consider topics that actually exist in current_topic_cols_set
        group_topics_existing = [
            topic for topic in group_topics_from_llm if topic in current_topic_cols_set
        ]
        # If the canonical name itself exists and is not already part of the group_topics_existing, add it
        if canonical_name_cleaned in current_topic_cols_set and canonical_name_cleaned not in group_topics_existing:
            group_topics_existing.append(canonical_name_cleaned)

        actual_group_topics = [col for col in group_topics_existing if col not in processed_source_cols]

        if actual_group_topics:
            for source_col in actual_group_topics:
                canonical_to_source_map[canonical_name_cleaned].append(source_col)
                processed_source_cols.add(source_col)
            canonical_cols_in_map.add(canonical_name_cleaned)

    final_topic_data = []
    final_topic_column_names = []

    for canonical_col, source_cols_list in canonical_to_source_map.items():
        if source_cols_list:
            merged_series = (merged_mlb_output_sparse_df[source_cols_list].sum(axis=1) > 0).astype('Sparse[int8]')
            merged_series.name = canonical_col
            final_topic_data.append(merged_series)
            final_topic_column_names.append(canonical_col)

    remaining_topic_cols = [
        col for col in current_topic_cols_set
        if col not in processed_source_cols and col not in canonical_cols_in_map
    ]

    for col in remaining_topic_cols:
        single_series = merged_mlb_output_sparse_df[col].astype('Sparse[int8]')
        single_series.name = col
        final_topic_data.append(single_series)
        final_topic_column_names.append(col)

    processed_fulldf_topics = pd.concat(final_topic_data, axis=1, copy=False)

    original_non_topic_cols = [col for col in fulldf.columns if not col.startswith('topic_') and col != 'main_topics_processed']
    processed_fulldf = pd.concat([fulldf[original_non_topic_cols], processed_fulldf_topics], axis=1, copy=False)

    return processed_fulldf, mlb_topics


def generate_additional_features(processed_fulldf: pd.DataFrame) -> (pd.DataFrame, MultiLabelBinarizer, MultiLabelBinarizer):
    """
    Generates additional features like expected_impact_X, project_length_days,
    continent_X, number_of_organizations, proportion_MSE,
    and performs final column dropping and type conversion.
    """
    log.info("Starting generation of additional features...")
    # MultiLabelBinarizer for expected_impact
    mlb_impact = MultiLabelBinarizer(sparse_output=True)
    processed_fulldf['expected_impact_processed'] = processed_fulldf['expected_impact'].apply(parse_impact)
    impact_labels = mlb_impact.fit_transform(processed_fulldf['expected_impact_processed'])
    impact_columns = [f'impact_{label.replace(" ", "_").replace("-", "_").lower()}' for label in mlb_impact.classes_]
    impact_df = pd.DataFrame.sparse.from_spmatrix(impact_labels, columns=impact_columns, index=processed_fulldf.index)
    processed_fulldf = pd.concat([processed_fulldf, impact_df], axis=1, copy=False)

    # Project Length Calculation
    processed_fulldf["startDate_dt"] = pd.to_datetime(processed_fulldf["startDate"], errors='coerce')
    processed_fulldf["endDate_dt"] = pd.to_datetime(processed_fulldf["endDate"], errors='coerce')
    processed_fulldf["project_length_days"] = (processed_fulldf["endDate_dt"] - processed_fulldf["startDate_dt"]).dt.days.fillna(0)
    processed_fulldf = processed_fulldf.drop(columns=["startDate_dt", "endDate_dt"])

    # Country to Continent Mapping
    country_to_continent = {
        'AD': 'Europe', 'AE': 'Asia', 'AF': 'Africa', 'AG': 'North America',
        'AI': 'North America', 'AL': 'Europe', 'AM': 'Asia', 'AO': 'Africa',
        'AQ': 'Antarctica', 'AR': 'South America', 'AS': 'Oceania', 'AT': 'Europe',
        'AU': 'Oceania', 'AW': 'North America', 'AX': 'Europe', 'AZ': 'Asia',
        'BA': 'Europe', 'BB': 'North America', 'BD': 'Asia', 'BE': 'Europe',
        'BF': 'Africa', 'BG': 'Europe', 'BH': 'Asia', 'BI': 'Africa',
        'BJ': 'Africa', 'BL': 'North America', 'BM': 'North America', 'BN': 'Asia',
        'BO': 'South America', 'BQ': 'North America', 'BR': 'South America', 'BS': 'North America',
        'BT': 'Asia', 'BV': 'Antarctica', 'BW': 'Africa', 'BY': 'Europe',
        'BZ': 'North America', 'CA': 'North America', 'CC': 'Asia', 'CD': 'Africa',
        'CF': 'Africa', 'CG': 'Africa', 'CH': 'Europe', 'CI': 'Africa',
        'CK': 'Oceania', 'CL': 'South America', 'CM': 'Africa', 'CN': 'Asia',
        'CO': 'South America', 'CP': 'North America', 'CR': 'North America', 'CU': 'North America',
        'CV': 'Africa', 'CW': 'North America', 'CX': 'Asia', 'CY': 'Europe',
        'CZ': 'Europe', 'DE': 'Europe', 'DJ': 'Africa', 'DK': 'Europe',
        'DM': 'North America', 'DO': 'North America', 'DZ': 'Africa', 'EC': 'South America',
        'EE': 'Europe', 'EG': 'Africa', 'EH': 'Africa', "EL": "Europe", 'ER': 'Africa',
        'ES': 'Europe', 'ET': 'Africa', 'FI': 'Europe', 'FJ': 'Oceania',
        'FK': 'South America', 'FM': 'Oceania', 'FO': 'Europe', 'FR': 'Europe',
        'GA': 'Africa', 'GD': 'North America', 'GE': 'Asia', 'GF': 'South America',
        'GG': 'Europe', 'GH': 'Africa', 'GI': 'Europe', 'GL': 'North America',
        'GM': 'Africa', 'GN': 'Africa', 'GP': 'North America', 'GQ': 'Africa',
        'GS': 'Antarctica', 'GT': 'North America', 'GU': 'Oceania', 'GW': 'Africa',
        'GY': 'South America', 'HK': 'Asia', 'HM': 'Antarctica', 'HN': 'North America',
        'HR': 'Europe', 'HT': 'North America', 'HU': 'Europe', 'ID': 'Asia',
        'IE': 'Europe', 'IL': 'Asia', 'IM': 'Europe', 'IN': 'Asia',
        'IO': 'Asia', 'IQ': 'Asia', 'IR': 'Asia', 'IS': 'Europe',
        'IT': 'Europe', 'JE': 'Europe', 'JM': 'North America', 'JO': 'Asia',
        'JP': 'Asia', 'KE': 'Africa', 'KG': 'Asia', 'KH': 'Asia',
        'KI': 'Oceania', 'KM': 'Africa', 'KN': 'North America', 'KP': 'Asia',
        'KR': 'Asia', 'KW': 'Asia', 'KY': 'North America', 'KZ': 'Asia',
        'LA': 'Asia', 'LB': 'Asia', 'LC': 'North America', 'LI': 'Europe',
        'LK': 'Asia', 'LR': 'Africa', 'LS': 'Africa', 'LT': 'Europe',
        'LU': 'Europe', 'LV': 'Europe', 'LY': 'Africa', 'MA': 'Africa',
        'MC': 'Europe', 'MD': 'Europe', 'ME': 'Europe', 'MF': 'North America',
        'MG': 'Africa', 'MH': 'Oceania', 'MK': 'Europe', 'ML': 'Africa',
        'MM': 'Asia', 'MN': 'Asia', 'MO': 'Asia', 'MP': 'Oceania',
        'MQ': 'North America', 'MR': 'Africa', 'MS': 'North America', 'MT': 'Europe',
        'MU': 'Africa', 'MV': 'Asia', 'MW': 'Africa', 'MX': 'North America',
        'MY': 'Asia', 'MZ': 'Africa', 'NC': 'Oceania', 'NE': 'Africa',
        'NF': 'Oceania', 'NG': 'Africa', 'NI': 'North America', 'NL': 'Europe',
        'NO': 'Europe', 'NP': 'Asia', 'NR': 'Oceania', 'NU': 'Oceania',
        'NZ': 'Oceania', 'OM': 'Asia', 'PA': 'North America', 'PE': 'South America',
        'PF': 'Oceania', 'PG': 'Oceania', 'PH': 'Asia', 'PK': 'Asia',
        'PL': 'Europe', 'PM': 'North America', 'PN': 'Oceania', 'PR': 'North America',
        'PS': 'Asia', 'PT': 'Europe', 'PW': 'Oceania', 'PY': 'South America',
        'QA': 'Asia', 'RE': 'Africa', 'RO': 'Europe', 'RS': 'Europe',
        'RU': 'Europe', 'RW': 'Africa', 'SA': 'Asia', 'SB': 'Oceania',
        'SC': 'Africa', 'SD': 'Africa', 'SE': 'Europe', 'SG': 'Asia',
        'SH': 'Africa', 'SI': 'Europe', 'SJ': 'Europe', 'SK': 'Europe',
        'SL': 'Africa', 'SM': 'Europe', 'SN': 'Africa', 'SO': 'Africa',
        'SR': 'South America', 'SS': 'Africa', 'ST': 'Africa', 'SV': 'North America',
        'SX': 'North America', 'SY': 'Asia', 'SZ': 'Africa', 'TC': 'North America',
        'TD': 'Africa', 'TF': 'Antarctica', 'TG': 'Africa', 'TH': 'Asia',
        'TJ': 'Asia', 'TK': 'Oceania', 'TL': 'Asia', 'TM': 'Asia',
        'TN': 'Africa', 'TO': 'Oceania', 'TR': 'Asia', 'TT': 'North America',
        'TV': 'Oceania', 'TW': 'Asia', 'TZ': 'Africa', 'UA': 'Europe',
        'UG': 'Africa', 'UK': 'Europe', 'UM': 'Oceania', 'US': 'North America',
        'UY': 'South America', 'UZ': 'Asia', 'VA': 'Europe', 'VC': 'North America',
        'VE': 'South America', 'VG': 'North America', 'VI': 'North America', 'VN': 'Asia',
        'VU': 'Oceania', 'WF': 'Oceania', 'WS': 'Oceania', 'XK': 'Europe',
        'YE': 'Asia', 'YT': 'Africa', 'ZA': 'Africa', 'ZM': 'Africa',
        'ZW': 'Africa'
    }
    processed_fulldf['continent'] = processed_fulldf['country'].apply(lambda code: country_to_continent.get(code, 'Unknown'))

    unique_continents_per_id = processed_fulldf.groupby('id')['continent'].agg(lambda x: list(x.unique()))
    processed_fulldf['all_continents'] = processed_fulldf['id'].map(unique_continents_per_id)

    # MultiLabelBinarizer for continents
    mlb_continents = MultiLabelBinarizer(sparse_output=True)
    processed_fulldf['continents_processed'] = processed_fulldf['all_continents'].apply(parse_continent_data)
    continent_labels = mlb_continents.fit_transform(processed_fulldf['continents_processed'])
    continents_columns = [f'continent_{label.replace(" ", "_").replace("-", "_").lower()}' for label in mlb_continents.classes_]
    continents_df = pd.DataFrame.sparse.from_spmatrix(continent_labels, columns=continents_columns, index=processed_fulldf.index)
    processed_fulldf = pd.concat([processed_fulldf, continents_df], axis=1, copy=False)

    # Organization counts
    # processed_fulldf['number_of_organizations'] = processed_fulldf.groupby('id')['id'].transform('count')
    # processed_fulldf['number_of_small_and_medium_orgs'] = processed_fulldf.groupby('id')['SME'].transform('sum')
    # processed_fulldf["proportion_MSE"] = processed_fulldf["number_of_small_and_medium_orgs"] / processed_fulldf['number_of_organizations']
        # already done in a previous function

    # Sort and Deduplicate
    processed_fulldf = processed_fulldf.sort_values(by="id")
    processed_fulldf = processed_fulldf.drop_duplicates(subset=["id"], keep="first")

    # Final Column Dropping
    drop_cols = ['id','status', 'startDate', 'endDate', 'objective_y','main_topics',
                        'expected_impact', 'semantic_summary', 'expected_impact_processed', 'country', 'all_continents', 'continent',
                        'continents_processed', 'nutsCode', 'activityType', 'endOfParticipation', 'order', 'role',
                        'main_topics_processed'] # Add the temporary column used for main_topics processing

    drop_cols = [col for col in drop_cols if col in processed_fulldf.columns]
    processed_fulldf = processed_fulldf.drop(columns=drop_cols, errors='ignore')

    # Type Conversion for sustainability (ensured int earlier, now just check if it needs to be cast to int again)
    if 'sustainability' in processed_fulldf.columns and processed_fulldf['sustainability'].dtype != int:
        processed_fulldf["sustainability"] = processed_fulldf["sustainability"].astype(int)

    log.info("Generation of additional features complete.")
    return processed_fulldf, mlb_impact, mlb_continents


def consolidate_infrequent_topics(processed_fulldf: pd.DataFrame, parameters: dict) -> (pd.DataFrame, list):
    """
    Identifies infrequent topic columns and consolidates them into 'topic_other'.
    Impact and continent columns will be left as is.
    """
    log.info("Starting consolidation of infrequent topics...")
    frequency_threshold_ratio = parameters['frequency_threshold_ratio']

    # ONLY identify columns that start with 'topic_' for consolidation
    topic_cols_to_check = [
        col for col in processed_fulldf.columns
        if col.startswith('topic_')
    ]

    if not topic_cols_to_check:
        log.warning("No topic columns found for frequency calculation. Skipping topic consolidation.")
        return processed_fulldf, [] # Return original df and empty list if no topics

    # Ensure all identified topic columns are Sparse[int8]
    for col in topic_cols_to_check:
        if not pd.api.types.is_sparse(processed_fulldf[col].dtype):
            processed_fulldf[col] = processed_fulldf[col].astype('Sparse[int8]')
            log.info(f"Converted {col} to Sparse[int8]")

    # Calculate topic frequencies based on sparse data
    current_topic_counts = processed_fulldf[topic_cols_to_check].sum()

    total_rows = len(processed_fulldf)
    infrequent_topic_columns = current_topic_counts[current_topic_counts / total_rows < frequency_threshold_ratio].index.tolist()
    log.info(f"Number of infrequent topic columns identified: {len(infrequent_topic_columns)}")

    if infrequent_topic_columns:
        log.info(f"Consolidating into 'topic_other': {len(infrequent_topic_columns)} columns")
        # Ensure 'topic_other' exists and is sparse
        if 'topic_other' not in processed_fulldf.columns:
            processed_fulldf['topic_other'] = pd.Series(0, index=processed_fulldf.index, dtype='Sparse[int8]')

        # Sum the infrequent columns (using .any(axis=1) for 'if any of these topics are present')
        infrequent_sum = processed_fulldf[infrequent_topic_columns].any(axis=1).astype('Sparse[int8]')

        # Merge existing 'topic_other' with the new infrequent sum
        processed_fulldf['topic_other'] = (processed_fulldf['topic_other'].astype(bool) | infrequent_sum.astype(bool)).astype('Sparse[int8]')

        # Drop the original infrequent topic columns
        processed_fulldf = processed_fulldf.drop(columns=infrequent_topic_columns, errors='ignore')
    else:
        log.info("No topic columns found below the frequency threshold. No consolidation performed.")
        # If 'topic_other' exists but no topics were consolidated, remove it if it's all zeros
        if 'topic_other' in processed_fulldf.columns and processed_fulldf['topic_other'].sum() == 0:
            processed_fulldf = processed_fulldf.drop(columns=['topic_other'], errors='ignore')


    # --- Column Reordering (Optional but good for consistency) ---
    # This section remains largely the same as it just reorders columns for output consistency.
    # It will include all 'impact_' and 'continent_' columns, whether or not they were infrequent.

    final_topic_cols = sorted([col for col in processed_fulldf.columns if col.startswith('topic_') and col != 'topic_other'])
    final_impact_cols = sorted([col for col in processed_fulldf.columns if col.startswith('impact_')])
    final_continent_cols = sorted([col for col in processed_fulldf.columns if col.startswith('continent_')])

    non_feature_cols_final = [
        col for col in processed_fulldf.columns
        if not (col.startswith('topic_') or col.startswith('impact_') or col.startswith('continent_'))
    ]

    ordered_feature_cols = final_topic_cols + final_impact_cols + final_continent_cols

    if 'topic_other' in processed_fulldf.columns:
        ordered_feature_cols.append('topic_other') # Ensure 'topic_other' is at the end if it exists

    columns_to_keep_final = non_feature_cols_final + ordered_feature_cols
    processed_fulldf = processed_fulldf[columns_to_keep_final]
    log.info("Consolidation of infrequent topics complete.")
    return processed_fulldf
