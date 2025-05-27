import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import logging

log = logging.getLogger(__name__)

def generate_topic_merge_library(
    initial_dataframe: pd.DataFrame, # This would be the input DataFrame containing 'main_topics'
    params: dict # You might pass similarity_threshold or model name via params
) -> pd.DataFrame:
    """
    Generates the topic merge library using SentenceTransformer and cosine similarity.

    Args:
        initial_dataframe (pd.DataFrame): DataFrame containing the 'main_topics' column.
        params (dict): Dictionary of parameters (e.g., similarity_threshold).

    Returns:
        pd.DataFrame: The DataFrame containing LLM-driven merge rules ('Canonical_Name' and source topics).
    """
    similarity_threshold = params.get("similarity_threshold", 0.50)
    model_name = params.get("sentence_transformer_model", "all-MiniLM-L6-v2")

    log.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    # Ensure 'main_topics' is object type for literal_eval and list manipulation
    df_topics = initial_dataframe.copy() # Work on a copy to avoid modifying original input
    df_topics['main_topics'] = df_topics['main_topics'].astype('object')

    # Safely evaluate string representations of lists and clean topics
    def preprocess_topics_list_for_embedding(topic_list):
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

    df_topics['main_topics_cleaned'] = df_topics['main_topics'].apply(preprocess_topics_list_for_embedding)

    # Get all unique topics
    all_topics = [topic for sublist in df_topics['main_topics_cleaned'] for topic in sublist]
    topics = sorted(list(set(all_topics))) # Get unique and sorted topics

    if not topics:
        log.warning("No unique topics found for embedding. Returning empty merge library.")
        return pd.DataFrame(columns=['Canonical_Name'])

    log.info(f"Generating embeddings for {len(topics)} unique topics.")
    embeddings = model.encode(topics, convert_to_tensor=True)

    log.info("Calculating cosine similarity matrix.")
    similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

    log.info(f"Identifying topic groups with similarity >= {similarity_threshold}")
    merged_topic_groups = []
    processed_topics_indices = set()

    for i, current_topic in enumerate(topics):
        if i in processed_topics_indices:
            continue

        current_group = [current_topic]
        processed_topics_indices.add(i)

        for j, other_topic in enumerate(topics):
            if i == j or j in processed_topics_indices:
                continue

            if similarity_matrix[i, j] >= similarity_threshold:
                current_group.append(other_topic)
                processed_topics_indices.add(j)

        # Ensure all topics are accounted for, either in a group or as a single topic
        if len(current_group) > 1:
            merged_topic_groups.append(current_group)
        else:
            # If a topic wasn't merged, add it as a single-element group
            merged_topic_groups.append([current_topic])


    log.info(f"Identified {len(merged_topic_groups)} topic groups/individual topics.")

    # Prepare data for DataFrame
    max_group_size = max(len(group) for group in merged_topic_groups) if merged_topic_groups else 0
    csv_data = []
    for group in merged_topic_groups:
        row = group + [None] * (max_group_size - len(group))
        csv_data.append(row)

    column_names = [f'Topic_{i+1}' for i in range(max_group_size)]
    df_merge_library = pd.DataFrame(csv_data, columns=column_names)

    df_merge_library['Canonical_Name'] = df_merge_library['Topic_1']
    df_merge_library = df_merge_library[['Canonical_Name'] + [col for col in df_merge_library.columns if col != 'Canonical_Name']]

    log.info("Generated LLM Merge Library DataFrame:")
    log.info(df_merge_library.head())

    return df_merge_library