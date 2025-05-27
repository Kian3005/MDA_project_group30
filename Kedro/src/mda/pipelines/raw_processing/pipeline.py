# src/mda/pipelines/raw_processing/pipeline.py

from kedro.pipeline import Pipeline, node

# --- THIS IS THE CRUCIAL CHANGE ---
# Import the 'nodes' module and give it an alias, e.g., 'rp_nodes' (raw_processing_nodes)
from .nodes import merge_and_initial_clean, handle_missing_values_and_adjustments, feature_engineering
# ----------------------------------

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the raw data processing pipeline.

    Args:
        kwargs: Not used in this pipeline, but often used for passing parameters.

    Returns:
        A Kedro Pipeline object.
    """
    return Pipeline(
        [
            node(
                # Now, you call the functions using the alias followed by the function name
                func=merge_and_initial_clean,
                inputs=[
                    "raw_project_data",
                    "raw_organization_data",
                    "raw_output_data"
                ],
                outputs="merged_initial_clean_data",
                name="merge_and_initial_clean_node",
                # tags=["raw_processing"]
            ),
            node(
                func=handle_missing_values_and_adjustments,
                inputs="merged_initial_clean_data",
                outputs="cleaned_data_with_adjustments",
                name="handle_missing_and_adjustments_node",
                # tags=["raw_processing"]
            ),
            node(
                func=feature_engineering,
                inputs="cleaned_data_with_adjustments",
                outputs="final_processed_data",
                name="feature_engineering_node",
                # tags=["raw_processing"]
            ),
        ]
    )