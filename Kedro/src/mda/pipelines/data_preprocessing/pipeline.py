# src/mda/pipelines/data_preprocessing/pipeline.py

from kedro.pipeline import Pipeline, node
from . import nodes as data_preprocessing_nodes

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the data preprocessing pipeline.

    Args:
        kwargs: Not used in this pipeline, but often used for passing parameters.

    Returns:
        A Kedro Pipeline object.
    """
    return Pipeline(
        [
            node(
                func=data_preprocessing_nodes.preprocess_main_topics,
                inputs=["final_processed_data", "topic_merging_library"], 
                outputs=["df_with_merged_topics", "initial_mlb_data_for_topics"],
                name="preprocess_main_topics_node",
                # tags=["data_preprocessing"]
            ),

            node(
                func=data_preprocessing_nodes.consolidate_infrequent_topics,
                inputs=["df_with_merged_topics", "params:infrequent_topic_params"],
                outputs="df_with_consolidated_topics",
                name="consolidate_infrequent_topics_node",
                # tags=["data_preprocessing"]
            ),
            node(
                func=data_preprocessing_nodes.generate_additional_features,
                inputs="df_with_consolidated_topics",
                outputs=[
                    "preprocessed_data_final",
                    "mlb_impact_model",
                    "mlb_continents_model"
                ],
                name="process_impact_and_continents_node",
            ),
        ]
    )