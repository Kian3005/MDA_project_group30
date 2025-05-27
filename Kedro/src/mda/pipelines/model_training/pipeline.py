# In src/mda/pipelines/model_training/pipeline.py

from kedro.pipeline import Pipeline, node
from . import nodes as model_training_nodes

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=model_training_nodes.preprocess_and_clean_data,
                inputs=["preprocessed_data_final", "params:outlier_removal_params"],
                outputs="df_for_model_training",
                name="preprocess_data_for_model_training_node",
            ),
            node(
                func=model_training_nodes.train_model,
                inputs=["df_for_model_training", "params:model_training_params"],
                outputs=[
                    "_trained_prediction_pipeline_intermediate", # RENAMED!
                    "_model_evaluation_metrics_intermediate"     # RENAMED!
                ],
                name="train_random_forest_model_node",
            ),
            node(
                func=model_training_nodes.save_model_artifacts,
                inputs=[
                    "_trained_prediction_pipeline_intermediate", # Input from previous node
                    "_model_evaluation_metrics_intermediate",    # Input from previous node
                    "preprocessed_data_final"
                ],
                outputs=[
                    "trained_prediction_pipeline",      # Final output name, matches catalog.yml
                    "model_evaluation_metrics",         # Final output name, matches catalog.yml
                    "ui_topics_list",
                    "ui_impacts_list",
                    "ui_continents_list",
                    "model_feature_columns_order"
                ],
                name="save_model_artifacts_node",
            ),
        ]
    )