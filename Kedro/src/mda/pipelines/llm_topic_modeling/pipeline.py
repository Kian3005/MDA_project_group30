from kedro.pipeline import Pipeline, node
from . import nodes as llm_topic_nodes

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=llm_topic_nodes.generate_topic_merge_library,
                inputs=["final_processed_data", "params:llm_merge_params"], # Assumes 'final_processed_data' has 'main_topics'
                outputs="topic_merging_library", # This must match the name in catalog.yml
                name="generate_topic_merge_library_node",
            ),
        ]
    )