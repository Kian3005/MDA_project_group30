# src/mda/pipeline_registry.py

from typing import Dict
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

# Import the create_pipeline functions directly from their modules
from mda.pipelines.raw_processing.pipeline import create_pipeline as create_raw_processing_pipeline
from mda.pipelines.data_preprocessing.pipeline import create_pipeline as create_data_preprocessing_pipeline
from mda.pipelines.llm_topic_modeling.pipeline import create_pipeline as create_llm_topic_modeling_pipeline 
from mda.pipelines.model_training.pipeline import create_pipeline as create_model_training_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Call the explicitly imported functions
    raw_processing_pipeline = create_raw_processing_pipeline()
    data_preprocessing_pipeline = create_data_preprocessing_pipeline()
    llm_topic_modeling_pipeline = create_llm_topic_modeling_pipeline()
    model_training_pipeline = create_model_training_pipeline()
    # --------------------------

    return {
        "__default__": raw_processing_pipeline + data_preprocessing_pipeline + llm_topic_modeling_pipeline + model_training_pipeline, # Ensure all are included in default, and in correct order
        "raw_processing": raw_processing_pipeline,
        "data_preprocessing": data_preprocessing_pipeline,
        "llm_topic_modeling": llm_topic_modeling_pipeline,
        "model_training": model_training_pipeline,
    }