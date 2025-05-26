"""
This is a boilerplate pipeline 'data_loading'
generated using Kedro 0.19.13
"""

from kedro.pipeline import Pipeline, pipeline, node
import pandas as pd

def load_organization(organization_data_path: str) -> pd.DataFrame:
    dfo = organization_data_path
    return dfo

def load_project(project_data_path: str) -> pd.DataFrame:
    dfp = project_data_path
    return dfp

def load_outputml(output_data_path: str) -> pd.DataFrame:
    dfoutput = output_data_path
    return dfoutput

def mergedfs(dfo: pd.DataFrame, dfp: pd.DataFrame, dfoutput: pd.DataFrame) -> pd.DataFrame:
    newdf = dfp.merge(dfoutput, on="id")
    # One to many merge newdf and dfo, excpect 100249 rows in fulldf
    fulldf = newdf.merge(dfo, left_on="id", right_on="projectID", validate="one_to_many")
    return fulldf

def drop_columns(fulldf: pd.DataFrame) -> pd.DataFrame:
    print(fulldf.info())
    # Delete active and nature (empty)
    fulldf = fulldf.drop(columns=["active", "nature"])
    # Drop col's based on initial exploration (old exploration files)
    fulldf = fulldf.drop(columns=["acronym", "title", "totalCost_x", "topics", "objective_x", "rcn_x", "grantDoi", "contentUpdateDate_x", \
                              "projectID", "projectAcronym", "organisationID", "vatNumber", "name", "shortName", "street", "postCode", \
                              "city", "geolocation", "organizationURL", "contactForm","contentUpdateDate_y", "rcn_y", "totalCost_y", \
                              "ecContribution", "netEcContribution", "ecSignatureDate", \
                              "masterCall", "subCall", "frameworkProgramme", "status"
                            ])
    return fulldf
    

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_organization,
            inputs="organization_data_path",
            outputs="dfo",
            name="load_organization",
            tags=["data_loading"]
        ),
        node(
            func=load_project,
            inputs="project_data_path",
            outputs="dfproject",
            name="load_project",
            tags=["data_loading"]
        ),
        node(
            func=load_outputml,
            inputs="output_data_path",
            outputs="dfoutput",
            name="load_outputml",
            tags=["data_loading"]
        ),
        node(
            func=mergedfs,
            inputs=["dfo", "dfproject", "dfoutput"],
            outputs="fulldf",
            name="merge_dataframes",
            tags=["data_loading"]
        ),
        node(
            func=drop_columns,
            inputs="fulldf",
            outputs="cleaned_df",
            name="drop_columns",
            tags=["data_loading"]
        )
    ])

