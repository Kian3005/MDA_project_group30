from shiny import App, ui, render, reactive
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer

# --- 1. Load Model Components ---
def load_model_components():
    """Loads the pre-trained model pipeline and transformers."""
    try:
        # Ensure this path is correct for your SAVED TUNED pipeline
        pipeline = joblib.load('tuned_random_forest_prediction_pipeline.joblib')

        mlb_topics = joblib.load('mlb_topics.joblib')
        mlb_impact = joblib.load('mlb_impact.joblib')
        mlb_continents = joblib.load('mlb_continents.joblib')
        infrequent_topics = joblib.load('infrequent_topics.joblib')

        # Load the final UI lists directly.
        all_mlb_topics = joblib.load('final_ui_topics.joblib')
        all_mlb_continents = joblib.load('final_ui_continents.joblib')

        feature_columns = joblib.load('feature_columns.joblib')

        # Access the OneHotEncoder transformer from the ColumnTransformer
        ohe_transformer = pipeline.named_steps['preprocessor'].named_transformers_['cat']

        unique_categories = {}
        if hasattr(ohe_transformer, 'categories_') and hasattr(ohe_transformer, 'feature_names_in_'):
            for feature_name, categories in zip(ohe_transformer.feature_names_in_, ohe_transformer.categories_):
                unique_categories[feature_name] = categories.tolist()
        else:
            print("Warning: OneHotEncoder does not have 'categories_' or 'feature_names_in_' attribute. Check your saved model.")
            unique_categories = {
                'legalBasis': [], 'fundingScheme': [], 'scientific_domain': [],
                'sustainability': [], 'problem_type': []
            }

        # --- Refined logic for all_mlb_impacts ---
        all_mlb_impacts_raw = mlb_impact.classes_.tolist()
        if "other" in all_mlb_impacts_raw:
            impacts_without_other_model = [impact for impact in all_mlb_impacts_raw if impact != "other"]
        else:
            impacts_without_other_model = all_mlb_impacts_raw
        sorted_impacts_model = sorted(impacts_without_other_model)
        all_mlb_impacts = sorted_impacts_model + ["other"]

        return pipeline, mlb_topics, mlb_impact, mlb_continents, infrequent_topics, feature_columns, unique_categories, all_mlb_topics, all_mlb_impacts, all_mlb_continents
    except FileNotFoundError as e:
        print(f"ERROR: Model component not found: {e}. Please ensure the training script has been run and files are in the correct directory.")
        raise
    except Exception as e:
        print(f"ERROR loading model components: {e}")
        raise

# Load components globally
pipeline, mlb_topics, mlb_impact, mlb_continents, infrequent_topics, feature_columns, unique_categories, all_mlb_topics, all_mlb_impacts, all_mlb_continents = load_model_components()


# --- 2. Define the User Interface (UI) ---
app_ui = ui.page_fluid(
    ui.h2("Project EC Max Contribution Predictor"),
    ui.markdown("Enter the details of a project to predict its estimated maximum EC contribution."),

    ui.hr(),

    ui.page_sidebar(
        ui.sidebar(
            ui.h4("Numerical Inputs"),
            ui.input_numeric("project_length_days", "Project Length (days)", value=365, min=0, max=3650),
            ui.input_numeric("number_of_organizations", "Number of Organizations", value=1, min=1, max=100),
            ui.input_numeric("proportion_of_small_and_medium_orgs", "Proportion of Small/Medium Organizations", value=0.5, min=0, max=1, step=0.01),

            ui.h4("Categorical Inputs"),
            ui.input_select("legal_basis", "Legal Basis", choices=unique_categories.get('legalBasis', [])),
            ui.input_select("funding_scheme", "Funding Scheme", choices=unique_categories.get('fundingScheme', [])),
            ui.input_select("scientific_domain", "Scientific Domain", choices=unique_categories.get('scientific_domain', [])),
            ui.input_select("sustainability", "Sustainability", choices=unique_categories.get('sustainability', [])),
            ui.input_select("problem_type", "Problem Type", choices=unique_categories.get('problem_type', [])),
        ),
        ui.h3("Topics, Impact, and Continents"),
        ui.layout_column_wrap(
            1/3,
            # Modified lines: Replace underscores with spaces for display
            ui.input_checkbox_group("selected_main_topics", "Main Topics (select one or more)", 
                                    choices=[topic.replace("_", " ") for topic in all_mlb_topics]),
            ui.input_checkbox_group("selected_expected_impact", "Expected Impact (select one or more)", 
                                    choices=[impact.replace("_", " ") for impact in all_mlb_impacts]),
            ui.input_checkbox_group("selected_continents", "Continents Involved (select one or more)", 
                                    choices=[continent.replace("_", " ") for continent in all_mlb_continents]),
        ),
        ui.br(),
        ui.input_action_button("predict_button", "Predict Contribution", class_="btn-primary"),
        ui.hr(),
        ui.h3("Prediction Result"),
        # --- NEW: Added output for CI ---
        ui.output_text("prediction_output"),
        ui.output_text("ci_output")
    )
)

def server(input, output, session):

    prediction_result = reactive.Value(None)
    ci_lower = reactive.Value(None)
    ci_upper = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.predict_button)
    def _():
        # --- Input Validation ---
        if not input.selected_main_topics():
            ui.notification_show("Please select at least one Main Topic.", type="warning")
            return
        if not input.selected_expected_impact():
            ui.notification_show("Please select at least one Expected Impact.", type="warning")
            return
        if not input.selected_continents():
            ui.notification_show("Please select at least one Continent Involved.", type="warning")
            return
        if not input.problem_type():
            ui.notification_show("Please select a Problem Type.", type="warning")
            return
        if not (0 <= input.proportion_of_small_and_medium_orgs() <= 1):
            ui.notification_show("Proportion of Small/Medium Orgs must be between 0 and 1.", type="warning")
            return

        try:
            input_row_data = {
                'project_length_days': input.project_length_days(),
                'number_of_organizations': input.number_of_organizations(),
                'proportion_of_small_and_medium_orgs': input.proportion_of_small_and_medium_orgs(),
                'legalBasis': input.legal_basis(),
                'fundingScheme': input.funding_scheme(),
                'scientific_domain': input.scientific_domain(),
                'sustainability': input.sustainability(),
                'problem_type': input.problem_type()
            }

            # --- Preprocessing for Topics ---
            processed_main_topics_for_mlb = []
            if "other" in input.selected_main_topics():
                processed_main_topics_for_mlb.append("other")
            else:
                for topic in input.selected_main_topics():
                    # Revert spaces back to underscores for model input
                    cleaned_topic = topic.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                    if f'topic_{cleaned_topic}' in infrequent_topics:
                        processed_main_topics_for_mlb.append("other")
                    else:
                        processed_main_topics_for_mlb.append(cleaned_topic)
                processed_main_topics_for_mlb = list(set(processed_main_topics_for_mlb))

            main_topics_binary_array = mlb_topics.transform([processed_main_topics_for_mlb]).toarray()[0]
            for i, topic_label in enumerate(mlb_topics.classes_):
                col_name = f'topic_{topic_label.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower()}'
                if col_name in feature_columns:
                    input_row_data[col_name] = main_topics_binary_array[i]

            # --- Preprocess Expected Impact ---
            selected_impact_lower = [item.lower() for item in input.selected_expected_impact()]
            processed_impact_for_mlb = []
            if "other" in selected_impact_lower:
                processed_impact_for_mlb = ["other"]
                if len(selected_impact_lower) > 1:
                    ui.notification_show("Selecting 'Other' for Expected Impact will consider only 'other' and set other specific impact indicators to zero.", type="info")
            else:
                for impact in input.selected_expected_impact():
                    # Revert spaces back to underscores for model input
                    cleaned_impact = impact.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                    if cleaned_impact in mlb_impact.classes_:
                        processed_impact_for_mlb.append(cleaned_impact)
                processed_impact_for_mlb = list(set(processed_impact_for_mlb))

            impact_binary_array = mlb_impact.transform([processed_impact_for_mlb]).toarray()[0]
            for i, impact_label in enumerate(mlb_impact.classes_):
                col_name = f'impact_{impact_label.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower()}'
                if col_name in feature_columns:
                    input_row_data[col_name] = impact_binary_array[i]

            # --- Preprocess Continents ---
            selected_continents_lower = [item.lower() for item in input.selected_continents()]
            processed_continents_for_mlb = []
            if "unknown" in selected_continents_lower:
                processed_continents_for_mlb.append("unknown")
                if len(selected_continents_lower) > 1:
                    ui.notification_show("Selecting 'unknown' for Continents will consider only 'unknown' and set other specific continent indicators to zero.", type="info")
            else:
                for continent in input.selected_continents():
                    # Revert spaces back to underscores for model input
                    cleaned_continent = continent.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                    if cleaned_continent in mlb_continents.classes_:
                        processed_continents_for_mlb.append(cleaned_continent)
                processed_continents_for_mlb = list(set(processed_continents_for_mlb))

            continents_binary_array = mlb_continents.transform([processed_continents_for_mlb]).toarray()[0]
            for i, continent_label in enumerate(mlb_continents.classes_):
                col_name = f'continent_{continent_label.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower()}'
                if col_name in feature_columns:
                    input_row_data[col_name] = continents_binary_array[i]

            # --- Create the final input DataFrame ---
            temp_df_dict = {col: 0 for col in feature_columns}
            for key, value in input_row_data.items():
                if key in temp_df_dict:
                    temp_df_dict[key] = value
            
            raw_input_df = pd.DataFrame([temp_df_dict])
            final_input_df = raw_input_df.reindex(columns=feature_columns, fill_value=0)

            # --- DEBUGGING STEP (KEEP FOR NOW) ---
            print("\n--- Debugging final_input_df ---")
            print("DataFrame head:\n", final_input_df.head())
            print("DataFrame dtypes:\n", final_input_df.dtypes)
            final_input_df.info()
            print("Any NaN values in numerical features? (Should be False for StandardScaler to work)")
            print(final_input_df.apply(lambda x: pd.isna(x).any()))
            missing_cols_in_input = set(feature_columns) - set(final_input_df.columns)
            if missing_cols_in_input:
                print(f"\nWARNING: Missing columns in final_input_df: {missing_cols_in_input}")
            extra_cols_in_input = set(final_input_df.columns) - set(feature_columns)
            if extra_cols_in_input:
                print(f"\nWARNING: Extra columns in final_input_df: {extra_cols_in_input}")
            print("----------------------------------\n")

            # Make prediction
            prediction = pipeline.predict(final_input_df)[0]
            prediction_result.set(prediction)

            # --- Calculate 90% Confidence Interval ---
            # Get the Random Forest Regressor from the pipeline
            rf_regressor = pipeline.named_steps['regressor_bayes_search'].best_estimator_

            # Get predictions from each individual tree
            individual_tree_predictions = []
            # First, preprocess the input data using the pipeline's preprocessor
            # We need to transform final_input_df without passing it through the entire pipeline again,
            # as the regressor_bayes_search step expects already preprocessed data.
            preprocessed_input = pipeline.named_steps['preprocessor'].transform(final_input_df)

            for tree in rf_regressor.estimators_:
                # Each tree expects the preprocessed data
                individual_tree_predictions.append(tree.predict(preprocessed_input)[0])

            # Convert to numpy array for percentile calculation
            individual_tree_predictions = np.array(individual_tree_predictions)

            # Calculate 2.5th and 97.5th percentiles for 80% CI
            lower_bound = np.percentile(individual_tree_predictions, 10)
            upper_bound = np.percentile(individual_tree_predictions, 90)

            ci_lower.set(lower_bound)
            ci_upper.set(upper_bound)

        except Exception as e:
            ui.notification_show(f"An error occurred: {e}", type="danger")
            prediction_result.set(None)
            ci_lower.set(None)
            ci_upper.set(None)
            print(f"Prediction error: {e}")

    @output
    @render.text
    def prediction_output():
        if prediction_result.get() is not None:
            return f"Predicted EC Max Contribution: €{prediction_result.get():,.2f}"
        else:
            return "Enter project details and click 'Predict' to see the result."

    @output
    @render.text
    def ci_output():
        if ci_lower.get() is not None and ci_upper.get() is not None:
            return f"80% Confidence Interval: €{ci_lower.get():,.2f} - €{ci_upper.get():,.2f}"
        else:
            return "" # Return empty string if no CI is available
# Create the Shiny app instance
app = App(app_ui, server)