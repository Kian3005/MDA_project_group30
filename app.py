# --- app.py (Your Shiny for Python Application) ---
from shiny import App, ui, render, reactive
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder # Required for unpickling
from sklearn.compose import ColumnTransformer # Required for unpickling
from sklearn.pipeline import Pipeline # Required for unpickling
from sklearn.ensemble import RandomForestRegressor # Required for unpickling
# from sklearn.decomposition import TruncatedSVD # NO LONGER NEEDED

# --- 1. Load Model Components ---
def load_model_components():
    """Loads the pre-trained model pipeline and transformers."""
    try:
        pipeline = joblib.load('prediction_pipeline.joblib')
        mlb_topics = joblib.load('mlb_topics.joblib')
        mlb_impact = joblib.load('mlb_impact.joblib')
        infrequent_topics = joblib.load('infrequent_topics.joblib')
        feature_columns = joblib.load('feature_columns.joblib')

        # Extract unique categories from the OneHotEncoder in the loaded pipeline
        # Find the 'cat' transformer using named_transformers_
        # This accesses the *fitted* transformer
        ohe_transformer = pipeline.named_steps['preprocessor'].named_transformers_['cat']

        if ohe_transformer is None: # This check might be redundant if 'cat' is guaranteed
            raise ValueError("OneHotEncoder transformer ('cat') not found in the pipeline!")

        # Map features to their categories using the features list from the preprocessor
        # and the categories_ from the OHE transformer
        unique_categories = {}
        # Get the feature names the OHE was trained on (from the pipeline's preprocessor)
        ohe_features_trained_on = pipeline.named_steps['preprocessor'].named_transformers_['cat'].feature_names_in_
        for i, feature_name in enumerate(ohe_features_trained_on):
            unique_categories[feature_name] = ohe_transformer.categories_[i].tolist()


        return pipeline, mlb_topics, mlb_impact, infrequent_topics, feature_columns, unique_categories
    except FileNotFoundError:
        print("ERROR: Model components not found. Please run the training script first.")
        raise
    except Exception as e:
        print(f"ERROR loading model components: {e}")
        raise

# Load components globally
pipeline, mlb_topics, mlb_impact, infrequent_topics, feature_columns, unique_categories = load_model_components()


# Prepare topic/impact options for the UI (mlb_topics.classes_ and mlb_impact.classes_ are already sorted)
# Ensure 'other' is in topics if it was used in training
all_mlb_topics = sorted(mlb_topics.classes_.tolist())
if "other" not in all_mlb_topics: # Add 'other' if it's not naturally a class (e.g., if no topics were infrequent during training)
    all_mlb_topics.append("other")
all_mlb_impacts = sorted(mlb_impact.classes_.tolist())


# --- 2. Define the User Interface (UI) ---
app_ui = ui.page_fluid(
    ui.h2("Project EC Max Contribution Predictor"),
    ui.markdown("Enter the details of a project to predict its estimated maximum EC contribution."),

    ui.hr(),

    ui.page_sidebar(
        ui.sidebar( # This is the sidebar content
            ui.h4("Numerical Inputs"),
            ui.input_numeric("project_length_days", "Project Length (days)", value=365, min=0, max=3650),
            ui.input_numeric("number_of_organizations", "Number of Organizations", value=1, min=1, max=100),
            # 'sustainability' is now categorical, so it should be a select input
            # ui.input_numeric("sustainability", "Sustainability Score (0.0 to 1.0)", value=0.5, min=0.0, max=1.0, step=0.01),
            # ui.input_radio_buttons("sme", "Is it an SME?", choices={"1": "Yes", "0": "No"}, selected="0"), # If SME is not in trained features, remove this

            ui.h4("Categorical Inputs"),
            ui.input_select("legal_basis", "Legal Basis", choices=unique_categories.get('legalBasis', [])),
            ui.input_select("funding_scheme", "Funding Scheme", choices=unique_categories.get('fundingScheme', [])),
            ui.input_select("scientific_domain", "Scientific Domain", choices=unique_categories.get('scientific_domain', [])),
            # Add 'sustainability' here as a categorical input if it's OHE'd
            ui.input_select("sustainability", "Sustainability", choices=unique_categories.get('sustainability', [])),
            # If 'continent' is a categorical feature, add it here:
            ui.input_select("continent", "Continent", choices=unique_categories.get('continent', [])),
            # If activityType is a categorical feature, add it here:
            # ui.input_select("activity_type", "Activity Type", choices=unique_categories.get('activityType', [])),
        ),
        # The main content goes directly after ui.sidebar() within ui.page_sidebar
        ui.h3("Topics and Impact"),
        ui.layout_column_wrap(
            1/2, # This means each element will take up 1/2 of the available width (creating two columns)
            ui.input_checkbox_group("selected_main_topics", "Main Topics (select one or more)", choices=all_mlb_topics),
            ui.input_checkbox_group("selected_expected_impact", "Expected Impact (select one or more)", choices=all_mlb_impacts),
        ),
        ui.br(),
        ui.input_action_button("predict_button", "Predict Contribution", class_="btn-primary"),
        ui.hr(),
        ui.h3("Prediction Result"),
        ui.output_text("prediction_output")
    )
)

def server(input, output, session):

    # Reactive value to store the prediction result
    prediction_result = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.predict_button) # This effect runs when the predict_button is clicked
    def _():
        # --- Input Validation ---
        if not input.selected_main_topics():
            ui.notification_show("Please select at least one Main Topic.", type="warning")
            return
        if not input.selected_expected_impact():
            ui.notification_show("Please select at least one Expected Impact.", type="warning")
            return

        try:
            # Prepare a dictionary for the input row
            input_row_data = {
                'project_length_days': input.project_length_days(),
                'number_of_organizations': input.number_of_organizations(),
                'legalBasis': input.legal_basis(),
                'fundingScheme': input.funding_scheme(),
                'scientific_domain': input.scientific_domain(),
                'continent': input.continent(),
                'sustainability': input.sustainability(),
            }

            # --- Preprocess Main Topics ---
            processed_main_topics_for_mlb = [
                topic.lower() if topic not in infrequent_topics else "other"
                for topic in input.selected_main_topics()
            ]
            main_topics_binary_array = mlb_topics.transform([processed_main_topics_for_mlb]).toarray()[0] # Get the single row
            for i, topic_label in enumerate(mlb_topics.classes_):
                col_name = f'topic_{topic_label.replace(' ', '_').replace('-', '_')}'
                input_row_data[col_name] = main_topics_binary_array[i]


            # --- Preprocess Expected Impact ---
            processed_impact_for_mlb = [
                impact.lower() for impact in input.selected_expected_impact()
            ]
            impact_binary_array = mlb_impact.transform([processed_impact_for_mlb]).toarray()[0] # Get the single row
            for i, impact_label in enumerate(mlb_impact.classes_):
                col_name = f'impact_{impact_label.replace(' ', '_').replace('-', '_')}'
                input_row_data[col_name] = impact_binary_array[i]

            # Create the final input DataFrame from the combined dictionary
            # This ensures we always have one row and correct columns
            final_input_df = pd.DataFrame([input_row_data], columns=feature_columns).fillna(0)

            # Important: Reindex to ensure column order and presence, filling NaNs (for unselected OHEs) with 0
            # .reindex ensures all columns from feature_columns are present.
            # .fillna(0) handles any columns that were in feature_columns but not explicitly
            # set in input_row_data (e.g., if a new category was added to training but not yet in UI options)
            final_input_df = final_input_df.reindex(columns=feature_columns, fill_value=0)


            # Make prediction
            prediction = pipeline.predict(final_input_df)[0]
            prediction_result.set(prediction) # Update the reactive value

        except Exception as e:
            ui.notification_show(f"An error occurred: {e}", type="danger")
            prediction_result.set(None) # Clear prediction on error
            print(f"Prediction error: {e}") # For debugging in console

    @output
    @render.text
    def prediction_output():
        if prediction_result.get() is not None:
            return f"### Predicted EC Max Contribution: €{prediction_result.get():,.2f}"
        else:
            return "Enter project details and click 'Predict' to see the result."

# Create the Shiny app instance
app = App(app_ui, server)