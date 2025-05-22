# --- app.py (Your Shiny for Python Application) ---
from shiny import App, ui, render, reactive
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder # Required for unpickling
from sklearn.compose import ColumnTransformer # Required for unpickling
from sklearn.pipeline import Pipeline # Required for unpickling
from sklearn.ensemble import RandomForestRegressor # Required for unpickling
from sklearn.decomposition import TruncatedSVD # Required for unpickling

# --- 1. Load Model Components (same as Streamlit example, but outside ui/server) ---
def load_model_components():
    """Loads the pre-trained model pipeline and transformers."""
    try:
        pipeline = joblib.load('prediction_pipeline.joblib')
        mlb_topics = joblib.load('mlb_topics.joblib')
        svd_topics = joblib.load('svd_topics.joblib')
        mlb_impact = joblib.load('mlb_impact.joblib')
        svd_impact = joblib.load('svd_impact.joblib')
        infrequent_topics = joblib.load('infrequent_topics.joblib')
        feature_columns = joblib.load('feature_columns.joblib')

        # Extract unique categories for OneHotEncoder from the loaded pipeline's preprocessor
        ohe_categories = pipeline.named_steps['preprocessor'].named_transformers_['cat'].categories_

        unique_categories = {
            'legalBasis': ohe_categories[0].tolist(),
            'fundingScheme': ohe_categories[1].tolist(),
            'total_scientific_domain': ohe_categories[2].tolist(), # Rename to match your code, assuming this is correct
            'activityType': ohe_categories[3].tolist(),
            'country': ohe_categories[4].tolist()
        }

        # IMPORTANT: Verify 'scientific_domain' from your training code matches 'total_scientific_domain' or adjust
        # It seems there might be a mismatch here based on the original training code.
        # If 'scientific_domain' was the actual column name for OHE, then keep it consistent.
        # Assuming the first element of ohe_categories refers to 'legalBasis', second to 'fundingScheme', etc.
        # Make sure this order matches what was passed to ColumnTransformer during training.
        # For example, if 'scientific_domain' is at index 2 for the 'cat' transformer:
        scientific_domain_index = -1
        for i, (name, transformer, features) in enumerate(pipeline.named_steps['preprocessor'].transformers):
            if name == 'cat':
                for j, feature in enumerate(features):
                    if feature == 'scientific_domain':
                        scientific_domain_index = j
                        break
            if scientific_domain_index != -1:
                break

        if scientific_domain_index != -1:
             unique_categories['scientific_domain'] = ohe_categories[scientific_domain_index].tolist()
        else:
             # Fallback or raise error if 'scientific_domain' isn't found
             print("Warning: 'scientific_domain' not found in OHE categories. Please check your training script's ColumnTransformer.")
             # You might want to handle this more robustly, e.g., by skipping this category
             unique_categories['scientific_domain'] = []


        return pipeline, mlb_topics, svd_topics, mlb_impact, svd_impact, infrequent_topics, feature_columns, unique_categories
    except FileNotFoundError:
        print("ERROR: Model components not found. Please run the training script first.")
        raise

pipeline, mlb_topics, svd_topics, mlb_impact, svd_impact, infrequent_topics, feature_columns, unique_categories = load_model_components()

# Prepare topic/impact options for the UI
all_mlb_topics = sorted(mlb_topics.classes_.tolist())
if "other" not in all_mlb_topics:
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
            ui.input_numeric("duration", "Duration (days)", value=365, min=0, max=3650),
            ui.input_numeric("sustainability", "Sustainability Score (0.0 to 1.0)", value=0.5, min=0.0, max=1.0, step=0.01),
            ui.input_radio_buttons("sme", "Is it an SME?", choices={"1": "Yes", "0": "No"}, selected="0"),
            ui.h4("Categorical Inputs"),
            ui.input_select("legal_basis", "Legal Basis", choices=unique_categories['legalBasis']),
            ui.input_select("funding_scheme", "Funding Scheme", choices=unique_categories['fundingScheme']),
            ui.input_select("scientific_domain", "Scientific Domain", choices=unique_categories['scientific_domain']),
            ui.input_select("activity_type", "Activity Type", choices=unique_categories['activityType']),
            ui.input_select("country", "Country", choices=unique_categories['country']),
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

# --- 3. Define the Server Logic ---
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
            # --- Preprocess Main Topics for SVD ---
            # Apply the same 'other' replacement logic as in training
            processed_main_topics_for_mlb = [
                topic.lower() if topic not in infrequent_topics else "other"
                for topic in input.selected_main_topics()
            ]
            # MultiLabelBinarizer transform
            main_topics_binary = mlb_topics.transform([processed_main_topics_for_mlb])
            # SVD transform
            main_topics_svd_features = svd_topics.transform(main_topics_binary)

            # Create DataFrame for SVD topics
            topic_svd_df = pd.DataFrame(
                main_topics_svd_features,
                columns=[f'topic_svd_{i+1}' for i in range(main_topics_svd_features.shape[1])]
            )

            # --- Preprocess Expected Impact for SVD ---
            processed_impact_for_mlb = [
                impact.lower() for impact in input.selected_expected_impact()
            ]
            impact_binary = mlb_impact.transform([processed_impact_for_mlb])
            impact_svd_features = svd_impact.transform(impact_binary)

            # Create DataFrame for SVD impacts
            impact_svd_df = pd.DataFrame(
                impact_svd_features,
                columns=[f'impact_svd_{i+1}' for i in range(impact_svd_features.shape[1])]
            )

            # --- Construct the input DataFrame for the pipeline ---
            input_data = pd.DataFrame({
                'duration': [input.duration()],
                'legalBasis': [input.legal_basis()],
                'fundingScheme': [input.funding_scheme()],
                'scientific_domain': [input.scientific_domain()],
                'sustainability': [input.sustainability()],
                'activityType': [input.activity_type()],
                'SME': [int(input.sme())], # Convert '0'/'1' string from radio buttons to int
                'country': [input.country()]
            })

            # Concatenate the SVD features
            final_input_df = pd.concat([input_data, topic_svd_df, impact_svd_df], axis=1)

            # Ensure the columns are in the exact order the model expects
            final_input_df = final_input_df[feature_columns]

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