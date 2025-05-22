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
        mlb_continents = joblib.load('mlb_continents.joblib') # Load mlb_continents
        infrequent_topics = joblib.load('infrequent_topics.joblib')
        feature_columns = joblib.load('feature_columns.joblib')

        # Extract unique categories from the OneHotEncoder in the loaded pipeline
        ohe_transformer = pipeline.named_steps['preprocessor'].named_transformers_['cat']

        unique_categories = {}
        if ohe_transformer is not None and hasattr(ohe_transformer, 'categories_'):
            ohe_features_trained_on = pipeline.named_steps['preprocessor'].named_transformers_['cat'].feature_names_in_
            for i, feature_name in enumerate(ohe_features_trained_on):
                unique_categories[feature_name] = ohe_transformer.categories_[i].tolist()


        return pipeline, mlb_topics, mlb_impact, mlb_continents, infrequent_topics, feature_columns, unique_categories # Return mlb_continents
    except FileNotFoundError:
        print("ERROR: Model components not found. Please run the training script first.")
        raise
    except Exception as e:
        print(f"ERROR loading model components: {e}")
        raise

# Load components globally
pipeline, mlb_topics, mlb_impact, mlb_continents, infrequent_topics, feature_columns, unique_categories = load_model_components() # Unpack mlb_continents

# Prepare topic/impact/continent options for the UI (mlb_classes_ are already sorted)
all_mlb_topics = sorted(mlb_topics.classes_.tolist())
if "other" not in all_mlb_topics: # Add 'other' if it's not naturally a class (e.g., if no topics were infrequent during training)
    all_mlb_topics.append("other")
all_mlb_impacts = sorted(mlb_impact.classes_.tolist())

# Prepare continent options for the UI
all_mlb_continents = sorted(mlb_continents.classes_.tolist())


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

            ui.h4("Categorical Inputs"),
            ui.input_select("legal_basis", "Legal Basis", choices=unique_categories.get('legalBasis', [])),
            ui.input_select("funding_scheme", "Funding Scheme", choices=unique_categories.get('fundingScheme', [])),
            ui.input_select("scientific_domain", "Scientific Domain", choices=unique_categories.get('scientific_domain', [])),
            ui.input_select("sustainability", "Sustainability", choices=unique_categories.get('sustainability', [])),
            # Remove ui.input_select("continent", ...) here as it's now handled by MLB
            # ui.input_select("continent", "Continent", choices=unique_categories.get('continent', [])),
        ),
        # The main content goes directly after ui.sidebar() within ui.page_sidebar
        ui.h3("Topics, Impact, and Continents"), # Updated heading
        ui.layout_column_wrap(
            1/3, # Adjust column wrap to accommodate continents (now 3 columns)
            ui.input_checkbox_group("selected_main_topics", "Main Topics (select one or more)", choices=all_mlb_topics),
            ui.input_checkbox_group("selected_expected_impact", "Expected Impact (select one or more)", choices=all_mlb_impacts),
            ui.input_checkbox_group("selected_continents", "Continents Involved (select one or more)", choices=all_mlb_continents), # NEW: Continent input
        ),
        ui.br(),
        ui.input_action_button("predict_button", "Predict Contribution", class_="btn-primary"),
        ui.hr(),
        ui.h3("Prediction Result"),
        ui.output_text("prediction_output")
    )
)

def server(input, output, session):

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
        if not input.selected_continents(): # You might want to make continent selection mandatory
            ui.notification_show("Please select at least one Continent.", type="warning")
            return

        try:
            # Prepare a dictionary for the input row
            input_row_data = {
                'project_length_days': input.project_length_days(),
                'number_of_organizations': input.number_of_organizations(),
                'legalBasis': input.legal_basis(),
                'fundingScheme': input.funding_scheme(),
                'scientific_domain': input.scientific_domain(),
                'sustainability': input.sustainability(),
                # Removed 'continent' from here as it's now handled by MLB
            }

            # --- Preprocess Main Topics (remains the same) ---
            processed_main_topics_for_mlb = [
                topic.lower() if topic not in infrequent_topics else "other"
                for topic in input.selected_main_topics()
            ]
            main_topics_binary_array = mlb_topics.transform([processed_main_topics_for_mlb]).toarray()[0]
            for i, topic_label in enumerate(mlb_topics.classes_):
                col_name = f'topic_{topic_label.replace(' ', '_').replace('-', '_')}'
                input_row_data[col_name] = main_topics_binary_array[i]


            # --- Preprocess Expected Impact (remains the same) ---
            processed_impact_for_mlb = [
                impact.lower() for impact in input.selected_expected_impact()
            ]
            impact_binary_array = mlb_impact.transform([processed_impact_for_mlb]).toarray()[0]
            for i, impact_label in enumerate(mlb_impact.classes_):
                col_name = f'impact_{impact_label.replace(' ', '_').replace('-', '_')}'
                input_row_data[col_name] = impact_binary_array[i]

            # --- NEW: Preprocess Continents using mlb_continents ---
            processed_continents_for_mlb = [
                continent.lower() for continent in input.selected_continents()
            ]
            continents_binary_array = mlb_continents.transform([processed_continents_for_mlb]).toarray()[0]
            for i, continent_label in enumerate(mlb_continents.classes_):
                col_name = f'continent_{continent_label.replace(' ', '_').replace('-', '_')}'
                input_row_data[col_name] = continents_binary_array[i]

            # Create the final input DataFrame from the combined dictionary
            final_input_df = pd.DataFrame([input_row_data], columns=feature_columns).fillna(0)

            # Important: Reindex to ensure column order and presence, filling NaNs (for unselected OHEs) with 0
            final_input_df = final_input_df.reindex(columns=feature_columns, fill_value=0)

            # Make prediction
            prediction = pipeline.predict(final_input_df)[0]
            prediction_result.set(prediction)

        except Exception as e:
            ui.notification_show(f"An error occurred: {e}", type="danger")
            prediction_result.set(None)
            print(f"Prediction error: {e}")

    @output
    @render.text
    def prediction_output():
        if prediction_result.get() is not None:
            return f"### Predicted EC Max Contribution: €{prediction_result.get():,.2f}"
        else:
            return "Enter project details and click 'Predict' to see the result."

# Create the Shiny app instance
app = App(app_ui, server)
