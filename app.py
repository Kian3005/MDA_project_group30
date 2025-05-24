from shiny import App, ui, render, reactive
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer # Assuming MLB is still used for topics/impact/continents

# --- 1. Load Model Components ---
def load_model_components():
    """Loads the pre-trained model pipeline and transformers."""
    try:
        # --- CHANGE 1: Load the refined pipeline ---
        pipeline = joblib.load('prediction_pipeline.joblib')

        # Load MLB components and other relevant objects that might have been part of feature engineering
        # These are likely still the same as before, as they process the raw input data
        mlb_topics = joblib.load('mlb_topics.joblib')
        mlb_impact = joblib.load('mlb_impact.joblib')
        mlb_continents = joblib.load('mlb_continents.joblib')
        infrequent_topics = joblib.load('infrequent_topics.joblib')

        # --- CHANGE 2: Load the feature_columns list from the refined model's training script ---
        # This list now contains only the features the refined model expects
        feature_columns = joblib.load('feature_columns.joblib') # This should be saved from the final X.columns in the training script

        # Extract unique categories from the OneHotEncoder in the loaded pipeline
        # This logic remains valid as the preprocessor structure is the same
        ohe_transformer = pipeline.named_steps['preprocessor'].named_transformers_['cat']

        unique_categories = {}
        if ohe_transformer is not None and hasattr(ohe_transformer, 'categories_'):
            # The feature names used by OHE within the refined pipeline
            ohe_features_trained_on = pipeline.named_steps['preprocessor'].named_transformers_['cat'].feature_names_in_
            for i, feature_name in enumerate(ohe_features_trained_on):
                unique_categories[feature_name] = ohe_transformer.categories_[i].tolist()


        return pipeline, mlb_topics, mlb_impact, mlb_continents, infrequent_topics, feature_columns, unique_categories
    except FileNotFoundError as e:
        print(f"ERROR: Model component not found: {e}. Please ensure the training script has been run and files are in the correct directory.")
        raise
    except Exception as e:
        print(f"ERROR loading model components: {e}")
        raise

# Load components globally
pipeline, mlb_topics, mlb_impact, mlb_continents, infrequent_topics, feature_columns, unique_categories = load_model_components()

# Prepare topic/impact/continent options for the UI (mlb_classes_ are already sorted)

# Get the raw list of topics from MLB classes
all_mlb_topics_raw = mlb_topics.classes_.tolist()

# Remove 'other' if it exists in the raw list to sort the rest independently
if "other" in all_mlb_topics_raw:
    topics_without_other = [topic for topic in all_mlb_topics_raw if topic != "other"]
else:
    topics_without_other = all_mlb_topics_raw # 'other' isn't there, so just use the raw list

# Sort the topics (excluding 'other')
sorted_topics_without_other = sorted(topics_without_other)

# Add 'other' to the very end of the sorted list
all_mlb_topics = sorted_topics_without_other + ["other"]

# Ensure 'other' is added if it wasn't originally in mlb_topics.classes_
if "other" not in all_mlb_topics:
    all_mlb_topics.append("other")


# Get the raw list of impacts from MLB classes
all_mlb_impacts_raw = mlb_impact.classes_.tolist()

# Remove 'other' if it exists in the raw list to sort the rest independently
if "other" in all_mlb_impacts_raw:
    impacts_without_other = [impact for impact in all_mlb_impacts_raw if impact != "other"]
else:
    impacts_without_other = all_mlb_impacts_raw # 'other' isn't there, so just use the raw list

# Sort the impacts (excluding 'other')
sorted_impacts_without_other = sorted(impacts_without_other)

# Add 'other' to the very end of the sorted list
all_mlb_impacts = sorted_impacts_without_other + ["other"]

# This ensures that "other" is always available as a choice in the UI
if "other" not in all_mlb_impacts:
    all_mlb_impacts.append("other")



# Get the raw list of continents from MLB classes
all_mlb_continents_raw = mlb_continents.classes_.tolist()

# Remove 'unknown' if it exists in the raw list to sort the rest independently
if "unknown" in all_mlb_continents_raw:
    continents_without_unknown = [continent for continent in all_mlb_continents_raw if continent != "unknown"]
else:
    continents_without_unknown = all_mlb_continents_raw # 'unknown' isn't there, so just use the raw list

# Sort the continents (excluding 'unknown')
sorted_continents_without_unknown = sorted(continents_without_unknown)

# Add 'unknown' to the very end of the sorted list
all_mlb_continents = sorted_continents_without_unknown + ["unknown"]

# This ensures that "unknown" is always available as a choice in the UI
if "unknown" not in all_mlb_continents:
    all_mlb_continents.append("unknown")



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

            ui.h4("Categorical Inputs"),
            ui.input_select("legal_basis", "Legal Basis", choices=unique_categories.get('legalBasis', [])),
            ui.input_select("funding_scheme", "Funding Scheme", choices=unique_categories.get('fundingScheme', [])),
            ui.input_select("scientific_domain", "Scientific Domain", choices=unique_categories.get('scientific_domain', [])),
            ui.input_select("sustainability", "Sustainability", choices=unique_categories.get('sustainability', [])),
        ),
        ui.h3("Topics, Impact, and Continents"),
        ui.layout_column_wrap(
            1/3, # Adjust column wrap to accommodate continents (now 3 columns)
            ui.input_checkbox_group("selected_main_topics", "Main Topics (select one or more)", choices=all_mlb_topics),
            ui.input_checkbox_group("selected_expected_impact", "Expected Impact (select one or more)", choices=all_mlb_impacts),
            ui.input_checkbox_group("selected_continents", "Continents Involved (select one or more)", choices=all_mlb_continents),
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

    # Reactive value to store the *internally managed* selection for topics
    # This prevents direct changes to input.selected_main_topics from immediately re-triggering
    # the logic that updates it.
    _selected_main_topics = reactive.Value([])
    _selected_expected_impact = reactive.Value([])
    _selected_continents = reactive.Value([])


    # Initialize the internal reactive values with the initial UI state if needed,
    # or let the first user interaction set them. For simplicity, we'll assume
    # they start empty and are updated by user input.

    # --- Reactive Logic for "Main Topics" ---
    @reactive.Effect
    @reactive.event(input.selected_main_topics) # Only run when the input changes
    def _():
        current_input_topics = input.selected_main_topics()
        # Get the previously set internal value, not necessarily the *last* input value
        current_internal_topics = _selected_main_topics.get()

        new_selection = list(current_input_topics) # Work with a mutable copy

        # Logic for "other" handling in Main Topics
        if "other" in new_selection:
            if len(new_selection) > 1:
                # If "other" is selected along with other topics, keep only "other"
                new_selection = ["other"]
                ui.update_checkbox_group(
                    "selected_main_topics",
                    session=session,
                    selected=new_selection # This will trigger the effect again, but the condition will then be met
                )
        elif "other" in current_internal_topics and "other" not in new_selection and len(current_internal_topics) == 1:
            # If "other" was the *only* selected item and it's now deselected, clear all
            new_selection = []
            ui.update_checkbox_group(
                "selected_main_topics",
                session=session,
                selected=new_selection
            )

        # Always update the internal reactive value to reflect the desired UI state
        _selected_main_topics.set(new_selection)


    # --- Reactive Logic for "Expected Impact" ---
    @reactive.Effect
    @reactive.event(input.selected_expected_impact)
    def _():
        current_input_impact = input.selected_expected_impact()
        current_internal_impact = _selected_expected_impact.get()

        new_selection = list(current_input_impact)

        if "other" in new_selection:
            if len(new_selection) > 1:
                new_selection = ["other"]
                ui.update_checkbox_group(
                    "selected_expected_impact",
                    session=session,
                    selected=new_selection
                )
        elif "other" in current_internal_impact and "other" not in new_selection and len(current_internal_impact) == 1:
            new_selection = []
            ui.update_checkbox_group(
                "selected_expected_impact",
                session=session,
                selected=new_selection
            )
        _selected_expected_impact.set(new_selection)


    # --- Reactive Logic for "Continents Involved" ---
    @reactive.Effect
    @reactive.event(input.selected_continents)
    def _():
        current_input_continents = input.selected_continents()
        current_internal_continents = _selected_continents.get()

        new_selection = list(current_input_continents)

        if "unknown" in new_selection:
            if len(new_selection) > 1:
                new_selection = ["unknown"]
                ui.update_checkbox_group(
                    "selected_continents",
                    session=session,
                    selected=new_selection
                )
        elif "unknown" in current_internal_continents and "unknown" not in new_selection and len(current_internal_continents) == 1:
            new_selection = []
            ui.update_checkbox_group(
                "selected_continents",
                session=session,
                selected=new_selection
            )
        _selected_continents.set(new_selection)


    @reactive.Effect
    @reactive.event(input.predict_button)
    def _():
        # --- Input Validation ---
        # Now use the *internal* reactive values for validation, as they reflect the controlled state
        if not _selected_main_topics.get():
            ui.notification_show("Please select at least one Main Topic.", type="warning")
            return
        if not _selected_expected_impact.get():
            ui.notification_show("Please select at least one Expected Impact.", type="warning")
            return
        if not _selected_continents.get():
            ui.notification_show("Please select at least one Continent Involved.", type="warning")
            return

        try:
            input_row_data = {
                'project_length_days': input.project_length_days(),
                'number_of_organizations': input.number_of_organizations(),
                'legalBasis': input.legal_basis(),
                'fundingScheme': input.funding_scheme(),
                'scientific_domain': input.scientific_domain(),
                'sustainability': input.sustainability(),
            }

            # --- Preprocess Main Topics (use the internal reactive value) ---
            if "other" in _selected_main_topics.get():
                processed_main_topics_for_mlb = ["other"]
            else:
                processed_main_topics_for_mlb = [
                    topic.lower() if topic not in infrequent_topics else "other"
                    for topic in _selected_main_topics.get()
                ]
            main_topics_binary_array = mlb_topics.transform([processed_main_topics_for_mlb]).toarray()[0]
            for i, topic_label in enumerate(mlb_topics.classes_):
                col_name = f'topic_{topic_label.replace(' ', '_').replace('-', '_')}'
                input_row_data[col_name] = main_topics_binary_array[i]


            # --- Conditional Preprocessing for Expected Impact (use the internal reactive value) ---
            if "other" in _selected_expected_impact.get():
                processed_impact_for_mlb = ["other"]
            else:
                processed_impact_for_mlb = [
                    impact.lower() for impact in _selected_expected_impact.get()
                ]
            impact_binary_array = mlb_impact.transform([processed_impact_for_mlb]).toarray()[0]
            for i, impact_label in enumerate(mlb_impact.classes_):
                col_name = f'impact_{impact_label.replace(' ', '_').replace('-', '_')}'
                input_row_data[col_name] = impact_binary_array[i]

            # --- Preprocess Continents (use the internal reactive value) ---
            if "unknown" in _selected_continents.get():
                processed_continents_for_mlb = ["unknown"]
            else:
                processed_continents_for_mlb = [
                    continent.lower() for continent in _selected_continents.get()
                ]
            continents_binary_array = mlb_continents.transform([processed_continents_for_mlb]).toarray()[0]
            for i, continent_label in enumerate(mlb_continents.classes_):
                col_name = f'continent_{continent_label.replace(' ', '_').replace('-', '_')}'
                input_row_data[col_name] = continents_binary_array[i]

            # --- IMPORTANT: Create the final input DataFrame ---
            processed_df = pd.DataFrame([input_row_data])
            final_input_df = processed_df.reindex(columns=feature_columns, fill_value=0)

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
            return f"Predicted EC Max Contribution: €{prediction_result.get():,.2f}"
        else:
            return "Enter project details and click 'Predict' to see the result."

# Create the Shiny app instance
app = App(app_ui, server)