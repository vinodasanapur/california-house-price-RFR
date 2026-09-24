from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


MODEL_PATH = Path(__file__).with_name("rfmodel.pkl")
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
OCEAN_LEVELS = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
NUMERIC_FEATURES = [
	"longitude",
	"latitude",
	"housing_median_age",
	"total_rooms",
	"total_bedrooms",
	"population",
	"households",
	"median_income",
]
MODEL_FEATURES = NUMERIC_FEATURES + [f"ocean_proximity_{level}" for level in OCEAN_LEVELS] + [
	"rooms_per_household",
	"bedrooms_per_room",
	"population_per_household",
]

st.set_page_config(
	page_title="California Home Value Predictor",
	page_icon="🏡",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
	html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
	h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
	.hero { padding: 2rem 0 1rem; border-bottom: 1px solid rgba(62, 83, 92, .18); margin-bottom: 1.5rem; }
	.eyebrow { color: #0b7a75; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; font-size: .76rem; }
	.hero h1 { font-size: clamp(2rem, 4vw, 3.8rem); line-height: 1.02; margin: .35rem 0 .8rem; color: #18333b; }
	.hero p { color: #567078; max-width: 700px; font-size: 1.05rem; }
	[data-testid="stMetricValue"] { color: #0b7a75; }
	.note { background: #edf7f4; border-left: 4px solid #0b7a75; padding: .8rem 1rem; color: #29494f; border-radius: 0 6px 6px 0; }
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"Saved model not found at {MODEL_PATH}")
	with MODEL_PATH.open("rb") as model_file:
		return pickle.load(model_file)


@st.cache_data
def load_scaler():
	"""Recreate the scaler used in the notebook before the forest was saved."""
	source = pd.read_csv(DATA_URL)
	source["total_bedrooms"] = source["total_bedrooms"].fillna(source["total_bedrooms"].median())
	encoded = pd.get_dummies(source, columns=["ocean_proximity"])
	encoded["rooms_per_household"] = encoded["total_rooms"] / encoded["households"]
	encoded["bedrooms_per_room"] = encoded["total_bedrooms"] / encoded["total_rooms"]
	encoded["population_per_household"] = encoded["population"] / encoded["households"]
	encoded = encoded.reindex(columns=MODEL_FEATURES, fill_value=0)
	return StandardScaler().fit(encoded)


def build_features(values):
	features = {key: float(values[key]) for key in NUMERIC_FEATURES}
	selected_ocean = values["ocean_proximity"]
	for level in OCEAN_LEVELS:
		features[f"ocean_proximity_{level}"] = int(level == selected_ocean)

	households = max(features["households"], 1e-9)
	total_rooms = max(features["total_rooms"], 1e-9)
	features["rooms_per_household"] = features["total_rooms"] / households
	features["bedrooms_per_room"] = features["total_bedrooms"] / total_rooms
	features["population_per_household"] = features["population"] / households
	return pd.DataFrame([[features[column] for column in MODEL_FEATURES]], columns=MODEL_FEATURES)


def money(value):
	return f"${value:,.0f}"


st.markdown(
	"""
	<div class="hero">
		<div class="eyebrow">Random forest regression · California</div>
		<h1>Estimate a district's home value.</h1>
		<p>Adjust the local housing profile and get an instant estimate from the saved model.</p>
	</div>
	""",
	unsafe_allow_html=True,
)

try:
	model = load_model()
except Exception as error:
	st.error(f"Could not load the saved model: {error}")
	st.stop()

with st.sidebar:
	st.header("Scenario")
	preset = st.selectbox(
		"Start with a profile",
		["Custom", "Urban / high income", "Inland family district", "Coastal community"],
	)
	presets = {
		"Urban / high income": [(-118.4, 34.1, 35, 2200, 420, 650, 400, 8.5, "<1H OCEAN")],
		"Inland family district": [(-117.2, 34.1, 25, 2600, 500, 900, 460, 5.0, "INLAND")],
		"Coastal community": [(-122.4, 37.8, 30, 1800, 350, 520, 300, 7.0, "NEAR BAY")],
	}
	if preset != "Custom":
		st.caption("Preset values are editable in the form.")
	st.divider()
	st.caption(f"Model: {type(model).__name__}")
	st.caption(f"Features expected: {getattr(model, 'n_features_in_', 'unknown')}")

defaults = presets.get(preset, [(-119.57, 35.63, 29, 2635, 537, 1425, 499, 3.87, "INLAND")])[0]
with st.form("prediction_form"):
	st.subheader("District profile")
	location_col, income_col = st.columns([2, 1])
	with location_col:
		longitude, latitude = st.columns(2)
		longitude = longitude.number_input("Longitude", -125.0, -114.0, float(defaults[0]), 0.01)
		latitude = latitude.number_input("Latitude", 32.0, 42.0, float(defaults[1]), 0.01)
	with income_col:
		median_income = st.number_input("Median income (10k USD)", 0.0, 20.0, float(defaults[7]), 0.05)

	age_col, rooms_col, bedrooms_col = st.columns(3)
	housing_median_age = age_col.number_input("Housing median age", 1.0, 100.0, float(defaults[2]), 1.0)
	total_rooms = rooms_col.number_input("Total rooms", 1.0, 100000.0, float(defaults[3]), 10.0)
	total_bedrooms = bedrooms_col.number_input("Total bedrooms", 1.0, 50000.0, float(defaults[4]), 10.0)
	population_col, households_col, ocean_col = st.columns(3)
	population = population_col.number_input("Population", 1.0, 100000.0, float(defaults[5]), 10.0)
	households = households_col.number_input("Households", 1.0, 50000.0, float(defaults[6]), 10.0)
	ocean_proximity = ocean_col.selectbox("Ocean proximity", OCEAN_LEVELS, index=OCEAN_LEVELS.index(defaults[8]))
	submitted = st.form_submit_button("Predict median house value", type="primary", use_container_width=True)

if submitted:
	values = locals()
	raw_features = build_features(values)
	try:
		scaled_features = load_scaler().transform(raw_features)
		prediction = float(model.predict(scaled_features)[0])
	except Exception as error:
		st.error(f"Prediction failed: {error}")
		st.stop()

	st.divider()
	result_col, detail_col = st.columns([1, 1.4])
	with result_col:
		st.metric("Estimated median value", money(prediction))
		st.caption("The training target is expressed in USD.")
	with detail_col:
		st.subheader("Derived inputs")
		st.dataframe(
			pd.DataFrame(
				{
					"Feature": ["Rooms / household", "Bedrooms / room", "Population / household"],
					"Value": [
						raw_features.at[0, "rooms_per_household"],
						raw_features.at[0, "bedrooms_per_room"],
						raw_features.at[0, "population_per_household"],
					],
				}
			).style.format({"Value": "{:.3f}"}),
			hide_index=True,
			use_container_width=True,
		)

	if hasattr(model, "feature_importances_"):
		st.subheader("What influences the model most")
		importance = pd.Series(model.feature_importances_, index=MODEL_FEATURES).sort_values(ascending=False).head(8)
		st.bar_chart(importance, horizontal=True, height=300)

with st.expander("About this model"):
	st.markdown(
		"""The app uses the Random Forest Regressor saved from the project notebook. """
		"""It applies the notebook's missing-value handling, one-hot encoding, engineered ratios, """
		"""and StandardScaler transformation before prediction."""
	)
