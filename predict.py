"""Predict yield for a new field using the trained model."""
import joblib
import pandas as pd

model = joblib.load("models/best_model.joblib")

sample = pd.DataFrame([{
    "Rainfall_mm": 1000,
    "Temperature_C": 27,
    "Humidity_pct": 70,
    "Soil_pH": 6.5,
    "Nitrogen": 90,
    "Phosphorus": 50,
    "Potassium": 55,
    "Fertilizer_kg_ha": 130,
    "Soil_Type": "Loamy",
    "Crop": "Rice",
}])

print(f"Predicted yield: {model.predict(sample)[0]:.2f} tonnes/hectare")
