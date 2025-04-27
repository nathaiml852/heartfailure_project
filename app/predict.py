# app/predict.py

import pandas as pd
import xgboost as xgb
from app.model_loader import load_model

model = load_model()

def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                        high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                        sex, smoking, time):
    input_dict = {
        "age": age,
        "anaemia": anaemia,
        "creatinine_phosphokinase": creatinine_phosphokinase,
        "diabetes": diabetes,
        "ejection_fraction": ejection_fraction,
        "high_blood_pressure": high_blood_pressure,
        "platelets": platelets,
        "serum_creatinine": serum_creatinine,
        "serum_sodium": serum_sodium,
        "sex": sex,
        "smoking": smoking,
        "time": time
    }
    df = pd.DataFrame([input_dict])
    dmatrix = xgb.DMatrix(df)
    prediction = int(model.predict(dmatrix)[0] >= 0.5)
    return "YES - Death event likely." if prediction == 1 else "NO - Death event not likely."