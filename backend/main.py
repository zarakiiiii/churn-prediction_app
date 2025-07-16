from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

#defining the schema with your real model features
class InputData(BaseModel):
    purchase_count: float
    customer_state: float
    last_review_score: float
    main_product_category: float
    last_product_category: float

#loading the trained model

import os

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "tuned_churn_model_pipeline2.pkl")
model = joblib.load(model_path)

@app.get("/")
def read_root():
    return {"message": "FastAPI backend is running!"}

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[
        data.purchase_count,
        data.customer_state,
        data.last_review_score,
        data.main_product_category,
        data.last_product_category
    ]])

    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}