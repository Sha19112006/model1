
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

app = FastAPI(title="LifeLink ML Engine", version="1.0")

#  trained model loading 
try:
    model = joblib.load("lifelink_model.pkl")
    print("LifeLink ML Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# rules
class Donor(BaseModel):
    donorId: str
    age: int
    weight_kg: float
    hemoglobin_g_dL: float
    total_donations: int
    days_since_last_donation: int

class RankingRequest(BaseModel):
    donors: List[Donor]

@app.get("/")
def home():
    return {"message": "LifeLink FastAPI ML Engine is running!"}

# 3. Ranking Endpoint
@app.post("/rank-donors")
def rank_donors(payload: RankingRequest):
    if not model:
        raise HTTPException(status_code=500, detail="ML Model is not loaded on the server.")
    
    if not payload.donors:
        return {"ranked_donors": []}
    
    
    # ['Age', 'Weight_kg', 'Hemoglobin_g_dL', 'Total_Donations', 'Days_Since_Last_Donation']
    features_list = [
        [
            d.age, 
            d.weight_kg, 
            d.hemoglobin_g_dL, 
            d.total_donations, 
            d.days_since_last_donation
        ] 
        for d in payload.donors
    ]
    

    probabilities = model.predict_proba(features_list)[:, 1]
    
    # Combine donors  probability
    ranked_donors = []
    for i, donor in enumerate(payload.donors):
        donor_dict = donor.dict()
        donor_dict["response_probability"] = round(float(probabilities[i]), 4)
        ranked_donors.append(donor_dict)
    
    
    ranked_donors = sorted(ranked_donors, key=lambda x: x["response_probability"], reverse=True)
    
    return {
        "status": "success",
        "total_evaluated": len(ranked_donors),
        "ranked_donors": ranked_donors
    }

