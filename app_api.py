import uvicorn # ASGI
from fastapi import FastAPI # FastAPI for building the API
from pydantic import BaseModel # Pydantic for request data validation
import torch # PyTorch for the model
import numpy as np
import pandas as pd
from src.components.models import EngageModel
from src.model_deployment import prepare_data_for_model
import pickle

# Initialize FastAPI app
app = FastAPI()

# Prediction output labels
level = ['Low', 'Medium', 'High']

# Path to the preprocessor object
p_path = 'artifacts\\preprocessor.pkl'

# Load the pre-trained model
model = EngageModel(11, 3, 'artifacts\\testModel.pth')
model.load_model()  

# Define the input data schema for the prediction endpoint
class ScoringItem(BaseModel):
    Age: int
    Gender: str
    Location: str
    GameGenre: str
    PlayTimeHours: float
    InGamePurchases: int
    GameDifficulty: str
    SessionsPerWeek: int
    AvgSessionDurationMinutes: int
    PlayerLevel: int
    AchievementsUnlocked: int

#  root endpoint
@app.get("/")
def read_root():
    return {"Message": "Game Engagement Model is running!"}

@app.get("/predict")
async def scoring_endpoint(item:ScoringItem):
    # Convert input data to dictionary
    data_point = item.dict()
    print(data_point)

    # Preprocess the data for the model
    x_tensor = prepare_data_for_model(data_point, p_path)

    # Make a prediction
    pred = model.predict(x_tensor)
    print(pred.item())

    # Return the predicted level as a response
    return {"Predicted Engagement Level:", level[pred.item()]}

"""
# To run the app directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_api:app", host="127.0.0.1", port=8000, reload=True)
"""
 

