import uvicorn # ASGI
from fastapi import FastAPI, Request # FastAPI(for building the API) and Request for handling web request
from pydantic import BaseModel # Pydantic for request data validation
from fastapi.responses import HTMLResponse # HTMLResponse to serve HTML pages
from fastapi.staticfiles import StaticFiles # To serve static files like CSS, JS, images
from fastapi.templating import Jinja2Templates # Jinja2 for rendering HTML templates
import torch # PyTorch for the model
import numpy as np
import pandas as pd
from src.components.models import EngageModel
from src.model_deployment import prepare_data_for_model
import pickle


# Initialize FastAPI app
app = FastAPI()

# Mount the static directory for css, js, images etc.
app.mount("/static", StaticFiles(directory="static", name="static"))

# Configure the templates directory
templates = Jinja2Templates(directory="templates")

# Prediction output labels
level = ['Low', 'Medium', 'High'] 

"""
# Path to the preprocessor object
p_path = 'artifacts\\preprocessor.pkl'
 
"""
# Load the pre-trained model
model = EngageModel(11, 3, 'artifacts\\testModel.pth')
model.load_model()  

 
"""
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
    """
 

#  root endpoint/ Home route serving the HTML page
@app.get("/", response_class=HTMLResponse)
def read_root(request=Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Game Engagement Model is running!"})

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request:Request):
    return templates.TemplateResponse("predict.html", {"request": request, "title": "Prediction Page"})

   
"""
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
 
# To run the app directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_api:app", host="127.0.0.1", port=8000, reload=True)
"""
 

