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
from fastapi import Form


# Initialize FastAPI app
app = FastAPI()

# Mount the static directory for css, js, images etc.
app.mount("/static", StaticFiles(directory="static", name="static"))

# Configure the templates directory
templates = Jinja2Templates(directory="templates")

# Prediction output labels
level = ['Low', 'Medium', 'High'] 

# Load the pre-trained model
model = EngageModel(11, 3, 'artifacts\\testModel.pth')
model.load_model()  

#  root endpoint -> Home Page
@app.get("/", response_class=HTMLResponse)
def read_root(request=Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Game Engagement Model is running!"})

# predict endpoint -> Prediction Form
@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request:Request):
    return templates.TemplateResponse("predict.html", {"request": request, "title": "Prediction Page"})

# Result endpoint -> Process data and show the result
@app.post("/result", response_class=HTMLResponse)
async def get_prediction(
    request : Request,
    Age: int = Form(...),
    Gender : str = Form(...),
    Location: str = Form(...),
    GameGenre: str = Form(...),
    PlayTimeHours: float = Form(...),
    InGamePurchases: int = Form(...),
    GameDifficulty: str = Form(...),
    SessionsPerWeek: int = Form(...),
    AvgSessionDurationMinutes: int = Form(...),
    PlayerLevel: int = Form(...),
    AchievementsUnlocked: int = Form(...)     

):
    # Collect form data into a dictionary
    user_data = {
        "Age": Age,
        "Gender": Gender,
        "Location": Location,
        "GameGenre": GameGenre,
        "PlayTimeHours": PlayTimeHours,
        "InGamePurchases": InGamePurchases,
        "GameDifficulty": GameDifficulty,
        "SessionsPerWeek": SessionsPerWeek,
        "AvgSessionDurationMinutes": AvgSessionDurationMinutes,
        "PlayerLevel": PlayerLevel,
        "AchievementsUnlocked": AchievementsUnlocked
    }
   
    preprocessed_data = prepare_data_for_model(user_data, 'artifacts\\preprocessor.pkl')

    # Make a prediction
    prediction = model.predict(preprocessed_data)
    predicted_level = level[prediction.item()]

    # Return result page with a prediction
    return templates.TemplateResponse("result.hrml", {"request": request, "result": predicted_level})
 

