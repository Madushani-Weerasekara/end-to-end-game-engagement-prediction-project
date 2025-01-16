import uvicorn # ASGI
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
from src.components.models import EngageModel
from src.model_deployment import prepare_data_for_model
import pickle

