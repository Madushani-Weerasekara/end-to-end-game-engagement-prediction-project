# Utils will have the all the common things that we are going to import or use
# Which will be used in entire application
# Read a dataset from database I can create a mongo db client here
# If I want to save the model into the cloud I can write the code here

import os
import sys

import pandas as pd
import numpy as np

from src. exception import CustomException
from src.logger import logging
import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
