import pandas as pd
import pickle
import torch

def load_object(path):
    with open(path, 'rb') as file:
        loaded_label_encoder = pickle.load(file)
        print('Preprocessor Loaded')
    return loaded_label_encoder

def prepare_data_for_model(data):
    # Converting incoming data to a DataFrame
    data_df = pd.DataFrame([data])

    label_mapping = {'High': 2  , 'Medium': 1, 'Low': 0}

    GameGenre = {
            'Strategy': 0,
            'Sports': 1,
            'Action': 2,
            'RPG': 3,
            'Simulation': 4
        }

    Location = {
            'Other': 0,
            'USA': 1,
            'Europe': 2,
            'Asia': 3
        }

    GameDifficulty = {
            'Medium': 1, 'Easy': 0, 'Hard':2
        }

    Gender = {'Male': 0, 'Female': 1}


    data_df['GameGenre'] = data_df['GameGenre'].map(GameGenre)

    data_df['Location'] = data_df['Location'].map(Location)

    data_df['GameDifficulty'] = data_df['GameDifficulty'].map(GameDifficulty)

    data_df['Gender'] = data_df['Gender'].map(Gender)    

    # Convert to numpy array and then to tensor

    x_numpy = data_df.to_numpy()
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)

    return x_tensor
         