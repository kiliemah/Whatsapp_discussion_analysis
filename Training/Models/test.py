import pickle as pkl
from keras.models import load_model

model = load_model("Training/Models/model.h5")

# File where the variable will be saved
model_file = 'Training/Models/model.pkl'

# Pickle the variable and save it to the file
with open(model_file, 'wb') as file:
    pkl.dump(model, file)