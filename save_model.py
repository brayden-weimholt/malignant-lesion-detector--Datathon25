import torch
import torchvision.models as models
import torch.nn as nn
import pickle

model_pkl_path = "model.pkl"
with open(model_pkl_path, 'rb') as file:
    model = pickle.load(file)

model_save_path = 'finetuned_model.pth'


torch.save(model.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")