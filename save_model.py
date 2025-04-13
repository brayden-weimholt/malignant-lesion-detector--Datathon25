import torch
import torchvision.models as models
import torch.nn as nn

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.7),
    nn.Linear(512, 1)
)


model_save_path = 'finetuned_model.pth'


torch.save(model.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")