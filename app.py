from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

app = Flask(__name__)

# Load model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.7),
    nn.Linear(512, 1)
)
model.load_state_dict(torch.load('finetuned_model.pth', map_location=torch.device('cpu')))
model.eval()

def user_preprocess(img):
    mm_to_inch = 0.0393701
    size_in_inches = 15 * mm_to_inch
    pixel_size = int(size_in_inches * 300)  # 300 DPI

    image = Image.open(img)
    image = image.resize((pixel_size, pixel_size), Image.LANCZOS)
    image_arr = np.array(image) / 255.0
    image_arr = np.expand_dims(image_arr, axis=0)
    image_tensor = torch.tensor(image_arr, dtype=torch.float32).permute(0, 3, 1, 2)
    return image_tensor

def predict_out(prediction):
    sigmoid = nn.Sigmoid()
    probabilities = sigmoid(prediction).detach().numpy()
    benign_prob, malignant_prob = float(probabilities[0][0]), float(1 - probabilities[0][0])
    if benign_prob > malignant_prob:
        return {"class": "benign"}
    else:
        return {"class": "malignant"}

@app.route('/', methods=['GET', 'POST'])
def user_upload():
    result = None
    if request.method == "POST":
        img_file = request.files['image']
        img = user_preprocess(img_file)
        if img is not None:
            with torch.no_grad():
                prediction = model(img)
            result = predict_out(prediction)
    return render_template('upload.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)