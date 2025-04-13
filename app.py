from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import pickle

app = Flask(__name__)

# Load model using pickle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
model.eval()

def user_preprocess(img):
    mm_to_inch = 0.0393701
    size_in_inches = 15 * mm_to_inch
    pixel_size = int(size_in_inches * 300)  # 300 DPI

    image = Image.open(img)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((pixel_size, pixel_size), Image.LANCZOS)

    image_arr = np.array(image) / 255.0

    image_tensor = torch.tensor(image_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    return image_tensor

def predict_out(prediction):
    sigmoid_pred = torch.sigmoid(prediction)

    predicted_class = int((sigmoid_pred > 0.5).detach().numpy()[0][0])
    print("Raw logit:", prediction)
    print("After sigmoid:", sigmoid_pred)
    print("Predicted class (0 for malignant, 1 for benign):", predicted_class)
    

    if sigmoid_pred < 0.01: 
        return {"class": "malignant"}
    else:
        return {"class": "benign"}

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