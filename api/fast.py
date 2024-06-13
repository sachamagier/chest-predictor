from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

import cv2

from fastapi.responses import JSONResponse


import json
import numpy as np
import cv2

app = FastAPI()

#open the model which is in the models folder .py file
app.state.model_binary = load_model('models/ADE_final_binary_model.keras')
app.state.model_all = load_model('models/best_only_disease_model.keras')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
async def index():
    return {"status": "ok"}

def preprocess_image(image):
    # Resize the image to 224x224
    image = tf.image.resize(image, [224, 224])
    # Ensure the image has 3 channels
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    # Normalize the image between 1 and -1
    image = (image / 127.5) - 1
    return image

@app.post('/predictions')
async def receive_image(img: UploadFile = File(...)):
    contents = await img.read()
    np_array = np.frombuffer(contents, np.uint8)
    cv2_img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Resize the image to the expected input size (224, 224)
    cv2_img_resized = cv2.resize(cv2_img, (224, 224))

    # If the model expects BGR order, use cv2_img_resized as it is.
    # If the model expects RGB order, convert it to RGB
    cv2_img_resized_rgb = cv2.cvtColor(cv2_img_resized, cv2.COLOR_BGR2RGB)

    # Normalize pixel values if the model expects them to be in the range [0, 1]
    cv2_img_resized_rgb_normalized = cv2_img_resized_rgb / 255.0

    input_tensor = tf.convert_to_tensor(cv2_img_resized_rgb_normalized, dtype='float32')

    # Expand dimensions to match the input shape expected by the model: (1, 224, 224, 3)
    input_tensor = tf.expand_dims(input_tensor,axis=0)




    #Predict the binary model
    predicted_labels_binary = app.state.model_binary.predict(input_tensor)
    predicted_labels_binary = predicted_labels_binary[0]


    if predicted_labels_binary[0] < 0.5:
        content = 'No disease detected, you are healthy!'
        return JSONResponse(content=content)

    # If the binary model predicts that the disease is found
    if predicted_labels_binary[0] >= 0.5:
        # Predict with the second model and return the disease name with the highest probability
        predicted_labels = app.state.model_all.predict(input_tensor)
        predicted_labels = predicted_labels[0]

        # Get the index of the highest probability
        predicted_label = np.argmax(predicted_labels)

        # Logging the predicted labels and the selected label
        print(f"Predicted labels: {predicted_labels}")
        print(f"Predicted label index: {predicted_label}")

        # mapping of indices to disease names
        disease_names = {
             0: 'Atelectasis', 1: 'Consolidation', 2: 'Infiltration', 3: 'Pneumothorax',
             4: 'Edema', 5: 'Emphysema', 6: 'Fibrosis', 7: 'Effusion', 8: 'Pneumonia',
             9: 'Pleural_Thickening', 10: 'Cardiomegaly', 11: 'Nodule', 12: 'Mass', 13: 'Hernia'
         }

        if predicted_label in disease_names:
             disease_name = disease_names[predicted_label]
             # response content
             content = f'We found a disease: {disease_name}. You should consult a doctor!'
        else:
             # This should not happen if the models are producing correct output
             content = 'Error: Disease label not found.'

        return JSONResponse(content=content)
