from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2

from fastapi.responses import JSONResponse


import json
import numpy as np
import cv2
import io

app = FastAPI()
app.state.model = load_model('models/ADE_final_model.keras')

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

#endoint

@app.post('/predictions')
async def receive_image(img: UploadFile = File(...)):
    contents = await img.read()
    np_array = np.frombuffer(contents, np.uint8)  # Updated from np.fromstring, which is deprecated
    cv2_img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)


    # Resize the image to the expected input size (224, 224)
    cv2_img_resized = cv2.resize(cv2_img, (224, 224))

    # If the model expects BGR order, use cv2_img_resized as it is.
    # If the model expects RGB order, convert it to RGB
    cv2_img_resized_rgb = cv2.cvtColor(cv2_img_resized, cv2.COLOR_BGR2RGB)

    cv2_img_resized_rgb_normalized = cv2_img_resized_rgb / 255.0

    # Expand dimensions to match the input shape expected by the model: (1, 224, 224, 3)
    input_tensor = tf.convert_to_tensor(cv2_img_resized_rgb_normalized)
    # Normalize pixel values if the model expects them to be in the range [0, 1]
    input_tensor = tf.expand_dims(input_tensor,axis=0)

    # Predict the label
    predicted_label = app.state.model.predict(input_tensor)

    # Convert the prediction to a human-readable format if necessary
    predicted_label = predicted_label.tolist()  # Convert to list for JSON serialization

    return JSONResponse(content={'message': predicted_label})
