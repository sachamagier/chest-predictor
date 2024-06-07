from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response, FileResponse
from tensorflow.keras.models import load_model
from fastapi.responses import JSONResponse


import json
import numpy as np
import cv2
import io

app = FastAPI()
app.state.model = load_model('/Users/sachamagier/code/sachamagier/chest-predictor/models/simple_model_best.h5')

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

    # Convert the image to grayscale if the model expects grayscale images
    cv2_img_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    # Resize the image to the expected input size (256, 256)
    cv2_img_gray_resized = cv2.resize(cv2_img_gray, (256, 256))

    # Expand dimensions to match the input shape expected by the model: (1, 256, 256, 1)
    input_image = np.expand_dims(cv2_img_gray_resized, axis=-1)
    input_image = np.expand_dims(input_image, axis=0)

    # Predict the label
    predicted_label = app.state.model.predict(input_image)

    treshold = 0.4
    predicted_label_binary = (predicted_label >= treshold).astype("int")

    #display the label name
    label_names = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding']

    #return the label if the prediction is above the treshold

    if predicted_label_binary[0][0] == 1:
        predicted_label = label_names[0]
    elif predicted_label_binary[0][1] == 1:
        predicted_label = label_names[1]
    elif predicted_label_binary[0][2] == 1:
        predicted_label = label_names[2]
    elif predicted_label_binary[0][3] == 1:
        predicted_label = label_names[3]
    elif predicted_label_binary[0][4] == 1:
        predicted_label = label_names[4]
    elif predicted_label_binary[0][5] == 1:
        predicted_label = label_names[5]
    elif predicted_label_binary[0][6] == 1:
        predicted_label = label_names[6]
    elif predicted_label_binary[0][7] == 1:
        predicted_label = label_names[7]
    elif predicted_label_binary[0][8] == 1:
        predicted_label = label_names[8]
    elif predicted_label_binary[0][9] == 1:
        predicted_label = label_names[9]
    elif predicted_label_binary[0][10] == 1:
        predicted_label = label_names[10]
    elif predicted_label_binary[0][11] == 1:
        predicted_label = label_names[11]
    elif predicted_label_binary[0][12] == 1:
        predicted_label = label_names[12]
    elif predicted_label_binary[0][13] == 1:
        predicted_label = label_names[13]
    else:
        predicted_label = label_names[14]


    # Return the prediction as JSON
    return JSONResponse(content={'message': predicted_label})





# # Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
# @app.post("/image_boxes")

# async def receive_image(img: UploadFile=File(...)):
#     # try:
#         #receiving the image + decoding it
#         contents = await img.read()

#         #creating a numpy array from a string of bytes
#         np_array = np.fromstring(contents,np.uint8)
#         cv2_img = cv2.imdecode(np_array,cv2.IMREAD_COLOR)

#         #adding our function that detects the location where the deseases is

#         #encoding and responding with an image

#         im = cv2.imencode('.png',cv2_img)[1]

#         return Response(content=im.tobytes(), media_type='image/png')

    # # except Exception as e:
    #     print(e)
    #     return {"status": "error", "message": "Unable to process the image"}
