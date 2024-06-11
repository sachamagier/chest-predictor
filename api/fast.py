from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2

from fastapi.responses import JSONResponse


import json
import numpy as np
import cv2
import tensorflow as tf

app = FastAPI()
<<<<<<< HEAD

#open the model which is in the models folder .py file
app.state.model = load_model('/Users/sachamagier/code/sachamagier/chest-predictor/models/cnn_keras (2).py')
=======
app.state.model = load_model('models/best_only_disease_model.keras')
>>>>>>> master

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

<<<<<<< HEAD
    # Expand dimensions to match the input shape expected by the model: (1, 224, 224, 3)
    input_image = np.expand_dims(cv2_img_resized, axis=0)

    # Predict the label
    predictions = app.state.model.predict(input_image)

    # Convert the predictions to a list
    prediction_list = predictions.tolist()

    # Create a response with the prediction percentages
    response = {
        "predictions": prediction_list[0]  # Assuming the model returns a single prediction
    }

    return JSONResponse(content=response)





# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from starlette.responses import JSONResponse
# from tensorflow.keras.models import load_model
# import numpy as np
# import cv2
# import tensorflow as tf

# app = FastAPI()
# app.state.model = load_model('/Users/sachamagier/code/sachamagier/chest-predictor/models/ADE_final_model.keras')

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# # Endpoint for https://your-domain.com/
# @app.get("/")
# async def index():
#     return {"status": "ok"}

# def preprocess_image(image):
#     # Resize the image to 224x224
#     image = tf.image.resize(image, [224, 224])
#     # Ensure the image has 3 channels
#     if image.shape[-1] == 1:
#         image = tf.image.grayscale_to_rgb(image)
#     return image

# @app.post('/predictions')
# async def receive_image(img: UploadFile = File(...)):
#     contents = await img.read()
#     np_array = np.frombuffer(contents, np.uint8)
#     cv2_img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

#     # Resize the image to the expected input size (224, 224)
#     cv2_img_resized = cv2.resize(cv2_img, (224, 224))

#     # Expand dimensions to match the input shape expected by the model: (1, 224, 224, 3)
#     input_image = np.expand_dims(cv2_img_resized, axis=0)

#     # Predict the label
#     predicted_label = app.state.model.predict(input_image)


#     # #return the label if the prediction is above the treshold






#     #threshold = 0.4
#     #predicted_label_binary = (predicted_label >= threshold).astype("int")

#     # # Display the label name
#     #label_names = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding']

#     # # Determine the predicted label based on the binary output
#     # for i, label in enumerate(predicted_label_binary[0]):
#     #     if label == 1:
#     #         predicted_label = label_names[i]
#     #         break
#     # else:
#     #     predicted_label = label_names[-1]

#     # #return the label if the prediction is above the treshold

#     # if predicted_label_binary[0][0] == 1:
#     #     predicted_label = label_names[0]
#     # elif predicted_label_binary[0][1] == 1:
#     #     predicted_label = label_names[1]
#     # elif predicted_label_binary[0][2] == 1:
#     #     predicted_label = label_names[2]
#     # elif predicted_label_binary[0][3] == 1:
#     #     predicted_label = label_names[3]
#     # elif predicted_label_binary[0][4] == 1:
#     #     predicted_label = label_names[4]
#     # elif predicted_label_binary[0][5] == 1:
#     #     predicted_label = label_names[5]
#     # elif predicted_label_binary[0][6] == 1:
#     #     predicted_label = label_names[6]
#     # elif predicted_label_binary[0][7] == 1:
#     #     predicted_label = label_names[7]
#     # elif predicted_label_binary[0][8] == 1:
#     #     predicted_label = label_names[8]
#     # elif predicted_label_binary[0][9] == 1:
#     #     predicted_label = label_names[9]
#     # elif predicted_label_binary[0][10] == 1:
#     #     predicted_label = label_names[10]
#     # elif predicted_label_binary[0][11] == 1:
#     #     predicted_label = label_names[11]
#     # elif predicted_label_binary[0][12] == 1:
#     #     predicted_label = label_names[12]
#     # elif predicted_label_binary[0][13] == 1:
#     #     predicted_label = label_names[13]
#     # else:
#     #     predicted_label = label_names[14]


#     # Return the prediction as JSON
#     return JSONResponse(content={'message': predicted_label})





# # # Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
# # @app.post("/image_boxes")

# # async def receive_image(img: UploadFile=File(...)):
# #     # try:
# #         #receiving the image + decoding it
# #         contents = await img.read()

# #         #creating a numpy array from a string of bytes
# #         np_array = np.fromstring(contents,np.uint8)
# #         cv2_img = cv2.imdecode(np_array,cv2.IMREAD_COLOR)

# #         #adding our function that detects the location where the deseases is

# #         #encoding and responding with an image

# #         im = cv2.imencode('.png',cv2_img)[1]

# #         return Response(content=im.tobytes(), media_type='image/png')

#     # # except Exception as e:
#     #     print(e)
#     #     return {"status": "error", "message": "Unable to process the image"}
=======
    # If the model expects BGR order, use cv2_img_resized as it is.
    # If the model expects RGB order, convert it to RGB
    cv2_img_resized_rgb = cv2.cvtColor(cv2_img_resized, cv2.COLOR_BGR2RGB)

    # Normalize pixel values if the model expects them to be in the range [0, 1]
    cv2_img_resized_rgb_normalized = cv2_img_resized_rgb / 255.0

    input_tensor = tf.convert_to_tensor(cv2_img_resized_rgb_normalized, dtype='float32')

    # Expand dimensions to match the input shape expected by the model: (1, 224, 224, 3)
    input_tensor = tf.expand_dims(input_tensor,axis=0)

    # Predict the label
    predicted_lab = app.state.model.predict(input_tensor)
    predicted_labels = predicted_lab[0]

    # Convert the prediction to a human-readable format if necessary
    predicted_label = predicted_labels.tolist()  # Convert to list for JSON serialization
    content={'Atelectasis': predicted_label[0],
               'Consolidation': predicted_label[1],
               'Infiltration': predicted_label[2],
               'Pneumothorax': predicted_label[3],
               'Edmema': predicted_label[4],
               'Emphysema': predicted_label[5],
               'fibrosis': predicted_label[6],
                'Effusion': predicted_label[7],
                'Pneumonia': predicted_label[8],
                'Pleural Thickening': predicted_label[9],
                'Cardiomegaly': predicted_label[10],
                'Nodule': predicted_label[11],
               'Mass': predicted_label[12],
               'Hernia': predicted_label[13]}


    return JSONResponse(content= content)
>>>>>>> master
