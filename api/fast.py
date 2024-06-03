# TODO: Import your package, replace this by explicit imports of what you need
from packagename.main import predict
from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

import numpy as np
import cv2
import io


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def index():
    return {'status': 'ok'
    }

# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.post("/iploading_image")

async def receive_image(img: UploadFile=File(...)):
    try:
        #receiving the image + decoding it
        contents = await img.read()

        #creating a numpy array from a string of bytes
        np_array = np.fromstring(contents,np.uint8)
        cv2_img = cv2.imdecode(np_array,cv2.IMREAD_COLOR)

        #adding our function that detects the location where the deseases is

        #encoding and responding with an image

        im = cv2.imencode('.png',cv2_img[1])

        return Response(content=im.tobytes(), media_type='image/png')

    except Exception as e:
        print(e)
        return {"status": "error", "message": "Unable to process the image"}
