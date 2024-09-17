# from fastapi import FastAPI,File,UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# import os
# app=FastAPI()

# app = FastAPI()

# # origins = [
# #     "http://localhost",
# #     "http://localhost:3000",
# # ]
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=origins,
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# MODEL=tf.keras.models.load_model(os.getcwd(),"potatoes.h5" )
# CLASS_NAMES=["Early Blight" , "Late Blight" , "Healthy"]

# @app.get("/ping")
# async def ping():
#     return "Hello, I am Alive"

# def read_file_as_image(data)->np.ndarray:
#     image=np.array(Image.open(BytesIO(data)))
#     return image

# @app.post('/predict')
# async def predict(
#     file:UploadFile=File(...)
# ):
#     image=read_file_as_image(await file.read())
#     image_batch=np.expand_dims(image,0)
#     predictions=MODEL.predict(image_batch)
#     predictions_class=CLASS_NAMES[np.argmax(predictions[0])]
#     confidence=np.max(predictions[0])
#     return{
#         "Class name":predictions_class,
#         "Confidence":float(confidence)
#     }

# if __name__=="__main__":
#     uvicorn.run(app,host="localhost", port=8080)

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# Uncomment and configure CORS if needed
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Load the model correctly


# MODEL = tf.keras.models.load_model(os.path.join(os.getcwd(), "potatoes.h5"))
# CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# @app.get("/ping")
# async def ping():
#     return "Hello, I am Alive"

# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

# @app.post('/predict')
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     image_batch = np.expand_dims(image, 0)
#     predictions = MODEL.predict(image_batch)
#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return {
#         "Class name": predicted_class,
#         "Confidence": float(confidence)
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8080)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# Load the model
MODEL = tf.keras.models.load_model("potatoes.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/", response_class=HTMLResponse)
async def get_form():
    # Serve the HTML form for uploading images
    content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Potato Disease Prediction</title>
    </head>
    <body>
        <h1>Upload Image for Potato Disease Prediction</h1>
        <form action="/predict" enctype="multipart/form-data" method="post">
            <label for="file">Select an image:</label>
            <input type="file" id="file" name="file" accept="image/*" required>
            <br><br>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the uploaded image
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    # Perform prediction
    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Return the result in HTML
    return HTMLResponse(content=f"""
        <h2>Prediction Results</h2>
        <p>Class: <strong>{predicted_class}</strong></p>
        <p>Confidence: <strong>{confidence * 100:.2f}%</strong></p>
        <br><a href="/">Go back</a>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)

