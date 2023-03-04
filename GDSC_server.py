from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from GDSC import predict_image_classification_sample as predict
from PIL import Image
import io
import cv2
import numpy as np

app = FastAPI()
# 꼭 GDSC_Solution_Challenge 폴더로 이동하고, 
# uvicorn GDSC_server:app --reload 로 실행
# 127.0.0.1/docs 로 들어가서 스웨거 테스트

# gray -> detect -> reduce the size


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    img = Image.open(file.file) 
    img = np.array(img)
 
    
    # Convert the image to black and white
    if len(img.shape) == 2:
        # Grayscale image has only one channel
        image_gray = img
    elif len(img.shape) == 3:
        # Color image has three channels
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(img)
    
    # Detect the biggest_face among the faces detected in the image
    biggest_face = biggest_face_image(img, faces)

    
    # Convert the image to a byte buffer
    retval, buffer = cv2.imencode('.png', biggest_face)
    binary_cv2 = buffer.tobytes()
    output = io.BytesIO(binary_cv2)
    img = Image.open(output)
    output = io.BytesIO()
    img.save(output, format='PNG')
    binary_pil = output.getvalue()

    return predict(file=binary_pil)


def biggest_face_image(image, faces):
    if len(faces) == 0:
        return None # or any other handling of this case
    areas = []
    for (x, y, w, h) in faces:
        area = w * h
        areas.append(area)
    max_area_index = np.argmax(areas)
    x, y, w, h = faces[max_area_index]
    biggest_face = image[y:y+h, x:x+w]
    return biggest_face