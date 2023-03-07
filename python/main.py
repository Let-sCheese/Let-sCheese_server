from fastapi import FastAPI, File, UploadFile,status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from google_module import predict_image_classification_sample as predict
from PIL import Image
import io
import cv2
import numpy as np
import logging
from pydantic import BaseModel
from typing import Union
from customized_exception import LowEmotionError,NoFaceException

app = FastAPI()
# 꼭 GDSC_Solution_Challenge 폴더로 이동하고, 
# uvicorn GDSC_server:app --reload 로 실행
# 127.0.0.1/docs 로 들어가서 스웨거 테스트

# gray -> detect -> reduce the size

class Classfication(BaseModel):
    emotion: Union[str,None]= None
    confidence: Union[float,None] = None
    is_higher_than_half: Union[bool, None] = None





@app.post("/uploadfile/",
        description="얼굴 감정 인식 API입니다.",response_model=Classfication) 
def create_upload_file(file: UploadFile = File(...)):
    try:
        binary_pil=find_face(file)
        if binary_pil is None:
            raise NoFaceException
        else:
            predict_result=predict(file=binary_pil)
            return Classfication(emotion=predict_result[0],confidence=predict_result[1],is_higher_than_half=(lambda x:True if x>=0.5 else False)(predict_result[1]))
    except (LowEmotionError, NoFaceException) as e:
        logging.warning(e)
        return JSONResponse(status_code=status.HTTP_406_NOT_ACCEPTABLE,content={"emotion": str(e), "confidence":None, "is_higher_than_half": None})
    except Exception as e:
        logging.warning(e)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=str(e))
    except:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content="INTERNAL_SERVER_ERROR")

    
    

def biggest_face_image(image, faces):
    if len(faces) == 0:
        return None  # or any other handling of this case
    areas = []
    for (x, y, w, h) in faces:
        area = w * h
        areas.append(area)
    max_area_index = np.argmax(areas)
    x, y, w, h = faces[max_area_index]
    biggest_face = image[y:y+h, x:x+w]
    return biggest_face

def find_face(file: UploadFile = File(...)):
    img = Image.open(file.file)
    img = np.array(img)
    # Convert the image to black and white
    if len(img.shape) == 2:
        # Grayscale image has only one channel
        image_gray = img
    elif len(img.shape) == 3:
        # Color image has three channels
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces
    faces = face_cascade.detectMultiScale(img)
    
    # Detect the biggest_face among the faces detected in the image
    biggest_face = biggest_face_image(img, faces)
    #logging.info(type(biggest_face))
    if biggest_face is None:
        return None
    # Convert the image to a byte buffer
    retval, buffer = cv2.imencode('.png', biggest_face)
    binary_cv2 = buffer.tobytes()
    output = io.BytesIO(binary_cv2)
    img = Image.open(output)
    output = io.BytesIO()
    img.save(output, format='PNG')
    binary_pil = output.getvalue()
    return binary_pil