from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from GDSC import predict_image_classification_sample as predict
from PIL import Image
import io
app = FastAPI()
# 꼭 GDSC_Solution_Challenge 폴더로 이동하고, 
# uvicorn GDSC_server:app --reload 로 실행
# 127.0.0.1/docs 로 들어가서 스웨거 테스트


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    img = Image.open(file.file)
    # Reduce the size of the image
    img = img.resize((224, 224))
    # Convert the image to black and white
    img = img.convert('L')
    output=io.BytesIO()
    img.save(output,"PNG")
    binary_pil=output.getvalue()

    #return predict(file=file.file.read())
    return predict(file=binary_pil)

