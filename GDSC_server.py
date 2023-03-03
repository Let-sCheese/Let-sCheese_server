from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from GDSC import predict_image_classification_sample as predict
from PIL import Image
import io
app = FastAPI()



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

