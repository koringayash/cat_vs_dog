import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.responses import HTMLResponse
from PIL import Image
import io
from model_builder import set_device,importing_model
from utils import func_load_model,custom_image

app = FastAPI()

def process_image(image: Image.Image) -> str:
    model = importing_model()
    model = func_load_model(model=model,path="vgg16_model.pth")
    image.save("geeks.jpg")
    return custom_image(model=model,lst=['cats','dogs'],path="geeks.jpg")


@app.get("/")
async def get_html():
    with open("upload.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=415, detail="Unsupported media type")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_output = process_image(image)
        return PlainTextResponse(content=processed_output, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
