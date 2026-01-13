from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import cv2
import os

from model import predict_mask

app = FastAPI()

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- FRONTEND SETUP ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "../Frontend")
)

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "../Frontend")),
    name="static"
)

# ---------- HOME PAGE ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- SEGMENTATION API ----------
@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    mask = predict_mask(image)

    _, encoded = cv2.imencode(".png", mask)
    return {"mask": encoded.tobytes().hex()}