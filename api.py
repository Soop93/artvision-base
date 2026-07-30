import io
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from artwork_core import propose_canvas, manual_canvas, search_artwork
from guide_core import generate_intro, answer_question

app = FastAPI(title="Guide Privé API")


def _b64_to_pil(data_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data_b64)))


def _pil_to_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class ProposeRequest(BaseModel):
    image: str  #Photo brute (base64), telle que prise par la caméra ou uploadée


class ManualRequest(BaseModel):
    image: str          #Même photo brute (base64) que pour /isolate/propose
    points: list        #4 points [[x, y], ...] cliqués sur l'image d'origine (coordonnées pleine résolution)


class IdentifyRequest(BaseModel):
    crop_image: str  #Image déjà isolée/redressée (base64), validée par l'utilisateur


class IntroRequest(BaseModel):
    artwork: dict


class AskRequest(BaseModel):
    question: str
    artwork: dict
    chat_history: list = []


#Étape 1 : Proposition automatique de découpe de la toile (GroundingDINO + SAM).
#Renvoyée à Streamlit pour validation par l'utilisateur (accepter ou cliquer les 4 coins).
@app.post("/isolate/propose")
async def isolate_propose(payload: ProposeRequest):
    image = _b64_to_pil(payload.image)
    crop, tag = propose_canvas(image)
    if crop is None:
        return {"status": tag, "tag": tag, "crop_image": None}
    return {"status": "ok", "tag": tag, "crop_image": _pil_to_b64(crop)}


#Étape 1Bis : découpe manuelle si la proposition automatique est refusée par l'utilisateur
#(4 coins cliqués côté Streamlit sur l'image d'origine).
@app.post("/isolate/manual")
async def isolate_manual(payload: ManualRequest):
    image = _b64_to_pil(payload.image)
    crop = manual_canvas(image, payload.points)
    return {"crop_image": _pil_to_b64(crop)}


#Étape 2 : Identification à partir de l'image isolée et validée (auto ou manuelle)
@app.post("/identify")
async def identify(payload: IdentifyRequest):
    crop = _b64_to_pil(payload.crop_image)
    top = search_artwork(crop)
    return {"artwork": top}


@app.post("/guide/intro")
async def guide_intro(payload: IntroRequest):
    return {"intro": generate_intro(payload.artwork)}


@app.post("/guide/ask")
async def guide_ask(payload: AskRequest):
    reply = answer_question(payload.question, payload.artwork, payload.chat_history)
    return {"reply": reply}
