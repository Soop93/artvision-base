import io
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from artwork_core import propose_canvas, manual_canvas, search_artwork
from guide_core import generate_intro, answer_question

app = FastAPI(title="Guide Privé API")


def _b64_to_pil(data_b64: str) -> Image.Image:
    """Décode en image PIL une chaîne base64 reçue de l'app.

    Côté serveur de pil_to_b64/b64_to_pil du front : l'app envoie les photos en
    base64 dans le JSON, l'API les redéserialise ici avant traitement.
    """
    return Image.open(io.BytesIO(base64.b64decode(data_b64)))


def _pil_to_b64(image: Image.Image) -> str:
    """Encode une image PIL en base64 JPEG pour la réponse JSON à l'app.

    Sert à renvoyer les images produites par l'API (toile isolée) que Streamlit
    réaffiche. Conversion RGB imposée (JPEG n'accepte pas de canal alpha). Transport
    uniquement, indépendant du preprocessing CLIP.
    """
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


@app.post("/isolate/propose")
async def isolate_propose(payload: ProposeRequest):
    """Étape 1 : propose un recadrage automatique de la toile (GroundingDINO + SAM).

    Reçoit la photo brute, délègue à propose_canvas, et renvoie la toile isolée à
    Streamlit pour validation par l'utilisateur (accepter ou cliquer les 4 coins).
    Si la détection échoue (crop None), renvoie le tag d'échec sans image.
    """
    image = _b64_to_pil(payload.image)
    crop, tag = propose_canvas(image)
    if crop is None:
        return {"status": tag, "tag": tag, "crop_image": None}
    return {"status": "ok", "tag": tag, "crop_image": _pil_to_b64(crop)}


@app.post("/isolate/manual")
async def isolate_manual(payload: ManualRequest):
    """Étape 1bis : recadrage manuel quand la proposition auto est refusée.

    Prend les 4 coins cliqués côté Streamlit sur l'image d'origine (coordonnées
    pleine résolution) et renvoie la toile redressée par homographie via
    manual_canvas.
    """
    image = _b64_to_pil(payload.image)
    crop = manual_canvas(image, payload.points)
    return {"crop_image": _pil_to_b64(crop)}


@app.post("/identify")
async def identify(payload: IdentifyRequest):
    """Étape 2 : identifie l'œuvre à partir de la toile isolée et validée.

    Reçoit le crop validé (auto ou manuel), appelle search_artwork (embedding CLIP
    puis plus proche voisin dans la base), et renvoie l'œuvre trouvée avec ses
    métadonnées.
    """
    crop = _b64_to_pil(payload.crop_image)
    top = search_artwork(crop)
    return {"artwork": top}


@app.post("/guide/intro")
async def guide_intro(payload: IntroRequest):
    """Génère le message d'accueil du guide sur l'œuvre identifiée.

    Appelle generate_intro (guide_core), qui lance le graphe RAG avec une question
    vide : le nœud de récupération cherche dans les PDF les extraits liés à l'œuvre
    (titre, artiste, période), puis le modèle rédige une présentation vivante de
    8-10 phrases terminée par une question ouverte pour inviter le visiteur à
    réagir. Renvoie ce texte d'introduction.
    """
    return {"intro": generate_intro(payload.artwork)}


@app.post("/guide/ask")
async def guide_ask(payload: AskRequest):
    """Répond à une question du visiteur sur l'œuvre (guide RAG conversationnel).

    Passe la question, les métadonnées de l'œuvre et l'historique de conversation à
    answer_question (guide_core) : le nœud de récupération cherche dans les PDF les
    extraits liés à l'œuvre et à la question, puis le modèle rédige une réponse
    développée. Le prompt lui demande de terminer par une question concrète sur le
    ressenti du visiteur (consigne, pas garantie stricte). Renvoie cette réponse.
    """
    reply = answer_question(payload.question, payload.artwork, payload.chat_history)
    return {"reply": reply}
