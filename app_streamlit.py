"""Front Streamlit de l'application : point d'entrée utilisateur.

Interface web où l'utilisateur prend ou importe la photo d'un tableau pour
l'identifier dans la collection, puis peut interroger un guide conversationnel sur
l'œuvre reconnue. Ce fichier ne contient aucune logique de vision ni de recherche :
il gère l'affichage et les interactions, et délègue tout le calcul à l'API FastAPI
sur http://localhost:8000 (isolation de la toile, identification CLIP, guide RAG).

Flux : l'utilisateur fournit une photo (onglet caméra ou import), l'API propose un
recadrage de la toile, l'utilisateur le valide ou ajuste les 4 coins à la main, puis
l'app lance l'identification et affiche l'œuvre trouvée avec ses métadonnées. Les
images circulent avec l'API en base64 (pil_to_b64 / b64_to_pil).
"""
import os
import io
import base64
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates


def _scroll_to_anchor(anchor_id: str):
    """Fait défiler la page jusqu'à l'élément d'ancre donné après le rendu.

    Injecte un petit script JS (via components.html) qui retrouve l'ancre dans la
    page et la ramène en haut de vue. Le setTimeout de 30 ms laisse Streamlit finir
    d'afficher les éléments avant le scroll. Sert à recentrer l'utilisateur sur le
    résultat après une interaction (identification, validation du crop) sans qu'il
    ait à faire défiler à la main.
    """
    components.html(
        f"""<script>
        setTimeout(function() {{
            var el = window.parent.document.getElementById('{anchor_id}');
            if (el) {{ el.scrollIntoView({{behavior: 'instant', block: 'start'}}); }}
        }}, 30);
        </script>""",
        height=0,
    )

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Guide Privé", layout="centered")

st.title("Identification d'œuvre")
st.caption("Prends une photo ou importe une photo d'un tableau pour l'identifier dans la collection.")


def pil_to_b64(image: Image.Image) -> str:
    """Encode une image PIL en chaîne base64 JPEG pour l'envoi à l'API.

    L'app parle à l'API en JSON, qui ne transporte pas de binaire : on sérialise
    donc l'image en base64. Conversion en RGB imposée (retire un éventuel canal
    alpha, incompatible JPEG). Encodage de transport uniquement, sans rapport avec
    le preprocessing CLIP qui a lieu côté serveur dans artwork_core.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def b64_to_pil(data_b64: str) -> Image.Image:
    """Décode une chaîne base64 en image PIL (opération inverse de pil_to_b64).

    Sert à reconstruire les images renvoyées par l'API (par exemple la toile isolée)
    pour les afficher dans Streamlit.
    """
    return Image.open(io.BytesIO(base64.b64decode(data_b64)))


#Création des deux onglets
tab_camera, tab_upload = st.tabs(["Prendre une photo", "Importer une image"])

image_input = None

#Module photo
with tab_camera:
    cam_photo = st.camera_input("Photographie l'œuvre")
    if cam_photo is not None:
        image_input = Image.open(cam_photo)

#Module importation
with tab_upload:
    uploaded_file = st.file_uploader("Choisis une image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)

#Mise en page modifiée si une image est fournie
if image_input is not None:
    st.divider()
    raw_b64 = pil_to_b64(image_input)

#Réinitialisation de l'état d'isolation dès qu'une NOUVELLE photo arrive (nouvelle prise/upload)
    if st.session_state.get("raw_image_b64") != raw_b64:
        st.session_state.raw_image_b64 = raw_b64
        st.session_state.isolation_stage = "propose"   # propose -> manual -> done
        st.session_state.manual_points = []
        st.session_state.last_click_id = None
        st.session_state.final_crop_b64 = None
        st.session_state.pop("guide_artwork_id", None)  #Force la régénération de l'intro pour la nouvelle œuvre

    st.image(image_input, caption="Photo fournie", use_container_width=True)

#Étape d'isolation de l'image (GroundingDINO + SAM), tant que l'utilisateur n'a pas validé
    if st.session_state.isolation_stage == "propose":
        with st.spinner("Détection de la toile (GroundingDINO + SAM)..."):
            propose_resp = requests.post(f"{API_URL}/isolate/propose", json={"image": raw_b64})
            propose_result = propose_resp.json()

        if propose_result["status"] != "ok":
            st.warning(f"Toile non détectée automatiquement ({propose_result['tag']}). "
                       f"Merci de cliquer les 4 coins de la toile.")
            st.session_state.isolation_stage = "manual"
            st.rerun()
        else:
            st.session_state.proposal_crop_b64 = propose_result["crop_image"]
            st.session_state.proposal_tag = propose_result["tag"]

            st.divider()
            st.markdown(f"### Proposition automatique ({st.session_state.proposal_tag})")
            proposal_img = b64_to_pil(st.session_state.proposal_crop_b64)
            st.image(proposal_img, caption="Toile isolée et redressée", use_container_width=True)

            col_valider, col_manuel = st.columns(2)
            with col_valider:
                if st.button("Valider cette découpe", use_container_width=True):
                    st.session_state.final_crop_b64 = st.session_state.proposal_crop_b64
                    st.session_state.isolation_stage = "done"
                    st.rerun()
            with col_manuel:
                if st.button("Cliquer les 4 coins moi-même", use_container_width=True):
                    st.session_state.isolation_stage = "manual"
                    st.session_state.manual_points = []
                    st.session_state.last_click_id = None
                    st.rerun()

#Clic manuel des 4 coins intérieurs de la toile (équivalent web du click_corners)
    elif st.session_state.isolation_stage == "manual":
        st.divider()
        st.markdown("### Clique les 4 coins intérieurs de la toile")
        st.caption("Clique dans l'ordre libre. Le polygone se ferme automatiquement au 4e point.")

        MAX_SIDE = 700  #Largeur d'affichage
        scale = min(1.0, MAX_SIDE / image_input.width)
        disp_w, disp_h = int(image_input.width * scale), int(image_input.height * scale)
        disp_img = image_input.resize((disp_w, disp_h)).convert("RGB")

#Dessin des points déjà cliqués + du polygone visible par l'utilisateur
        draw_img = disp_img.copy()
        draw = ImageDraw.Draw(draw_img)
        disp_points = [(int(x * scale), int(y * scale)) for x, y in st.session_state.manual_points]
        for i, (x, y) in enumerate(disp_points):
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 0, 0))
            draw.text((x + 8, y - 8), str(i + 1), fill=(255, 255, 0))
        if len(disp_points) >= 2:
            draw.line(disp_points + ([disp_points[0]] if len(disp_points) == 4 else []),
                      fill=(0, 255, 0), width=2)

        st.markdown('<div id="corner-click-anchor"></div>', unsafe_allow_html=True)
#Fix pour que l'interface ne saute pas dans Streamlit à chaque sélection de coin.
        coords = streamlit_image_coordinates(draw_img, key="manual_click")
        if coords is not None and len(st.session_state.manual_points) < 4:
            click_id = (coords["x"], coords["y"])
            if st.session_state.get("last_click_id") != click_id:
                st.session_state.last_click_id = click_id
#streamlit_image_coordinates renvoie les coordonnées dans le référentiel de l'image affichée
#Remise à l'échelle de l'image d'origine (coordonnées pleine résolution attendues par warp())
                ox, oy = coords["x"] / scale, coords["y"] / scale
                st.session_state.manual_points.append((ox, oy))
                st.rerun()

        _scroll_to_anchor("corner-click-anchor")
        st.caption(f"{len(st.session_state.manual_points)} / 4 points")

        col_annuler, col_valider, col_recommencer = st.columns(3)
        with col_annuler:
            if st.button("Annuler dernier point", use_container_width=True,
                        disabled=not st.session_state.manual_points):
                st.session_state.manual_points.pop()
                st.rerun()
        with col_valider:
            if st.button("Valider (4 points)", use_container_width=True,
                        disabled=len(st.session_state.manual_points) != 4):
                with st.spinner("Redressement de la toile..."):
                    manual_resp = requests.post(f"{API_URL}/isolate/manual", json={
                        "image": raw_b64,
                        "points": st.session_state.manual_points,
                    })
                    st.session_state.final_crop_b64 = manual_resp.json()["crop_image"]
                st.session_state.isolation_stage = "done"
                st.rerun()
        with col_recommencer:
            if st.button("Recommencer", use_container_width=True):
                st.session_state.manual_points = []
                st.session_state.last_click_id = None
                st.rerun()

#Isolation validée (auto ou manuelle)
    elif st.session_state.isolation_stage == "done" and st.session_state.final_crop_b64:
        cropped_image = b64_to_pil(st.session_state.final_crop_b64)

        st.divider()
        col_input, col_crop = st.columns(2)
        with col_input:
            st.image(image_input, caption="Photo fournie", use_container_width=True)
        with col_crop:
            st.image(cropped_image, caption="Toile isolée (validée)", use_container_width=True)

        if st.button("↺ Refaire la découpe"):
            st.session_state.isolation_stage = "propose"
            st.session_state.manual_points = []
            st.rerun()

        with st.spinner("Recherche de l'embedding le plus proche (CLIP)..."):
            identify_resp = requests.post(f"{API_URL}/identify", json={"crop_image": st.session_state.final_crop_b64})
            top = identify_resp.json()["artwork"]  #top => metadonnées de l'oeuvre la plus proche

        st.divider()

        if not top:
            st.error("Aucune correspondance trouvée.")
        else:
            st.success("Œuvre identifiée")

#Création des colonnes pour afficher l'image et les métadonnées cote à cote
            col_official, col_meta = st.columns([1, 1])
            with col_official:
                if os.path.exists(top["image_path"]):
                    st.image(top["image_path"], caption="Oeuvre originale", use_container_width=True)
                else:
                    st.error(f"Oeuvre originale introuvable : {top['image_path']}")

            with col_meta:
                st.markdown(f"### {top['title']}")
                st.markdown(f"**Artiste :** {top['artist']}")
                if top.get("dated"):
                    st.markdown(f"**Date :** {top['dated']}")
                if top.get("period") or top.get("culture"):
                    period_culture = " · ".join(filter(None, [top.get("period"), top.get("culture")]))
                    st.markdown(f"**Courant / culture :** {period_culture}")
                if top.get("medium"):
                    st.markdown(f"**Médium :** {top['medium']}")
                if top.get("technique"):
                    st.markdown(f"**Technique :** {top['technique']}")
                if top.get("dimensions"):
                    st.markdown(f"**Dimensions :** {top['dimensions']}")
                if top.get("classification"):
                    st.markdown(f"**Classification :** {top['classification']}")
                if top.get("department"):
                    st.markdown(f"**Département :** {top['department']}")
                if top.get("creditline"):
                    st.markdown(f"**Provenance :** {top['creditline']}")
                st.markdown(f"**ID :** {top['artwork_id']}")

#Affichage en dessous de l'image originale et des métadonnées d'une description si elle existe
            if top.get("description") or top.get("labeltext"):
                st.divider()
                if top.get("description"):
                    st.markdown("**Description**")
                    st.write(top["description"])
                if top.get("labeltext"):
                    st.markdown("**Texte du cartel**")
                    st.write(top["labeltext"])

#Génération de l'intro du guide LLM (RAG sur les PDF)
            st.divider()
            st.markdown("### Introduction du guide")

            if "guide_artwork_id" not in st.session_state or st.session_state.guide_artwork_id != top["artwork_id"]:
                with st.spinner("Préparation de l'introduction"):
                    try:
                        intro_resp = requests.post(f"{API_URL}/guide/intro", json={"artwork": top})
                        intro = intro_resp.json()["intro"]
                    except Exception as e:
                        intro = f"Le guide n'a pas pu se connecter à l'API ({e}). " \
                                f"Vérifie que le serveur FastAPI est bien lancé (`uvicorn api:app`)."
                st.session_state.guide_artwork_id = top["artwork_id"]  #Mémorisation pour ne pas régénérer à chaque interaction
                st.session_state.guide_intro = intro
                st.session_state.chat_history = []

            st.write(st.session_state.guide_intro)

#Zone de chat : historique + nouvelle question
            st.markdown("### Posez vos questions")

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            question = st.chat_input("Posez une question sur cette œuvre...")
            if question:
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.write(question)

                with st.chat_message("assistant"):
                    with st.spinner("Recherche..."):
                        try:
                            ask_resp = requests.post(f"{API_URL}/guide/ask", json={
                                "question": question,
                                "artwork": top,
                                "chat_history": st.session_state.chat_history[:-1],
                            })
                            reply = ask_resp.json()["reply"]  #Appel RAG + LLM via l'API
                        except Exception as e:
                            reply = f"Erreur de connexion à l'API : {e}"
                    st.write(reply)

                st.session_state.chat_history.append({"role": "assistant", "content": reply})
