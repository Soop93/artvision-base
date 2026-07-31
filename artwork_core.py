import os
import sqlite3
import chromadb
import cv2
import numpy as np
import torch
import clip
from PIL import Image
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          SamModel, SamProcessor)

DB_NAME = "harvard_clean.db"
VECTOR_STORE_PATH = "vector_store_images"
COLLECTION_NAME = "artworks_collection"
IMG_DIR = "images_clean"

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_DEVICE = None


def _get_clip_model():
    """Charge CLIP ViT-B/32 une seule fois et le réutilise (singleton).

    Le modèle et son preprocess sont lourds : on les garde dans des variables
    globales pour ne pas les recharger à chaque appel. Choisit le GPU si disponible,
    sinon le CPU. Renvoie (modèle, preprocess, device). Même modèle et même
    preprocess que la vectorisation de la base : c'est le contrat CLIP côté requête.
    """
    global _CLIP_MODEL, _CLIP_PREPROCESS, _DEVICE
    if _CLIP_MODEL is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=_DEVICE)
    return _CLIP_MODEL, _CLIP_PREPROCESS, _DEVICE

def _models_cached(repo_ids):   # repo_ids : liste d'identifiants Hugging Face (e.g. IDEA-Research/grounding-dino-tiny)
    """True si tous les repos sont déjà dans le cache HF (aucun besoin réseau)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    return all(isinstance(try_to_load_from_cache(r, "config.json"), str) for r in repo_ids)


GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"

# Hors-ligne automatique si les modèles sont déjà téléchargés : coupe le ping HF + le warning.
# Respecte un HF_HUB_OFFLINE fixé manuellement, et laisse le 1er téléchargement se faire.
# Mode en ligne forcé pour HF_HUB_OFFLINE = 0
if os.environ.get("HF_HUB_OFFLINE") is None and _models_cached([GDINO_ID, SAM_ID]):
    os.environ["HF_HUB_OFFLINE"] = "1"


# ---------- conversions PIL <-> BGR (l'app manipule du PIL, vs du BGR uint8) ----------
def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """PIL RGB -> ndarray BGR uint8 (le format natif manipulé par CanvasIsolator / OpenCV)."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """ndarray BGR uint8 -> PIL RGB (pour ré-affichage Streamlit / renvoi API)."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


class CanvasIsolator:
    """Isole la toile seule d'une photo de tableau, redressée de face.

    OBJECTIF : à partir d'une photo de tableau (prise de biais, avec cadre et fond),
    produire une image de la TOILE SEULE, redressée de face, le domaine attendu par
    la base. Sans ça, CLIP encoderait aussi le cadre et le mur.

    DEUX MODÈLES, DEUX RÔLES (téléchargés au 1er lancement depuis Hugging Face) :
      - GroundingDINO (détection guidée par texte) : on lui donne un prompt TEXTE
        ("a painting"), il renvoie une BOÎTE englobant le tableau. Il sait "où", mais
        grossièrement (un rectangle).
      - SAM (Segment Anything Model, segmentation) : on lui donne un prompt
        GÉOMÉTRIQUE (un point ou une boîte), il renvoie un MASQUE précis de l'objet.
        Il sait "la forme". SAM n'a pas de prompt texte.

    PIPELINE propose() :
      1. GroundingDINO donne une boîte autour du tableau.
      2. SAM avec un POINT au centre de la boîte donne un masque de la toile. Si ce
         masque est trop petit (aire < area_thresh x aire boîte), c'est qu'on a
         attrapé un SOUS-objet (un visage, une zone) : on bascule sur SAM avec la
         BOÎTE comme prompt (englobe tout le tableau).
      3. masque converti en quadrilatère à 4 coins (mask_to_quad).
      4. warp homographique (warp) pour obtenir la toile redressée de face.

    Sortie : image couleur pleine résolution uint8 (BGR, cohérent OpenCV). AUCUN
    resize ni normalisation ici : le resize 224 et la normalisation restent la
    responsabilité de CLIP (preprocess), pour rester cohérent avec la base.

    Côté web (Streamlit) : si l'utilisateur juge la proposition insuffisante, il
    clique les 4 coins sur l'image d'origine (composant streamlit-image-coordinates),
    puis warp() est rappelé avec ces 4 points (même géométrie qu'en interne).
    """

    def __init__(self, prompt="a painting", area_thresh=0.60, device=None,
                 gdino_id=GDINO_ID, sam_id=SAM_ID, detect_max_side=1024):
        """Charge GroundingDINO et SAM une seule fois ; l'instance traite ensuite n
        images sans recharger (d'où l'usage en singleton, cf. _get_canvas_isolator).

        Paramètres clés :
          - prompt : requête texte donnée à GroundingDINO pour localiser le tableau.
          - area_thresh : seuil du ratio aire(masque point)/aire(boîte) dans
            propose(). Fixé à 0.60 ; en dessous, on considère que le point a attrapé
            un sous-objet et on bascule sur la boîte. Filet de sécurité = clic
            manuel. Si incohérence à l'enrichissement, tester [0.45, 0.50, 0.55].
          - detect_max_side : côté max pour la détection/segmentation (accélère, cf.
            _resize_for_detection) ; le warp final reste en pleine résolution.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt = prompt
        self.area_thresh = area_thresh
        self.detect_max_side = detect_max_side   # taille max (côté) pour la détection/segmentation :
                                                  # pour accélérer le traitement
        print(f"chargement des modèles sur {self.device} ...")
        self.gproc = AutoProcessor.from_pretrained(gdino_id)
        self.gmodel = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to(self.device).eval()
        self.sproc = SamProcessor.from_pretrained(sam_id)
        self.smodel = SamModel.from_pretrained(sam_id).to(self.device).eval()  # Rappel : désactive dropout ou BatchNorm par exemple si nécessaire
        print("modèles prêts.")

    # ---------- géométrie ----------
    @staticmethod   # Fonction géométrique
    def order_points(pts):
        """Range 4 points quelconques dans l'ordre TL, TR, BR, BL, à n'importe quelle rotation.

        Méthode : tri angulaire (atan2) autour du centroïde donne un ordre cyclique convexe
        valable quelle que soit l'orientation, puis départ au coin haut-gauche (x+y minimal).
        En repère image (y vers le bas), atan2 croissant tourne dans le sens horaire, d'où
        TL, TR, BR, BL. Marche quel que soit l'ordre des clics.

        Remplace l'ancienne astuce somme/différence, qui dégénérait sur un losange à ~45°
        (deux coins de même somme et même différence, un point compté deux fois, homographie
        effondrée, crop vide). Fix du 2026-07-30.
        """
        pts = np.asarray(pts, dtype="float32")
        c = pts.mean(axis=0)                            # centroïde du quadrilatère
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        pts = pts[np.argsort(ang)]                      # ordre angulaire = polygone convexe sans croisement
        start = int(np.argmin(pts.sum(axis=1)))         # coin haut-gauche = somme x+y minimale
        return np.roll(pts, -start, axis=0).astype("float32")   # TL, TR, BR, BL (sens horaire, repère image)

    def warp(self, img_bgr, quad):   # self pour appeler order_points
        """Redresse la toile de face par transformation homographique (perspective).

        On calcule la largeur/hauteur cibles à partir des côtés du quadrilatère, on construit un
        rectangle destination, puis cv2 trouve la matrice 3x3 (getPerspectiveTransform) qui envoie
        les 4 coins du tableau (vu de biais) sur ce rectangle -> la toile devient rectangulaire.
        """
        rect = self.order_points(quad)
        tl, tr, br, bl = rect
        W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))   # largeur = plus long des 2 côtés horiz.
                                                                        # Plus facile d'agrandir que de réduire, d'où le max()
        H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))   # hauteur = plus long des 2 côtés vert.
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], "float32")   # rectangle cible. Les indices pixels (0, taille - 1)
        M = cv2.getPerspectiveTransform(rect, dst)                       # matrice de perspective 3x3
        return cv2.warpPerspective(img_bgr, M, (W, H))

    @staticmethod
    def clean_mask(mask, open_px=15):  # open_px : taille du noyau morphologique
        """Nettoie le masque SAM binaire : ouverture morphologique + garde le plus gros bloc.

        - ouverture (érosion puis dilatation) : rase les fines bavures du masque (ex. débordement
          sur le cadre) sans déformer la grande forme.
        - composantes connexes : on ne garde que le plus gros bloc, pour éliminer les îlots parasites.
        """
        m = (mask.astype(np.uint8)) * 255     # conversion du masque en niveau de gris
        k = max(3, open_px | 1)                      # noyau impair >= 3 (| 1 force l'imparité) -> Le centre du noyau doit être bien défini.
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n > 2:                                    # >2 = fond + plusieurs blocs -> on garde le plus gros
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))   # +1 car l'indice 0 = le fond
            m = np.where(lab == big, 255, 0).astype(np.uint8)
        return m

    @staticmethod
    def mask_to_quad(mask_bin):
        """Convertit un masque binaire en 4 coins (le quadrilatère de la toile).

        On prend le plus grand contour, puis on l'approxime par un polygone (approxPolyDP) en
        augmentant la tolérance `eps` jusqu'à tomber sur 4 sommets convexes = un quadrilatère.
        Si aucune tolérance ne donne 4 coins nets, fallback : le rectangle orienté minimal
        (minAreaRect) qui englobe le contour.
        """
        cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Contours extérieurs + On garde que les sommets
        if not cnts:
            return None                                          # masque vide -> Filet de sécurité manuel avec propose()
        c = max(cnts, key=cv2.contourArea)                       # le plus grand contour = la toile
        if cv2.contourArea(c) <= 0:                              # Garde-fou si aire = 0
            return None
        peri = cv2.arcLength(c, True)                            # périmètre, sert d'échelle à eps
        for eps in np.linspace(0.01, 0.08, 8):                  # tolérance croissante 1%..8% du périmètre
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):  # On s'assure de ne pas avoir de coin concave (type flèche)
                return approx.reshape(4, 2).astype("float32")   # 4 coins convexes trouvés
        return cv2.boxPoints(cv2.minAreaRect(c)).astype("float32")   # fallback : rectangle englobant
                                                                    # minAreaRect trouve le plus petit rectangle qui englobe le contour
                                                                    # boxPoints en extrait les 4 coins

    @staticmethod
    def _resize_for_detection(img_bgr, max_side):
        """Réduit l'image avant détection/segmentation si son plus grand côté dépasse max_side.

        - Pourquoi : GroundingDINO + SAM sont des transformers ; leur coût (temps, VRAM)
          grimpe vite avec la résolution. Une photo 3000-4000px alourdit l'inférence pour
          rien, la précision utile plafonne avant.
        - Retour : (image_redimensionnée, scale) ; scale=1.0 si déjà assez petite.
        - Garantie : le warp final reste sur l'image ORIGINALE (cf. propose()), donc ce
          resize n'affecte que la vitesse, jamais la qualité de la toile.
        """
        h, w = img_bgr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale >= 1.0:
            return img_bgr, 1.0
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return small, scale

    # ---------- détection / segmentation ----------  Utilisation des modèles
    def detect_box(self, img_bgr, box_thr=0.25, text_thr=0.25):
        """GroundingDINO : renvoie la meilleure boîte [x0,y0,x1,y1] en pixels, ou None.

        Applique le prompt texte, filtre par les seuils box/text, et garde la boîte de
        plus haut score. None si rien ne passe les seuils (déclenche le repli manuel).
        doc : https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/grounding-dino#transformers.GroundingDinoImageProcessor
        """
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        inputs = self.gproc(images=rgb, text=[[self.prompt]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gmodel(**inputs)
        res = self.gproc.post_process_grounded_object_detection(
            outputs, threshold=box_thr, text_threshold=text_thr, target_sizes=[(H, W)])[0]
        if len(res["scores"]) == 0:
            return None
        best = int(res["scores"].argmax())
        return [int(v) for v in res["boxes"][best].tolist()]

    def _sam_mask(self, img_bgr, points=None, boxes=None):
        """Lance SAM avec un prompt géométrique (point OU boîte) et renvoie le meilleur masque booléen.
        Docs : https://huggingface.co/docs/transformers/tasks/mask_generation
               https://huggingface.co/docs/transformers/model_doc/sam

        SAM propose 3 masques candidats par prompt (avec un score IoU estimé chacun) ; on garde
        celui de meilleur score. `points` et `boxes` sont les deux modes de prompt utilisés par propose().
        """
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.sproc(rgb, input_points=points, input_boxes=boxes,
                            return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.smodel(**inputs)
        masks = self.sproc.image_processor.post_process_masks(   # rééchantillonne les masques à la résolution de l'image d'origine
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())
        iou = outputs.iou_scores[0, 0]
        return masks[0][0][int(iou.argmax())].numpy().astype(bool)   # le meilleur des 3 candidats :

    def sam_point_to_quad(self, img_bgr, pt):
        """SAM en mode POINT : segmente autour du point `pt`, nettoie le masque, en tire
        un quadrilatère. Renvoie (quad, mask_bin) ; mask_bin sert au calcul de l'aire
        dans propose() pour décider si on bascule sur le mode boîte.
        """
        mask = self._sam_mask(img_bgr, points=[[[int(pt[0]), int(pt[1])]]])  # Appelle le mode point de SAM
                                                                            # [[[..]]] -> (batch image = 1, objets = 1, points de l'objet = (x,y)
        mask_bin = self.clean_mask(mask)
        return self.mask_to_quad(mask_bin), mask_bin   # renvoit tuple (quad, mask_bin), mask_bin nécessaire au calcul de l'aire dans propose()

    def sam_box_to_quad(self, img_bgr, box):
        """SAM en mode BOÎTE : segmente dans la boîte `box`, nettoie le masque, en tire un
        quadrilatère. Voie de secours de propose() quand le mode point n'a attrapé qu'un
        sous-objet ; la boîte force SAM à englober tout le tableau. Renvoie (quad, mask_bin).
        """
        mask = self._sam_mask(img_bgr, boxes=[[box]])  # Mode Boxes de SAM
        mask_bin = self.clean_mask(mask)
        return self.mask_to_quad(mask_bin), mask_bin

    def propose(self, img_bgr):
        """Proposition automatique de toile redressée. Renvoie (toile_bgr, tag) ou (None, raison).

        Stratégie point-puis-boîte :
          1. GroundingDINO localise le tableau (boîte). Pas de boîte -> on abandonne (clic manuel).
          2. SAM avec un POINT au centre : masque précis. On mesure le ratio aire(masque)/aire(boîte).
          3. Si le masque remplit assez la boîte (ratio >= area_thresh) -> c'est bien la toile : on garde.
             Sinon le point a attrapé un sous-objet (visage, détail) -> on relance SAM avec la BOÎTE,
             qui force à segmenter tout le tableau.
        Le `tag` (ex. "point 87%", "box (point 20%)") trace quelle voie a été prise, utile au debug.
        """
        small_bgr, scale = self._resize_for_detection(img_bgr, self.detect_max_side)  # détection sur image réduite (vitesse)
        box = self.detect_box(small_bgr)
        if box is None:
            return None, "no-box"                                # DINO n'a rien trouvé
        x0, y0, x1, y1 = box
        box_area = max(1, (x1 - x0) * (y1 - y0))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2                  # centre de la boîte = indice point
        quad, mask_bin = self.sam_point_to_quad(small_bgr, (cx, cy))
        ratio = (mask_bin > 0).sum() / box_area                 # à quel point le masque remplit la boîte
        if quad is not None and ratio >= self.area_thresh:
            return self.warp(img_bgr, quad / scale), f"point {ratio:.0%}"   # voie point : masque suffisant, warp en pleine résolution
        quad, _ = self.sam_box_to_quad(small_bgr, box)          # voie box : masque trop petit -> secours
        if quad is None:
            return None, "no-mask"                               # SAM n'a rien rendu -> clic manuel
        return self.warp(img_bgr, quad / scale), f"box (point {ratio:.0%})"

    # ---------- validation manuelle (équivalent web de click_corners / show_and_decide) ----------
    def warp_from_points(self, img_bgr, points):
        """Redresse la toile à partir de 4 points cliqués par l'utilisateur (ordre libre, coords
        pleine résolution de l'image d'origine). Pendant web du mode 'cliquer les 4 coins' :
        même géométrie que l'auto (order_points + warp), seule la capture des clics change
        (composant Streamlit au lieu d'une fenêtre OpenCV)."""
        quad = np.array(points, dtype="float32")
        return self.warp(img_bgr, quad)


# ---------- singleton, comme CLIP (cf. _get_clip_model) : chargé une fois, réutilisé ensuite ----------
_CANVAS_ISOLATOR = None


def _get_canvas_isolator():
    """Charge le CanvasIsolator (GroundingDINO + SAM) une seule fois et le réutilise
    (singleton, même logique que _get_clip_model). Évite de recharger ces modèles
    lourds à chaque requête.
    """
    global _CANVAS_ISOLATOR
    if _CANVAS_ISOLATOR is None:
        _CANVAS_ISOLATOR = CanvasIsolator()
    return _CANVAS_ISOLATOR

def propose_canvas(pil_image: Image.Image):
    """Point d'entrée API de la découpe automatique. Prend l'image PIL reçue, la
    convertit en BGR, lance CanvasIsolator.propose(), et renvoie (toile_pil, tag) ou
    (None, tag) en cas d'échec.

    Le tag trace la voie prise ("point 87%", "no-box"...) : Streamlit s'en sert pour
    informer l'utilisateur ou lui proposer le clic manuel.
    """
    isolator = _get_canvas_isolator()
    img_bgr = pil_to_bgr(pil_image)
    toile_bgr, tag = isolator.propose(img_bgr)
    if toile_bgr is None:
        return None, tag
    return bgr_to_pil(toile_bgr), tag


def manual_canvas(pil_image: Image.Image, points):
    """Point d'entrée API de la découpe manuelle. L'utilisateur a cliqué les 4 coins
    de la toile dans Streamlit (coords pleine résolution de l'image d'origine) ; on
    applique le même redressement homographique que propose() (warp_from_points).
    Renvoie la toile isolée en PIL.
    """
    isolator = _get_canvas_isolator()
    img_bgr = pil_to_bgr(pil_image)
    toile_bgr = isolator.warp_from_points(img_bgr, points)
    return bgr_to_pil(toile_bgr)


def embed_image(cropped_image: Image.Image):
    """Calcule l'embedding CLIP d'une toile déjà isolée et redressée.

    La découpe (auto ou manuelle) est une étape séparée, validée par l'utilisateur
    avant d'arriver ici. Applique le preprocess CLIP puis encode_image, et renvoie le
    vecteur en liste Python. Même modèle et même preprocess que la vectorisation de la
    base : c'est ce qui garantit la comparabilité des distances (contrat CLIP).
    """
    model, preprocess, device = _get_clip_model()
    tensor = preprocess(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
    return features.cpu().numpy().tolist()[0]


def search_artwork(cropped_image: Image.Image):
    """Identifie l'œuvre à partir de la toile isolée et validée.

    Chaîne : embedding CLIP (embed_image), puis plus proche voisin dans ChromaDB
    (métrique L2 par défaut), puis lecture des métadonnées complètes dans la base
    SQLite harvard_clean via l'artwork_id, assemblées en dict pour l'affichage
    Streamlit (avec image_path vers images_clean).

    Piste (cf. README) : renvoyer un top-k avec distances, un seuil de confiance et
    l'écart 1er/2e candidat, pour signaler les cas ambigus au lieu de trancher
    systématiquement au top-1.
    """
    embedding = embed_image(cropped_image)

    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["metadatas"]
    )
#Extraction des metadatas de la ChromaDB (et notamment l'artworkID)
    metadatas = results["metadatas"][0]
    if not metadatas:
        return None

    artwork_id = metadatas[0].get("artwork_id")
#Extraction des metadonnées de la SQLITE3 harvard_clean
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        '''SELECT id, title, artist, url, dated, medium, technique, classification,
                   period, century, culture, dimensions, description, labeltext,
                   department, creditline
            FROM artworks WHERE id = ?''',
        (artwork_id,)
    )

#Création d'un dictionnaire pour l'affichage dans Stremlit
    columns = ["id", "title", "artist", "url", "dated", "medium", "technique", "classification",
               "period", "century", "culture", "dimensions", "description", "labeltext",
               "department", "creditline"]
    row = cursor.fetchone()
    conn.close()
    meta = dict(zip(columns, row)) if row else {}

    best_candidate = {
        "artwork_id": artwork_id,
        "title": meta.get("title", "?"),
        "artist": meta.get("artist", "?"),
        "url": meta.get("url"),
        "dated": meta.get("dated"),
        "medium": meta.get("medium"),
        "technique": meta.get("technique"),
        "classification": meta.get("classification"),
        "period": meta.get("period"),
        "century": meta.get("century"),
        "culture": meta.get("culture"),
        "dimensions": meta.get("dimensions"),
        "description": meta.get("description"),
        "labeltext": meta.get("labeltext"),
        "department": meta.get("department"),
        "creditline": meta.get("creditline"),
        "image_path": f"{IMG_DIR}/{artwork_id}.jpg",
    }

    return best_candidate
