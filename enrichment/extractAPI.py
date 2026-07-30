"""
extractAPI.py — Extraction incrémentale Harvard Art Museums -> SQLite + images_clean/.

Première brique du pipeline d'enrichissement (exécuté en local). Récupère de NOUVELLES
œuvres via l'API Harvard, télécharge leur image de référence, insère leurs métadonnées.
Enchaîné ensuite par augmentDB.py (variantes) puis vectorisationDB.py (embeddings CLIP).

============================ MODIFICATIONS vs l'original ============================
(1) CLÉ API EN VARIABLE D'ENV (plus aucune clé en dur).
(2) CAP « 50 » -> « +N NOUVELLES par run » AVEC PAGINATION (la base grossit au fil des runs).
(3) SESSION HTTP ROBUSTE (retry/backoff + User-Agent) pour un run non surveillé.
(4) GARDE-FOU ARTISTE (is_by_artist) — CRITIQUE :
    `q=person:X` est une recherche PLEIN-TEXTE floue. En page 1 les top résultats sont
    les bons ; en paginant plus profond, Harvard renvoie des œuvres contenant juste 'van'
    (van Dyck, van Rijn…), du MAUVAIS artiste. L'original s'en sortait par chance (page 1
    seulement). On vérifie donc côté client, via le champ `people` du record, que
    l'artiste visé est bien le PEINTRE (rôle 'Artist') avant d'insérer. Corrige aussi la
    métadonnée `artist` qui était sinon fausse (artiste de la boucle stocké aveuglément).
====================================================================================

Contrat INCHANGÉ : aucun traitement d'image ici. Images de base = reproductions propres.

Env : HARVARD_API_KEY (requis), DB_NAME, IMG_DIR, MAX_NEW_PER_RUN (10), PAGE_SIZE (25),
      MAX_PAGES (20).
"""
import os
import time
import sqlite3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Config : tout surchargeable par variable d'env (secret Space / env local) ---
API_KEY = os.environ.get("HARVARD_API_KEY")
if not API_KEY:
    raise SystemExit("HARVARD_API_KEY manquante (variable d'env / secret). Aucune clé en dur.")

BASE_URL = "https://api.harvardartmuseums.org/object"
DB_NAME  = os.environ.get("DB_NAME", "harvard_clean.db")
IMG_DIR  = os.environ.get("IMG_DIR", "images_clean")
MAX_NEW_PER_RUN = int(os.environ.get("MAX_NEW_PER_RUN", "10"))
PAGE_SIZE       = int(os.environ.get("PAGE_SIZE", "25"))
MAX_PAGES       = int(os.environ.get("MAX_PAGES", "20"))

ARTISTS = [
    "Vincent van Gogh", "Claude Monet", "Pablo Picasso", "Edgar Degas",
    "Paul Cézanne", "Henri Matisse", "Pierre-Auguste Renoir", "Gustave Courbet",
    "Édouard Manet", "Georges Seurat", "Simone Martini", "John Singleton Copley",
]


def make_session():
    """Session HTTP réutilisable avec retry auto (MODIF 3) : ré-essaie tout seul sur
    429/5xx ou coupure de connexion, User-Agent navigateur (le serveur d'images refuse
    parfois les clients 'python'). Utile pour un run non surveillé."""
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "Mozilla/5.0 (M10-artvision enrichment)"})
    return s


def is_by_artist(item, artist):
    """Garde-fou (MODIF 4) : True seulement si `artist` est bien le PEINTRE de l'œuvre.

    On inspecte le champ `people` du record et on n'accepte que si l'artiste visé y
    figure avec le rôle 'Artist' (donc créateur — pas 'After', 'Sitter', 'Former
    Attribution'…). C'est ce qui filtre les faux positifs 'van' des pages profondes.
    """
    target = artist.strip().lower()
    for p in (item.get("people") or []):
        if (p.get("name") or "").strip().lower() == target \
                and (p.get("role") or "").strip().lower() == "artist":
            return True
    return False


def ensure_schema(conn):
    """Crée la table `artworks` si absente (schéma identique à l'original)."""
    conn.execute('''CREATE TABLE IF NOT EXISTS artworks (
                        id INTEGER PRIMARY KEY,
                        title TEXT, artist TEXT, url TEXT, dated TEXT,
                        datebegin INTEGER, dateend INTEGER, medium TEXT, technique TEXT,
                        classification TEXT, period TEXT, century TEXT, culture TEXT,
                        dimensions TEXT, description TEXT, labeltext TEXT,
                        department TEXT, creditline TEXT,
                        inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
                      )''')
    conn.commit()


def fetch_page(session, artist, page):
    """Renvoie les records d'UNE page (MODIF 2 : pagination). [] = fin de catalogue ou échec."""
    params = {"apikey": API_KEY, "hasimage": 1, "classification": "Paintings",
              "q": f"person:{artist}", "size": PAGE_SIZE, "page": page}
    try:
        resp = session.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("records", [])
    except Exception as e:
        print(f"  Échec requête {artist} p{page} : {e}")
        return []

# Modif / PoC8
def best_image_url(item):
    """URL image redimensionnée via IIIF (800px large) au lieu de full/full : réduit le
    payload et calme le serveur d'images Harvard (429). 800px suffit (CLIP resize -> 224).
    Fallback: primaryimageurl tel quel."""
    url = item.get("primaryimageurl") or ""
    return url.replace("/full/full/", "/full/800,/") if "/full/full/" in url else url


def save_artwork(cursor, session, item, artist):
    """Télécharge l'image + insère les métadonnées. Renvoie True si AJOUTÉE.

    `artist` a déjà été VÉRIFIÉ par is_by_artist en amont, donc la métadonnée stockée est
    correcte. `INSERT OR IGNORE` + test existing_ids garantissent l'idempotence.
    """
    obj_id  = item.get("id")
    img_url = item.get("primaryimageurl")          # métadonnée stockée (inchangée)
    dl_url  = best_image_url(item)                  # version 800px à télécharger (calme le 429)
    img_path = os.path.join(IMG_DIR, f"{obj_id}.jpg")
    try:
        time.sleep(2)  
        r = session.get(dl_url, timeout=30)
        r.raise_for_status()
        with open(img_path, "wb") as f:
            f.write(r.content)
        time.sleep(1)
    except Exception as e:
        print(f"  Téléchargement échoué pour {obj_id} : {e}")
        return False
    cursor.execute(
        '''INSERT OR IGNORE INTO artworks
           (id, title, artist, url, dated, datebegin, dateend, medium, technique,
            classification, period, century, culture, dimensions, description,
            labeltext, department, creditline)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (obj_id, item.get("title"), artist, img_url, item.get("dated"),
         item.get("datebegin"), item.get("dateend"), item.get("medium"),
         item.get("technique"), item.get("classification"), item.get("period"),
         item.get("century"), item.get("culture"), item.get("dimensions"),
         item.get("description"), item.get("labeltext"), item.get("department"),
         item.get("creditline")))
    return cursor.rowcount > 0


def main():
    """Ajoute jusqu'à MAX_NEW_PER_RUN œuvres nouvelles, du BON artiste, sans doublon."""
    session = make_session()
    os.makedirs(IMG_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    ensure_schema(conn)

    cursor.execute("SELECT id FROM artworks")
    existing_ids = {row[0] for row in cursor.fetchall()}
    print(f"{len(existing_ids)} œuvres déjà en base. Objectif : +{MAX_NEW_PER_RUN} max ce run.")

    added = 0
    for artist in ARTISTS:
        if added >= MAX_NEW_PER_RUN:
            break
        print(f"Extraction : {artist}")
        for page in range(1, MAX_PAGES + 1):
            if added >= MAX_NEW_PER_RUN:
                break
            records = fetch_page(session, artist, page)
            if not records:
                break
            for item in records:
                if added >= MAX_NEW_PER_RUN:
                    break
                obj_id = item.get("id")
                if obj_id in existing_ids or not item.get("primaryimageurl"):
                    continue
                if not is_by_artist(item, artist):
                    continue  # match flou -> mauvais artiste -> rejeté
                if save_artwork(cursor, session, item, artist):
                    existing_ids.add(obj_id)
                    added += 1
                    print(f"  [+{added}/{MAX_NEW_PER_RUN}] {item.get('title')}")
            conn.commit()

    conn.close()
    print(f"Terminé. {added} nouvelles œuvres ajoutées. Total en base : {len(existing_ids)}.")


if __name__ == "__main__":
    main()