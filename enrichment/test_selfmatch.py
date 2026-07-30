"""test_selfmatch.py — preuve de cohérence de l'enrichissement.

Ré-embedde l'image d'une œuvre fraîchement ajoutée avec EXACTEMENT le même contrat que
vectorisationDB.py (clip ViT-B/32, encode_image BRUT, sans normalisation) et vérifie
qu'elle se retrouve elle-même en rang 1 à distance ~0 dans vector_store_images. C'est la
preuve que le nouveau contenu vit dans le même espace que la requête du Space
(artwork_core.search_artwork) -> comparaison index/requête intacte.
"""
import sys, glob, os, chromadb, torch, clip
from PIL import Image

STORE, COLL, IMGDIR = "vector_store_images", "artworks_collection", "images_clean"

# ID à tester : passé en argument, sinon on prend la 1re œuvre présente dans images_clean/
# (portable : plus d'ID en dur qui casse si l'œuvre n'existe plus).
if len(sys.argv) > 1:
    ARTWORK_ID = int(sys.argv[1])
else:
    imgs = sorted(glob.glob(f"{IMGDIR}/*.jpg"))
    if not imgs:
        raise SystemExit(f"Aucune image dans {IMGDIR}/ — base vide ?")
    ARTWORK_ID = int(os.path.splitext(os.path.basename(imgs[0]))[0])
    print(f"Aucun ID fourni : test sur l'œuvre {ARTWORK_ID} (1re de {IMGDIR}/).")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

img = Image.open(f"{IMGDIR}/{ARTWORK_ID}.jpg")
with torch.no_grad():
    emb = model.encode_image(preprocess(img).unsqueeze(0).to(device)).cpu().numpy().tolist()[0]

col = chromadb.PersistentClient(path=STORE).get_collection(COLL)
res = col.query(query_embeddings=[emb], n_results=5, include=["metadatas", "distances"])

print(f"Requête = image de l'œuvre {ARTWORK_ID}")
for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
    flag = "  <-- self" if meta.get("artwork_id") == ARTWORK_ID else ""
    print(f"  dist={dist:.4f} | id={meta.get('artwork_id')} | {meta.get('title')}{flag}")

top = res["metadatas"][0][0]
assert top.get("artwork_id") == ARTWORK_ID, "ÉCHEC : l'œuvre ne ressort pas en rang 1"
print(f"OK — rang 1, distance {res['distances'][0][0]:.4f} -> cohérence prouvée.")