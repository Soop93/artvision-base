import chromadb
import torch
import clip
from PIL import Image
import os
import sqlite3

DB_NAME = "harvard_clean.db"
COLLECTION_NAME = "artworks_collection"

#Client Chroma : métrique par défaut = L2 (euclidienne), sur embeddings CLIP bruts. La requête
#(artwork_core) fait pareil : cohérence index/requête garantie. Cosinus = piste (cf. README).
chroma_client = chromadb.PersistentClient(path="vector_store_images")
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#Jointure SQL entre les métadonnées et les images à vectoriser
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute('''
    SELECT v.variant_id, v.artwork_id, v.variant_type, v.file_path, a.title
    FROM artwork_variants v
    JOIN artworks a ON a.id = v.artwork_id
''')
rows = cursor.fetchall()
conn.close()

#Récupération des IDs et contrôle de la présence ou non des vecteurs
already_indexed = set(collection.get()['ids'])
to_process = [r for r in rows if r[0] not in already_indexed]
print(f"À vectoriser : {len(to_process)} / {len(rows)}")

indexed_count = 0

#Boucle sur chaque variante et contrôle de l'existence de l'image
for variant_id, artwork_id, variant_type, file_path, title in to_process:
    if not os.path.exists(file_path):
        print(f"Fichier manquant pour {variant_id}, ignoré.")
        continue

    #Vectorisation image par image (pas de batching) : simple et suffisant à cette échelle ;
    #batcher accélérerait sur une grosse base (piste).
    #Calcul de l'embedding CLIP
    try:
        image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).cpu().numpy().tolist()[0]
    except Exception as e:
        print(f"Échec vectorisation {variant_id} : {e}")
        continue

    #Insertion du vecteur dans ChromaDB
    collection.upsert(
        ids=[variant_id],
        embeddings=[embedding],
        metadatas=[{
            "artwork_id": artwork_id,
            "title": title,
            "variant_type": variant_type,
        }]
    )
    indexed_count += 1
    print(f"Indexé : {variant_id} ({variant_type}) — {title}")

print(f"\nTerminé. {indexed_count} nouvelles variantes indexées.")
