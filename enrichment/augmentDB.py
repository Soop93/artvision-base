import os
import sqlite3
import cv2
import random
import albumentations as A

DB_NAME = "harvard_clean.db"
IMG_DIR = "images_clean"
AUG_DIR = "images_augmented"
N_CLONES_PER_ARTWORK = 3
SEED = 42   # graine fixe : augmentation reproductible d'un run à l'autre (sinon des embeddings différents à chaque enrichissement)

#Pipeline de degradation Albumentations avec des probabilités d'applications.
#seed=SEED : pipeline de dégradation déterministe (reproductibilité).
DEGRADATION_PIPELINE = A.Compose([
    A.OneOf([
        A.MotionBlur(blur_limit=7, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    ], p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.Perspective(scale=(0.05, 0.15), p=0.6),
    A.ImageCompression(quality_range=(30, 70), p=0.8),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.5),
], seed=SEED)

#Recadrage aléatoire 75-95 % : simule une photo cadrée un peu large ou un peu serrée sur la
#toile. Choix de robustesse ; à l'inférence la toile est isolée entière et redressée, donc on
#reste proche du domaine tant que le crop ne coupe pas le motif central.
def random_crop(image, min_ratio=0.75, max_ratio=0.95):
    h, w = image.shape[:2]
    ratio = random.uniform(min_ratio, max_ratio)
    new_h, new_w = int(h * ratio), int(w * ratio)
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    return image[top:top + new_h, left:left + new_w]


def main():
    random.seed(SEED)   # contrôle l'aléa de random_crop (random.uniform/randint)

    os.makedirs(AUG_DIR, exist_ok=True)
    #Création de la table Artwork_variants
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS artwork_variants (
                        variant_id TEXT PRIMARY KEY,
                        artwork_id INTEGER NOT NULL,
                        variant_type TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        FOREIGN KEY (artwork_id) REFERENCES artworks(id)
                      )''')
    #Récupération des ID des oeuvres
    cursor.execute("SELECT id, title FROM artworks")
    rows = cursor.fetchall()
    print(f"Génération de clones pour {len(rows)} œuvres ({N_CLONES_PER_ARTWORK} clones/œuvre)...")

    total_generated = 0
    #Contrôle de la présence de l'oeuvre
    for artwork_id, title in rows:
        src_path = os.path.join(IMG_DIR, f"{artwork_id}.jpg")
        if not os.path.exists(src_path):
            print(f"Image manquante pour {artwork_id} ({title}), ignorée.")
            continue

        image = cv2.imread(src_path)              # OpenCV lit en BGR
        if image is None:
            print(f"Lecture impossible pour {artwork_id}, ignorée.")
            continue
        #Passage en RGB : tout le pipeline d'augmentation travaille en RGB, cohérent avec la
        #vectorisation (PIL/CLIP lit en RGB). Retour en BGR juste avant cv2.imwrite (qui attend du BGR).
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3, "image attendue en 3 canaux"

        cursor.execute(
            "INSERT OR IGNORE INTO artwork_variants (variant_id, artwork_id, variant_type, file_path) "
            "VALUES (?, ?, 'original', ?)",
            (f"{artwork_id}_orig", artwork_id, src_path)
        )

        for n in range(N_CLONES_PER_ARTWORK):
            variant_id = f"{artwork_id}_clone{n}"
            clone_path = os.path.join(AUG_DIR, f"{variant_id}.jpg")

            cursor.execute("SELECT 1 FROM artwork_variants WHERE variant_id = ?", (variant_id,))
            if cursor.fetchone() and os.path.exists(clone_path):
                continue
            #Application du recadrage auto et du pipeline de dégradation
            try:
                cropped = random_crop(image_rgb)
                degraded = DEGRADATION_PIPELINE(image=cropped)["image"]
                #Retour BGR pour l'écriture disque seulement ; relu BGR->RGB à la vectorisation.
                degraded_bgr = cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR)
                cv2.imwrite(clone_path, degraded_bgr)
            except Exception as e:
                print(f"Échec génération {variant_id} : {e}")
                continue

            cursor.execute(
                "INSERT OR IGNORE INTO artwork_variants (variant_id, artwork_id, variant_type, file_path) "
                "VALUES (?, ?, 'augmented', ?)",
                (variant_id, artwork_id, clone_path)
            )
            total_generated += 1

        conn.commit()
        print(f"{title} ({artwork_id}) : clones générés")

    conn.close()
    print(f"\nTerminé. {total_generated} clones générés.")


if __name__ == "__main__":
    main()
