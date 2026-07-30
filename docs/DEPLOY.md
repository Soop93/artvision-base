# Déploiement Hugging Face — Récapitulatif (PoC8 → artvision-space)

**Date :** 2026-07-28
**Contexte :** bootcamp Jedha Lead Track IA — déploiement de PoC8 (reconnaissance d'œuvres + guide de musée).

> Mise à jour (2026-07-30) — ce document décrit le premier déploiement, celui où la base était
> embarquée dans l'image et où le Space ne dépendait pas d'AWS. Depuis, deux choses ont changé :
> la base ne vit plus dans l'image mais sur S3, que le Space va chercher à son démarrage (voir
> start.sh et docs/enrichment.md), et le Space de référence est maintenant
> guillaumegab93/artvision-aws. Tout le reste du récit ci-dessous — montage de l'image,
> orchestration des trois process, journal des erreurs de build — reste valable.

---

## 0. TL;DR
PoC8 (Streamlit + FastAPI + CLIP/GroundingDINO/SAM + guide Ollama/RAG) déployé sur un unique Docker Space Hugging Face GPU. ⚠️ État initial : base bundlée dans l'image, sans AWS ; depuis, la base est tirée de S3 au boot (voir bandeau ci-dessus et `docs/enrichment.md`).

- **URL :** https://huggingface.co/spaces/guillaumegab93/artvision (Public)
- **Hardware :** Nvidia T4 small (~0,40 $/h) — compte HF **PRO**.
- **Aucune modification du code applicatif** : seulement de l'empaquetage + 1 épinglage de dépendance.

---

## 1. Architecture cible
Un seul conteneur, un seul port public (7860), **3 process** orchestrés par `start.sh` :

| Process | Adresse | Rôle |
|---|---|---|
| Ollama | localhost:11434 | qwen2.5:7b-instruct-q5_K_M (chat) + nomic-embed-text (embeddings RAG) |
| FastAPI (uvicorn) | 127.0.0.1:8000 (interne) | `api.py` |
| Streamlit | 0.0.0.0:7860 (public) | `app_streamlit.py` |

Les `localhost` en dur du code fonctionnent tels quels car tout est colocalisé → **zéro refactor réseau**. Base (`harvard_clean.db` + `vector_store_images` + `vector_store_text` + `images_clean`) bundlée dans le repo (stockage inclus au PRO). ⚠️ Obsolète : la base vit désormais sur S3, tirée au boot par `start.sh`. HTTPS auto → `st.camera_input` OK.

---

## 2. Le dossier de déploiement `artvision-space`
Copie de PoC8 nettoyée (`robocopy ... /XD images_augmented __pycache__ /COPY:DT /R:1 /W:1` — flags OneDrive).

**Inclus (runtime) :** `api.py`, `app_streamlit.py`, `artwork_core.py`, `guide_core.py`, `harvard_clean.db`, `images_clean/` (46 img), `vector_store_images/`, `vector_store_text/`, `requirements.txt`
**+ gardés pour reproductibilité :** `augmentDB.py`, `extractAPI.py`, `vectorisationDB.py`, `ingest_pdfs.py`, `pdfs/`

**Exclu du dossier (vs PoC8) :**
- `images_augmented/` (185 Mo) — clones ayant servi à remplir la base vectorielle, inutiles au runtime.
- `__pycache__/` — cache Python.

---

## 3. Fichiers AJOUTÉS (nouveaux, absents de PoC8)

### `Dockerfile`
Recette de l'image. Étapes clés :
- Base **`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`** (torch 2.6 préinstallé CUDA → pas de réinstall qui casserait le GPU).
- apt : `git` (install CLIP), `curl` (Ollama + health-checks), `libglib2.0-0` (opencv-headless).
- Install Ollama (binaire système).
- Utilisateur non-root **`user`** (uid 1000, exigé par HF Spaces).
- `pip install -r requirements.txt`.
- **BAKE 1** : télécharge DINO + SAM + CLIP dans l'image au build → présents à chaque boot (disque du Space non-persistant).
- **BAKE 2** : lance Ollama et `pull` qwen2.5:7b-instruct-q5_K_M + nomic-embed-text dans l'image.
- `chown -R user /home/user/app` (repassage root ponctuel) avant `COPY` → écritures runtime possibles.
- `COPY` code + base ; `sed` sur `start.sh` (neutralise les `\r` Windows) ; retour `USER user`.
- `EXPOSE 7860` ; `CMD ["bash", "start.sh"]`.

### `start.sh`
Orchestration des 3 process au démarrage (voir §6).

### `README.md`
Header YAML HF **obligatoire** : `sdk: docker`, `app_port: 7860`, `title`, `emoji` + description.

### `.dockerignore`
Exclut de l'**IMAGE** (pas du repo) : `.git`, `__pycache__`, build scripts, `pdfs/`, `README`/`Dockerfile`/`.dockerignore`. → image plus légère ; le repo garde tout.

### `.gitattributes`
Créé par `git lfs track` : `*.jpg` `*.jpeg` `*.png` `*.db` `*.sqlite3` `*.bin` → stockés en **Git-LFS** (hors historique git).

---

## 4. Fichiers MODIFIÉS depuis PoC8

### `requirements.txt`
- `transformers>=4.44.0`  →  **`transformers==5.14.1`** (épinglé).
- **Raison :** `artwork_core.py` utilise l'API GroundingDINO `text=[[prompt]]` de transformers **5.x**. Épingler garantit la repro et évite qu'un futur build casse. transformers 5.x exige **torch ≥ 2.5** (d'où la base torch 2.6).

### Code applicatif : **AUCUNE modification**
`api.py`, `app_streamlit.py`, `artwork_core.py`, `guide_core.py` sont **inchangés**.
**Conséquence :** la **cohérence index/requête CLIP est intacte** (même code d'embedding qu'à la construction de la base). Le déploiement n'a touché qu'à l'emballage, jamais à la logique.

---

## 5. Journal des erreurs de build & correctifs (chronologique)
Trois erreurs, trois causes racines, trois fixes :

**1. `ImportError: cannot import name 'DTensor' from 'torch.distributed.tensor'`** (bake vision)
Cause : transformers (dernière version) trop *récent* pour torch 2.4.1 de la base initiale (le `DTensor` public existe depuis torch ≥ 2.5).
Fix intermédiaire : pin `transformers==4.45.2` → a provoqué l'erreur #3.

**2. `sed: couldn't open temporary file /home/user/app/... : Permission denied`** (étape sed)
Cause : `WORKDIR` crée `/home/user/app` appartenant à **root** ; `user` ne peut pas y écrire.
Fix : `USER root ; chown -R user:user /home/user/app` avant `COPY`, puis retour `USER user`. Corrige aussi les écritures **runtime** (journal SQLite, WAL Chroma, cache Streamlit) qui auraient buté sur le même mur.

**3. `TypeError: TextEncodeInput must be Union[...]`** (propose côté API ; `JSONDecodeError` côté Streamlit)
Cause : `transformers==4.45.2` trop *vieux* pour l'API `text=[[prompt]]` du code (API 5.x).
Fix final : **`transformers==5.14.1` + base `torch 2.6`** (`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`). Aligne torch et transformers sur ce que le code attend.

**Leçon :** la version torch (figée par l'image de base) et transformers doivent être choisies **ensemble**. Laisser flotter un seul des deux = rupture d'API silencieuse.

---

## 6. Ce que fait `start.sh` (détail)
Chef d'orchestre du conteneur : lance les 3 process dans l'ordre imposé par leurs dépendances (Streamlit → API → Ollama) et maintient le conteneur en vie.

1. `set -euo pipefail` — garde-fous : arrêt à la 1re erreur, pas de variable non définie masquée, pipes stricts.
2. **Ollama** : `ollama serve &` (arrière-plan) puis boucle `until curl .../api/tags` = **attente active** jusqu'à ce qu'il réponde (modèles bakés → quelques secondes).
3. **FastAPI** : `uvicorn api:app --host 127.0.0.1 --port 8000 &` (interne, jamais exposé), puis attente de `/docs`.
4. **Streamlit** : `streamlit run ... --server.port 7860 --server.address 0.0.0.0 --headless --enableCORS false --enableXsrfProtection false`, **au premier plan** → tient le conteneur en vie ; seul port public ; les flags CORS/XSRF permettent upload + caméra dans l'iframe HF.

---

## 7. Décisions clés
- **Bake des modèles au build** : boots rapides et déterministes malgré le disque non-persistant du Space (pattern stop/start).
- **Un seul conteneur** : les `localhost` du code restent valides ; surface d'attaque réduite (API interne sur 127.0.0.1).
- **Validation sur CPU gratuit avant GPU** : socle vert sans brûler de GPU.
- **T4 GPU pour la latence** (Ollama + DINO/SAM/CLIP accélérés). Facturé en continu → **PAUSE après usage**.

---

## 8. État actuel
- Chaîne complète validée **end-to-end** : isolation (DINO+SAM) → identification (CLIP) → guide (RAG + LLM).
- Space **Public**, sur **T4**.
- ⚠️ **PAUSER le Space après chaque session** (Settings → Pause the Space) — sinon facturation continue ~0,40 $/h.

---

## 9. Prochaines étapes possibles
- Mesurer les **latences réelles** sur T4 (isolation, identification, guide).
- Optionnel « plomberie » : hybride HF + AWS (persistance vector store, S3 pour images), GitHub Action de déploiement auto.
- Optimisation : persistance des modèles / réduction de la taille d'image.

---

## 10. Commandes utiles
- **Déployer une modif** : `git add -A && git commit -m "..." && git push space main` (build auto sur HF).
- **Mettre en pause** : Space → Settings → *Pause the Space*.
- **Relancer** : Space → *Restart* (boot ~1-2 min, modèles bakés).
