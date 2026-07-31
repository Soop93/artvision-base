---
title: ArtVision
emoji: 🖼️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ArtVision : reconnaissance d'œuvres d'art + guide de musée

Projet de fin de module du bootcamp Jedha – Lead Track IA, réalisé en trinôme. On prend une
photo d'un tableau : l'application isole la toile, l'identifie dans la collection du Harvard
Art Museums, puis un guide conversationnel la commente. L'interface, le code et la documentation
sont en français.

## Comment ça marche

Trois étapes s'enchaînent. D'abord l'isolation de la toile : GroundingDINO localise le tableau
dans la photo, SAM en extrait le masque précis, et une homographie le redresse de face — on
n'envoie à CLIP que la toile, sans le cadre ni le mur. Vient ensuite l'identification : CLIP
(ViT-B/32) vectorise la toile et on cherche l'œuvre la plus proche par distance L2 (la métrique
par défaut de ChromaDB, appliquée aux embeddings CLIP bruts) dans une base ChromaDB d'embeddings
de la collection Harvard. Enfin le guide : un RAG (LangChain /
LangGraph avec Ollama en local) répond aux questions sur l'œuvre à partir de fiches PDF.

## Architecture

```
Photo ──> Streamlit (UI, :7860) ──> FastAPI (:8000)
                                      ├─ isolation : GroundingDINO + SAM
                                      ├─ identif.  : CLIP + ChromaDB (vector_store_images)
                                      └─ guide     : RAG (Ollama + ChromaDB vector_store_text)
```

Le tout est déployé dans un seul Docker Space Hugging Face sur GPU T4, avec trois process
orchestrés par `start.sh` (Ollama, FastAPI, Streamlit). Les détails sont dans
[`docs/DEPLOY.md`](docs/DEPLOY.md).

## Où sont les données

Pas dans ce dépôt : il ne contient que le code. La base — SQLite Harvard, images, vector
stores — vit sur AWS S3, que le Space va chercher à son démarrage. C'est la séparation code /
données habituelle, et ça évite de gonfler l'historique git avec des binaires.

L'enrichissement de la base (récupérer de nouvelles œuvres via l'API Harvard, les vectoriser
avec le même CLIP que l'inférence, puis renvoyer le tout sur S3) tourne en local, et non sur
un serveur cloud : le serveur d'images de Harvard bloque les IP de datacenter et renvoie un
429 depuis AWS. Le contexte et la procédure sont dans [`docs/enrichment.md`](docs/enrichment.md).

Une contrainte domine tout le projet : l'index (enrichissement) et la requête (inférence)
doivent utiliser le même modèle CLIP, le même preprocessing et la même métrique. Une
divergence casserait la comparaison en silence, sans lever la moindre erreur.

## Stack

Streamlit, FastAPI, OpenAI CLIP (ViT-B/32), GroundingDINO, SAM, ChromaDB, Ollama (qwen2.5),
LangChain / LangGraph, Docker, AWS (S3 et IAM).

CLIP vient de l'implémentation OpenAI d'origine (`git+.../CLIP.git`) ; `transformers` (5.x) ne
sert qu'à GroundingDINO et SAM, pas à CLIP.

## Lancer en local

Il faut Python 3.12, Ollama installé, et la base présente en local (voir
[`docs/enrichment.md`](docs/enrichment.md) pour la reconstruire). Ensuite, dans deux terminaux :

```bash
pip install -r requirements.txt
uvicorn api:app --port 8000      # l'API
streamlit run app_streamlit.py   # l'interface
```

## Enrichir la base

Le dossier `enrichment/` contient la chaîne `extractAPI.py` → `augmentDB.py` →
`vectorisationDB.py`, qui tire la base de S3, va chercher de nouvelles œuvres chez Harvard, les
augmente, les vectorise et repousse le tout sur S3. Elle s'exécute en local, parce que Harvard
throttle les IP cloud, et se configure par variables d'environnement — la clé API Harvard et
les accès S3 ne sont jamais dans le code ni dans git. La procédure complète est dans
[`docs/enrichment.md`](docs/enrichment.md).

## Limites connues

CLIP mesure une similarité sémantique, pas l'identité d'une instance précise : deux œuvres très
proches (même artiste, même cadrage) peuvent être confondues, et aucun seuil ne rattrape
complètement ce cas (des pistes pour améliorer ce point sont listées plus bas). Par ailleurs,
une minorité des reproductions Harvard incluent le cadre, ce
qui crée un écart avec les requêtes où la toile est isolée — on a recadré à la main les rares
cas repérés. Enfin, l'enrichissement doit partir d'une IP résidentielle : le serveur d'images
de Harvard bloque les IP de datacenter (429 systématique depuis AWS, même avec une Elastic IP
dédiée), d'où son exécution en local décrite dans [`docs/enrichment.md`](docs/enrichment.md).

## Pistes d'amélioration

Fiabiliser l'identification, aujourd'hui limitée au voisin le plus proche en L2 sans score ni seuil :

- Passer tout le pipeline en normalisation L2 + similarité cosinus (index et requête), plus stable que la L2 brute.
- Déterminer un seuil de confiance en dessous duquel l'œuvre n'est pas confirmée.
- Ajouter un critère d'écart (gap) entre le 1er et le 2e candidat : quand le gap est faible (œuvres qui se ressemblent), proposer les deux plutôt que trancher.
- Calibrer ces deux seuils empiriquement sur un jeu de test dédié : images propres de `images_clean`, photos des mêmes œuvres trouvées en ligne (variations couleur/texture), et images hors base pour mesurer les faux positifs.

## Crédits

Données : [Harvard Art Museums API](https://harvardartmuseums.org/collections/api). Projet
réalisé en trinôme dans le cadre du bootcamp Jedha – Lead Track IA.
