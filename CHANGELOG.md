# Changelog

Évolutions notables du projet, de la plus récente à la plus ancienne.

## [0.2.0] - 2026-07-30
### Ajouté
- Le Space tire la base depuis AWS S3 au démarrage (`start.sh`, étape 0) au lieu de la
  bundler dans l'image, ce qui pousse la séparation code / données jusqu'au bout.
- `docs/enrichment.md` : procédure d'enrichissement en local et diagnostic du throttle du
  serveur d'images Harvard.
- Nouveaux artistes 19e / domaine public dans l'extraction (Simone Martini, John Singleton
  Copley).

### Modifié
- Enrichissement déplacé sur poste local (pull S3, extraction Harvard, augmentation,
  vectorisation CLIP, push S3), au lieu d'une instance EC2 autonome.

### Corrigé
- Crop manuel : `order_points` dégénérait sur un quadrilatère en losange (~45°) — deux coins
  partageant la même somme et la même différence, ce qui effondrait l'homographie (aplat uni).
  Remplacé par un tri angulaire autour du centroïde, robuste à toute rotation.

### Abandonné
- Automatisation de l'enrichissement sur EC2 (cron hebdomadaire) : le serveur d'images de
  Harvard bloque les IP de datacenter (429 systématique depuis AWS, y compris avec une Elastic
  IP dédiée). Démarche conservée pour trace dans `docs/archive/config_ec2.md`. EC2 retiré de
  l'architecture.

## [0.1.0] - 2026-07-29
### Ajouté
- Reconnaissance d'œuvres : isolation (GroundingDINO + SAM) puis identification
  (CLIP ViT-B/32 + ChromaDB, distance L2).
- Guide conversationnel (RAG : LangChain / LangGraph + Ollama).
- Déploiement Docker sur Hugging Face Space (GPU T4).
- Pipeline d'enrichissement de la base (`enrichment/`) : extraction Harvard (avec garde-fou
  artiste + pagination), augmentation, vectorisation CLIP.
- Séparation code / données : code sur GitHub, données sur AWS S3.
- Hygiène de dépôt : README détaillé, LICENSE (MIT), CHANGELOG, dossier `docs/`.
