#!/usr/bin/env bash
set -euo pipefail

# 0) Base tirée de S3 au boot (source de vérité, enrichie en local puis poussée sur S3).
#    Le Space ne bake plus la base dans l'image -> il la télécharge à chaque démarrage.
#    Accès en LECTURE SEULE via l'utilisateur IAM `artvision-space-reader`, dont les clés
#    sont fournies par les *secrets* du Space (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).
: "${S3_BUCKET:?S3_BUCKET manquant (à définir dans les variables du Space)}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-eu-west-3}}"

echo ">> pull de la base depuis s3://$S3_BUCKET/base/ ..."
aws s3 cp   "s3://$S3_BUCKET/base/harvard_clean.db"   harvard_clean.db
aws s3 sync "s3://$S3_BUCKET/base/images_clean"        images_clean
aws s3 sync "s3://$S3_BUCKET/base/vector_store_images" vector_store_images
aws s3 sync "s3://$S3_BUCKET/base/vector_store_text"   vector_store_text

# Garde-fou : base réellement présente. Mieux vaut échouer franchement au boot qu'un
# Space qui démarre sur une base vide et répond n'importe quoi en silence.
test -s harvard_clean.db                 || { echo "!! harvard_clean.db absent/vide après sync"; exit 1; }
test -s vector_store_images/chroma.sqlite3 || { echo "!! vector_store_images vide après sync (identification cassée)"; exit 1; }
test -s vector_store_text/chroma.sqlite3   || { echo "!! vector_store_text vide après sync (guide RAG cassé)"; exit 1; }
echo ">> base prête : $(ls images_clean | wc -l) images_clean."

# 1) Ollama en arrière-plan (modèles déjà bakés dans l'image)
echo ">> démarrage d'Ollama..."
ollama serve &
until curl -sf http://localhost:11434/api/tags >/dev/null; do sleep 1; done
echo ">> Ollama prêt."

# 2) API FastAPI en arrière-plan (interne, jamais exposée publiquement)
echo ">> démarrage de l'API..."
uvicorn api:app --host 127.0.0.1 --port 8000 &
until curl -sf http://127.0.0.1:8000/docs >/dev/null; do sleep 1; done
echo ">> API prête."

# 3) Streamlit au premier plan -> port public 7860 du Space
echo ">> démarrage de Streamlit..."
# CORS + XSRF désactivés pour l'iframe du Space HF (l'app tourne embarquée dans huggingface.co).
#   - CORS (Cross-Origin Resource Sharing) : le navigateur bloque les requêtes entre origines
#     différentes ; l'iframe HF et le serveur Streamlit n'ont pas la même origine.
#   - XSRF (Cross-Site Request Forgery) : jeton anti-forge de Streamlit qui casse l'upload de
#     fichiers et la caméra en contexte cross-origin.
# Relâchement assumé : sans ça upload/caméra ne marchent pas, et l'API sensible reste interne (127.0.0.1).
streamlit run app_streamlit.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false