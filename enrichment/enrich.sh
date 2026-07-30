#!/usr/bin/env bash
# enrich.sh — orchestrateur d'enrichissement (exécuté en local).
# Pull base S3 -> extract -> augment -> vectorise -> push base S3.
set -euo pipefail
cd "$(dirname "$0")"            # ~/artvision : code + base au même endroit

# Charge .env (HARVARD_API_KEY, S3_BUCKET, MAX_NEW_PER_RUN…)
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# [0] Base à jour depuis S3 (source de vérité)
if [ -n "${S3_BUCKET:-}" ]; then
  echo "[0/4] Pull base <- s3://$S3_BUCKET/base/"
  aws s3 cp   "s3://$S3_BUCKET/base/harvard_clean.db" harvard_clean.db
  aws s3 sync "s3://$S3_BUCKET/base/images_clean"        images_clean
  aws s3 sync "s3://$S3_BUCKET/base/vector_store_images" vector_store_images
fi

echo "[1/4] Extraction Harvard (+${MAX_NEW_PER_RUN:-10} max)"
python3 extractAPI.py
echo "[2/4] Augmentation"
python3 augmentDB.py
echo "[3/4] Vectorisation CLIP"
python3 vectorisationDB.py

# [4] Base enrichie -> S3
if [ -n "${S3_BUCKET:-}" ]; then
  echo "[4/4] Push base -> s3://$S3_BUCKET/base/"
  aws s3 cp   harvard_clean.db      "s3://$S3_BUCKET/base/harvard_clean.db"
  aws s3 sync images_clean          "s3://$S3_BUCKET/base/images_clean"        --delete
  aws s3 sync vector_store_images   "s3://$S3_BUCKET/base/vector_store_images" --delete
  date -u +%Y-%m-%dT%H:%M:%SZ | aws s3 cp - "s3://$S3_BUCKET/base/READY"
fi
echo "Enrichissement terminé."