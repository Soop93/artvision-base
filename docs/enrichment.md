# Enrichir la base

Ce document explique comment ajouter de nouvelles œuvres à la base ArtVision. Le principe est
simple : on récupère des œuvres via l'API Harvard, on les vectorise avec CLIP, et on repousse
la base enrichie sur S3, d'où le Space la tire à son démarrage.

```
[pull S3] -> extractAPI.py -> augmentDB.py -> vectorisationDB.py -> [push S3] -> Restart du Space
```

Comme partout dans le projet, le code est sur GitHub, les données sur S3, et aucun secret ne
traîne dans git.

## Pourquoi ça tourne en local

Le pipeline devait au départ tourner tout seul sur une EC2, avec un cron hebdomadaire. On a dû
abandonner cette idée : le serveur d'images de Harvard (ids.lib.harvard.edu) bloque les IP de
datacenter et renvoie un 429 dès la première image quand on télécharge depuis AWS — même avec
une Elastic IP fraîche, ce qui montre que le blocage porte sur la plage AWS et pas sur une
adresse en particulier. Les mêmes images se téléchargent sans souci depuis une IP
résidentielle. On enrichit donc depuis un poste local. Toute la démarche EC2 est conservée
dans [archive/config_ec2.md](archive/config_ec2.md) pour référence.

## Ce qu'il faut avant de commencer

Un environnement Python avec les mêmes dépendances que l'index (voir
`enrichment/requirements-enrich.txt`) : torch et torchvision (la version CPU suffit), OpenAI
CLIP, chromadb, opencv-python, albumentations et requests. Pour vérifier que tout est en place :

```powershell
python -c "import torch, torchvision, clip, chromadb, cv2, albumentations; print('OK', torch.__version__); print('ViT-B/32', 'ViT-B/32' in clip.available_models())"
```

Il faut aussi la clé API Harvard dans l'environnement, jamais dans git :

```powershell
$env:HARVARD_API_KEY = "<clé>"
```

Et enfin un accès S3 en écriture. En local, on n'a pas le rôle IAM de l'instance : on passe par
un utilisateur IAM dédié, artvision-local-enrich, qui porte la policy artvision-base-write
(GetObject, PutObject, DeleteObject et ListBucket sur le bucket artvision-base), configuré avec
`aws configure`. Pour vérifier qu'il lit bien le bucket :

```powershell
aws s3 ls s3://artvision-base/base/ --region eu-west-3
```

## La procédure

Tout se lance depuis la racine du dépôt, puisque c'est là que vit la base, les scripts étant
dans `enrichment/`. Les commandes ci-dessous sont en PowerShell : on enrichit depuis un poste
Windows local. `enrichment/enrich.sh` fait exactement la même chaîne (pull S3, extract, augment,
vectorise, push S3) en bash, pour un shell Unix, c'est la même procédure, pas une seconde
source de vérité.

```powershell
$REGION = "eu-west-3"
$env:HARVARD_API_KEY = "<clé>"
$env:MAX_NEW_PER_RUN = "10"

# On tire la base à jour depuis S3 (sans --delete, on ajoute par-dessus)
aws s3 cp   "s3://artvision-base/base/harvard_clean.db" harvard_clean.db            --region $REGION
aws s3 sync "s3://artvision-base/base/images_clean"        images_clean             --region $REGION
aws s3 sync "s3://artvision-base/base/vector_store_images" vector_store_images      --region $REGION

# Extraction, augmentation, vectorisation
python enrichment\extractAPI.py
python enrichment\augmentDB.py
python enrichment\vectorisationDB.py

# On repousse la base enrichie (--delete uniquement sur ce qu'on gère)
aws s3 cp   harvard_clean.db      "s3://artvision-base/base/harvard_clean.db"                  --region $REGION
aws s3 sync images_clean          "s3://artvision-base/base/images_clean"        --delete      --region $REGION
aws s3 sync vector_store_images   "s3://artvision-base/base/vector_store_images" --delete      --region $REGION
[DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ") | aws s3 cp - "s3://artvision-base/base/READY" --region $REGION
```

Une fois le push terminé, on redémarre le Space artvision-aws pour qu'il retire la base à jour —
il ne le fait qu'au démarrage. Dans les logs, on doit voir passer `>> base prête : N images_clean.`

Un point de prudence : ne jamais synchroniser `vector_store_text/` ni `pdfs/` avec `--delete`.
Ils appartiennent au guide RAG et ne sont pas gérés par l'enrichissement ; le `--delete` ne doit
porter que sur la base, les images et le vector store des images.

## La cohérence, le point à ne pas rater

La vectorisation doit utiliser exactement le même CLIP, le même preprocessing et la même
métrique que l'inférence du Space (OpenAI CLIP ViT-B/32, preprocess par défaut, embeddings bruts
et distance L2 : la métrique ChromaDB par défaut). C'est déjà ce
que fait `vectorisationDB.py`, donc il ne faut surtout pas y toucher. Une divergence casserait
la comparaison en silence, sans lever la moindre erreur — c'est la contrainte numéro un du
projet.

## Ajouter un nouvel artiste

Il suffit d'éditer la liste ARTISTS dans `enrichment/extractAPI.py`. Trois précautions à garder
en tête. Rester sur des œuvres du domaine public, car les 20e et 21e siècles sont restreints
côté Harvard pour des raisons de droits. Respecter l'orthographe exacte du nom, parce que le
garde-fou is_by_artist le compare à l'identique au champ people.name avec le rôle Artist, et
qu'une simple faute donne zéro œuvre insérée. Et placer le nouveau nom en tête de liste, pour
l'atteindre avant les artistes déjà épuisés.

## Après usage

Penser à révoquer la clé d'accès de artvision-local-enrich : ce sont des clés long-terme qui
restent sur le poste. L'accès en lecture du Space (artvision-space-reader), lui, reste en place.
Côté coût, l'enrichissement ne coûte rien : poste local et stockage S3 négligeable. Quant au
dossier images_augmented, il est régénéré localement et n'est ni poussé sur S3 ni lu par le
Space.
