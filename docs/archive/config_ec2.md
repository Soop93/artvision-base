> # Approche abandonnée — conservée pour trace
>
> Ce document décrit l'enrichissement automatique sur EC2 (cron hebdomadaire). On l'a
> finalement abandonné : le serveur d'images de Harvard (ids.lib.harvard.edu) throttle les
> IP de datacenter et renvoie un 429 dès la première image depuis AWS, y compris avec une
> Elastic IP dédiée — le blocage se situe au niveau de la plage AWS, on l'a vérifié. Les
> mêmes images se téléchargent normalement depuis une IP résidentielle.
>
> L'enrichissement tourne maintenant en local ; la procédure à jour est dans
> [../enrichment.md](../enrichment.md).
>
> Ce qui suit est la configuration EC2 d'origine, laissée telle quelle : création de
> l'instance, swap mémoire, rôle IAM, environnement Python, orchestration par enrich.sh et
> mise en place du cron. On la garde comme mémo de la démarche testée, utile si le diagnostic
> du 429 est abordé en soutenance.

---

# Configuration de l'EC2 d'enrichissement

L'idée de départ était simple : une petite instance qui, une fois par semaine, va chercher
de nouvelles œuvres chez Harvard, les vectorise avec CLIP et repousse la base à jour sur S3.
Le Space n'a alors qu'à retirer cette base à son démarrage. Tout le code vient de GitHub,
toutes les données vivent sur S3, et aucun secret ne traîne en clair sur la machine :
l'accès à S3 passe par le rôle IAM de l'instance, et la clé Harvard est saisie à la main
dans un fichier .env local. Voici comment l'instance était montée, si jamais il fallait la
recréer.

## L'instance

Une t3.micro sous Amazon Linux 2023, nommée artvision-enrich, avec un disque de 20 Gio (gp3),
dans la région de Paris (eu-west-3, zone c). Le t3.micro reste dans le free tier, ce qui
suffit largement pour un run hebdomadaire. Le Security Group n'ouvre que le port SSH (22), et
uniquement depuis mon IP.

Le point important, c'est le profil IAM attaché à l'instance : artvision-ec2-enrich, qui porte
la policy artvision-base-write (GetObject, PutObject, DeleteObject et ListBucket sur le bucket
artvision-base). C'est lui qui autorise la machine à lire et écrire sur S3 sans qu'aucune clé
AWS ne soit stockée dessus.

L'IP publique change à chaque stop/start (pas d'Elastic IP), il faut donc la relire dans la
console avant de se reconnecter :

```bash
ssh -i "chemin/vers/artvision-key.pem" ec2-user@<IP_PUBLIQUE>
```

## Le swap

La t3.micro n'a qu'1 Go de RAM, ce qui ne suffit pas à charger torch et CLIP. On ajoute donc
4 Go de swap, rendus permanents pour survivre à un reboot :

```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=4096 status=progress
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h   # doit montrer ~4 Go de swap
```

## Les paquets et le code

aws-cli et Python 3.9 sont déjà présents sur l'AMI. Il ne manque que git et la librairie
graphique dont OpenCV a besoin :

```bash
sudo dnf install -y git mesa-libGL
```

Le code vient du dépôt public (pas d'authentification), les données non — elles sont sur S3 :

```bash
cd ~
git clone https://github.com/Soop93/artvision-base.git artvision
```

## L'environnement Python

Un venv isolé, avec torch et torchvision en version CPU (inutile d'embarquer CUDA ici), puis
les dépendances d'enrichissement, et enfin CLIP installé sans ses dépendances pour ne pas
écraser le torch CPU :

```bash
cd ~/artvision/enrichment
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-enrich.txt
pip install git+https://github.com/openai/CLIP.git --no-deps
```

Un petit import pour vérifier que tout tient :

```bash
python -c "import torch, torchvision, clip, chromadb, cv2, albumentations; print('OK', torch.__version__)"
```

## La configuration

Le fichier ~/artvision/enrichment/.env, créé directement sur l'instance et jamais versionné :

```
S3_BUCKET=artvision-base
AWS_DEFAULT_REGION=eu-west-3
MAX_NEW_PER_RUN=10
HARVARD_API_KEY=<saisie à la main, jamais collée dans un fichier suivi>
```

La clé Harvard est entrée via read, pour ne jamais la laisser dans l'historique shell ni dans
un objet S3.

## Ce que fait enrich.sh

Le script enchaîne toute la chaîne : il tire la base depuis S3, lance l'extraction des
nouvelles œuvres, l'augmentation, la vectorisation, puis repousse la base sur S3 en terminant
par un marqueur READY — écrit en dernier, donc présent seulement si le run est allé au bout.
La vectorisation utilise exactement le même CLIP et le même preprocessing que l'inférence du
Space ; c'est la seule chose à ne surtout pas toucher.

## Le cron

Le service cron n'est pas actif par défaut sur Amazon Linux 2023, il faut l'installer et
l'activer :

```bash
sudo dnf install -y cronie
sudo systemctl enable --now crond
```

L'entrée elle-même lance enrich.sh chaque lundi à 3h :

```
0 3 * * 1 PATH=/home/ec2-user/venv/bin:$PATH bash /home/ec2-user/artvision/enrichment/enrich.sh >> /home/ec2-user/enrich.log 2>&1
```

On force le PATH sur le venv parce que le cron n'hérite pas de l'environnement interactif —
sans ça, python3 serait le Python système, sans torch ni CLIP. La sortie complète part dans
~/enrich.log. À noter que l'instance est en UTC : 3h du matin ici, c'est 5h à Paris l'été.

## Quelques pièges rencontrés

Comme l'instance tourne en UTC, toutes les heures (cron, timestamps S3) sont à convertir pour
l'heure de Paris. L'IP publique est éphémère, à relire dans la console à chaque redémarrage.
Le dossier images_augmented persiste entre les runs sur le disque de l'instance, ce qui permet
à l'augmentation de rester incrémentale et de sauter les variantes déjà générées. Côté coût,
le t3.micro reste dans le free tier même allumé en continu, cron hebdo compris.

## Ce qui avait été validé

Un run manuel puis un run en conditions cron ont exécuté toute la chaîne sans erreur : pull,
extraction, augmentation, vectorisation, push, avec le marqueur READY écrit sur S3. chromadb
rouvrait bien le vector store tiré de S3 (compatibilité de format confirmée), et la cohérence
index/requête avait été prouvée par un self-match : une œuvre ré-embeddée ressortait en rang 1,
distance 0. Toute la plomberie fonctionnait — c'est uniquement le throttle du serveur d'images
de Harvard sur les IP AWS qui a fini par rendre l'approche inexploitable.
