# Base PyTorch + CUDA : torch/torchvision déjà compilés pour GPU.
# On évite ainsi de réinstaller torch (et de risquer une build CPU qui casserait le GPU).
# torch 2.6 (>=2.5) : requis pour transformers 5.x qui importe torch.distributed.tensor.DTensor
# (absent de torch 2.4). Le code applicatif utilise l'API GroundingDINO texte de transformers 5.x.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# --- Dépendances système ---
#  git        -> requis pour "pip install git+https://.../CLIP.git"
#  curl       -> installe Ollama + sert aux boucles d'attente
#  libglib2.0 -> requis par opencv-python-headless à l'import (libgthread)
#  awscli     -> `aws s3 sync` de la base depuis S3 au boot (start.sh, étape 0)
# DEBIAN_FRONTEND=noninteractive : awscli tire tzdata qui, sinon, demande la zone
# géographique en interactif et bloque le build (pas de stdin en build HF). -> UTC par défaut.
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git curl libglib2.0-0 awscli && \
    rm -rf /var/lib/apt/lists/*

# --- Ollama (binaire système, nécessite root) ---
RUN curl -fsSL https://ollama.com/install.sh | sh

# --- Utilisateur non-root (uid 1000) : recommandé par HF Spaces ---
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

# --- Dépendances Python (en root, dans l'env système de l'image) ---
# torch/torchvision de requirements.txt sont "déjà satisfaits" -> non réinstallés.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /home/user/app
USER user

# --- BAKE 1 : modèles vision (DINO + SAM + CLIP) dans le cache de l'image ---
# Téléchargés une fois au build (CPU) -> présents à chaque boot, plus de re-download.
# artwork_core passera alors en HF_HUB_OFFLINE tout seul (cache détecté).
# Note : clip.load met son cache dans /home/user/.cache/clip (bake sous USER user) -> relu tel
# quel au runtime, pas de re-download.
RUN python -c "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor; \
AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); \
AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny'); \
SamProcessor.from_pretrained('facebook/sam-vit-base'); \
SamModel.from_pretrained('facebook/sam-vit-base')" && \
    python -c "import clip; clip.load('ViT-B/32', device='cpu')"

# --- BAKE 2 : modèles Ollama (chat + embeddings RAG) dans l'image ---
# Serveur temporaire pendant le build, on attend qu'il réponde, puis on pull.
RUN nohup ollama serve >/dev/null 2>&1 & \
    for i in $(seq 1 30); do curl -sf http://localhost:11434/api/tags && break; sleep 1; done && \
    ollama pull qwen2.5:7b-instruct-q5_K_M && \
    ollama pull nomic-embed-text

# --- Code applicatif + base bundlée (copié en dernier pour le cache de layers) ---
# WORKDIR a créé /home/user/app en ROOT -> on lui rend la propriété avant de copier,
# sinon `user` ne peut pas y écrire (sed ci-dessous au build ; journal SQLite,
# WAL Chroma et cache Streamlit au runtime). Repassage root ponctuel, puis retour user.
USER root
RUN chown -R user:user /home/user/app
COPY --chown=user . /home/user/app

# Neutralise d'éventuels \r (fins de ligne Windows) restés dans start.sh
RUN sed -i 's/\r$//' /home/user/app/start.sh

# Retour à l'utilisateur non-root pour l'exécution
USER user

# Port public du Space (Streamlit s'y bindera dans start.sh)
EXPOSE 7860

CMD ["bash", "start.sh"]