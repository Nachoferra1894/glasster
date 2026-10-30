# ---------- Stage 1: build the Vite frontend ----------
FROM node:20-alpine AS frontend
WORKDIR /app

RUN corepack enable
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY . .
RUN pnpm build   # outputs to /app/dist


# ---------- Stage 2: Flask + backgroundremover runtime ----------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5010

# opencv (pulled in by backgroundremover) needs these system libs;
# curl is used to pre-download the u2net model weights
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image ~2GB smaller than the CUDA default
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY server/requirements.txt server/
RUN pip install -r server/requirements.txt

COPY server/ server/
COPY --from=frontend /app/dist dist/

# Pre-download the model by running the real CLI once on a tiny image so the
# first user request isn't a 170MB+ download (`|| true` so build never fails)
RUN set -eux; \
    python -c "from PIL import Image; Image.new('RGB',(64,64),(200,100,50)).save('/tmp/warmup.png')"; \
    backgroundremover -i /tmp/warmup.png -o /tmp/warmup_out.png || true

EXPOSE 5010

CMD ["python", "server/server.py"]
