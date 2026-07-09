# Transcription gateway: FastAPI + Modal CLI (per-user BYOK ASR)
FROM python:3.11-slim

# curl for the container healthcheck; ca-certificates for outbound HTTPS (Modal/OpenEMR)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Brings in transcribe.py AND modal_asr.py (the script `modal deploy` ships per user)
COPY . .

# HOST=0.0.0.0 so the reverse proxy in another container can reach it.
# MODAL_BIN points at the CLI installed by `pip install modal`.
ENV HOST=0.0.0.0 \
    PORT=8001 \
    MODAL_BIN=/usr/local/bin/modal

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["python", "transcribe.py"]
