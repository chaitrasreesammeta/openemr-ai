# SOAP summarization: FastAPI + ChromaDB client + Groq (per-user BYOK)
FROM python:3.11-slim

# libgomp1: required by onnxruntime (ChromaDB default all-MiniLM embeddings, computed client-side)
# curl: container healthcheck; ca-certificates: outbound HTTPS (Groq/OpenEMR)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0 \
    PORT=8002

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8002/health || exit 1

CMD ["python", "summarize.py"]
