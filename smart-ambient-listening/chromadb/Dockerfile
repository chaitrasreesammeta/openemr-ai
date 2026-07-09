# ChromaDB vector store: bootstraps the 1,000 MIMIC-IV SOAP schemas, then serves over HTTP
FROM python:3.11-slim

# libgomp1: onnxruntime (all-MiniLM embeddings computed at bootstrap); curl: healthcheck
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# chromadb_script.py + patient_notes_structure.json + entrypoint
COPY . .
RUN chmod +x /app/entrypoint.sh

# chromadb_script.py writes PersistentClient(path="./chroma_data"); the server reads the same dir.
ENV CHROMA_DATA=/app/chroma_data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
  CMD curl -fsS http://localhost:8000/api/v2/heartbeat || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
