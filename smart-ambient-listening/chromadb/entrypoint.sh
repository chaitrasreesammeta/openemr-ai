#!/bin/sh
# Bootstrap the "medical-schemas" collection on first boot (empty volume), then run the server.
# The volume is mounted at /app/chroma_data, which is exactly where chromadb_script.py writes.
set -e

DATA="${CHROMA_DATA:-/app/chroma_data}"

if [ ! -f "$DATA/chroma.sqlite3" ]; then
  echo "[chroma] empty data dir, bootstrapping medical-schemas (1,000 SOAP schemas)..."
  python chromadb_script.py
  echo "[chroma] bootstrap complete."
else
  echo "[chroma] existing data found at $DATA, skipping bootstrap."
fi

echo "[chroma] starting server on 0.0.0.0:8000 (path=$DATA)"
exec chroma run --host 0.0.0.0 --port 8000 --path "$DATA"
