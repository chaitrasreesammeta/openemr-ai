# Docker Compose deployment

Brings up the **entire** Smart Ambient Listening stack (OpenEMR, MariaDB, ChromaDB, the
transcription and summarization gateways, and a Caddy reverse proxy) with one command.

Everything is published on **one HTTPS origin** through Caddy. That matters because the SMART
on FHIR app is same-origin by design: it calls `/transcription`, `/api/summarize`, and
OpenEMR's `/apis` (FHIR) + `/oauth2` endpoints all off `window.location.origin`, and the
browser microphone only works in a secure (HTTPS) context.

```
                        ┌──────────────── Caddy (:443, TLS) ────────────────┐
   browser  ──https──▶  │  /app/*           → static SMART app (this repo)  │
                        │  /transcription/* → transcription:8001            │
                        │  /api/summarize/* → summarize:8002                │
                        │  everything else  → openemr:443 (FHIR/OAuth/UI)   │
                        └───────────────────────────────────────────────────┘
   summarize ──▶ chromadb:8000 (RAG)   summarize/transcription ──▶ openemr (token check)
   openemr ──▶ mysql:3306
```

## Quick start

```bash
cd smart-ambient-listening
cp .env.example .env          # then edit SITE_ADDRESS (and passwords for anything public)
docker compose up -d --build
```

First boot takes a few minutes: OpenEMR self-configures against MariaDB and ChromaDB
bootstraps the 1,000 MIMIC-IV SOAP schemas into its volume. Watch progress with
`docker compose ps` and `docker compose logs -f openemr chromadb`.

Then open `https://<SITE_ADDRESS>/`. With `SITE_ADDRESS=localhost` the cert is self-signed,
so accept the browser warning.

## Restoring your existing OpenEMR config/volume

The compose declares the same volume names the upstream OpenEMR production compose uses:
`sitevolume` (→ `/var/www/localhost/htdocs/openemr/sites`) and `databasevolume`
(→ MariaDB `/var/lib/mysql`). Restore into those **before** first boot and OpenEMR comes up
already-configured, including the registered SMART client, which lives in the database:

```bash
docker volume create smart-ambient-listening_databasevolume
docker volume create smart-ambient-listening_sitevolume
# ...restore your dumps into those volumes, then:
docker compose up -d --build
```

Make sure the `MYSQL_*` values in `.env` match the credentials your restored database expects.

## Registering the SMART app (only if not already in your restored DB)

Admin → System → **API Clients** → register:

- **Launch URI:** `https://<SITE_ADDRESS>/app/launch.html`
- **Redirect URI:** `https://<SITE_ADDRESS>/app/index.html`
- **Scopes:** `launch patient/Patient.read patient/Condition.read patient/Procedure.read patient/Observation.read patient/MedicationRequest.read patient/AllergyIntolerance.read patient/Encounter.read openid fhirUser api:oemr user/custom.read user/custom.write user/soap_note.write`
- Public client (not confidential).

Per-user Modal and Groq API keys are entered per clinician under User Settings → Custom
(the services retrieve them from OpenEMR per request, so it is BYOK and nothing is stored server-side).

## Ports

Only Caddy publishes to the host (`80`, `443`). Everything else talks over the internal
compose network by service name: `openemr`, `mysql`, `chromadb:8000`,
`transcription:8001`, `summarize:8002`.

## Production notes

- **Real cert:** set `SITE_ADDRESS` to a public domain with an A record and open 80/443;
  Caddy provisions Let's Encrypt automatically.
- **Outbound internet:** the transcription (Modal) and summarization (Groq) gateways need it.
- **Change default passwords** in `.env` before exposing anything publicly.
- ChromaDB data persists in the `chromadata` volume; to re-bootstrap, remove it.

## What differs from a bare `openemr/openemr` run

- The SMART app is served at `/app` on the OpenEMR origin (via Caddy), so there is no editing of the
  OpenEMR image.
- Two small, backward-compatible code changes let the Python services run cross-container:
  they now honor `HOST` (bind `0.0.0.0`) and `CHROMA_HOST`/`CHROMA_PORT`. Defaults are
  unchanged, so the systemd deployment in `config/SETUP.md` still works as before.
