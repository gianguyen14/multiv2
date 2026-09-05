# Chi Lăng — Video Retrieval Frontend

## Overview

Chi Lăng is a vanilla HTML/CSS/JavaScript operator console for the Video Retrieval
project. It supports four search modes with a single existing backend contract:

- Textual KIS
- Video Q&A
- TRAKE
- Image Search

The frontend is plain HTML/CSS/JS — no Node.js, npm, bundler, or frontend
package-manager dependency is required. It is served directly by the FastAPI
backend.

## Serve from the real backend (recommended)

The FastAPI backend serves the whole frontend at its root and exposes the static
assets on `/styles/*` and `/scripts/*`:

```bash
python projectctl.py dev --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

`GET /` serves `frontend/src/index.html`, which links:

- `/styles/main.css`
- `/scripts/api.js`
- `/scripts/app.js`
- `/scripts/shortcuts.js`

All `/api/*` calls are same-origin relative fetches, so no CORS or proxy is needed.

## Alternative: dev runner on port 3000

`run_dev.py` (standard-library only) starts the existing backend, serves
`frontend/src/` on port 3000, and proxies `/api/*` to the backend on port 8000.
Because the backend already serves the assets, this is an optional convenience.
Press `Ctrl+C` to stop both servers.

```bash
python run_dev.py
```

Open:

```text
http://127.0.0.1:3000
```

## Backend integration

### Text search — `POST /api/search`

KIS:

```json
{"query": "<text>", "query_type": "kis", "top_k": 100}
```

Q&A:

```json
{"query": "<question>", "query_type": "qa", "top_k": 100}
```

TRAKE:

```json
{"query_type": "trake", "events": ["<event 1>", "<event 2>"], "top_k": 100}
```

Frontend constraints: at most 20 TRAKE events; every event non-empty and at most
2000 characters; event order preserved exactly.

### Image search — `POST /api/search/image?top_k=100`

multipart/form-data with field name `file`. The browser generates the multipart
boundary (JavaScript never sets `Content-Type` manually). Client-side pre-checks:
JPEG/PNG/WebP, maximum 15 MiB. The backend remains authoritative for parsing and
image validation.

## URL policy

Frontend JavaScript uses relative URLs only (`/api/search`, `/api/search/image?top_k=100`)
and never hard-codes a host or port. Result `image_url` values are assigned directly
to image `src`; the frontend never trims, rewrites, or derives frame URLs, and it
treats backend frame identifiers (`frame_id`, `frame_ids`) as authoritative.

## Safe DOM behavior

The runtime uses `textContent` + `document.createElement` and wires events with
`addEventListener`. It does not use `innerHTML`, `eval`, or inline event-handler
attributes.

## Keyboard workflow

`1/2/3/4` switch modes; `/` focuses the active input; `Enter` submits; `Alt+↑`/`Alt+↓`
add/remove a TRAKE event; arrow keys navigate results; `C` copies the selected
result as a competition submission row.

## Submission copy

- KIS: `video_id,frame_id`
- Q&A: `video_id,frame_id,answer` (answer required, ≤ 100 chars)
- TRAKE: `video_id,frame1,...,frameN` (`frame_ids` required, must match submitted
  event count, each a non-negative integer)
- Image Search has no submission-copy action.

## Folder structure

```text
frontend/
└── src/
    ├── index.html
    ├── styles/
    │   └── main.css
    └── scripts/
        ├── api.js
        ├── app.js
        └── shortcuts.js
```