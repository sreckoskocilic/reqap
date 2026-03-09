# reqap — Text Analysis Service

Upload a book, script, article, or any long-form text — get a comprehensive AI-generated report. Runs as a streaming web portal or a CLI tool.

---

## Features

**Upload & formats**
- Drag-and-drop `.epub` or `.txt` upload — books, scripts, articles, reports, screenplays, anything long-form
- Live streaming output — see the report as it's written

**Section selection**
- After upload, the text is parsed and all chapters/sections are listed with token counts
- Select any subset before generating — useful for targeted analysis or large texts

**10 report types**
- Book Review, Summary, Chapter Summaries, Character Map, Timeline, Themes & Motifs, Key Quotes, Reading Guide, Argument Map, Action Items
- Each type uses a tailored prompt and focused notes extraction

**Smart reading strategy**
- Single-pass when the book fits in the model's context window
- Multi-pass for large books: extract notes per chapter group, then synthesise — no truncation

**Six LLM modes — three free**
- Gemini 2.0 Flash (1M context), Groq (128K), free hybrid, offline Ollama, Claude Opus, Claude hybrid

**Deployment**
- Kubernetes-ready with HPA autoscaling, Ollama sidecar, and full minikube guide
- Docker single-container for local use
- CLI tool (`review.py`) works standalone without the service

---

## LLM Modes

| Mode | Notes extraction | Synthesis | Cost |
|---|---|---|---|
| `free-gemini` | Gemini 2.0 Flash | Gemini 2.0 Flash | Free — 1M ctx, best free option |
| `free-hybrid` | Groq llama3.1-8b | Gemini 2.0 Flash | Free — fast notes + quality synthesis |
| `free-groq` | Groq llama3.1-8b | Groq llama3.3-70b | Free — 128K ctx |
| `offline` | Ollama (local) | Ollama (local) | Free — no internet |
| `online` | Claude Opus | Claude Opus | Paid — best quality |
| `hybrid` | Claude Haiku | Claude Opus | Paid — cheaper than online |

Default is `free-gemini`. Switch per-request from the portal UI.

---

## Report Types

Select the report type from the portal UI alongside the LLM mode. Each type uses a tailored prompt.

| Type | Label | Best for |
|---|---|---|
| `review` | Book Review | Fiction / non-fiction — overview, style, themes, rating |
| `summary` | Summary | Quick retelling of the narrative arc |
| `chapter-summaries` | Chapter Summaries | Per-chapter breakdown |
| `characters` | Character Map | Profiles, arcs, relationships |
| `timeline` | Timeline | Chronological event extraction |
| `themes` | Themes & Motifs | Symbols, ideas, philosophical content |
| `quotes` | Key Quotes | 15-20 significant passages with context |
| `reading-guide` | Reading Guide | 20-25 book club discussion questions |
| `argument-map` | Argument Map | Non-fiction — thesis, evidence, counterarguments |
| `action-items` | Action Items | Self-help/business — principles, checklists, exercises |

Default is `review`.

---

## Free API Keys

| Provider | Link | Used for |
|---|---|---|
| Gemini | https://aistudio.google.com/app/apikey | `free-gemini`, `free-hybrid` |
| Groq | https://console.groq.com/keys | `free-groq`, `free-hybrid` |
| Anthropic | https://console.anthropic.com | `online`, `hybrid` |

---

## Project Structure

```
reqap/
├── review.py                  # CLI — works standalone
├── requirements.txt           # EPUB + LLM deps
├── requirements-service.txt   # FastAPI + uvicorn
├── Dockerfile
├── app/
│   ├── config.py              # all settings via env vars
│   ├── main.py                # FastAPI app
│   ├── epub/parser.py         # EPUB parsing, chapter grouping
│   ├── llm/
│   │   ├── base.py            # LLMBackend protocol
│   │   ├── claude.py          # Anthropic SDK backend
│   │   ├── ollama.py          # Ollama backend
│   │   ├── openai_compat.py   # Generic OpenAI-compat (Groq, Gemini, etc.)
│   │   └── router.py          # mode → backend mapping
│   ├── review/engine.py       # orchestration, SSE event generator
│   ├── api/routes.py          # POST /api/reviews/stream, GET /api/health
│   └── static/index.html      # web portal
└── k8s/
    ├── namespace.yaml
    ├── configmap.yaml
    ├── secret.yaml             # template only — never commit real keys
    ├── deployment.yaml
    ├── service.yaml
    ├── hpa.yaml
    └── ollama/
        ├── pvc.yaml
        ├── deployment.yaml
        └── service.yaml
```

---

## CLI Usage

```bash
pip install -r requirements.txt

# Basic review (prints to stdout)
python review.py book.epub

# Save to file
python review.py book.epub -o review.md

# Show model's thinking process
python review.py book.epub --thinking

# Offline with local Ollama
python review.py book.epub --offline
python review.py book.epub --offline --model llama3.2 --context 32768

# Custom Ollama URL
python review.py book.epub --offline --ollama-url http://192.168.1.10:11434
```

Set your API key before running:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # online / hybrid
export GEMINI_API_KEY=AIza...         # free-gemini / free-hybrid
export GROQ_API_KEY=gsk_...           # free-groq / free-hybrid
```

---

## Local Dev (service)

```bash
pip install -r requirements.txt -r requirements-service.txt

export GEMINI_API_KEY=AIza...
export LLM_MODE=free-gemini

uvicorn app.main:app --reload
# open http://localhost:8000
```

---

## Docker

```bash
docker build -t reqap:latest .

docker run -p 8000:8000 \
  -e GEMINI_API_KEY=AIza... \
  -e LLM_MODE=free-gemini \
  reqap:latest

# open http://localhost:8000
```

---

## Kubernetes — Minikube

### Prerequisites

- minikube
- kubectl
- Docker

### 1 — Start minikube

```bash
minikube start --cpus=2 --memory=4096
```

### 2 — Point Docker at minikube's daemon

```bash
eval $(minikube docker-env)
# Windows PowerShell: minikube docker-env | Invoke-Expression
```

Run this in every new terminal before `docker build`.

### 3 — Build the image

```bash
docker build -t reqap:latest .
```

### 4 — Create namespace and secret

```bash
kubectl apply -f k8s/namespace.yaml

kubectl create secret generic reqap-secrets \
  --namespace reqap \
  --from-literal=GEMINI_API_KEY=AIza... \
  --from-literal=GROQ_API_KEY=gsk_...
```

For paid modes also add `--from-literal=ANTHROPIC_API_KEY=sk-ant-...`.

### 5 — Deploy

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# HPA requires metrics-server
minikube addons enable metrics-server
kubectl apply -f k8s/hpa.yaml
```

### 6 — Open the portal

```bash
# Option A — opens browser automatically
minikube service reqap-external -n reqap

# Option B — manual port-forward
kubectl port-forward -n reqap svc/reqap 8080:80
# open http://localhost:8080
```

### Useful commands

```bash
# Watch pods
kubectl get pods -n reqap -w

# Logs
kubectl logs -n reqap -l app=reqap -f

# HPA status
kubectl get hpa -n reqap

# Rebuild and redeploy after code changes
eval $(minikube docker-env)
docker build -t reqap:latest .
kubectl rollout restart deployment/reqap -n reqap
```

---

## Kubernetes — With Ollama (offline/hybrid mode)

Requires at least 8 GB RAM allocated to minikube:

```bash
minikube start --cpus=4 --memory=8192
```

Deploy Ollama:

```bash
kubectl apply -f k8s/ollama/pvc.yaml
kubectl apply -f k8s/ollama/deployment.yaml
kubectl apply -f k8s/ollama/service.yaml

# Watch the init container pull the model (~2 GB)
kubectl logs -n reqap -l app=ollama -c pull-model -f
```

Switch to a mode that uses Ollama:

```bash
kubectl patch configmap reqap-config -n reqap \
  --patch '{"data":{"LLM_MODE":"free-hybrid"}}'

kubectl rollout restart deployment/reqap -n reqap
```

To change the Ollama model, edit `k8s/ollama/deployment.yaml` (the `pull-model` init container command) and `k8s/configmap.yaml` (`OLLAMA_MODEL`), then reapply.

---

## Configuration Reference

All settings are environment variables, loaded from the ConfigMap in Kubernetes or a `.env` file locally.

| Variable | Default | Description |
|---|---|---|
| `LLM_MODE` | `free-gemini` | `online` / `hybrid` / `offline` / `free-gemini` / `free-groq` / `free-hybrid` |
| `GEMINI_API_KEY` | — | Required for `free-gemini`, `free-hybrid` |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `GEMINI_CONTEXT_TOKENS` | `900000` | Effective context limit |
| `GROQ_API_KEY` | — | Required for `free-groq`, `free-hybrid` |
| `GROQ_NOTES_MODEL` | `llama-3.1-8b-instant` | Groq model for notes extraction |
| `GROQ_SYNTHESIS_MODEL` | `llama-3.3-70b-versatile` | Groq model for final review |
| `GROQ_CONTEXT_TOKENS` | `128000` | Effective context limit |
| `ANTHROPIC_API_KEY` | — | Required for `online`, `hybrid` |
| `CLAUDE_OPUS_MODEL` | `claude-opus-4-6` | Opus model for synthesis |
| `CLAUDE_HAIKU_MODEL` | `claude-haiku-4-5` | Haiku model for notes (hybrid) |
| `HYBRID_NOTES_BACKEND` | `haiku` | `haiku` or `ollama` |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama service URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OLLAMA_CONTEXT_TOKENS` | `8192` | Ollama model context size |
| `UPLOAD_MAX_MB` | `50` | Max EPUB file size |

---

## Security Notes

- Never commit real API keys. `k8s/secret.yaml` is a template only — use `kubectl create secret` or an external secrets operator.
- `.env` is in `.gitignore`. Keep it there.
- The container runs as non-root (`uid=1000`).
- Uploaded EPUBs are written to a tmpfs `emptyDir` volume and deleted immediately after the review stream closes.
