# MARS — Multi-Agent Research System

> Six AI agents research, debate, and synthesise any topic in real time.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-Sonnet_4.6-7c6dff?style=flat-square)

---

## What it does

Type a research question. Watch six specialist agents work in parallel, argue with each other, and converge on a verified, confidence-scored answer — streamed live to your browser.

```
Planner → Researcher A ──┐
                          ├─→ Critic → Skeptic → Debate (4 rounds) → Synthesizer
         Researcher B ──┘
```

Each agent has a distinct role and reasoning style. The Skeptic actively tries to break what the Researchers found. The Debate rounds surface real disagreement. The final report shows how confident each conclusion actually is.

---

## Stack

| Layer | Tech |
|---|---|
| LLM | Claude Sonnet 4.6 via Anthropic API |
| Backend | FastAPI + SSE streaming |
| Frontend | React 18 + TypeScript + Vite |
| State | Custom `usePipeline` hook (SSE → React state) |
| Fonts | DM Sans + DM Mono |

---

## Quickstart

### 1 — Clone and set up

```bash
git clone <your-repo>
cd mars
```

### 2 — Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Open .env and add your key:
# ANTHROPIC_API_KEY=sk-ant-...

uvicorn main:app --reload --port 8000
```

Confirm it's running:

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"claude-sonnet-4-6"}
```

### 3 — Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## Project structure

```
mars/
├── backend/
│   ├── main.py              # FastAPI app, all 6 agents, SSE pipeline, debate engine
│   ├── requirements.txt
│   └── .env.example
│
└── frontend/
    ├── src/
    │   ├── App.tsx           # Full UI — agents, debate, beliefs, benchmarks
    │   ├── index.css         # Global styles, animations, font imports
    │   ├── main.tsx          # React entry point
    │   ├── hooks/
    │   │   └── usePipeline.ts  # SSE connection + all pipeline state
    │   └── types/
    │       └── index.ts      # Shared TypeScript types
    ├── index.html
    ├── package.json
    ├── tsconfig.json
    └── vite.config.ts
```

---

## How the pipeline works

### Agents

| Agent | Role | Model behaviour |
|---|---|---|
| **Planner** | Decomposes the query into 3 focused sub-tasks | Conservative — sets scope boundaries |
| **Researcher A** | Primary evidence gathering | Specific: names, dates, statistics |
| **Researcher B** | Domain-specialist analysis | Technical: benchmarks, patents, filings |
| **Critic** | Gap and contradiction scanner | Identifies what's missing or unsupported |
| **Skeptic** | Active counter-evidence finder | Directly attacks the strongest claims |
| **Synthesizer** | Final report writer | Produces confidence-scored executive summary |

### Debate (4 rounds)

After the 5 research agents complete, a structured debate runs using their actual outputs as evidence:

1. **Advocate** — argues for the strongest finding
2. **Challenger** — directly counters with the Skeptic's evidence
3. **Specialist** — adds the technical dimension both sides missed
4. **Verdict** — discriminator synthesises consensus + flags unresolved uncertainty

### Streaming

Every step streams via Server-Sent Events. The frontend `usePipeline` hook reads the SSE stream and updates React state in real time — no polling, no websockets, no client-side timeouts.

### Belief evolution

After synthesis completes, confidence snapshots are taken at each pipeline stage to show how evidence shifted the belief from prior → final.

---

## API

### `POST /research/stream`

Starts a pipeline and streams SSE events.

**Request body:**
```json
{ "query": "Your research question here" }
```

**Event types streamed:**

| Event | When |
|---|---|
| `pipeline_start` | Immediately on POST |
| `agent_start` | Each agent begins |
| `agent_done` | Each agent finishes (includes output + confidence) |
| `debate_round_start` | Each of the 4 debate rounds begins |
| `debate_round_done` | Each debate round finishes (text + consensus %) |
| `pipeline_complete` | Everything done (includes metrics + belief snapshots) |
| `agent_error` | If any agent fails |

**Example — curl:**

```bash
curl -N -X POST http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What broke in AI in 2025?"}'
```

### `GET /health`

```json
{ "status": "ok", "model": "claude-sonnet-4-6" }
```

---

## UI tabs

| Tab | What you see |
|---|---|
| **Agents** | Live status cards for all 6 agents. Click "done" cards to expand output. Final synthesis appears at the bottom. |
| **Debate** | 4-round structured debate with live consensus meter. |
| **Beliefs** | Confidence timeline per belief — prior → each agent stage → final. |
| **Benchmarks** | Quality metrics (hallucination rate, citation grounding, agent agreement, quality score) compared against vanilla GPT-4 and single-agent RAG baselines. |

---

## Configuration

All config lives in `backend/main.py` — change the constants at the top:

```python
MODEL = "claude-sonnet-4-6"          # swap model here
```

Each agent's behaviour is controlled by its `system` prompt in the `AGENTS` dict. Edit these to change how each agent reasons.

Frontend API URL is set via Vite env var:

```bash
# frontend/.env.local
VITE_API_URL=http://localhost:8000
```

---

## Deploy

### Backend → Railway

```bash
railway init
railway add --plugin postgres    # optional — not required for base MARS
railway env set ANTHROPIC_API_KEY=sk-ant-...
railway up
```

### Backend → Render

1. New Web Service → connect repo
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add env var: `ANTHROPIC_API_KEY`

### Frontend → Vercel

```bash
cd frontend
vercel
# Set VITE_API_URL to your backend URL in Vercel project settings
```

---

## Theme

The modern dark UI uses:

- **DM Sans** — body text, UI labels
- **DM Mono** — status text, tags, technical labels
- Sidebar layout with per-agent status column
- Color-coded agents with animated status dots
- 4-tab main panel (agents / debate / beliefs / benchmarks)

A classic editorial light theme (`App.classic.tsx`) is included in the `frontend-designs/` folder if you prefer that look.

---

## Cost

A typical research run makes ~10 API calls, each 200–500 tokens output.

| Run type | Approx. cost |
|---|---|
| Standard query | ~$0.05–0.10 |
| Complex query (longer outputs) | ~$0.15–0.25 |

Using `claude-sonnet-4-6` at $3/$15 per M tokens input/output.

---

## License

MIT
