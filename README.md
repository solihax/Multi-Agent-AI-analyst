# Multi-Agent AI Analyst — project structure

Your original single notebook is now split into modules. Upload the whole
folder to your Colab working directory (same place as `main.ipynb`), then
just run `main.ipynb` top to bottom — it imports everything else.

```
config.py       -> API key prompts + llm_flash / llm_lite setup
state.py        -> AgentState TypedDict, new_state(), get_llm_text()
data_setup.py   -> downloads churn CSV, builds SQLite db, writes churn report
vectorstore.py  -> Qdrant collections (RAG docs + conversation memory), embedder
agents.py       -> retriever_agent, web_agent, make_data_agent, make_code_agent
graph.py        -> supervisor, critic, generate node, build_graph()
main.ipynb      -> slim notebook that wires it all together and runs it
```

## What changed from your original notebook

- **Fixed the API key bug.** `input("sk-...")` and `input("tvly-...")` were
  passing the actual key as the *prompt text* shown on screen — not as the
  value read in. That meant the keys were sitting in plain text in your
  notebook, and the code wasn't even using them correctly. Now it's
  `input("Enter your API key: ")`, which is what actually works.
  **You should still revoke and regenerate both keys**, since the old ones
  were exposed in the notebook file.
- **Kept only the final supervisor.** Your notebook had two versions of
  `supervisor()` (cell 14, then a memory-aware redefinition in cell 19).
  `graph.py` uses the memory-aware one — the one that resolves follow-ups
  like "what about that" using `get_relevant_memory()`.
- **Agents that need shared objects (`llm`, `df`, `db`) are factory
  functions** (`make_data_agent(db, llm_lite)`, etc.) instead of relying on
  notebook globals. Same behavior, just doesn't break if you reorder cells.
- Removed the duplicate `!pip install tavily-python` cell — merged into one
  install cell up top.

## Rebuilding the graph after edits

LangGraph compiles `app` once. If you change any function inside `agents.py`
or `graph.py`, re-import it (Colab: restart-and-run or `importlib.reload`)
and re-run the "Build the graph" cell — just editing the file on disk won't
update an already-compiled `app`.

## Known bug to fix (per your error-analysis notes)

The supervisor sometimes under-routes, and `generate_answer` can hallucinate
facts the critic doesn't catch. Good places to dig, now that they're
isolated:
- `graph.py` → `make_supervisor()` — routing prompt/logic
- `graph.py` → `make_critic()` — verification strictness

## Model config (per mentor's model_list)

The proxy currently only exposes `gemini-flash-lite` (chat) and
`gemini-embedding` (embeddings) — no separate full `gemini-flash`. Both
`llm_flash` and `llm_lite` point at `gemini-flash-lite`, and Qdrant
embeddings now come from the proxy (`gemini-embedding`) instead of a local
SentenceTransformer, via `config.get_embedder()` / `vectorstore.init_embedder()`.

---

## Deploying: backend on Render, frontend on Vercel

```
backend/    -> FastAPI wrapper around the graph, deploys to Render
frontend/   -> single static index.html, deploys to Vercel
```

### 1. Backend → Render

1. Push the whole project folder to a GitHub repo (or just the `backend/`
   folder as its own repo — either works).
2. On [render.com](https://render.com): **New → Web Service**, connect the repo.
3. If your repo root *is* `backend/`, leave settings default. If `backend/`
   is a subfolder, set **Root Directory** to `backend`.
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Under **Environment**, add:
   - `GEMINI_API_KEY` = your key
   - `TAVILY_API_KEY` = your key (optional)
7. Deploy. First boot takes a bit longer — it downloads the churn CSV and
   rebuilds the vector index on startup. Once it's up, check
   `https://<your-service>.onrender.com/health` — should return
   `{"status": "ok", "graph_ready": true}`.
8. Test the API directly:
   ```bash
   curl -X POST https://<your-service>.onrender.com/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "How many customers have churned?"}'
   ```

Render's free tier spins down after inactivity, so the first request after
idle time will be slow (cold start rebuilds everything in step 7). That's
expected — not a bug.

### 2. Frontend → Vercel

1. `frontend/index.html` is a single static file — no build step, no
   `package.json` needed.
2. On [vercel.com](https://vercel.com): **New Project**, import the repo,
   set **Root Directory** to `frontend`. Vercel will detect it as a static
   site automatically (framework preset: "Other").
3. Deploy. Open the resulting URL, paste your Render backend URL into the
   "Backend URL" field at the top of the page (it's saved in your browser
   for next time), and ask a question.
4. Optional hardening once you have your Vercel URL: in
   `backend/main.py`, change `allow_origins=["*"]` to
   `allow_origins=["https://your-app.vercel.app"]` and redeploy the
   backend, so only your frontend can call the API.

### Notes

- The backend's `config.py` reads `GEMINI_API_KEY`/`TAVILY_API_KEY` directly
  from environment variables — no `input()` prompts, since Render runs
  non-interactively. This is separate from the notebook's `config.py`,
  which still prompts (fine for Colab).
- `backend/` has its own copies of `state.py`, `data_setup.py`,
  `vectorstore.py`, `agents.py`, `graph.py` so it can be deployed
  standalone. If you change agent logic in the notebook version, copy the
  same change into `backend/` before redeploying.
