# TicketSense — IT Support Ticket Classification AI Agent (UC01)

An AI assistant that analyses IT support tickets and explains:

- **Issue type**: Technical / Non-Technical
- **Department** with primary responsibility (from the dataset's departments)
- **Urgency**: High / Medium / Low
- **Findings** (Jira team): Technical / Billing / Marketing / Sales / Support
- **Severity** (Jira): Critical / Highest / High / Medium / Low
- **Reasoning**, citing similar historical tickets

It also answers follow-ups (likely causes, next steps, "why not High?"), questions about the dataset,
and politely declines off-topic requests.

## How it works

1. `scripts/build_kb.py` cleans `data/synthetic_data.xlsx` into `data/tickets.json`. Every ticket shares
   the same 28 lines of troubleshooting boilerplate, so only the first, issue-specific line is kept as the
   summary; the boilerplate is stored once as the standard procedure.
2. For each question, `api/chat.py` retrieves the most similar tickets with TF-IDF (plus ticket-ID lookup
   and a few IT synonyms), then sends Gemini a grounded prompt: classification rules, the full 100-ticket
   catalog, the top matches and the chat history.
3. Gemini (temperature 0.2) answers in a fixed format; the UI shows the answer and retrieved ticket IDs.

```
api/chat.py          serverless endpoint POST /api/chat (retrieval + prompt + Gemini)
public/index.html    chat UI
data/tickets.json    knowledge base (generated)
scripts/build_kb.py  regenerates the knowledge base from the Excel file
dev_server.py        local server (no Vercel CLI needed)
vercel.json          Vercel function config
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your key in .env
python dev_server.py        # open http://localhost:8000
```

Uses a Google AI Studio (Gemini API) key. Set `GEMINI_MODEL` to use a different Gemini model, or
`USE_VERTEXAI=true` for a Vertex AI express-mode key.

To rebuild the knowledge base: `pip install openpyxl && python scripts/build_kb.py`.

## Deploy to Vercel

1. Push this folder to a GitHub repository.
2. In Vercel: **Add New → Project**, import the repository (framework preset: **Other**).
3. Under **Environment Variables**, add `GEMINI_API_KEY` (and optionally `GEMINI_MODEL`).
4. Deploy. The UI is served from `public/`, and `api/chat.py` runs as a Python serverless function.
