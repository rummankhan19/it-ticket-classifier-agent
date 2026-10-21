"""IT Support Ticket Classification Agent - Vercel serverless endpoint (POST /api/chat).

Flow: user message -> retrieve similar tickets from the knowledge base (TF-IDF)
-> grounded prompt (rules + full ticket catalog + top matches + chat history)
-> Gemini -> answer + cited ticket IDs.
"""
import json
import math
import os
import re
from collections import Counter
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from google import genai
from google.genai import types

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "tickets.json"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
TIMEOUT_MS = 25_000
TOP_K = 5
MAX_HISTORY = 12

# ---------------------------------------------------------------- knowledge base
KB = json.loads(KB_PATH.read_text())
TICKETS = KB["tickets"]
STOPWORDS = set("""a an the and or of to in on for is are was were be been it its this that
with as at by from not cannot can't unable user users our we i my me how what which should
ticket issue please help""".split())


SYNONYMS = {
    "vm": "virtual machine", "down": "unavailable unreachable", "outage": "unavailable",
    "laptop": "equipment", "slow": "slowly performance", "login": "log", "pwd": "password",
    "mfa": "authentication", "2fa": "authentication", "invoice": "billing", "charged": "charge billing",
    "wifi": "wi fi", "hacked": "suspicious security", "breach": "suspicious security",
}


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    words += [s for w in words if w in SYNONYMS for s in SYNONYMS[w].split()]
    return [t for t in words if t not in STOPWORDS and len(t) > 1]


def _doc_text(t):
    return f"{t['subject']} {t['summary']} {t['department'].replace('/', ' ')}"


_DOC_TOKENS = [Counter(tokenize(_doc_text(t))) for t in TICKETS]
_DF = Counter(tok for doc in _DOC_TOKENS for tok in doc)
_IDF = {tok: math.log((1 + len(TICKETS)) / (1 + df)) + 1 for tok, df in _DF.items()}


def _vector(counts):
    vec = {tok: tf * _IDF.get(tok, 0.0) for tok, tf in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {tok: v / norm for tok, v in vec.items()}


_DOC_VECS = [_vector(c) for c in _DOC_TOKENS]


def retrieve(query, k=TOP_K):
    """Return tickets explicitly referenced by ID, then the most similar ones."""
    ids = {m.upper() for m in re.findall(r"tkt-\d{3}", query, re.I)}
    picked = [t for t in TICKETS if t["id"] in ids]
    q = _vector(Counter(tokenize(query)))
    scored = sorted(
        ((sum(w * d.get(tok, 0.0) for tok, w in q.items()), t) for d, t in zip(_DOC_VECS, TICKETS)),
        key=lambda x: x[0], reverse=True,
    )
    for score, t in scored:
        if len(picked) >= k + len(ids):
            break
        if score > 0.05 and t not in picked:
            picked.append(t)
    return picked


# ---------------------------------------------------------------- prompt
def _catalog():
    lines = [f"{t['id']} | {t['subject']} | {t['summary']} | Department: {t['department']}" for t in TICKETS]
    counts = Counter(t["department"] for t in TICKETS)
    dept_list = ", ".join(f"{d} ({n})" for d, n in counts.most_common())
    return "\n".join(lines), dept_list


CATALOG, DEPARTMENTS = _catalog()
PROCEDURE = " ".join(KB["standard_procedure"])

SYSTEM_PROMPT = f"""You are TicketSense, an AI assistant specialised ONLY in IT support ticket analysis and classification.
Your knowledge source is the organisation's historical ticket dataset below (100 synthetic tickets). Ground every
classification in it, and cite the ticket IDs (e.g. TKT-001) that support your decision.

## Classification framework
For any ticket or issue you are asked to classify, provide ALL of the following:
1. **Issue Type**: Technical (systems, software, infrastructure, network, security, data, access/identity)
   or Non-Technical (billing, payments, subscriptions, HR, administration, facilities, equipment, policies).
2. **Department**: the team with PRIMARY responsibility, chosen from the dataset departments:
   {DEPARTMENTS}.
   Use the most specific matching department label from the dataset (e.g. "Network / VPN" over "Network").
3. **Urgency** (core scale): High / Medium / Low.
   - High: production/critical service down, security incident or suspected breach, data loss, many users or
     customers blocked, revenue/financial-integrity impact, no workaround.
   - Medium: degraded performance, a single team/user blocked on a business task, intermittent failures,
     a workaround exists, time-sensitive non-technical request (e.g. disputed charge, payroll error).
   - Low: informational queries, how-to questions, routine requests, cosmetic issues, single user with minimal impact.
4. **Findings** (Jira team field): one of Technical, Billing, Marketing, Sales, Support.
   Technical = any technical issue; Billing = invoices, charges, refunds, payments, subscriptions;
   Marketing = campaigns, promotions, brand/website content; Sales = quotes, pricing, licences purchase, new deals;
   Support = HR, administration, facilities, general requests and anything else non-technical.
5. **Severity** (Jira-style): Critical / Highest / High / Medium / Low.
   Map from urgency and impact: Critical = High urgency + outage/security breach/data loss at wide scale;
   Highest = High urgency with narrower scope; High = upper Medium (business blocked, no workaround);
   Medium = Medium; Low = Low.
6. **Reasoning**: 2-4 sentences explaining the decision from the FULL ticket context (impact, scope, criticality,
   environment such as production vs test), not just keywords. Mention the similar dataset tickets.

Format classifications as a short markdown list with those six bold labels, then a "Similar tickets" line.

## Other questions you handle
- Likely causes and recommended troubleshooting/next steps for a ticket. Base next steps on the organisation's
  standard procedure: {PROCEDURE}
- Questions about the dataset (counts, which tickets belong to a department, comparisons, explaining a ticket by ID).
- Follow-up questions referring to earlier messages ("why not High?", "what if only one user is affected?").
  Re-evaluate and explain what changed.

## Rules
- Stay within the IT support ticket domain. Politely decline unrelated requests and say what you can help with.
- Do not invent facts, ticket IDs, departments or data that are not in the dataset. If the dataset does not
  contain the answer, say so. Possible causes may draw on general IT knowledge but present them as possibilities.
- The dataset has no urgency labels: urgency/severity are your assessment, so justify them.
- If a ticket description is too vague to classify confidently, give your best classification, state the
  assumption, and ask one clarifying question.
- Be concise and professional.

## Ticket dataset (ID | Subject | Summary | Department)
{CATALOG}
"""


# ---------------------------------------------------------------- LLM
_CLIENT = None


def _client():
    # Keep one long-lived client: the SDK closes its HTTP connection when a Client is garbage-collected.
    global _CLIENT
    if _CLIENT is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        # Google AI Studio (Gemini API) key; set USE_VERTEXAI=true to use a Vertex AI express-mode key instead.
        _CLIENT = genai.Client(vertexai=os.environ.get("USE_VERTEXAI", "").lower() == "true", api_key=key,
                               http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    return _CLIENT


def answer(message, history=None):
    history = (history or [])[-MAX_HISTORY:]
    recent_user_text = " ".join(h["content"] for h in history if h.get("role") == "user")[-500:]
    matches = retrieve(f"{message} {recent_user_text}")
    context = "\n".join(f"{t['id']} | {t['subject']} | {t['summary']} | Department: {t['department']}"
                        for t in matches) or "(no close matches)"

    contents = [
        types.Content(role="user" if h["role"] == "user" else "model", parts=[types.Part(text=h["content"])])
        for h in history if h.get("content")
    ]
    contents.append(types.Content(role="user", parts=[types.Part(
        text=f"Most similar dataset tickets for this question:\n{context}\n\nUser question: {message}")]))

    client = _client()
    try:
        # Thinking disabled for fast, consistent responses (fits Vercel's function time limit).
        resp = client.models.generate_content(model=MODEL, contents=contents, config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=0)))
    except Exception:
        # Primary model busy/unavailable: retry once on the lighter fallback model.
        resp = client.models.generate_content(model=FALLBACK_MODEL, contents=contents, config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, temperature=0.2))
    return {"reply": resp.text, "sources": [t["id"] for t in matches]}


# ---------------------------------------------------------------- HTTP handler (Vercel)
class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            message = (data.get("message") or "").strip()
            if not message:
                return self._send(400, {"error": "Message is required."})
            self._send(200, answer(message, data.get("history")))
        except Exception as exc:  # surface a readable error to the UI
            self._send(500, {"error": str(exc)})

    def do_GET(self):
        self._send(200, {"status": "ok", "tickets": len(TICKETS), "model": MODEL})
