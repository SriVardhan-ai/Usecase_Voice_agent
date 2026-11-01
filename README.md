# Evernorth Health Services: website + voice assistant (student demo)

```
.
├── web/index.html          # the website (voice assistant widget built in)
├── server.py               # serves the site + /api/token (LiveKit token)
├── voice_agent/            # your LiveKit voice agent (unchanged)
├── langchain_agent/        # your LangChain tools + RAG (unchanged)
├── ingest.py, config.py    # your ingestion + settings (unchanged)
└── docs/                   # PDFs the agent answers from
```

## Run (three terminals, or three steps)
```bash
cp .env.example .env            # fill in LiveKit + Groq keys
uv sync                         # or: pip install -e .
python ingest.py                # once, builds the Qdrant index from docs/

python -m voice_agent.main dev  # terminal 1: the agent (use `dev`, not `console`)
uvicorn server:app --reload --port 8000   # terminal 2: the website
```
Open http://localhost:8000 and press "Talk to us". Mic access needs localhost or HTTPS.

## Demo script
- "What plans do you offer?" / "Tell me about Evernorth Select 5L"
- "Check my policy, my number is 9876543210" (also 9123456789, 8765432109)
- "I need a plan for my family with maternity cover"
- "My claim was rejected, I want to talk to someone" (creates a ticket)

## Deploy
Host `server.py` (Render/Railway/Fly) with the LiveKit env vars, and run the agent as a second service with
`python -m voice_agent.main start`. Keys stay server-side.
