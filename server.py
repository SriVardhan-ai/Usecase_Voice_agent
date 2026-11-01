"""Serves the Evernorth website and issues LiveKit tokens for the voice assistant.

Run:  uvicorn server:app --reload --port 8000
The agent runs separately:  python -m voice_agent.main dev
"""
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from livekit import api

load_dotenv()

app = FastAPI(title="Evernorth Health Services demo")


@app.get("/api/token")
def token():
    """One fresh room per call; the agent worker auto-joins new rooms."""
    room = f"support-{uuid.uuid4().hex[:8]}"
    jwt = (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(f"member-{uuid.uuid4().hex[:6]}")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    return {"url": os.environ["LIVEKIT_URL"], "token": jwt, "room": room}


@app.get("/")
def index():
    return FileResponse("web/index.html")


app.mount("/static", StaticFiles(directory="web"), name="static")
