# server.py
from fastapi import FastAPI

app = FastAPI(title="castbooster-cloud-worker", version="0.1.0")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
