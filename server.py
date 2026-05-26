# server.py
from fastapi import FastAPI, Depends

from auth import require_api_key

app = FastAPI(title="castbooster-cloud-worker", version="0.1.0")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/_test_protected", dependencies=[Depends(require_api_key)])
async def _test_protected():
    return {"authorized": True}
