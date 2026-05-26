# auth.py
import os

from fastapi import Header, HTTPException, status


async def require_api_key(authorization: str | None = Header(default=None)):
    """Dependency injection for Bearer auth.

    Reads CLOUD_API_KEY from env at request time (so tests can monkeypatch).
    """
    expected_key = os.environ.get("CLOUD_API_KEY", "")
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CLOUD_API_KEY not configured on worker",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )
    provided = authorization.removeprefix("Bearer ").strip()
    if provided != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
