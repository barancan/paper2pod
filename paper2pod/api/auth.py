"""Bearer token authentication dependency for /v1 routes (except /v1/health)."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_security = HTTPBearer(auto_error=False)


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),  # noqa: B008
) -> None:
    expected_token = request.app.state.api_auth_token
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or credentials.credentials != expected_token
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
