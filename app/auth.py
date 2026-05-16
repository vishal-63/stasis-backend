import logging
from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwk, jwt
from jose.utils import base64url_decode
from functools import lru_cache
import json

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
bearer_scheme = HTTPBearer(auto_error=True)


class AuthenticatedUser:
    def __init__(self, user_id: str, email: str | None, role: str):
        self.user_id = user_id
        self.email = email
        self.role = role


def fetch_jwks(supabase_url: str) -> dict:
    """
    Always fetch fresh JWKS — no caching.   
    Supabase rotates keys and a stale cache causes auth failures.
    """
    jwks_url = f"{supabase_url}/auth/v1/.well-known/jwks.json"
    response = httpx.get(jwks_url, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data


def verify_jwt(token: str, settings: Settings) -> dict:
    """
    Verify a Supabase JWT.
    Raises HTTPException on any failure — never returns None.
    """

    # ── Try JWKS verification ──────────────────────────────────────
    try:
        jwks = fetch_jwks(settings.supabase_url)
        keys = jwks.get("keys", [])

        if not keys:
            logger.warning("JWKS returned no keys")
        else:
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")
            alg = unverified_header.get("alg", "")

            # Find matching key by kid, or try all keys if no kid
            matching_keys = [
                k for k in keys
                if not kid or k.get("kid") == kid
            ]

            if not matching_keys:
                logger.warning(f"No JWKS key matched kid={kid}")
            else:
                for key_data in matching_keys:
                    try:
                        key_alg = key_data.get("alg", alg)

                        signing_key = jwk.construct(key_data, algorithm=key_alg)
                        payload = jwt.decode(
                            token,
                            signing_key.to_dict(),
                            algorithms=[key_alg],
                            options={
                                "verify_exp": True,
                                "verify_aud": False,
                                "require": ["sub", "exp", "iat"],
                            },
                        )
                        logger.info("JWKS verification succeeded")
                        return _validate_payload(payload)

                    except ExpiredSignatureError:
                        logger.warning("Token expired")
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token has expired",
                            headers={"WWW-Authenticate": "Bearer"},
                        )
                    except Exception as e:
                        logger.warning(f"Key attempt failed: {type(e).__name__}: {e}")
                        continue

    except HTTPException:
        raise  # re-raise expiry and other HTTP exceptions
    except Exception as e:
        logger.warning(f"JWKS fetch/parse failed: {type(e).__name__}: {e}")

def _validate_payload(payload: dict) -> dict:
    """Validate role claim. Raises HTTPException if invalid."""
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    role = payload.get("role", "")
    if role != "authenticated":
        logger.warning(f"JWT rejected: unexpected role '{role}'")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    return payload


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials, Security(bearer_scheme)
    ],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AuthenticatedUser:
    payload = verify_jwt(credentials.credentials, settings)

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing subject",
        )

    return AuthenticatedUser(
        user_id=user_id,
        email=payload.get("email"),
        role=payload.get("role", "authenticated"),
    )


CurrentUser = Annotated[AuthenticatedUser, Depends(get_current_user)]