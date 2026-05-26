from __future__ import annotations

import hmac
from typing import Any

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .config import FinanceSettings

TOKEN_SALT = "finance-api-token-v1"


def verify_pin(pin: str, settings: FinanceSettings) -> bool:
    if not settings.pin_configured:
        return False

    return hmac.compare_digest(pin.strip(), settings.pin)


def create_access_token(settings: FinanceSettings) -> str:
    serializer = URLSafeTimedSerializer(settings.token_secret, salt=TOKEN_SALT)
    return serializer.dumps({"sub": "finance", "householdId": settings.household_id})


def verify_access_token(token: str, settings: FinanceSettings) -> dict[str, Any] | None:
    serializer = URLSafeTimedSerializer(settings.token_secret, salt=TOKEN_SALT)

    try:
        payload = serializer.loads(token, max_age=settings.token_max_age_seconds)
    except (BadSignature, SignatureExpired):
        return None

    if not isinstance(payload, dict) or payload.get("sub") != "finance":
        return None

    return payload
