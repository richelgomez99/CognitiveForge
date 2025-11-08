"""
API Key Authentication for CognitiveForge FastAPI endpoints.

Implements FR-T2-012: API key authentication via X-API-Key header
Implements FR-T2-013: 401 response for missing/invalid API keys
Implements FR-T2-014: Public health endpoint (no auth required)
"""

import os
from fastapi import HTTPException, Header
from typing import Optional


def verify_api_key(api_key: str) -> bool:
    """
    Verify the provided API key against the configured API_KEY environment variable.
    
    This is a simple string comparison for MVP authentication. For production,
    consider using hashed keys or OAuth2.
    
    Args:
        api_key: The API key to verify
    
    Returns:
        True if the API key is valid, False otherwise
    
    Example:
        >>> os.environ["API_KEY"] = "test_secret_key"
        >>> verify_api_key("test_secret_key")
        True
        >>> verify_api_key("wrong_key")
        False
    """
    expected_key = os.getenv("API_KEY")
    
    if not expected_key:
        raise ValueError(
            "API_KEY environment variable not set. "
            "Please configure API_KEY in .env file."
        )
    
    return api_key == expected_key


def get_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """
    FastAPI dependency for API key authentication.
    
    Extracts the API key from the X-API-Key header and validates it.
    Raises HTTPException with 401 status if validation fails.
    
    This dependency should be used with FastAPI's Depends() for endpoints
    that require authentication.
    
    Args:
        x_api_key: API key from X-API-Key header (injected by FastAPI)
    
    Returns:
        The validated API key string
    
    Raises:
        HTTPException: 401 if API key is missing or invalid
    
    Example:
        >>> from fastapi import Depends
        >>> @app.get("/protected")
        >>> def protected_endpoint(api_key: str = Depends(get_api_key)):
        ...     return {"message": "Access granted"}
    """
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key"
        )
    
    if not verify_api_key(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key"
        )
    
    return x_api_key

