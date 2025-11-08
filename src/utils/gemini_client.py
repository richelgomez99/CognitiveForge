"""
Gemini API client wrapper with retry logic and structured output support.

Uses the new unified Google Gen AI SDK (google-genai) for improved
structured output handling and better authentication support.
"""

import os
import time
import logging
from typing import Dict, Any, Type
from pydantic import BaseModel

# New unified SDK
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiRateLimitError(Exception):
    """Raised when Gemini API rate limit is exhausted after all retries."""
    pass


def call_gemini_with_retry(
    model_id: str,
    prompt: str,
    response_schema: Type[BaseModel],
    max_retries: int = 3
) -> str:
    """
    Call Gemini API with exponential backoff retry on HTTP 429 errors.
    
    Uses the new unified Google Gen AI SDK with native Pydantic support.
    No schema cleaning needed - the SDK handles Pydantic models natively!
    
    Implements FR-T1-016: Exponential backoff retry on HTTP 429 (rate limit) errors
    with delays of 1s, 2s, 4s (max 3 retries).
    
    Args:
        model_id: Gemini model ID (e.g., "gemini-2.5-pro")
        prompt: The prompt text to send to Gemini
        response_schema: Pydantic BaseModel class for structured output
        max_retries: Maximum number of retry attempts (default: 3)
    
    Returns:
        str: JSON string response from Gemini
    
    Raises:
        GeminiRateLimitError: If rate limit persists after all retries
        ValueError: If model_id is invalid or prompt is empty
        Exception: For other API errors
    
    Example:
        >>> from src.models import Thesis
        >>> response = call_gemini_with_retry("gemini-2.5-pro", prompt, Thesis)
        >>> thesis = Thesis.model_validate_json(response)
    """
    # Validate inputs
    if not model_id or not model_id.strip():
        raise ValueError("model_id cannot be empty")
    if not prompt or not prompt.strip():
        raise ValueError("prompt cannot be empty")
    if not response_schema:
        raise ValueError("response_schema cannot be empty")
    
    # Initialize client with API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key)
    
    # Retry delays: 1s, 2s, 4s
    retry_delays = [1, 2, 4]
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling Gemini model '{model_id}' (attempt {attempt + 1}/{max_retries})")
            
            # Generate content with structured output (native Pydantic support!)
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=response_schema  # Pass Pydantic model directly!
                )
            )
            
            logger.info(f"✅ Gemini API call successful (model: {model_id})")
            
            # Return the JSON text
            return response.text
            
        except Exception as e:
            error_message = str(e).lower()
            error_code = getattr(e, 'code', None)
            
            # Check if it's a rate limit error (429)
            is_rate_limit = (
                '429' in error_message or 
                'rate limit' in error_message or 
                'quota' in error_message or
                error_code == 429
            )
            
            if is_rate_limit and attempt < max_retries - 1:
                delay = retry_delays[attempt]
                logger.warning(
                    f"Rate limit hit (HTTP 429). Retrying in {delay}s... "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
                continue
            elif is_rate_limit:
                # Exhausted all retries
                error_msg = f"Gemini API rate limit exceeded after {max_retries} retries."
                logger.error(error_msg)
                raise GeminiRateLimitError(error_msg) from e
            else:
                # Non-rate-limit error
                logger.error(f"Gemini API error: {e}")
                raise
    
    # Should never reach here
    raise Exception("Failed to call Gemini API after multiple retries")


def call_gemini_simple(model_id: str, prompt: str, max_retries: int = 3) -> str:
    """
    Call Gemini API for simple text generation (no structured output).
    
    Useful for testing or non-structured queries.
    
    Args:
        model_id: Gemini model ID (e.g., "gemini-2.5-flash")
        prompt: The prompt text
        max_retries: Maximum retry attempts
    
    Returns:
        str: Generated text response
    
    Raises:
        GeminiRateLimitError: If rate limit persists after all retries
        Exception: For other API errors
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key)
    retry_delays = [1, 2, 4]
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt
            )
            return response.text
            
        except Exception as e:
            error_message = str(e).lower()
            is_rate_limit = '429' in error_message or 'rate limit' in error_message
            
            if is_rate_limit and attempt < max_retries - 1:
                delay = retry_delays[attempt]
                logger.warning(f"Rate limit hit. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            elif is_rate_limit:
                raise GeminiRateLimitError(
                    f"Gemini API rate limit exceeded after {max_retries} retries."
                ) from e
            else:
                raise
    
    raise Exception("Failed to call Gemini API")
