"""
Model configuration helper for per-agent Gemini model selection.

This module implements FR-T1-014/015: Per-agent environment variable configuration
with fallback to global default.
"""

import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed Gemini model IDs (updated for Gemini 2.x - 1.5 variants deprecated)
# Reference: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
ALLOWED_MODELS = [
    # Gemini 2.5 series (current stable as of June 2025)
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-image",
    
    # Gemini 2.0 series (stable)
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    
    # Auto-updated aliases (always point to latest stable)
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# Default model if no configuration found
DEFAULT_MODEL = "gemini-2.5-pro"


def get_agent_model(agent_role: str) -> str:
    """
    Get the Gemini model ID for a specific agent role.
    
    Implements FR-T1-014: Per-agent model configuration via environment variables
    Implements FR-T1-015: Fallback to global GEMINI_MODEL, then default
    
    Resolution order:
    1. GEMINI_MODEL_{ROLE} (e.g., GEMINI_MODEL_ANALYST)
    2. GEMINI_MODEL (global fallback)
    3. DEFAULT_MODEL ("gemini-2.5-pro")
    
    Args:
        agent_role: Agent role name (e.g., "analyst", "skeptic", "synthesizer")
    
    Returns:
        str: Validated Gemini model ID
    
    Raises:
        ValueError: If agent_role is empty
    
    Example:
        >>> model = get_agent_model("analyst")
        >>> # Returns "gemini-2.5-pro" (or value from GEMINI_MODEL_ANALYST env var)
    """
    if not agent_role or not agent_role.strip():
        raise ValueError("agent_role cannot be empty")
    
    # Normalize role name to uppercase for env var lookup
    role_upper = agent_role.strip().upper()
    
    # Try agent-specific env var first (e.g., GEMINI_MODEL_ANALYST)
    agent_env_var = f"GEMINI_MODEL_{role_upper}"
    model_id = os.getenv(agent_env_var)
    
    if model_id:
        logger.info(f"Using agent-specific model for {agent_role}: {model_id} (from {agent_env_var})")
    else:
        # Fall back to global GEMINI_MODEL
        model_id = os.getenv("GEMINI_MODEL")
        
        if model_id:
            logger.info(f"Using global model for {agent_role}: {model_id} (from GEMINI_MODEL)")
        else:
            # Fall back to default
            model_id = DEFAULT_MODEL
            logger.info(f"Using default model for {agent_role}: {model_id} (no env vars set)")
    
    # Validate model ID
    model_id = model_id.strip()
    
    if model_id not in ALLOWED_MODELS:
        logger.warning(
            f"Invalid model ID '{model_id}' for {agent_role}. "
            f"Allowed models: {', '.join(ALLOWED_MODELS)}. "
            f"Falling back to default: {DEFAULT_MODEL}"
        )
        model_id = DEFAULT_MODEL
    
    return model_id


def get_available_models() -> list[str]:
    """
    Get list of all available Gemini model IDs.
    
    Returns:
        list[str]: List of allowed model IDs
    """
    return ALLOWED_MODELS.copy()


def validate_model_id(model_id: str) -> bool:
    """
    Check if a model ID is valid.
    
    Args:
        model_id: Model ID to validate
    
    Returns:
        bool: True if model ID is in the allowed list
    """
    return model_id in ALLOWED_MODELS

