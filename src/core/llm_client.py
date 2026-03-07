"""
Centralized LLM client interface for the Dynamic Causal Character Graphs system.

Handles:
- API key configuration (GEMINI_API_KEY)
- Model selection (default: gemini-2.0-flash-exp)
- Structured generation (JSON output)
- Text generation
"""

import os
import json
import logging
from typing import Any, Dict, Optional, Type, TypeVar
import google.generativeai as genai
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL_NAME = "gemini-2.0-flash-exp"

def get_api_key() -> Optional[str]:
    """Retrieve the API key from environment variables."""
    return os.getenv("GEMINI_API_KEY")

def configure_client() -> bool:
    """
    Configure the Google Generative AI client.
    Returns True if successful, False otherwise.
    """
    api_key = get_api_key()
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment variables. LLM features will be disabled.")
        return False
    
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logger.error(f"Failed to configure Gemini client: {e}")
        return False

def generate_text(
    prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> Optional[str]:
    """
    Generate plain text response from the LLM.
    
    Returns None if the client is not configured or generation fails.
    """
    if not configure_client():
        return None

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        return response.text
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        return None

def generate_structured(
    prompt: str,
    response_schema: Dict[str, Any],
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.2
) -> Optional[Dict[str, Any]]:
    """
    Generate a JSON structured response using the model's JSON mode capabilities.
    
    The prompt should explicitly ask for JSON output matching the schema.
    """
    if not configure_client():
        return None

    full_prompt = (
        f"{prompt}\n\n"
        f"You must respond with valid JSON matching this schema:\n"
        f"{json.dumps(response_schema, indent=2)}\n"
        f"Response:"
    )

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json"
            )
        )
        
        # Parse the JSON response
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response from LLM.")
            return None
            
    except Exception as e:
        logger.error(f"Structured generation failed: {e}")
        return None
