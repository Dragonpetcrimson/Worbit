import os
import logging
import keyring
import dotenv

def get_openai_api_key():
    """Get the OpenAI API key from environment variable or keyring."""
    # First try to get from environment variable
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    # If not found in environment, try loading from .env file
    if not api_key:
        try:
            dotenv.load_dotenv()
            api_key = os.environ.get("OPENAI_API_KEY", "")
        except Exception:
            pass
    
    # If still not found, try keyring
    if not api_key:
        try:
            keyring_key = keyring.get_password("orbit_analyzer", "openai_api_key")
            if keyring_key:
                api_key = keyring_key
        except Exception as e:
            logging.warning(f"Could not access system keyring: {str(e)}")
    
    # If no key found, log a warning
    if not api_key:
        logging.warning("OpenAI API key not found. GPT-based analysis will not be available.")
        return ""  # Return empty string instead of None
    
    return api_key