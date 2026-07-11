"""
Shared API-key resolution for the HTTP layer.

The frontend never sends a real key when one is configured in ``.env``: it
sends the ``__USE_ENV__`` sentinel instead (see ``ApiKeyUtils.getValue`` in
``api-key-utils.js``). Every endpoint that accepts a per-request key must turn
that sentinel — and any empty value — back into the configured key.

This logic was previously copy-pasted into four blueprints
(translation/sample/config/glossary). Diverging copies caused issue #200:
the glossary NER endpoint forwarded the literal ``__USE_ENV__`` to Gemini,
which rejected it as an invalid key. Keep the single source of truth here.
"""
import os
from src.common.provider_metadata import PROVIDER_ENV_VARS, provider_env_var

# Marker the frontend sends when the key field is empty but a key is
# configured in .env. Resolved back to the real env value server-side.
USE_ENV_SENTINEL = '__USE_ENV__'

def resolve_api_key(value, env_var_name, config_default=''):
    """Resolve a per-request API-key value to the key to actually use.

    A real key (anything truthy that isn't the ``__USE_ENV__`` sentinel) is
    returned unchanged — including multi-key, comma-separated strings, which
    provider constructors split for key rotation. Otherwise the value falls
    back to the environment variable, then to ``config_default``.

    Args:
        value: Value from the request (a real key, ``'__USE_ENV__'``, or empty).
        env_var_name: Env var to fall back to. Empty/None skips the env lookup.
        config_default: Last-resort default (e.g. the value loaded into the
            config module at import time) when the env var is unset.

    Returns:
        The resolved key string (possibly empty if nothing is configured).
    """
    if value and value != USE_ENV_SENTINEL:
        return value
    if not env_var_name:
        return config_default
    return os.getenv(env_var_name, config_default)
