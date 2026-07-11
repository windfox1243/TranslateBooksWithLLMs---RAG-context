"""Shared provider credential metadata used by HTTP and core runtimes."""

PROVIDER_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "poe": "POE_API_KEY",
    "nim": "NIM_API_KEY",
}

KEY_REQUIRED_PROVIDERS = {
    "gemini", "openrouter", "mistral", "deepseek", "poe", "nim",
}


def provider_env_var(provider):
    return PROVIDER_ENV_VARS.get((provider or "").lower(), "")
