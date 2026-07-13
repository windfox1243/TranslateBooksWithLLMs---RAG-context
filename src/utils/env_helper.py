"""
Utility to help users configure .env file
"""
import os
import sys
from pathlib import Path


def _get_config_dir():
    """Get directory for configuration files"""
    return Path.cwd()


_DEFAULT_COMPACT_ENV_VALUES = {
    "LLM_PROVIDER": "ollama",
    "DEFAULT_MODEL": "qwen3:14b",
    "API_ENDPOINT": "http://localhost:11434/api/generate",
    "OLLAMA_API_ENDPOINT": "http://localhost:11434/api/generate",
    "PORT": "5000",
    "HOST": "127.0.0.1",
    "OUTPUT_DIR": "translated_files",
    "DEFAULT_SOURCE_LANGUAGE": "",
    "DEFAULT_TARGET_LANGUAGE": "",
    "REQUEST_TIMEOUT": "300",
    "MAX_TOKENS_PER_CHUNK": "450",
    "OLLAMA_NUM_CTX": "4096",
    "AUTO_ADJUST_CONTEXT": "true",
    "PARALLEL_TRANSLATIONS": "1",
    "ENABLE_CHUNK_REFLECTION": "false",
    "NOVEL_CONTEXT_PROMPT_MAX_TOKENS": "1800",
    "NOVEL_CONTEXT_UPDATE_INTERVAL": "1",
    "NOVEL_CONTEXT_SOURCE_MEMORY_CHARS": "6000",
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": "",
    "OPENROUTER_API_KEY": "",
    "MISTRAL_API_KEY": "",
    "DEEPSEEK_API_KEY": "",
    "POE_API_KEY": "",
    "NIM_API_KEY": "",
    "GEMINI_MODEL": "gemini-2.0-flash",
    "OPENROUTER_MODEL": "anthropic/claude-4.5-haiku",
    "MISTRAL_MODEL": "mistral-large-latest",
    "DEEPSEEK_MODEL": "deepseek-v4-pro",
    "POE_MODEL": "Claude-Sonnet-4",
    "NIM_MODEL": "meta/llama-3.1-8b-instruct",
    "EDITOR_PROVIDER": "",
    "EDITOR_MODEL": "",
}

_COMPACT_ENV_LAYOUT = [
    "LLM_PROVIDER",
    "DEFAULT_MODEL",
    "API_ENDPOINT",
    "OLLAMA_API_ENDPOINT",
    "PORT",
    "HOST",
    "OUTPUT_DIR",
    "DEFAULT_SOURCE_LANGUAGE",
    "DEFAULT_TARGET_LANGUAGE",
    "REQUEST_TIMEOUT",
    "MAX_TOKENS_PER_CHUNK",
    "OLLAMA_NUM_CTX",
    "AUTO_ADJUST_CONTEXT",
    "PARALLEL_TRANSLATIONS",
    "ENABLE_CHUNK_REFLECTION",
    "EDITOR_PROVIDER",
    "EDITOR_MODEL",
    "NOVEL_CONTEXT_PROMPT_MAX_TOKENS",
    "NOVEL_CONTEXT_UPDATE_INTERVAL",
    "NOVEL_CONTEXT_SOURCE_MEMORY_CHARS",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "MISTRAL_API_KEY",
    "DEEPSEEK_API_KEY",
    "POE_API_KEY",
    "NIM_API_KEY",
]


def ensure_env_defaults(env_file: Path) -> list[str]:
    """Ensure newly introduced default configuration keys exist in an existing .env file.

    Preserves all existing keys, comments, and values. Only appends keys that are
    missing from the file.
    """
    if not env_file.exists():
        return []

    content = env_file.read_text(encoding="utf-8")
    existing_keys = set()
    for line in content.splitlines():
        line_str = line.strip()
        if line_str and not line_str.startswith("#") and "=" in line_str:
            key = line_str.split("=", 1)[0].strip()
            existing_keys.add(key)

    added_keys = []
    lines_to_append = []

    for key in _COMPACT_ENV_LAYOUT:
        if key and not key.startswith("#") and key not in existing_keys:
            added_keys.append(key)
            lines_to_append.append(f"{key}={_DEFAULT_COMPACT_ENV_VALUES.get(key, '')}")

    if lines_to_append:
        prefix = "\n" if not content.endswith("\n") else ""
        append_block = prefix + "# Auto-added settings from software upgrade\n" + "\n".join(lines_to_append) + "\n"
        env_file.write_text(content + append_block, encoding="utf-8")

    return added_keys


def render_compact_env(overrides: dict | None = None) -> str:
    """Render a short, practical .env file.

    The full commented reference lives in .env.example. This generated .env is
    intentionally concise so users only see values they are likely to edit.
    """
    values = dict(_DEFAULT_COMPACT_ENV_VALUES)
    for key, value in (overrides or {}).items():
        if key in values:
            values[key] = "" if value is None else str(value)

    layout_keys = {item for item in _COMPACT_ENV_LAYOUT if item and not item.startswith("#")}
    lines = ["# TranslateBook configuration. Full reference: .env.example"]
    for item in _COMPACT_ENV_LAYOUT:
        if item == "":
            lines.append("")
        elif item.startswith("#"):
            lines.append(item)
        else:
            lines.append(f"{item}={values[item]}")

    extra_keys = [
        key
        for key in (overrides or {})
        if key in values and key not in layout_keys and values[key]
    ]
    if extra_keys:
        lines.append("")
        for key in extra_keys:
            lines.append(f"{key}={values[key]}")

    return "\n".join(lines).rstrip() + "\n"


def write_compact_env(env_file: Path, overrides: dict | None = None) -> None:
    """Write a concise .env file to disk."""
    env_file.write_text(render_compact_env(overrides), encoding="utf-8")


def create_env_file(force: bool = False) -> bool:
    """
    Create a concise .env file.

    Args:
        force: If True, overwrites existing .env file

    Returns:
        bool: True if file was created, False otherwise
    """
    config_dir = _get_config_dir()
    env_file = config_dir / '.env'

    if env_file.exists() and not force:
        print(f"❌ .env file already exists at: {env_file.absolute()}")
        print("   Use force=True to overwrite")
        return False

    try:
        write_compact_env(env_file)
        print("✅ Created concise .env")
        print(f"   Location: {env_file.absolute()}")
        if (config_dir / '.env.example').exists():
            print("   Full option reference: .env.example")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env: {e}")
        return False


def create_env_from_template(force: bool = False) -> bool:
    """Backward-compatible alias for create_env_file()."""
    return create_env_file(force=force)


def validate_env_config(verbose: bool = True) -> dict:
    """
    Validate current environment configuration and return status

    Args:
        verbose: If True, prints detailed information

    Returns:
        dict: Status information about configuration
    """
    from dotenv import load_dotenv
    load_dotenv()

    config_dir = _get_config_dir()
    status = {
        'env_exists': (config_dir / '.env').exists(),
        'issues': [],
        'warnings': [],
        'config': {}
    }

    # Check critical configuration
    api_endpoint = os.getenv('API_ENDPOINT', 'http://localhost:11434/api/generate')
    llm_provider = os.getenv('LLM_PROVIDER', 'ollama')
    default_model = os.getenv('DEFAULT_MODEL', 'qwen3:14b')
    gemini_key = os.getenv('GEMINI_API_KEY', '')
    openai_key = os.getenv('OPENAI_API_KEY', '')

    status['config'] = {
        'api_endpoint': api_endpoint,
        'llm_provider': llm_provider,
        'model': default_model,
        'port': os.getenv('PORT', '5000'),
    }

    # Validate provider-specific requirements
    if llm_provider == 'gemini' and not gemini_key:
        status['issues'].append("Gemini provider selected but GEMINI_API_KEY is not set")

    if llm_provider == 'openai' and not openai_key:
        status['issues'].append("OpenAI provider selected but OPENAI_API_KEY is not set")

    if llm_provider == 'ollama' and 'localhost' not in api_endpoint and '127.0.0.1' not in api_endpoint:
        status['warnings'].append(f"Using remote Ollama server: {api_endpoint}")

    # Check if using defaults (likely means no .env)
    if api_endpoint == 'http://localhost:11434/api/generate' and not status['env_exists']:
        status['warnings'].append("Using default localhost configuration - may not be correct")

    if verbose:
        print("\n" + "="*70)
        print("🔍 CONFIGURATION VALIDATION")
        print("="*70)
        print(f"\n📁 .env file exists: {'✅ Yes' if status['env_exists'] else '❌ No (using defaults)'}")
        print(f"\n⚙️  Current Configuration:")
        print(f"   • LLM Provider: {llm_provider}")
        print(f"   • API Endpoint: {api_endpoint}")
        print(f"   • Model: {default_model}")
        print(f"   • Port: {status['config']['port']}")

        if status['issues']:
            print(f"\n❌ CRITICAL ISSUES:")
            for issue in status['issues']:
                print(f"   • {issue}")

        if status['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in status['warnings']:
                print(f"   • {warning}")

        if not status['issues'] and not status['warnings']:
            print(f"\n✅ Configuration looks good!")

        print("="*70 + "\n")

    return status


def interactive_env_setup():
    """
    Interactive setup wizard for .env configuration
    """
    print("\n" + "="*70)
    print("🛠️  INTERACTIVE .ENV SETUP WIZARD")
    print("="*70)

    config_dir = _get_config_dir()
    env_file = config_dir / '.env'

    if env_file.exists():
        response = input("\n.env file already exists. Overwrite? (yes/no): ").strip().lower()
        if response != 'yes':
            print("❌ Setup cancelled")
            return

    print("\n📋 Please provide the following information:")
    print("   (Press Enter to use default values shown in brackets)\n")

    # Collect configuration
    config = {}

    print("1️⃣  LLM Provider")
    config['LLM_PROVIDER'] = input("   Provider (ollama/gemini/openai) [ollama]: ").strip() or 'ollama'

    if config['LLM_PROVIDER'] == 'ollama':
        config['API_ENDPOINT'] = input("   Ollama API endpoint [http://localhost:11434/api/generate]: ").strip() or 'http://localhost:11434/api/generate'
        config['DEFAULT_MODEL'] = input("   Model name [qwen3:14b]: ").strip() or 'qwen3:14b'

    elif config['LLM_PROVIDER'] == 'gemini':
        config['GEMINI_API_KEY'] = input("   Gemini API Key: ").strip()
        config['GEMINI_MODEL'] = input("   Gemini Model [gemini-2.0-flash]: ").strip() or 'gemini-2.0-flash'

    elif config['LLM_PROVIDER'] == 'openai':
        config['OPENAI_API_KEY'] = input("   OpenAI API Key: ").strip()
        config['API_ENDPOINT'] = input("   API endpoint [https://api.openai.com/v1/chat/completions]: ").strip() or 'https://api.openai.com/v1/chat/completions'
        config['DEFAULT_MODEL'] = input("   Model [gpt-4o]: ").strip() or 'gpt-4o'

    config['PORT'] = input("\n2️⃣  Web server port [5000]: ").strip() or '5000'
    config['DEFAULT_SOURCE_LANGUAGE'] = input("3️⃣  Default source language [English]: ").strip() or 'English'
    config['DEFAULT_TARGET_LANGUAGE'] = input("4️⃣  Default target language [Chinese]: ").strip() or 'Chinese'

    overrides = {
        'LLM_PROVIDER': config['LLM_PROVIDER'],
        'PORT': config['PORT'],
        'DEFAULT_SOURCE_LANGUAGE': config['DEFAULT_SOURCE_LANGUAGE'],
        'DEFAULT_TARGET_LANGUAGE': config['DEFAULT_TARGET_LANGUAGE'],
    }
    if config['LLM_PROVIDER'] == 'ollama':
        overrides['API_ENDPOINT'] = config['API_ENDPOINT']
        overrides['OLLAMA_API_ENDPOINT'] = config['API_ENDPOINT']
        overrides['DEFAULT_MODEL'] = config['DEFAULT_MODEL']
    elif config['LLM_PROVIDER'] == 'gemini':
        overrides['GEMINI_API_KEY'] = config.get('GEMINI_API_KEY', '')
        overrides['GEMINI_MODEL'] = config.get('GEMINI_MODEL', 'gemini-2.0-flash')
    elif config['LLM_PROVIDER'] == 'openai':
        overrides['OPENAI_API_KEY'] = config.get('OPENAI_API_KEY', '')
        overrides['API_ENDPOINT'] = config.get(
            'API_ENDPOINT',
            'https://api.openai.com/v1/chat/completions',
        )
        overrides['DEFAULT_MODEL'] = config.get('DEFAULT_MODEL', 'gpt-4o')

    # Write .env file
    try:
        write_compact_env(env_file, overrides)

        print("\n✅ .env file created successfully!")
        print(f"   Location: {env_file.absolute()}")
        print("   Full option reference: .env.example")
        print("\n💡 You can edit this file manually at any time to adjust settings.\n")

    except Exception as e:
        print(f"\n❌ Failed to create .env file: {e}\n")


def cleanup_legacy_env_flags() -> None:
    """Automatically detect and remove obsolete environment flags (such as USE_LLM_SANITIZER) from .env files on disk."""
    legacy_keys = {"USE_LLM_SANITIZER"}
    config_dir = _get_config_dir()
    for env_filename in [".env", ".env.example"]:
        env_path = config_dir / env_filename
        if env_path.exists():
            try:
                content = env_path.read_text(encoding="utf-8")
                lines = content.splitlines()
                new_lines = [
                    line for line in lines
                    if not any(
                        line.strip().startswith(key + "=") or line.strip() == key
                        for key in legacy_keys
                    )
                ]
                if len(new_lines) != len(lines):
                    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            except Exception:
                pass


# Auto-run legacy flag cleanup on import/load
cleanup_legacy_env_flags()


if __name__ == '__main__':
    """Allow running this script standalone for configuration"""
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'create':
            raise SystemExit(0 if create_env_file() else 1)
        elif command == 'validate':
            validate_env_config()
        elif command == 'setup':
            interactive_env_setup()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: create, validate, setup")
    else:
        print("\nUsage:")
        print("  python -m src.utils.env_helper create   - Create concise .env")
        print("  python -m src.utils.env_helper validate - Check current configuration")
        print("  python -m src.utils.env_helper setup    - Interactive setup wizard")
