import tempfile
from pathlib import Path
from src.utils.env_helper import ensure_env_defaults, write_compact_env

def test_ensure_env_defaults():
    with tempfile.TemporaryDirectory() as tmp_dir:
        env_file = Path(tmp_dir) / ".env"
        env_file.write_text("LLM_PROVIDER=gemini\nGEMINI_API_KEY=test_key\n", encoding="utf-8")

        added = ensure_env_defaults(env_file)
        assert "ENABLE_CHUNK_REFLECTION" in added
        assert "USE_LLM_SANITIZER" in added

        content = env_file.read_text(encoding="utf-8")
        assert "LLM_PROVIDER=gemini" in content
        assert "GEMINI_API_KEY=test_key" in content
        assert "ENABLE_CHUNK_REFLECTION=false" in content
        assert "USE_LLM_SANITIZER=false" in content

        # Running again should not re-add existing keys
        added_second = ensure_env_defaults(env_file)
        assert len(added_second) == 0
