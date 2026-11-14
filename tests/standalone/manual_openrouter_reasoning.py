"""
Live check that OpenRouter models stop reasoning when TBL asks them to.

Runs one real translation per model through the project's factory and prints
the resolved `reasoning` override plus the token counts (including the
reasoning tokens OpenRouter reports), then repeats each model with its own
default so the difference is visible in the same output. This is the test that
proves the feature works against the live API, beyond the metadata mapping
pinned by tests/unit/test_openrouter_reasoning.py.

Requires OPENROUTER_API_KEY in .env. Costs well under a cent (one short prompt
per model and mode).

Run from repo root:
    python tests/standalone/manual_openrouter_reasoning.py

Pass model ids to check others:
    python tests/standalone/manual_openrouter_reasoning.py z-ai/glm-5.3-flash
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.console import ensure_utf8_stdio

ensure_utf8_stdio()

from src.core.llm.factory import create_llm_provider

# Reasoning on by default with an effort list (DeepSeek), on by default with a
# token budget only (Qwen), and mandatory reasoning (Gemini).
DEFAULT_MODELS = [
    "deepseek/deepseek-v4.1-flash",
    "qwen/qwen3.7-flash",
    "google/gemini-3.8-flash",
]

SYSTEM_PROMPT = "You are a professional literary translator. Output only the translation."
USER_PROMPT = (
    "Translate the following Chinese text into Spanish. "
    "Return only the translation, wrapped in <TRANSLATED></TRANSLATED>.\n\n"
    "<TRANSLATE>他推开门，看见院子里站着一个穿白衣的少年，手里握着一把旧剑。"
    "月光落在剑鞘上，像一层薄薄的霜。</TRANSLATE>"
)


async def run_one(model: str, disable_thinking: bool) -> None:
    provider = create_llm_provider(
        "openrouter",
        model=model,
        openrouter_disable_thinking=disable_thinking,
    )
    try:
        override = await provider._get_reasoning_override()
        # The provider prints the per-request line with the reasoning count.
        response = await provider.generate(USER_PROMPT, system_prompt=SYSTEM_PROMPT)
        if response is None:
            print(f"  {model:32s} FAILED (no response)")
            return
        print(f"  {model:32s} reasoning={override or '(model default)'}")
        print(f"  {'':32s} prompt={response.prompt_tokens} "
              f"completion={response.completion_tokens} "
              f"chars={len(response.content)}")
        print(f"  {'':32s} {response.content.strip()[:120]}")
    finally:
        await provider.close()


async def main() -> None:
    models = sys.argv[1:] or DEFAULT_MODELS

    print("Reasoning disabled (TBL default):")
    for model in models:
        await run_one(model, True)

    print("\nModel defaults (OPENROUTER_DISABLE_THINKING=false):")
    for model in models:
        await run_one(model, False)


if __name__ == "__main__":
    asyncio.run(main())
