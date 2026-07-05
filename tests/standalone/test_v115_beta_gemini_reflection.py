"""
Standalone test script to verify v1.15.0-beta.1 LLM Chunk Reflection Engine
with Gemini API keys loaded from F:\\TranslateBook_Data\\.env.
"""

import asyncio
import sys
from pathlib import Path
from dotenv import dotenv_values

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from src.core.llm import create_llm_provider
from src.prompts.prompts import generate_chunk_reflection_prompt
from src.utils.universal_addressing_engine import UniversalAddressingEngine


async def main():
    print("=================================================================")
    print("Testing v1.15.0-beta.1 LLM Reflection & Register Alignment Engine")
    print("=================================================================\n")

    # 1. Test Incompatible Register Pair Repair
    engine = UniversalAddressingEngine(language="vi")
    s, t, v = engine.validate_and_repair_pair("tôi", "ngươi", "Villain", "Hero")
    print(f"[1] Python Harmonious Repair Test: ('tôi', 'ngươi') -> ('{s}', '{t}')")
    assert s == "ta" and t == "ngươi", "Failed register pair repair!"
    print("    PASSED 100%\n")

    # 2. Test Gemini API Reflection Pass
    f_env = dotenv_values("F:/TranslateBook_Data/.env")
    key = f_env.get("GEMINI_API_KEY")
    if not key:
        print("FAIL: GEMINI_API_KEY not found in F:\\TranslateBook_Data\\.env")
        return

    print("[2] Initializing Gemini Provider (gemini-2.0-flash-lite)...")
    provider = create_llm_provider("gemini", api_key=key, model="gemini-2.0-flash-lite")

    source_chunk = "「私、トミオトレーナーのことが大好きなの」と、グラスワンダーは静かに微笑んだ。"
    draft_translation = "Grass Wonder mỉm cười nhẹ nhàng: 'Em yêu HLV Tomio lắm.'"
    novel_lore = "Grass Wonder: Female, Umamusume. Tomio Momozawa: Male Trainer."

    ref_prompt = generate_chunk_reflection_prompt(
        source_chunk=source_chunk,
        draft_translation=draft_translation,
        target_language="Vietnamese",
        novel_context=novel_lore,
    )

    print("\n[3] Executing Senior Editor Reflection Pass on draft chunk...")
    print(f"    Draft: \"{draft_translation}\"")
    
    try:
        res = await provider.generate(ref_prompt.user)
        if res and res.content:
            print("\n================ SENIOR EDITOR CRITIQUE RESULT ================")
            print(res.content.strip())
            print("===============================================================")
        else:
            print("Notice: API request returned empty response.")
    except Exception as e:
        print(f"Notice: Rate limit or API exception encountered: {e}")


if __name__ == "__main__":
    asyncio.run(main())
