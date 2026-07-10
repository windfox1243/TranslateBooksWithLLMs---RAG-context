"""
Translation module for LLM communication
"""
import inspect
import json
import time
import re
from dataclasses import dataclass
from tqdm.auto import tqdm

from src.config import (
    DEFAULT_MODEL, TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT, SENTENCE_TERMINATORS,
    THINKING_MODELS, ADAPTIVE_CONTEXT_INITIAL_THINKING
)
from src.prompts.prompts import (
    REFLECTION_CONTRACT_VERSION,
    REFLECTION_JSON_TAG_IN,
    REFLECTION_JSON_TAG_OUT,
    REFLECTION_PROMPT_VERSION,
    generate_translation_prompt,
    generate_refinement_prompt,
)
from .llm_client import default_client, create_llm_client, LLMResponse
from .llm import ContextOverflowError, RepetitionLoopError, RateLimitError
from .post_processor import clean_translated_text
from .context_optimizer import (
    AdaptiveContextManager,
    INITIAL_CONTEXT_SIZE,
    CONTEXT_STEP
)
from .progress_tracker import TokenProgressTracker
from .chunking.token_chunker import TokenChunker
from typing import List, Dict, Tuple, Optional, Any, Callable
from src.utils.progress_logging import emit_progress_log


# Configuration for context overflow recovery
MAX_CHUNK_REDUCTION_ATTEMPTS = 3
CHUNK_REDUCTION_FACTOR = 0.6  # Reduce to 60% of original size each attempt
MIN_CHUNK_CHARACTERS = 200  # Minimum chunk size to attempt translation


@dataclass
class ReflectionResult:
    """Parsed Senior Editor reflection result."""

    status: str
    issues: List[Dict[str, Any]]
    raw_text: str = ""
    parse_status: str = "empty"

    @property
    def needs_repair(self) -> bool:
        return self.status == "needs_repair" and bool(self.issues)


class ReflectionValidationError(RuntimeError):
    """Raised when a contract-v2 editor gate cannot validate a chunk."""


def _build_chunk_glossary_block(
    chunk_content: str,
    prompt_options: Optional[dict],
    log_callback=None,
    runtime_state: Optional[dict] = None,
) -> str:
    """
    Filter the active glossary against the current chunk and render a prompt block.

    Reads `glossary_terms` (dict source -> target) and optional `glossary_config`
    (GlossaryConfig) from prompt_options. Returns "" when no glossary is active
    or no terms match this chunk.

    When the per-chunk cap is hit and `warn_on_cap` is enabled, logs a single
    warning per job. The dedupe flag lives in `runtime_state` (a transient dict
    owned by the caller) so it never leaks into the persisted prompt_options
    snapshot. If runtime_state is None, a fresh local dict is used (warning
    won't be deduped across calls — fine for ad-hoc uses).
    """
    if not prompt_options:
        return ""
    terms = prompt_options.get("glossary_terms")
    if not terms:
        return ""
    try:
        from src.core.glossary import filter_glossary, build_glossary_block, GlossaryConfig
    except ImportError:
        return ""
    config = prompt_options.get("glossary_config") or GlossaryConfig()
    if not getattr(config, "source_language", ""):
        try:
            config.source_language = str(prompt_options.get("source_language") or "")
        except Exception:
            pass
    filtered, capped = filter_glossary(chunk_content, terms, config)

    if runtime_state is None:
        runtime_state = {}

    if capped and config.warn_on_cap and not runtime_state.get("glossary_cap_warned"):
        runtime_state["glossary_cap_warned"] = True
        emit_progress_log(
            log_callback,
            "glossary_capped",
            f"⚠️ Glossary cap reached: more than {config.max_entries} terms matched in a single chunk. "
            f"Excess entries are dropped — increase `max_entries` if you need full coverage.",
            layer="glossary",
            data={"max_entries": config.max_entries},
        )

    if not filtered:
        return ""

    metadata = prompt_options.get("glossary_term_metadata") or None
    return build_glossary_block(filtered, term_metadata=metadata)


def split_chunk_for_retry(main_content: str, target_ratio: float = 0.5) -> Tuple[str, str]:
    """
    Split a chunk into two parts for retry after context overflow.

    Tries to split at a sentence boundary near the target ratio.

    Args:
        main_content: The text content to split
        target_ratio: Target position for split (0.5 = middle)

    Returns:
        Tuple of (first_half, second_half)
    """
    if not main_content.strip():
        return main_content, ""

    lines = main_content.split('\n')
    if len(lines) <= 1:
        # For single line, split at sentence boundary or middle
        target_pos = int(len(main_content) * target_ratio)

        # Look for sentence terminators near target position
        best_split = target_pos
        for terminator in SENTENCE_TERMINATORS:
            # Search in a window around target position
            search_start = max(0, target_pos - 100)
            search_end = min(len(main_content), target_pos + 100)
            search_area = main_content[search_start:search_end]

            term_pos = search_area.rfind(terminator)
            if term_pos != -1:
                actual_pos = search_start + term_pos + len(terminator)
                if abs(actual_pos - target_pos) < abs(best_split - target_pos):
                    best_split = actual_pos

        return main_content[:best_split].strip(), main_content[best_split:].strip()

    # For multi-line content, split at line boundaries
    target_line = int(len(lines) * target_ratio)

    # Look for a sentence-ending line near target
    best_line = target_line
    for i in range(max(0, target_line - 5), min(len(lines), target_line + 5)):
        line_stripped = lines[i].strip()
        if line_stripped and line_stripped.endswith(SENTENCE_TERMINATORS):
            best_line = i + 1
            break

    first_half = '\n'.join(lines[:best_line])
    second_half = '\n'.join(lines[best_line:])

    return first_half.strip(), second_half.strip()





async def _make_llm_request_with_adaptive_context(
    main_content: str,
    context_before: str,
    context_after: str,
    previous_translation_context: str,
    source_language: str,
    target_language: str,
    model: str,
    llm_client,
    log_callback,
    has_placeholders: bool,
    prompt_options: dict = None,
    context_manager: AdaptiveContextManager = None,
    placeholder_format: Optional[Tuple[str, str]] = None,
    runtime_state: Optional[dict] = None,
) -> Tuple[Optional[str], str, Optional[LLMResponse]]:
    """
    Make LLM request with adaptive context sizing.

    This function uses the AdaptiveContextManager to:
    1. Start with a small context
    2. Retry with larger context if needed
    3. Return token usage info for the manager to learn from

    Args:
        main_content: Text to translate
        context_before: Context before main content
        context_after: Context after main content
        previous_translation_context: Previous translation for consistency
        source_language: Source language
        target_language: Target language
        model: LLM model name
        llm_client: LLM client instance
        log_callback: Logging callback function
        has_placeholders: If True, includes placeholder preservation instructions (for EPUB HTML tags)
        prompt_options: Optional dict with prompt customization options
        context_manager: AdaptiveContextManager for context sizing

    Returns:
        Tuple of (translated_text or None, actual_content_translated, LLMResponse)
    """
    current_content = main_content
    remaining_content = ""
    all_translations = []
    reduction_attempt = 0
    last_response: Optional[LLMResponse] = None

    while current_content.strip():
        try:
            # Build the per-chunk glossary block (empty if no glossary configured)
            glossary_block = _build_chunk_glossary_block(
                current_content, prompt_options, log_callback=log_callback,
                runtime_state=runtime_state,
            )

            # Generate prompts
            prompt_pair = generate_translation_prompt(
                current_content,
                context_before,
                context_after,
                previous_translation_context,
                source_language,
                target_language,
                has_placeholders=has_placeholders,
                prompt_options=prompt_options,
                placeholder_format=placeholder_format,
                glossary_block=glossary_block,
            )

            # Log the request
            if log_callback and reduction_attempt == 0:
                emit_progress_log(
                    log_callback,
                    "llm_request",
                    "Sending request to LLM",
                    layer="translation",
                    data={
                        'system_prompt': prompt_pair.system,
                        'user_prompt': prompt_pair.user,
                        'model': model,
                    },
                )

            start_time = time.time()
            client = llm_client or default_client

            # Set context from manager if available
            if context_manager and hasattr(client, 'context_window'):
                new_ctx = context_manager.get_context_size()
                if client.context_window != new_ctx:
                    if log_callback:
                        log_callback("context_update",
                            f"📐 Updating context window: {client.context_window} → {new_ctx}")
                    else:
                        tqdm.write(f"\n📐 Context: {client.context_window} → {new_ctx}")
                client.context_window = new_ctx

            llm_response = await client.generate(
                prompt_pair.user, system_prompt=prompt_pair.system
            )
            execution_time = time.time() - start_time

            if not llm_response:
                return None, main_content, None

            last_response = llm_response
            full_raw_response = llm_response.content

            if not full_raw_response or not full_raw_response.strip():
                # Empty/null content: the model returned nothing. This is almost
                # always a refusal or a provider-side moderation/policy block on
                # sensitive content (not a truncation), so retrying with a larger
                # context would not help. Fail this unit cleanly with a clear hint
                # instead of crashing later on None slicing.
                if log_callback:
                    log_callback("empty_llm_response",
                        "⚠️ Model returned an empty response (0 tokens). This usually "
                        "means the model refused or filtered this chunk — common with "
                        "sensitive/policy-flagged content or provider-side moderation. "
                        "Try a different model.")
                return None, main_content, last_response

            # Check if we should retry with larger context (adaptive strategy)
            if context_manager and llm_response.was_truncated:
                if context_manager.should_retry_with_larger_context(
                    llm_response.was_truncated, llm_response.context_used
                ):
                    context_manager.increase_context()
                    continue  # Retry with larger context

            # Log the response
            if log_callback:
                emit_progress_log(
                    log_callback,
                    "llm_response",
                    "LLM Response received",
                    layer="translation",
                    data={
                        'response': full_raw_response,
                        'execution_time': execution_time,
                        'model': model,
                        'tokens': {
                            'prompt': llm_response.prompt_tokens,
                            'completion': llm_response.completion_tokens,
                            'total': llm_response.context_used,
                            'limit': llm_response.context_limit,
                        },
                    },
                )

            # Extract translation
            translated_text = client.extract_translation(full_raw_response)

            if translated_text:
                all_translations.append(translated_text)
            else:
                # Extraction failed - tags not found or malformed
                if log_callback:
                    log_callback("translation_extraction_failed",
                        "⚠️ WARNING: Failed to extract translation (tags not found or malformed)")
                    log_callback("translation_extraction_failed_preview",
                        f"Response preview (first 300 chars): {full_raw_response[:300]}")

                # Implicit truncation detection: model started <TRANSLATION> but hit EOS before </TRANSLATION>
                stripped_response = full_raw_response.strip()
                if (stripped_response.startswith(TRANSLATE_TAG_IN) and not stripped_response.endswith(TRANSLATE_TAG_OUT)):
                    if context_manager and context_manager.should_retry_with_larger_context(True, llm_response.context_used):
                        if log_callback:
                            log_callback("implicit_truncation_retry",
                                "🔄 Model stopped before closing tag. Retrying with larger context...")
                        context_manager.increase_context()
                        continue  # Retry with larger context

                # For EPUB with placeholders, failing to extract is CRITICAL
                # because using the raw response would include <TRANSLATION> tags in the HTML
                if has_placeholders:
                    if log_callback:
                        log_callback("epub_extraction_critical_fail",
                            "CRITICAL: Cannot use raw response for EPUB (would corrupt HTML structure)")
                    return None, main_content, last_response

                # For plain text, try fallback to raw response (legacy behavior)
                if current_content not in full_raw_response:
                    if log_callback:
                        log_callback("using_raw_response_fallback",
                            "Using raw response as fallback (plain text mode)")
                    all_translations.append(full_raw_response.strip())
                    if last_response:
                        last_response.was_fallback = True
                else:
                    # Response contains input - this is an error
                    if log_callback:
                        log_callback("llm_prompt_in_response_warning",
                            "WARNING: LLM response seems to contain input. Discarded.")
                    return None, main_content, last_response

            # If we had remaining content from a previous split, translate it
            if remaining_content.strip():
                current_content = remaining_content
                remaining_content = ""
                # Update context for continuity
                if all_translations:
                    words = all_translations[-1].split()
                    previous_translation_context = " ".join(words[-25:]) if len(words) > 25 else all_translations[-1]
                reduction_attempt = 0  # Reset for new content
                continue

            # Success - combine all translations
            combined = "\n".join(all_translations) if all_translations else None
            return combined, main_content, last_response

        except RepetitionLoopError as e:
            # Repetition loop detected - this typically happens with thinking models
            # when context window is too small. Try increasing context.
            if context_manager:
                old_context = context_manager.get_context_size()
                # Force a larger context increase for repetition loops
                context_manager.increase_context()
                context_manager.increase_context()  # Double increase for repetition loops
                new_context = context_manager.get_context_size()

                if new_context > old_context:
                    if log_callback:
                        log_callback("repetition_loop_retry",
                            f"🔄 Repetition loop detected! Increasing context from {old_context} to {new_context} tokens")
                    else:
                        tqdm.write(f"\n🔄 Repetition loop - increasing context to {new_context}")
                    continue  # Retry with larger context

            # No context manager or can't increase further
            if log_callback:
                log_callback("repetition_loop_fatal",
                    f"⚠️ Repetition loop detected and cannot recover. "
                    f"Try manually increasing OLLAMA_NUM_CTX. Error: {e}")
            else:
                tqdm.write(f"\n⚠️ Repetition loop detected - increase OLLAMA_NUM_CTX")
            return None, main_content, last_response

        except ContextOverflowError as e:
            # If we have a context manager, try increasing context
            if context_manager and context_manager.should_retry_with_larger_context(True, 0):
                context_manager.increase_context()
                continue  # Retry with larger context

            reduction_attempt += 1

            if reduction_attempt > MAX_CHUNK_REDUCTION_ATTEMPTS:
                if log_callback:
                    log_callback("context_overflow_fatal",
                        f"⚠️ Context overflow: Max reduction attempts ({MAX_CHUNK_REDUCTION_ATTEMPTS}) "
                        f"exceeded. Original error: {e}")
                else:
                    tqdm.write(f"\n⚠️ Context overflow after {MAX_CHUNK_REDUCTION_ATTEMPTS} reduction attempts")
                return None, main_content, last_response

            # Calculate new reduction factor
            reduction_factor = CHUNK_REDUCTION_FACTOR ** reduction_attempt

            if log_callback:
                log_callback("context_overflow_retry",
                    f"⚠️ Context overflow detected! Reducing chunk to {reduction_factor*100:.0f}% "
                    f"(attempt {reduction_attempt}/{MAX_CHUNK_REDUCTION_ATTEMPTS})")
            else:
                tqdm.write(f"\n⚠️ Context overflow - reducing chunk (attempt {reduction_attempt})")

            # Split the content
            first_part, second_part = split_chunk_for_retry(current_content, reduction_factor)

            if len(first_part) < MIN_CHUNK_CHARACTERS and not all_translations:
                # Can't reduce further without losing too much content
                if log_callback:
                    log_callback("context_overflow_fatal",
                        f"⚠️ Cannot reduce chunk further (min size: {MIN_CHUNK_CHARACTERS} chars)")
                return None, main_content, last_response

            current_content = first_part
            # Accumulate remaining content for later
            if second_part.strip():
                remaining_content = second_part + ("\n" + remaining_content if remaining_content else "")

    # Shouldn't reach here normally
    return "\n".join(all_translations) if all_translations else None, main_content, last_response


# Legacy wrapper for backward compatibility

async def generate_translation_request(main_content, context_before, context_after, previous_translation_context,
                                       source_language=None, target_language=None, model=DEFAULT_MODEL,
                                       llm_client=None, log_callback=None, has_placeholders=False,
                                       prompt_options=None, context_manager: AdaptiveContextManager = None,
                                       placeholder_format: Optional[Tuple[str, str]] = None):
    """
    Generate translation request to LLM API with automatic context overflow handling.

    Args:
        main_content (str): Text to translate
        context_before (str): Context before main content
        context_after (str): Context after main content
        previous_translation_context (str): Previous translation for consistency
        source_language (str): Source language
        target_language (str): Target language
        model (str): LLM model name
        llm_client: LLM client instance
        log_callback (callable): Logging callback function
        has_placeholders (bool): If True, includes placeholder preservation instructions
        prompt_options (dict): Optional dict with prompt customization options
        context_manager (AdaptiveContextManager): Optional context manager for adaptive retry on overflow
        placeholder_format (Tuple[str, str]): Optional tuple of (prefix, suffix) for placeholders.
            e.g., ('[', ']') for [0] format or ('[[', ']]') for [[0]] format

    Returns:
        str: Translated text or None if failed
    """
    if prompt_options is None:
        prompt_options = {}
    else:
        prompt_options = dict(prompt_options)
    if source_language and not prompt_options.get("source_language"):
        prompt_options["source_language"] = source_language
    if target_language and not prompt_options.get("target_language"):
        prompt_options["target_language"] = target_language

    # Skip LLM translation for single character or empty chunks
    if len(main_content.strip()) <= 1:
        if log_callback:
            log_callback("skip_translation", f"Skipping LLM for single/empty character: '{main_content}'")
        return main_content

    # Use the adaptive context handler
    translated_text, _, _ = await _make_llm_request_with_adaptive_context(
        main_content=main_content,
        context_before=context_before,
        context_after=context_after,
        previous_translation_context=previous_translation_context,
        source_language=source_language,
        target_language=target_language,
        model=model,
        llm_client=llm_client,
        log_callback=log_callback,
        has_placeholders=has_placeholders,
        prompt_options=prompt_options,
        context_manager=context_manager,
        placeholder_format=placeholder_format
    )

    if translated_text:
        return translated_text
    else:
        err_msg = "ERROR: LLM API request failed"
        if log_callback:
            log_callback("llm_api_error", err_msg)
        else:
            tqdm.write(f"\n{err_msg}")
        return None



async def _make_refinement_request(
    draft_translation: str,
    context_before: str,
    context_after: str,
    previous_refined_context: str,
    target_language: str,
    model: str,
    llm_client,
    log_callback,
    has_placeholders: bool,
    prompt_options: dict = None,
    context_manager: AdaptiveContextManager = None,
    runtime_state: Optional[dict] = None,
    **kwargs
) -> Tuple[Optional[str], Optional[LLMResponse]]:
    """
    Make LLM request for refinement pass.

    Similar to translation request but uses the refinement prompt.

    Args:
        draft_translation: First-pass translation to refine
        context_before: Previously refined text for context
        context_after: Text appearing after for context
        previous_refined_context: Last refined text for consistency
        target_language: Target language
        model: LLM model name
        llm_client: LLM client instance
        log_callback: Logging callback function
        has_placeholders: If True, includes placeholder preservation instructions
        prompt_options: Optional dict with prompt customization options
        context_manager: AdaptiveContextManager for context sizing
        dynamic_context: Optional decompressed dynamic relationship state snapshot

    Returns:
        Tuple of (refined_text or None, LLMResponse)
    """
    # Extract refinement instructions from prompt_options
    refinement_instructions = prompt_options.get('refinement_instructions', '') if prompt_options else ''

    # Filter the glossary against the DRAFT (target language) — terms that survived
    # the first pass are the ones we want to keep stable through refinement.
    glossary_block = _build_chunk_glossary_block(
        draft_translation, prompt_options, log_callback=log_callback,
        runtime_state=runtime_state,
    )

    # Inject historical/full context if provided.
    dynamic_context = kwargs.get('dynamic_context')
    context_content = kwargs.get('context_content')
    local_prompt_options = dict(prompt_options) if prompt_options else {}
    if context_content:
        from src.utils.novel_context import normalize_refinement_context
        local_prompt_options['novel_context'] = normalize_refinement_context(
            context_content,
            local_prompt_options.get('novel_context', ''),
        )
    elif dynamic_context:
        base_context = local_prompt_options.get('novel_context', '')
        if base_context:
            from src.utils.novel_context import normalize_refinement_context
            local_prompt_options['novel_context'] = normalize_refinement_context(
                dynamic_context,
                base_context,
            )

    # Generate refinement prompts
    prompt_pair = generate_refinement_prompt(
        draft_translation=draft_translation,
        context_before=context_before,
        context_after=context_after,
        previous_refined_context=previous_refined_context,
        target_language=target_language,
        has_placeholders=False,
        prompt_options=local_prompt_options,
        additional_instructions=refinement_instructions,
        glossary_block=glossary_block,
    )

    client = llm_client or default_client
    last_response: Optional[LLMResponse] = None

    # Retry loop with adaptive context (mirrors translation logic)
    while True:
        try:
            # Log the request
            if log_callback:
                emit_progress_log(
                    log_callback,
                    "refinement_request",
                    "Sending refinement request to LLM",
                    layer="refinement",
                    data={
                        'system_prompt': prompt_pair.system,
                        'user_prompt': prompt_pair.user,
                        'model': model,
                    },
                )

            start_time = time.time()

            # Set context from manager if available
            if context_manager and hasattr(client, 'context_window'):
                new_ctx = context_manager.get_context_size()
                if client.context_window != new_ctx:
                    if log_callback:
                        log_callback("context_update",
                            f"📐 Refinement context window: {client.context_window} → {new_ctx}")
                    client.context_window = new_ctx

            llm_response = await client.make_request(
                prompt_pair.user, model, system_prompt=prompt_pair.system
            )
            execution_time = time.time() - start_time

            if not llm_response:
                return None, None

            last_response = llm_response

            # Check if we should retry with larger context (adaptive strategy)
            if context_manager and llm_response.was_truncated:
                if context_manager.should_retry_with_larger_context(
                    llm_response.was_truncated, llm_response.context_used
                ):
                    context_manager.increase_context()
                    continue  # Retry with larger context

            full_raw_response = llm_response.content

            # Log the response
            if log_callback:
                emit_progress_log(
                    log_callback,
                    "refinement_response",
                    "Refinement response received",
                    layer="refinement",
                    data={
                        'response': full_raw_response,
                        'execution_time': execution_time,
                        'model': model,
                        'tokens': {
                            'prompt': llm_response.prompt_tokens,
                            'completion': llm_response.completion_tokens,
                            'total': llm_response.context_used,
                            'limit': llm_response.context_limit,
                        },
                    },
                )

            # Extract refined text
            refined_text = client.extract_translation(full_raw_response)

            if refined_text:
                return refined_text, llm_response
            else:
                # Fallback to raw response if no tags found
                if draft_translation not in full_raw_response:
                    return full_raw_response.strip(), llm_response
                else:
                    if log_callback:
                        log_callback("refinement_warning",
                            "WARNING: Refinement response contains input. Using original.")
                    return None, llm_response

        except RepetitionLoopError as e:
            # Repetition loop detected - try increasing context (double increase)
            if context_manager:
                old_context = context_manager.get_context_size()
                context_manager.increase_context()
                context_manager.increase_context()  # Double increase for repetition loops
                new_context = context_manager.get_context_size()

                if new_context > old_context:
                    if log_callback:
                        log_callback("refinement_repetition_retry",
                            f"🔄 Refinement repetition loop! Increasing context from {old_context} to {new_context} tokens")
                    continue  # Retry with larger context

            # No context manager or can't increase further
            if log_callback:
                log_callback("refinement_error",
                    f"⚠️ Refinement repetition loop, cannot recover: {e}")
            return None, last_response

        except ContextOverflowError as e:
            # Context overflow - try increasing context
            if context_manager and context_manager.should_retry_with_larger_context(True, 0):
                context_manager.increase_context()
                if log_callback:
                    log_callback("refinement_overflow_retry",
                        f"⚠️ Refinement context overflow! Retrying with context {context_manager.get_context_size()}")
                continue  # Retry with larger context

            # Can't increase further
            if log_callback:
                log_callback("refinement_error",
                    f"⚠️ Refinement context overflow, cannot recover: {e}")
            return None, last_response


async def refine_chunks(
    translated_chunks: List[str],
    original_chunks: List[Dict],
    target_language: str,
    model_name: str,
    api_endpoint: str,
    log_callback=None,
    stats_callback=None,
    check_interruption_callback=None,
    llm_provider="ollama",
    gemini_api_key=None,
    openai_api_key=None,
    openrouter_api_key=None,
    mistral_api_key=None,
    deepseek_api_key=None,
    poe_api_key=None,
    nim_api_key=None,
    context_window=2048,
    auto_adjust_context=True,
    prompt_options=None,
    dynamic_contexts=None,
    context_tracker=None,
) -> List[str]:
    """
    Refine translated chunks with a second pass for literary quality improvement.

    This function takes already-translated chunks and runs them through a
    refinement prompt that focuses on improving literary quality, natural flow,
    and stylistic excellence.

    Args:
        translated_chunks: List of translated text strings from first pass
        original_chunks: Original chunk dictionaries (for context structure)
        target_language: Target language name
        model_name: LLM model name
        api_endpoint: API endpoint        log_callback: Logging callback
        stats_callback: Statistics update callback
        check_interruption_callback: Interruption check callback
        llm_provider: LLM provider name
        gemini_api_key: Gemini API key
        openai_api_key: OpenAI API key
        openrouter_api_key: OpenRouter API key
        context_window: Initial context window size
        auto_adjust_context: Enable adaptive context adjustment
        prompt_options: Optional dict with prompt customization options

    Returns:
        List of refined text strings
    """
    total_chunks = len(translated_chunks)
    refined_parts = []
    last_refined_context = ""
    # Transient per-job state (e.g. glossary cap warning dedupe) — never persisted.
    runtime_state: dict = {}

    # Single-phase refinement tracker (the workflow phase, when this runs as the
    # second pass of a translate→refine job, is tagged at the handler seam).
    progress_tracker = TokenProgressTracker()
    progress_tracker.start()
    token_counter = TokenChunker(max_tokens=800)
    for chunk_text in translated_chunks:
        token_count = token_counter.count_tokens(chunk_text)
        progress_tracker.register_chunk(token_count)

    if log_callback:
        log_callback("refinement_start", f"✨ Starting refinement pass ({total_chunks} chunks)...")

    # Determine if model is a thinking model for initial context sizing
    is_known_thinking_model = any(tm in model_name.lower() for tm in THINKING_MODELS)

    # Refinement needs MORE context than translation because:
    # - The prompt includes the already-translated text (input)
    # - Plus context before/after
    # - Plus instructions
    # So we start with at least 4096 or the user's context_window, whichever is larger
    REFINEMENT_MIN_CONTEXT = 4096

    if auto_adjust_context:
        if is_known_thinking_model:
            initial_context = max(ADAPTIVE_CONTEXT_INITIAL_THINKING, REFINEMENT_MIN_CONTEXT)
        else:
            initial_context = max(INITIAL_CONTEXT_SIZE * 2, REFINEMENT_MIN_CONTEXT)
    else:
        initial_context = max(context_window, REFINEMENT_MIN_CONTEXT)

    # Create LLM client
    llm_client = create_llm_client(
        llm_provider, gemini_api_key, api_endpoint, model_name,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        context_window=initial_context, log_callback=log_callback
    )

    if context_tracker is None:
        from src.utils.novel_context import RefinementContextTracker
        context_tracker = RefinementContextTracker(
            prompt_options=prompt_options or {},
            historical_contexts=dynamic_contexts or [],
            log_callback=log_callback,
        )

    # Create adaptive context manager for Ollama
    context_manager = None
    if llm_provider == "ollama" and auto_adjust_context:
        from .context_optimizer import MAX_CONTEXT_SIZE
        context_manager = AdaptiveContextManager(
            initial_context=initial_context,
            context_step=CONTEXT_STEP,
            max_context=MAX_CONTEXT_SIZE,
            log_callback=log_callback
        )
        if log_callback:
            log_callback("refinement_context", f"📐 Refinement context: starting at {initial_context} tokens (min for refinement: {REFINEMENT_MIN_CONTEXT})")

    # Detect thinking model status
    if llm_client and llm_provider == "ollama":
        await llm_client.detect_thinking_model()

    try:
        iterator = tqdm(
            enumerate(translated_chunks),
            total=total_chunks,
            desc=f"Refining {target_language} translation",
            unit="seg"
        ) if not log_callback else enumerate(translated_chunks)

        for i, draft_text in iterator:
            # Check for interruption
            if check_interruption_callback and check_interruption_callback():
                if log_callback:
                    log_callback("refinement_interrupted",
                        f"Refinement interrupted at chunk {i+1}/{total_chunks}")
                else:
                    tqdm.write(f"\nRefinement interrupted at chunk {i+1}/{total_chunks}")
                # Add remaining unrefined chunks as-is
                for remaining in translated_chunks[i:]:
                    refined_parts.append(remaining)
                break

            # Progress update (token-based)
            # Measure refinement time for this chunk
            chunk_start_time = time.time()
            if log_callback:
                log_callback(
                    "refinement_chunk_start",
                    f"🪄 Refining chunk {i+1}/{total_chunks}...",
                )

            # Skip empty chunks
            if not draft_text.strip():
                refined_parts.append(draft_text)
                chunk_elapsed = time.time() - chunk_start_time
                progress_tracker.mark_completed(i, chunk_elapsed)
                if stats_callback:
                    stats_callback(progress_tracker.get_stats().to_dict())
                if log_callback:
                    log_callback(
                        "refinement_chunk_complete",
                        f"✅ Refinement chunk {i+1}/{total_chunks} complete (empty chunk kept).",
                    )
                continue

            # Skip very short content
            if len(draft_text.strip()) <= 1:
                refined_parts.append(draft_text)
                chunk_elapsed = time.time() - chunk_start_time
                progress_tracker.mark_completed(i, chunk_elapsed)
                if stats_callback:
                    stats_callback(progress_tracker.get_stats().to_dict())
                if log_callback:
                    log_callback(
                        "refinement_chunk_complete",
                        f"✅ Refinement chunk {i+1}/{total_chunks} complete (short text kept).",
                    )
                continue

            # Get context from original chunks if available
            context_before = ""
            context_after = ""
            if i < len(original_chunks):
                context_before = original_chunks[i].get("context_before", "")
                context_after = original_chunks[i].get("context_after", "")

            context_content = await context_tracker.next_context(
                text=draft_text,
                llm_client=llm_client,
                model_name=model_name,
                target_language=target_language,
                display_index=i + 1,
                total_chunks=total_chunks,
                scene_key=(
                    original_chunks[i].get("chapter_index")
                    if i < len(original_chunks)
                    else None
                ),
            )
            local_prompt_options = dict(prompt_options or {})
            dialogue_attribution = None
            if i < len(original_chunks):
                dialogue_attribution = original_chunks[i].get(
                    "dialogue_attribution"
                )
            if not dialogue_attribution:
                dialogue_attribution = getattr(
                    context_tracker,
                    "current_dialogue_attribution",
                    None,
                )
            if dialogue_attribution:
                local_prompt_options["dialogue_attribution"] = (
                    dialogue_attribution
                )
            else:
                local_prompt_options.pop("dialogue_attribution", None)

            # Make refinement request
            try:
                refined_text, llm_response = await _make_refinement_request(
                    draft_translation=draft_text,
                    context_before=context_before,
                    context_after=context_after,
                    previous_refined_context=last_refined_context,
                    target_language=target_language,
                    model=model_name,
                    llm_client=llm_client,
                    log_callback=log_callback,
                    has_placeholders=False,
                    prompt_options=local_prompt_options,
                    context_manager=context_manager,
                    runtime_state=runtime_state,
                    context_content=context_content,
                )
            except RateLimitError as e:
                if log_callback:
                    retry_msg = f" (retry after ~{e.retry_after}s)" if e.retry_after else ""
                    log_callback("rate_limit_pause",
                        f"⏸️ Rate limited by {e.provider or 'API'}{retry_msg}. "
                        f"Auto-pausing refinement at chunk {i+1}/{total_chunks}...")
                # Add remaining unrefined chunks as-is
                for remaining in translated_chunks[i:]:
                    refined_parts.append(remaining)
                raise  # Re-raise to handlers.py

            # Record success in context manager
            if refined_text is not None and llm_response and context_manager:
                context_manager.record_success(
                    prompt_tokens=llm_response.prompt_tokens,
                    completion_tokens=llm_response.completion_tokens,
                    context_limit=llm_response.context_limit
                )

            chunk_elapsed = time.time() - chunk_start_time

            if refined_text is not None:
                # Clean the refined text
                refined_text = clean_translated_text(refined_text)
                refined_parts.append(refined_text)
                progress_tracker.mark_completed(i, chunk_elapsed)

                # Update context for next chunk
                words = refined_text.split()
                if len(words) > 25:
                    last_refined_context = " ".join(words[-25:])
                else:
                    last_refined_context = refined_text
                if log_callback:
                    log_callback(
                        "refinement_chunk_complete",
                        f"✅ Refinement chunk {i+1}/{total_chunks} complete.",
                    )
            else:
                # Keep original translation if refinement fails
                if log_callback:
                    log_callback("refinement_chunk_failed",
                        f"Refinement failed for chunk {i+1}, keeping original translation")
                refined_parts.append(draft_text)
                progress_tracker.mark_failed(i)
                last_refined_context = ""

            if stats_callback:
                stats_callback(progress_tracker.get_stats().to_dict())

    finally:
        if llm_client:
            await llm_client.close()

    stats = progress_tracker.get_stats()
    if log_callback:
        log_callback("refinement_complete",
            f"✨ Refinement complete: {stats.completed_chunks} refined, {stats.failed_chunks} kept original")

    return refined_parts


def _filter_actionable_critiques(critique_text: str) -> list[str]:
    """Extract actionable defect bullet points from critique text, filtering out positive validation commentary."""
    if not critique_text or not critique_text.strip():
        return []

    positive_indicators = [
        "this is correct", "is correct", "well-handled", "well handled",
        "correctly applied", "no error", "no issues", "is valid",
        "passes validation", "no changes needed", "no repair needed",
    ]

    lines = [line.strip() for line in critique_text.splitlines() if line.strip()]
    actionable = []
    for line in lines:
        if line.startswith(('*', '-', '•', '>')) or (len(line) > 2 and line[0].isdigit() and line[1] in ('.', ')')):
            clean = line.lstrip("*•-> 0123456789.)").strip()
            if clean.endswith(":") and (" " not in clean or clean.isupper() or len(clean.split()) <= 3):
                continue
            if clean:
                lower_clean = clean.lower()
                if any(pos in lower_clean for pos in positive_indicators):
                    continue
                actionable.append(clean)
    return actionable


def _strip_json_code_fence(text: str) -> str:
    """Remove a surrounding JSON code fence when present."""
    stripped = (text or "").strip()
    fence_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fence_match.group(1).strip() if fence_match else stripped


def _reflection_json_candidates(text: str) -> List[str]:
    """Return possible JSON payloads from a Senior Editor response."""
    if not text or not text.strip():
        return []

    candidates: List[str] = []
    tag_pattern = (
        rf"{re.escape(REFLECTION_JSON_TAG_IN)}\s*(.*?)\s*"
        rf"{re.escape(REFLECTION_JSON_TAG_OUT)}"
    )
    for match in re.finditer(tag_pattern, text, flags=re.IGNORECASE | re.DOTALL):
        candidates.append(match.group(1).strip())

    stripped = _strip_json_code_fence(text)
    candidates.append(stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start:end + 1])

    unique: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _load_reflection_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse the first valid Senior Editor JSON object from a response."""
    for candidate in _reflection_json_candidates(text):
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


def _normalize_reflection_issue(raw_issue: Any) -> Optional[Dict[str, Any]]:
    """Normalize one structured reflection issue into the internal shape."""
    if isinstance(raw_issue, str):
        instruction = raw_issue.strip()
        if not instruction:
            return None
        return {
            "category": "other",
            "severity": "major",
            "source_quote": "",
            "draft_quote": "",
            "instruction": instruction,
            "draft_replacement": None,
            "glossary_update": None,
            "term_replacement": None,
        }

    if not isinstance(raw_issue, dict):
        return None

    instruction = str(raw_issue.get("instruction") or "").strip()
    if not instruction:
        return None

    glossary_update = raw_issue.get("glossary_update")
    if not isinstance(glossary_update, dict):
        glossary_update = raw_issue.get("term_replacement")
    if isinstance(glossary_update, dict):
        source = str(glossary_update.get("source") or "").strip()
        target = str(glossary_update.get("target") or "").strip()
        glossary_update = (
            {"source": source, "target": target}
            if source and target
            else None
        )
    else:
        glossary_update = None

    draft_replacement = raw_issue.get("draft_replacement")
    if isinstance(draft_replacement, dict):
        draft_span = str(
            draft_replacement.get("draft")
            or draft_replacement.get("source")
            or draft_replacement.get("from")
            or ""
        ).strip()
        replacement = str(
            draft_replacement.get("replacement")
            or draft_replacement.get("target")
            or draft_replacement.get("to")
            or ""
        ).strip()
        draft_replacement = (
            {"draft": draft_span, "replacement": replacement}
            if draft_span and replacement
            else None
        )
    else:
        draft_replacement = None

    return {
        "category": str(raw_issue.get("category") or "other").strip() or "other",
        "severity": str(raw_issue.get("severity") or "major").strip() or "major",
        "source_quote": str(raw_issue.get("source_quote") or "").strip(),
        "draft_quote": str(raw_issue.get("draft_quote") or "").strip(),
        "instruction": instruction,
        "draft_replacement": draft_replacement,
        "glossary_update": glossary_update,
        "term_replacement": glossary_update,
    }


def _issue_requires_draft_replacement(issue: Dict[str, Any]) -> bool:
    """Return whether an issue describes a direct, locally verifiable edit."""

    category = str(issue.get("category") or "other").casefold()
    return bool(
        str(issue.get("draft_quote") or "").strip()
        and category not in {"omission", "placeholder", "format"}
        and not issue.get("deterministic")
    )


def _reflection_contract_incomplete(result: ReflectionResult) -> bool:
    if result.status == "needs_repair" and not result.issues:
        return True
    return any(
        _issue_requires_draft_replacement(issue)
        and not issue.get("draft_replacement")
        for issue in result.issues
    )


def parse_reflection_result(reflection_text: str) -> ReflectionResult:
    """Parse structured Senior Editor output with a conservative legacy fallback."""
    raw_text = reflection_text or ""
    if not raw_text.strip():
        return ReflectionResult("no_issues", [], raw_text, "empty")

    data = _load_reflection_json_object(raw_text)
    had_json_markers = (
        REFLECTION_JSON_TAG_IN in raw_text
        or REFLECTION_JSON_TAG_OUT in raw_text
        or raw_text.strip().startswith("{")
    )
    if data is not None:
        raw_status = str(data.get("status") or "").strip().lower().replace("-", "_")
        raw_issues = data.get("issues") if isinstance(data.get("issues"), list) else []
        issues = [
            issue for issue in (
                _normalize_reflection_issue(item) for item in raw_issues
            )
            if issue is not None
        ]
        if raw_status not in {"no_issues", "needs_repair"}:
            raw_status = "needs_repair" if issues else "no_issues"
        if issues and raw_status == "no_issues":
            raw_status = "needs_repair"
        parse_status = "json"
        if not issues and raw_status == "needs_repair":
            parse_status = "incomplete_json"
        return ReflectionResult(raw_status, issues, raw_text, parse_status)

    if raw_text.strip().upper() == "NO_ISSUES":
        return ReflectionResult("no_issues", [], raw_text, "legacy_no_issues")

    legacy_items = _filter_actionable_critiques(raw_text)
    if legacy_items:
        issues = [
            {
                "category": "other",
                "severity": "major",
                "source_quote": "",
                "draft_quote": "",
                "instruction": item,
                "draft_replacement": None,
                "glossary_update": None,
                "term_replacement": None,
            }
            for item in legacy_items
        ]
        return ReflectionResult("needs_repair", issues, raw_text, "legacy_bullets")

    if had_json_markers:
        return ReflectionResult("no_issues", [], raw_text, "invalid_json")
    return ReflectionResult("no_issues", [], raw_text, "legacy_no_action")


def _reflection_feedback_payload(result: ReflectionResult) -> str:
    """Serialize parsed reflection issues for the repair prompt."""
    return json.dumps(
        {
            "status": result.status,
            "issues": result.issues,
        },
        ensure_ascii=False,
        indent=2,
    )


def format_critique_tldr(critique_text: str, max_bullets: int = 3, max_len: int = 120) -> str:
    """Format a multi-line Senior Editor critique into a concise single-line TL;DR log message."""
    data = _load_reflection_json_object(critique_text)
    if data and isinstance(data.get("issues"), list):
        issue_summaries = []
        for issue in data.get("issues") or []:
            normalized = _normalize_reflection_issue(issue)
            if normalized:
                issue_summaries.append(normalized["instruction"])
        if issue_summaries:
            summaries = []
            for clean in issue_summaries[:max_bullets]:
                if len(clean) > max_len:
                    clean = clean[:max_len].rstrip() + "..."
                summaries.append(f"• {clean}")
            remaining = len(issue_summaries) - max_bullets
            if remaining > 0:
                summaries.append(f"(+{remaining} more)")
            return " | ".join(summaries)

    bullet_items = _filter_actionable_critiques(critique_text)

    if not bullet_items:
        lines = [line.strip() for line in (critique_text or "").splitlines() if line.strip()]
        compact = " ".join(lines)
        return compact[:180] + "..." if len(compact) > 180 else compact

    summaries = []
    for clean in bullet_items[:max_bullets]:
        if len(clean) > max_len:
            clean = clean[:max_len].rstrip() + "..."
        summaries.append(f"• {clean}")

    remaining = len(bullet_items) - max_bullets
    if remaining > 0:
        summaries.append(f"(+{remaining} more)")

    return " | ".join(summaries)


def _is_valid_glossary_term(term: str) -> bool:
    """Check if an extracted string is a valid glossary term candidate and not a full sentence or dialogue quote."""
    if not term:
        return False
    term_str = term.strip()
    if not (1 <= len(term_str) <= 60):
        return False
    # Reject dialogue/sentence punctuation and control characters
    if any(char in term_str for char in ['?', '!', '…', '\n', '\r', '\t', ';']):
        return False
    if '...' in term_str:
        return False
    words = term_str.split()
    # Reject sentences ending with a period/comma if multi-word
    if (term_str.endswith('.') or term_str.endswith(',')) and len(words) > 2:
        return False
    # Reject multi-word text containing sentence punctuation like period or comma
    if len(words) > 3 and (term_str.count(',') > 0 or term_str.count('.') > 0):
        return False
    return True


def extract_term_replacements_from_critique(critique: str) -> List[Tuple[str, str]]:
    """
    Extract term replacement pairs ordered by Senior Editor critique text.
    For example:
    - Change all instances of "Học viện Đào tạo Mã nương Nhật Bản" to "Học viện Tracen"
    - Replace "Tracen Academy" with "Học viện Tracen"
    - Change "Mã nương" -> "Umamusume"
    """
    if not critique or not critique.strip():
        return []

    results: List[Tuple[str, str]] = []
    structured = _load_reflection_json_object(critique)
    if structured and isinstance(structured.get("issues"), list):
        for issue in structured.get("issues") or []:
            if not isinstance(issue, dict):
                continue
            replacement = issue.get("glossary_update")
            if not isinstance(replacement, dict):
                replacement = issue.get("term_replacement")
            if not isinstance(replacement, dict):
                continue
            src_term = str(replacement.get("source") or "").strip()
            tgt_term = str(replacement.get("target") or "").strip()
            if (
                src_term
                and tgt_term
                and src_term != tgt_term
                and _is_valid_glossary_term(src_term)
                and _is_valid_glossary_term(tgt_term)
            ):
                results.append((src_term, tgt_term))

    clean_text = critique.replace("**", "")

    patterns = [
        r'(?:change|replace|convert)\s+(?:all\s+instances\s+of\s+)?["`„«]([^"`„»]+)["`„»]\s+(?:to|with)\s+["`„«]([^"`„»]+)["`„»]',
        r'["`„«]([^"`„»]+)["`„»]\s*(?:->|=>)\s*["`„«]([^"`„»]+)["`„»]',
        r'(?:change|replace|convert)\s+["`„«]([^"`„»]+)["`„»]\s*(?:->|=>)\s*["`„«]([^"`„»]+)["`„»]',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, clean_text, re.IGNORECASE):
            src_term = match.group(1).strip()
            tgt_term = match.group(2).strip()
            if src_term and tgt_term and src_term != tgt_term:
                if _is_valid_glossary_term(src_term) and _is_valid_glossary_term(tgt_term):
                    results.append((src_term, tgt_term))

    from src.utils.novel_context import _is_inverted_target_to_source_glossary_pair

    deduped: List[Tuple[str, str]] = []
    seen = set()
    for src, tgt in results:
        if _is_inverted_target_to_source_glossary_pair(src, tgt):
            src, tgt = tgt, src
        key = (src.casefold(), tgt.casefold())
        if key not in seen:
            seen.add(key)
            deduped.append((src, tgt))
    return deduped


def _render_reflection_novel_context(
    novel_context: str,
    prompt_options: Optional[Dict[str, Any]],
    source_chunk: str,
    draft_translation: str,
) -> str:
    """Render the same selective novel-context view for reflection and repair."""
    options = dict(prompt_options or {})
    if novel_context and not options.get("novel_context"):
        options["novel_context"] = novel_context

    raw_context = str(options.get("novel_context") or novel_context or "")
    active_speaker = None
    attribution = options.get("dialogue_attribution") or {}
    if isinstance(attribution, dict):
        state_after = attribution.get("state_after") or {}
        if isinstance(state_after, dict):
            active_speaker = state_after.get("speaker")

    active_context = ""
    if raw_context.strip():
        try:
            from src.utils.novel_context import render_novel_context_for_prompt

            rendered = render_novel_context_for_prompt(
                raw_context,
                reference_text="\n".join(
                    part for part in (source_chunk, draft_translation) if part
                ),
                max_tokens=options.get("novel_context_prompt_max_tokens"),
                selective=options.get("novel_context_selective_injection", True),
                active_speaker=active_speaker,
            )
            active_context = rendered.strip() or raw_context.strip()
        except Exception:
            active_context = raw_context.strip()
    directed_context = str(options.get("directed_addressing_context") or "").strip()
    relationship_context = str(options.get("relationship_context") or "").strip()
    blocks = []
    if directed_context:
        blocks.append(
            "# STRUCTURED DIRECTED ADDRESSING RULES\n"
            f"{directed_context}"
        )
    if relationship_context:
        blocks.append(
            "# STRUCTURED RELATIONSHIP CONTEXT\n"
            f"{relationship_context}"
        )
    if active_context:
        blocks.append(
            "# ACTIVE MARKDOWN NOVEL CONTEXT\n"
            f"{active_context}"
        )
    return "\n\n".join(blocks).strip()


async def _generate_editor_response(
    llm_client: Any,
    prompt: str,
    system_prompt: str,
    model_name: str,
    temperature: float,
):
    """Call the available LLM client method for reflection/repair."""
    if hasattr(llm_client, "generate_async"):
        result = llm_client.generate_async(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model_name,
            temperature=temperature,
        )
    elif hasattr(llm_client, "generate"):
        result = llm_client.generate(
            prompt,
            system_prompt=system_prompt,
            model=model_name,
            temperature=temperature,
        )
    elif hasattr(llm_client, "make_request"):
        result = llm_client.make_request(
            prompt,
            model_name,
            system_prompt=system_prompt,
        )
    else:
        raise AttributeError("LLM client has no supported generation method")

    if inspect.isawaitable(result):
        return await result
    return result


async def run_chunk_reflection_pass(
    source_chunk: str,
    draft_translation: str,
    target_language: str,
    model_name: str,
    llm_client: Optional[Any] = None,
    novel_context: str = "",
    custom_instructions: str = "",
    glossary_block: str = "",
    log_callback: Optional[Callable] = None,
    context_session: Optional[Any] = None,
    prompt_options: Optional[Dict[str, Any]] = None,
    repair_validator: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    """Run a 2-pass Senior Translation Editor reflection & repair evaluation on a draft chunk."""
    from src.prompts.prompts import generate_chunk_reflection_prompt, generate_chunk_repair_prompt
    from src.core.llm import TranslationExtractor
    from src.utils.addressing_schema import context_contract_version
    from src.utils.translation_quality import (
        find_source_residue,
        residue_findings_to_editor_issues,
        validate_editor_repair,
    )

    if not draft_translation or not draft_translation.strip() or not llm_client:
        return draft_translation

    options = prompt_options or {}
    contract_v2 = context_contract_version(options) >= 2
    source_language = str(options.get("source_language") or "")
    glossary_terms = (
        options.get("glossary_terms")
        if isinstance(options.get("glossary_terms"), dict)
        else {}
    )
    protected_terms = [
        str(item) for item in options.get("active_character_names") or [] if item
    ]
    protected_terms.extend(
        str(item) for item in options.get("preserved_terms") or [] if item
    )
    try:
        from src.utils.novel_context import (
            _character_profile_map,
            character_alias_map,
            extract_global_lore,
        )

        raw_lore = extract_global_lore(
            str(options.get("novel_context") or novel_context or "")
        )
        protected_terms.extend(
            str(profile.get("name") or key)
            for key, profile in _character_profile_map(raw_lore).items()
        )
        aliases = character_alias_map(raw_lore)
        protected_terms.extend(str(item) for item in aliases.keys())
        protected_terms.extend(str(item) for item in aliases.values())
    except Exception:
        pass
    residue_findings = []
    if bool(options.get("source_residue_validation", contract_v2)):
        residue_findings = find_source_residue(
            source_chunk,
            draft_translation,
            source_language=source_language,
            target_language=target_language,
            protected_terms=protected_terms,
            glossary_terms=glossary_terms,
        )
    deterministic_findings = json.dumps(
        [finding.to_dict() for finding in residue_findings],
        ensure_ascii=False,
        indent=2,
    ) if residue_findings else ""

    if log_callback:
        emit_progress_log(
            log_callback,
            "reflection_start",
            "Running 2-pass Senior Editor reflection pass...",
            layer="senior_editor_reflection",
        )
        emit_progress_log(
            log_callback,
            "reflection_prompt_contract",
            "Senior Editor reflection contract active.",
            layer="senior_editor_reflection",
            data={
                "prompt_version": REFLECTION_PROMPT_VERSION,
                "contract_version": REFLECTION_CONTRACT_VERSION,
            },
        )

    active_novel_context = _render_reflection_novel_context(
        novel_context=novel_context,
        prompt_options=prompt_options,
        source_chunk=source_chunk,
        draft_translation=draft_translation,
    )

    reflection_pair = generate_chunk_reflection_prompt(
        source_chunk=source_chunk,
        draft_translation=draft_translation,
        target_language=target_language,
        novel_context=active_novel_context,
        custom_instructions=custom_instructions,
        glossary_block=glossary_block,
        deterministic_findings=deterministic_findings,
    )

    try:
        response = await _generate_editor_response(
            llm_client=llm_client,
            prompt=reflection_pair.user,
            system_prompt=reflection_pair.system,
            model_name=model_name,
            temperature=0.2,
        )
        critique = (response.content or "").strip() if response and getattr(response, "content", None) else ""
    except Exception as e:
        if log_callback:
            emit_progress_log(
                log_callback,
                "reflection_failed",
                f"Senior Editor reflection failed: {e}",
                layer="senior_editor_reflection",
            )
        if contract_v2 and residue_findings:
            raise ReflectionValidationError(
                "Senior Editor failed while deterministic source residue remained."
            ) from e
        return draft_translation

    reflection_result = parse_reflection_result(critique)
    if log_callback:
        emit_progress_log(
            log_callback,
            "reflection_parse_status",
            f"Senior Editor reflection parsed with {reflection_result.parse_status}.",
            layer="senior_editor_reflection",
            data={
                "prompt_version": REFLECTION_PROMPT_VERSION,
                "contract_version": REFLECTION_CONTRACT_VERSION,
                "parse_status": reflection_result.parse_status,
            },
        )

    def reflection_contract_invalid(result: ReflectionResult) -> bool:
        return bool(
            result.parse_status in {"invalid_json", "empty", "incomplete_json"}
            or (contract_v2 and _reflection_contract_incomplete(result))
        )

    if reflection_contract_invalid(reflection_result):
        if log_callback:
            emit_progress_log(
                log_callback,
                "reflection_parse_retry",
                "Senior Editor returned malformed or incomplete JSON; retrying once with the strict editor contract.",
                layer="senior_editor_reflection",
                data={
                    "prompt_version": REFLECTION_PROMPT_VERSION,
                    "contract_version": REFLECTION_CONTRACT_VERSION,
                },
            )
        retry_user_prompt = (
            f"{reflection_pair.user}\n\n"
            "Your previous response was not valid JSON. Return only one valid "
            f"JSON object inside {REFLECTION_JSON_TAG_IN} and "
            f"{REFLECTION_JSON_TAG_OUT}. Every direct correction with a draft_quote "
            "must include draft_replacement with exact draft and replacement spans. "
            "Do not include prose, markdown, or bullets."
        )
        try:
            retry_response = await _generate_editor_response(
                llm_client=llm_client,
                prompt=retry_user_prompt,
                system_prompt=reflection_pair.system,
                model_name=model_name,
                temperature=0.0,
            )
            retry_critique = (
                (retry_response.content or "").strip()
                if retry_response and getattr(retry_response, "content", None)
                else ""
            )
            retry_result = parse_reflection_result(retry_critique)
            if log_callback:
                emit_progress_log(
                    log_callback,
                    "reflection_parse_retry_status",
                    f"Senior Editor reflection retry parsed with {retry_result.parse_status}.",
                    layer="senior_editor_reflection",
                    data={
                        "prompt_version": REFLECTION_PROMPT_VERSION,
                        "contract_version": REFLECTION_CONTRACT_VERSION,
                        "parse_status": retry_result.parse_status,
                    },
                )
            if not reflection_contract_invalid(retry_result):
                critique = retry_critique
                reflection_result = retry_result
        except Exception as e:
            if log_callback:
                emit_progress_log(
                    log_callback,
                    "reflection_parse_retry_failed",
                    f"Senior Editor reflection JSON retry failed: {e}",
                    layer="senior_editor_reflection",
                )

    if reflection_contract_invalid(reflection_result):
        if log_callback:
            emit_progress_log(
                log_callback,
                "reflection_parse_failed",
                "Senior Editor reflection contract remained invalid or incomplete after retry.",
                layer="senior_editor_reflection",
                data={
                    "prompt_version": REFLECTION_PROMPT_VERSION,
                    "contract_version": REFLECTION_CONTRACT_VERSION,
                    "parse_status": reflection_result.parse_status,
                },
            )
        if contract_v2:
            raise ReflectionValidationError(
                "Senior Editor reflection contract remained invalid after retry."
            )
        return draft_translation

    if (
        reflection_result.parse_status.startswith("legacy")
        or reflection_result.parse_status == "invalid_json"
    ) and log_callback:
        emit_progress_log(
            log_callback,
            "reflection_parse_fallback",
            f"Senior Editor reflection used {reflection_result.parse_status} parsing.",
            layer="senior_editor_reflection",
        )

    if residue_findings:
        existing_spans = {
            str(issue.get("draft_quote") or "").casefold().strip()
            for issue in reflection_result.issues
        }
        mandatory_issues = [
            issue for issue in residue_findings_to_editor_issues(residue_findings)
            if str(issue.get("draft_quote") or "").casefold().strip()
            not in existing_spans
        ]
        if mandatory_issues:
            reflection_result = ReflectionResult(
                "needs_repair",
                [*reflection_result.issues, *mandatory_issues],
                reflection_result.raw_text,
                reflection_result.parse_status,
            )
            if log_callback:
                emit_progress_log(
                    log_callback,
                    "source_residue_blocker",
                    f"Detected {len(mandatory_issues)} mandatory source-residue repair issue(s).",
                    level="warning",
                    layer="senior_editor_reflection",
                    data={"findings": [finding.to_dict() for finding in residue_findings]},
                )

    if not reflection_result.needs_repair:
        if log_callback:
            log_callback("reflection_complete", "Senior Editor reflection complete: No issues found.")
        return draft_translation

    from src.utils.unified_logger import get_logger, LogType
    get_logger().info(f"Senior Editor critique:\n{critique.strip()}", log_type=LogType.GENERAL)

    if log_callback:
        tldr_summary = format_critique_tldr(critique)
        log_callback("reflection_critique", f"Senior Editor critique: {tldr_summary}")

    critique_feedback = _reflection_feedback_payload(reflection_result)
    term_pairs = extract_term_replacements_from_critique(critique_feedback)
    if not term_pairs:
        term_pairs = extract_term_replacements_from_critique(critique)
    validation_errors: List[str] = []
    for repair_attempt in range(2):
        repair_feedback = critique_feedback
        if validation_errors:
            repair_feedback += (
                "\n\nPREVIOUS REPAIR VALIDATION FAILURES:\n- "
                + "\n- ".join(validation_errors)
                + "\nReturn a corrected complete translation."
            )
        repair_pair = generate_chunk_repair_prompt(
            source_chunk=source_chunk,
            draft_translation=draft_translation,
            critique_feedback=repair_feedback,
            target_language=target_language,
            custom_instructions=custom_instructions,
            glossary_block=glossary_block,
            novel_context=active_novel_context,
        )
        validation_errors = []
        try:
            repair_response = await _generate_editor_response(
                llm_client=llm_client,
                prompt=repair_pair.user,
                system_prompt=repair_pair.system,
                model_name=model_name,
                temperature=0.3 if repair_attempt == 0 else 0.0,
            )
            raw_content = (
                repair_response.content
                if repair_response and getattr(repair_response, "content", None)
                else ""
            )
            repaired_text = (
                llm_client.extract_translation(raw_content)
                if hasattr(llm_client, "extract_translation")
                else None
            )
            if not repaired_text:
                extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
                repaired_text = extractor.extract(raw_content)
            repaired_text = str(repaired_text or "").strip()
            if not repaired_text:
                validation_errors.append("repair output was empty")
            if repaired_text and repair_validator:
                try:
                    adapter_feedback = repair_validator(repaired_text)
                except Exception as validation_error:
                    adapter_feedback = f"repair validator raised {validation_error}"
                if adapter_feedback:
                    validation_errors.append(str(adapter_feedback))
            if repaired_text and contract_v2:
                validation_errors.extend(validate_editor_repair(
                    repaired_text,
                    reflection_result.issues,
                    source_text=source_chunk,
                    source_language=source_language,
                    target_language=target_language,
                    protected_terms=protected_terms,
                    glossary_terms=glossary_terms,
                ))
            if repaired_text and not validation_errors:
                if term_pairs and context_session and hasattr(
                    context_session,
                    "register_editor_terms",
                ):
                    context_session.register_editor_terms(term_pairs)
                if log_callback:
                    emit_progress_log(
                        log_callback,
                        "repair_applied",
                        "Applied and validated Senior Editor repair fixes.",
                        layer="senior_editor_repair",
                    )
                return repaired_text
        except Exception as e:
            validation_errors.append(f"repair call failed: {type(e).__name__}: {e}")

        if log_callback:
            emit_progress_log(
                log_callback,
                "repair_validation_failed",
                (
                    "Senior Editor repair failed validation; retrying once."
                    if repair_attempt == 0
                    else "Senior Editor repair remained invalid after retry."
                ),
                level="warning",
                layer="senior_editor_repair",
                data={"attempt": repair_attempt + 1, "reasons": validation_errors},
            )

    has_blocking_issue = any(
        str(issue.get("severity") or "major").casefold() in {"blocker", "major"}
        for issue in reflection_result.issues
    )
    if contract_v2 and has_blocking_issue:
        raise ReflectionValidationError(
            "Senior Editor repair did not pass validation: "
            + "; ".join(validation_errors[:3])
        )
    return draft_translation
