"""
Translation module for LLM communication
"""
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from src.config import (
    ADAPTIVE_CONTEXT_INITIAL_THINKING,
    DEFAULT_MODEL,
    EDITOR_LOG_FULL_RESPONSE,
    PLACEHOLDER_PREFIX,
    PLACEHOLDER_SUFFIX,
    SENTENCE_TERMINATORS,
    THINKING_MODELS,
    TRANSLATE_TAG_IN,
    TRANSLATE_TAG_OUT,
)
from src.core.common.source_residue import (
    detect_source_residue,
    residue_log_message,
    residue_score,
)
from src.core.editor.contracts import ReflectionValidationError

# Re-exported: these moved to the editor package but callers and tests still
# import them from this module.
from src.core.editor.prompting import (
    _build_focused_locator_retry_prompt,
    _build_missing_replacement_retry_prompt,
    _render_reflection_novel_context,
    compose_reflection_prompts,
)
from src.core.editor.unit_repair import UNIT_REPAIR_KIND
from src.core.llm.call_ledger import PURPOSE_EDITOR, PURPOSE_REFINEMENT, llm_purpose
from src.prompts.prompts import (
    REFLECTION_CONTRACT_VERSION,
    REFLECTION_JSON_TAG_IN,
    REFLECTION_JSON_TAG_OUT,
    REFLECTION_PROMPT_VERSION,
    generate_refinement_prompt,
    generate_translation_prompt,
)
from src.utils.progress_logging import emit_progress_log

from .chunking.token_chunker import TokenChunker
from .context_optimizer import (
    CONTEXT_STEP,
    INITIAL_CONTEXT_SIZE,
    AdaptiveContextManager,
)
from .llm import ContextOverflowError, RateLimitError, RepetitionLoopError
from .llm.exceptions import StructuredOutputSchemaError
from .llm_client import LLMResponse, create_llm_client
from .post_processor import clean_translated_text
from .progress_tracker import TokenProgressTracker

# Configuration for context overflow recovery
MAX_CHUNK_REDUCTION_ATTEMPTS = 3
CHUNK_REDUCTION_FACTOR = 0.6  # Reduce to 60% of original size each attempt
MIN_CHUNK_CHARACTERS = 200  # Minimum chunk size to attempt translation
_EDITOR_SCHEMA_UNSUPPORTED = set()
# Rejections that never named the schema, counted per provider/model/endpoint.
# One of those is not evidence the model cannot answer a schema -- a bad key,
# a transient fault and a genuine schema failure all arrive as the same generic
# INVALID_ARGUMENT -- but a second one in the same process is.
_EDITOR_SCHEMA_GENERIC_REJECTIONS: Dict[Any, int] = {}
_EDITOR_SCHEMA_GENERIC_LIMIT = 2


def _classify_editor_exception(exc: BaseException) -> str:
    """Map terminal editor errors without persisting provider response bodies."""

    explicit = str(getattr(exc, "failure_class", "") or "")
    if explicit:
        return explicit
    if isinstance(exc, RateLimitError):
        return "provider_rate_limit"
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in {401, 403}:
        return "provider_auth"
    if status == 402:
        return "provider_quota"
    if status == 429:
        return "provider_rate_limit"
    name = type(exc).__name__.casefold()
    if "timeout" in name or "connect" in name or "network" in name:
        return "transport"
    if isinstance(exc, StructuredOutputSchemaError):
        return "schema_rejected"
    return "transport"


@dataclass
class ReflectionResult:
    """Parsed Senior Editor reflection result."""

    status: str
    issues: List[Dict[str, Any]]
    raw_text: str = ""
    parse_status: str = "empty"
    voice_observations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def needs_repair(self) -> bool:
        return self.status == "needs_repair" and bool(self.issues)


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
        from src.core.glossary import (
            GlossaryConfig,
            build_glossary_block,
            filter_glossary,
        )
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





def _require_client(llm_client):
    """Return llm_client, or fail clearly when none was given.

    These request helpers used to fall back to the module-global Ollama
    default_client, so a cloud job whose client was lost silently sent its
    text to a local Ollama endpoint. (Ported from upstream c7e5cb64.)
    """
    if llm_client is None:
        raise ValueError(
            "No LLM client was provided for this request. Create one with "
            "create_llm_client() for the job's provider."
        )
    return llm_client


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
            client = _require_client(llm_client)

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
            # when context window is too small. Try increasing context, but only a
            # bounded number of times: a loop that survives larger windows is the
            # model looping on this chunk, not a context size problem.
            if context_manager and context_manager.should_retry_repetition_loop():
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

def _placeholder_tokens(text: str, placeholder_format: Optional[Tuple[str, str]]) -> List[str]:
    """Sorted placeholder tokens of `text`, used to compare two answers."""
    prefix, suffix = placeholder_format or (PLACEHOLDER_PREFIX, PLACEHOLDER_SUFFIX)
    return sorted(re.findall(re.escape(prefix) + r"\d+" + re.escape(suffix), text or ""))


async def _retry_on_source_residue(
    translated_text: str,
    *,
    main_content: str,
    source_language: str,
    target_language: str,
    log_callback,
    has_placeholders: bool,
    prompt_options: Optional[dict],
    placeholder_format: Optional[Tuple[str, str]],
    **request_kwargs,
) -> str:
    """Re-ask once when an answer still holds source-language text (issue #270).

    The retry states what went wrong in the prompt. The answer with the least
    residue wins; in placeholder mode a retry that lost or invented a
    placeholder never replaces an answer that kept them all, since the caller
    would then throw away a usable chunk. Every finding earns this retry,
    including the retry-only ones (echo, leftover Latin text); callers then
    run `detect_blocking_residue` on what comes back and record the chunk as
    untranslated only for a blocking finding.
    """
    if (prompt_options or {}).get('source_residue_retry'):
        return translated_text
    report = detect_source_residue(main_content, translated_text, source_language, target_language)
    if report is None:
        return translated_text

    if log_callback:
        log_callback("source_residue_retry",
                     f"⚠️ {residue_log_message(report, source_language)} - retrying once")

    retry_options = {**(prompt_options or {}), 'source_residue_retry': True}
    retried = await generate_translation_request(
        main_content,
        source_language=source_language,
        target_language=target_language,
        log_callback=log_callback,
        has_placeholders=has_placeholders,
        prompt_options=retry_options,
        placeholder_format=placeholder_format,
        **request_kwargs,
    )
    if not retried:
        return translated_text

    if has_placeholders:
        expected = _placeholder_tokens(main_content, placeholder_format)
        first_ok = _placeholder_tokens(translated_text, placeholder_format) == expected
        retry_ok = _placeholder_tokens(retried, placeholder_format) == expected
        if first_ok and not retry_ok:
            return translated_text
        if retry_ok and not first_ok:
            return retried

    retry_report = detect_source_residue(main_content, retried, source_language, target_language)
    if residue_score(retry_report) < residue_score(report):
        if log_callback and retry_report is None:
            log_callback("source_residue_recovered", "✓ Retry translated the leftover text")
        return retried
    return translated_text


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

    # Use the adaptive context handler. Each chunk gets its own context growth
    # budget, and a chunk that still fails gives back what it grew.
    if context_manager:
        context_manager.begin_request()
    try:
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
            placeholder_format=placeholder_format,
        )
    except BaseException:
        if context_manager:
            context_manager.abandon_request()
        raise
    if not translated_text and context_manager:
        context_manager.abandon_request()

    if translated_text:
        return await _retry_on_source_residue(
            translated_text,
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
            placeholder_format=placeholder_format,
        )
    else:
        err_msg = "ERROR: LLM API request failed"
        if log_callback:
            log_callback("llm_api_error", err_msg)
        else:
            tqdm.write(f"\n{err_msg}")
        return None



@llm_purpose(PURPOSE_REFINEMENT)
async def _refinement_request_attempt(
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

    client = _require_client(llm_client)
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
            # Repetition loop detected - try increasing context (double increase),
            # a bounded number of times per chunk
            if context_manager and context_manager.should_retry_repetition_loop():
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


async def _make_refinement_request(**kwargs) -> Tuple[Optional[str], Optional[LLMResponse]]:
    """Refine one chunk with its own context growth budget.

    Wraps _refinement_request_attempt: the context manager (if any) starts a
    fresh per-chunk budget, and a chunk that still fails gives back the
    context it grew, so one bad chunk cannot inflate every later request.
    """
    context_manager = kwargs.get('context_manager')
    if context_manager:
        context_manager.begin_request()
    try:
        refined_text, llm_response = await _refinement_request_attempt(**kwargs)
    except BaseException:
        if context_manager:
            context_manager.abandon_request()
        raise
    if refined_text is None and context_manager:
        context_manager.abandon_request()
    return refined_text, llm_response


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

    # Create LLM client. A failure must stop the pass: there is no safe
    # fallback client for an arbitrary provider.
    try:
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
    except Exception as e:
        if log_callback:
            log_callback("refinement_client_failed",
                f"❌ Could not create the LLM client for '{llm_provider}': {e}")
        raise RuntimeError(f"Could not create the LLM client for '{llm_provider}': {e}") from e

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
            local_prompt_options.update({
                "chunk_index": i,
                "editor_phase": "refinement",
            })
            from src.utils.narrator_voice import build_narrator_voice_context
            voice_db = local_prompt_options.get("_checkpoint_db")
            narrative_voice_context = build_narrator_voice_context(
                str(local_prompt_options.get("translation_id") or ""),
                voice_db,
                chunk_index=i,
                target_language=target_language,
            )
            if narrative_voice_context:
                local_prompt_options["narrative_voice_context"] = narrative_voice_context
            if i < len(original_chunks):
                local_prompt_options["chapter_index"] = original_chunks[i].get(
                    "chapter_index"
                )
                local_prompt_options["scene_key"] = str(
                    original_chunks[i].get("scene_key") or ""
                )
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
            from src.core.context.unit_pipeline import prepare_unit_prompt_options
            local_prompt_options = prepare_unit_prompt_options(
                local_prompt_options,
                unit_index=i,
                phase="refinement",
                file_type=str(local_prompt_options.get("file_type") or "txt"),
                source_text=(
                    str(original_chunks[i].get("source_content")
                        or original_chunks[i].get("original_text") or "")
                    if i < len(original_chunks) else ""
                ),
                target_language=target_language,
                dialogue_attribution=dialogue_attribution,
                chapter_index=(
                    original_chunks[i].get("chapter_index")
                    if i < len(original_chunks) else None
                ),
                scene_key=(
                    str(original_chunks[i].get("scene_key") or "")
                    if i < len(original_chunks) else ""
                ),
                nearby_source=(context_before, context_after),
            )

            # Run the same structured Senior Editor used by translation.
            try:
                from src.utils.novel_context import normalize_refinement_context
                from src.utils.translation_quality import (
                    validate_plain_refinement_structure,
                )

                if context_content:
                    local_prompt_options["novel_context"] = normalize_refinement_context(
                        context_content,
                        local_prompt_options.get("novel_context", ""),
                    )
                source_chunk = ""
                if i < len(original_chunks):
                    source_chunk = str(
                        original_chunks[i].get("source_content")
                        or original_chunks[i].get("original_text")
                        or ""
                    )
                local_prompt_options["editor_source_mode"] = (
                    "checkpoint" if source_chunk.strip() else "monolingual"
                )
                local_prompt_options.setdefault("source_language", str(
                    (prompt_options or {}).get("source_language") or ""
                ))
                glossary_block = _build_chunk_glossary_block(
                    source_chunk or draft_text,
                    local_prompt_options,
                    log_callback=log_callback,
                    runtime_state=runtime_state,
                )
                refined_text = await run_chunk_reflection_pass(
                    source_chunk=source_chunk,
                    draft_translation=draft_text,
                    target_language=target_language,
                    model_name=model_name,
                    llm_client=llm_client,
                    novel_context=local_prompt_options.get("novel_context", ""),
                    custom_instructions=str(
                        local_prompt_options.get("refinement_instructions") or ""
                    ),
                    glossary_block=glossary_block,
                    log_callback=log_callback,
                    prompt_options=local_prompt_options,
                    repair_validator=lambda repaired: validate_plain_refinement_structure(
                        draft_text,
                        repaired,
                    ),
                )
                llm_response = None
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
            except ReflectionValidationError as exc:
                refined_text = None
                llm_response = None
                if log_callback:
                    emit_progress_log(
                        log_callback,
                        "refinement_editor_invalid",
                        "Senior Editor refinement was invalid; keeping the incoming draft.",
                        level="warning",
                        layer="senior_editor_refinement",
                        data=getattr(exc, "diagnostics", {}),
                    )

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
                refinement_quality = str(
                    getattr(refined_text, "quality_status", "passed") or "passed"
                )
                refinement_validation = getattr(refined_text, "editor_validation", None)
                refined_text = clean_translated_text(refined_text)
                refined_parts.append(refined_text)
                progress_tracker.mark_completed(i, chunk_elapsed)
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

            from src.core.refine.persistence import save_refinement_unit
            persisted_text = refined_text if refined_text is not None else draft_text
            save_refinement_unit(
                prompt_options,
                unit_index=i,
                source_text=(
                    str(original_chunks[i].get("source_content")
                        or original_chunks[i].get("original_text") or "")
                    if i < len(original_chunks) else ""
                ),
                refined_text=persisted_text,
                status=("completed" if refined_text is not None else "failed"),
                quality_status=(
                    refinement_quality if refined_text is not None else "review_required"
                ),
                metadata={
                    "editor_validation": refinement_validation
                    if refined_text is not None and isinstance(refinement_validation, dict)
                    else {},
                    "chapter_index": (
                        original_chunks[i].get("chapter_index")
                        if i < len(original_chunks) else None
                    ),
                },
            )

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


# Rungs the editor writes instead of the three the contract offers. Severity
# no longer decides whether a finding may be repaired, so this costs nothing at
# the gate -- but a database holding `major`, `high` and `medium` side by side
# cannot be counted, and one measured book answered with all three.
_SEVERITY_SYNONYMS = {
    "critical": "blocker",
    "severe": "blocker",
    "high": "major",
    "moderate": "minor",
    "medium": "minor",
    "low": "minor",
    "trivial": "minor",
    "nit": "minor",
    "info": "minor",
}


def _normalized_severity(value: Any) -> str:
    """Fold an editor's severity word onto the contract's three rungs."""

    text = str(value or "").strip().casefold()
    if not text:
        return "major"
    if text in {"blocker", "major", "minor"}:
        return text
    return _SEVERITY_SYNONYMS.get(text, "major")


def _normalize_reflection_issue(raw_issue: Any) -> Optional[Dict[str, Any]]:
    """Normalize one structured reflection issue into the internal shape."""
    if isinstance(raw_issue, str):
        instruction = raw_issue.strip()
        if not instruction:
            return None
        return {
            "category": "other",
            "severity": "major",
            "confidence": 0.8,
            "repair_kind": "rewrite",
            "source_quote": "",
            "draft_quote": "",
            "instruction": instruction,
            "draft_replacement": None,
            "glossary_update": None,
            "term_replacement": None,
        }

    if not isinstance(raw_issue, dict):
        return None

    # Accept older provider-emitted aliases at the parser boundary. The prompt
    # contract remains canonical, but a semantically complete issue must not be
    # discarded merely because a compact model used its familiar field names.
    instruction = str(
        raw_issue.get("instruction")
        or raw_issue.get("reason")
        or raw_issue.get("description")
        or ""
    ).strip()
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
            {
                "draft": draft_span,
                "replacement": replacement,
                **(
                    {"occurrence_index": draft_replacement.get("occurrence_index")}
                    if draft_replacement.get("occurrence_index") is not None
                    else {}
                ),
            }
            if draft_span and replacement
            else None
        )
    else:
        replacement_text = str(draft_replacement or "").strip()
        draft_span = str(raw_issue.get("draft_quote") or "").strip()
        draft_replacement = (
            {"draft": draft_span, "replacement": replacement_text}
            if draft_span and replacement_text else None
        )

    try:
        confidence = max(0.0, min(1.0, float(raw_issue.get("confidence", 0.8))))
    except (TypeError, ValueError):
        confidence = 0.8

    category = str(
        raw_issue.get("category") or raw_issue.get("type") or "other"
    ).strip() or "other"
    severity = _normalized_severity(raw_issue.get("severity"))
    # The unit contract answers a different question -- which unit, and how it
    # should read -- so an issue that came back in that shape keeps it. There is
    # no span to locate and no locator to validate; both are computed later from
    # the rewrite itself.
    unit_id = str(raw_issue.get("unit_id") or "").strip().upper()
    if unit_id and "rewritten_unit" in raw_issue:
        return {
            "issue_id": str(
                raw_issue.get("issue_id") or raw_issue.get("id") or ""
            ).strip(),
            "segment_id": "",
            "unit_id": unit_id,
            "category": category,
            "severity": severity,
            "confidence": confidence,
            "repair_kind": (
                "review_only" if confidence < 0.80 else UNIT_REPAIR_KIND
            ),
            "source_quote": "",
            "draft_quote": "",
            "instruction": instruction,
            "rewritten_unit": str(raw_issue.get("rewritten_unit") or ""),
            "draft_replacement": None,
            "glossary_update": glossary_update,
            "term_replacement": glossary_update,
        }
    repair_kind = str(raw_issue.get("repair_kind") or "").strip().casefold()
    # Certainty decides whether an edit may be applied unattended; severity
    # decides only how much the defect costs the reader. Downgrading every
    # `minor` here overrode the editor's own choice of repair, so a defect it
    # had located exactly and was sure of arrived unrepairable no matter what
    # the gate downstream allowed.
    if repair_kind not in {"local_replace", "rewrite", "review_only"}:
        if confidence < 0.80:
            repair_kind = "review_only"
        elif draft_replacement:
            repair_kind = "local_replace"
        else:
            repair_kind = "rewrite"
    if confidence < 0.80:
        repair_kind = "review_only"
    return {
        "issue_id": str(
            raw_issue.get("issue_id") or raw_issue.get("id") or ""
        ).strip(),
        "segment_id": str(
            raw_issue.get("segment_id") or raw_issue.get("draft_id") or ""
        ).strip().upper(),
        "category": category,
        "severity": severity,
        "confidence": confidence,
        "repair_kind": repair_kind,
        "source_quote": str(raw_issue.get("source_quote") or "").strip(),
        "draft_quote": str(raw_issue.get("draft_quote") or "").strip(),
        "instruction": instruction,
        "draft_replacement": draft_replacement,
        "glossary_update": glossary_update,
        "term_replacement": glossary_update,
    }


def _issue_requires_draft_replacement(issue: Dict[str, Any]) -> bool:
    """Return whether an issue describes a direct, locally verifiable edit."""

    return str(issue.get("repair_kind") or "").casefold() == "local_replace"


def _reflection_contract_incomplete(result: ReflectionResult) -> bool:
    if result.status == "needs_repair" and not result.issues:
        return True
    return any(
        _issue_requires_draft_replacement(issue)
        and not issue.get("draft_replacement")
        for issue in result.issues
    )


def _reflection_contract_invalid(result: ReflectionResult) -> bool:
    """True when an editor answer cannot be used as a reflection verdict."""
    return bool(
        result.parse_status in {"invalid_json", "empty", "incomplete_json"}
        or _reflection_contract_incomplete(result)
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
        for index, issue in enumerate(issues, start=1):
            if not issue.get("issue_id"):
                issue["issue_id"] = f"issue-{index}"
        if raw_status not in {"no_issues", "needs_repair"}:
            raw_status = "needs_repair" if issues else "no_issues"
        if issues and raw_status == "no_issues":
            raw_status = "needs_repair"
        parse_status = "json"
        if not issues and raw_status == "needs_repair":
            parse_status = "incomplete_json"
        raw_voice = (
            data.get("voice_observations")
            if isinstance(data.get("voice_observations"), list) else []
        )
        return ReflectionResult(
            raw_status, issues, raw_text, parse_status, raw_voice,
        )

    if raw_text.strip().upper() == "NO_ISSUES":
        return ReflectionResult("no_issues", [], raw_text, "legacy_no_issues")

    legacy_items = _filter_actionable_critiques(raw_text)
    if legacy_items:
        issues = [
            {
                "category": "other",
                "severity": "major",
                "confidence": 0.8,
                "repair_kind": "rewrite",
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


def log_full_editor_response(
    log_callback: Optional[Callable],
    stage: str,
    raw_text: str,
    chunk_index: Optional[int] = None,
) -> None:
    """Print the editor's answer verbatim when the operator asked to see it.

    The TL;DR log summarises an empty verdict and a thorough audit to the same
    nothing, so a contract change that quietly stops the editor finding defects
    is invisible from the logs alone. This is the escape hatch for reading what
    the model actually said, gated because it is thousands of characters a chunk.
    """

    if not EDITOR_LOG_FULL_RESPONSE:
        return
    emit_progress_log(
        log_callback,
        "reflection_full_response",
        f"Senior Editor raw response ({stage}):\n{raw_text or '<empty>'}",
        layer="senior_editor_reflection",
        chunk_index=chunk_index,
        data={"stage": stage, "response_chars": len(raw_text or "")},
    )


# Lives with the rest of the glossary rules now. It is a pure string predicate
# with no engine dependency, and the novel-context session needed it too --
# which had src/utils importing from src/core to get it. Re-exported here so
# every existing src.core.translator._is_valid_glossary_term reference resolves.
from src.utils.novel_context.glossary import (  # noqa: E402
    _is_valid_glossary_term,
    is_valid_glossary_term,
)


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


async def _generate_editor_response(
    llm_client: Any,
    prompt: str,
    system_prompt: str,
    model_name: str,
    temperature: float,
    max_output_tokens: Optional[int] = None,
    response_schema: Optional[Dict[str, Any]] = None,
    stage: str = "",
    thinking_level: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    thinking_enabled: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
):
    """Call the available LLM client method for reflection/repair."""
    if hasattr(llm_client, "generate_async"):
        result = llm_client.generate_async(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_schema=response_schema,
            stage=stage,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
        )
    elif hasattr(llm_client, "generate"):
        from src.core.llm import LLMGenerationOptions

        signature = inspect.signature(llm_client.generate)
        kwargs = {"system_prompt": system_prompt}
        if "model" in signature.parameters:
            kwargs["model"] = model_name
        if "temperature" in signature.parameters:
            kwargs["temperature"] = temperature
        if "generation_options" in signature.parameters:
            kwargs["generation_options"] = LLMGenerationOptions(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_schema=response_schema,
                stage=stage,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
            )
        result = llm_client.generate(prompt, **kwargs)
    elif hasattr(llm_client, "make_request"):
        from src.core.llm import LLMGenerationOptions

        signature = inspect.signature(llm_client.make_request)
        kwargs = {"system_prompt": system_prompt}
        if "generation_options" in signature.parameters:
            kwargs["generation_options"] = LLMGenerationOptions(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_schema=response_schema,
                stage=stage,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
            )
        result = llm_client.make_request(prompt, model_name, **kwargs)
    else:
        raise AttributeError("LLM client has no supported generation method")

    if inspect.isawaitable(result):
        return await result
    return result


@llm_purpose(PURPOSE_EDITOR)
async def _run_chunk_reflection_pass_impl(
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
    from src.utils.editor_diagnostics import response_hash
    from src.utils.translation_quality import (
        apply_editor_patches,
        apply_repairs_that_hold,
        filter_protected_span_editor_issues,
        filter_source_contradicting_name_edits,
        normalize_unique_issue_locators,
        replacement_already_in_draft,
        residue_findings_to_editor_issues,
        validate_editor_repair,
        validate_issue_locators,
    )

    if not draft_translation or not draft_translation.strip() or not llm_client:
        return draft_translation

    from src.core.editor import review_required_translation
    from src.core.editor.preflight import run_editor_preflight
    from src.core.editor.requests import EditorRequestSession

    options = prompt_options or {}
    # Client selection, request budget, schema/truncation fallbacks and token
    # accounting live in the session; this function decides what to ask.
    session = EditorRequestSession(
        options=options,
        llm_client=llm_client,
        model_name=model_name,
        target_language=target_language,
        log_callback=log_callback,
    )
    editor_output_tokens = session.output_tokens
    recorder = session.recorder
    reflection_components = session.reflection_components
    record_attempt = session.record_attempt
    generate_editor = session.generate

    result_state = "unchanged_draft"
    resolved_issue_count = 0
    unresolved_issue_count = 0
    # Counts findings that were reported but will not reach the draft. Every
    # site that *removes* an issue from the result adds to it; issues that stay
    # in the result are counted once at the end, where the actionable filter
    # runs, so nothing is credited twice.
    warning_count = 0
    # Assigned once the editor's answer is parsed; a run that never got that
    # far reported nothing.
    llm_issue_count = 0
    review_issue_count = 0
    max_automatic_repair_attempts = 3
    automatic_repair_attempts = 0
    narrator_conformance: Dict[str, Any] = {"status": "not_checked"}
    initial_narrator_finding_count = 0

    def finish_run(outcome: str, **payload: Any) -> None:
        # Reads the run's outcome state at call time, so every exit records
        # the counts as they stand when it is taken.
        session.finish(
            outcome,
            defaults={
                "result_state": result_state,
                "resolved_issue_count": resolved_issue_count,
                "unresolved_issue_count": unresolved_issue_count,
                "warning_count": warning_count,
                "llm_issue_count": llm_issue_count,
            },
            narrator_conformance=narrator_conformance,
            automatic_repair_attempts=automatic_repair_attempts,
            max_automatic_repair_attempts=max_automatic_repair_attempts,
            deterministic_count=sum(
                1 for finding in residue_findings if finding.blocking
            ) + initial_narrator_finding_count,
            payload=payload,
        )

    preflight = run_editor_preflight(
        source_chunk,
        draft_translation,
        target_language,
        novel_context,
        options,
        log_callback,
    )
    # Unpacked into locals rather than read through `preflight` because the
    # closures defined above (finish_run, record_attempt) already capture these
    # names from this scope.
    draft_translation = preflight.draft_translation
    source_available = preflight.source_available
    source_language = preflight.source_language
    glossary_terms = preflight.glossary_terms
    protected_terms = preflight.protected_terms
    residue_findings = preflight.residue_findings
    narrator_conformance = preflight.narrator_conformance
    narrator_patches = preflight.narrator_patches
    remaining_narrator_blockers = preflight.remaining_narrator_blockers
    initial_narrator_finding_count = preflight.initial_narrator_finding_count
    deterministic_findings = preflight.deterministic_findings
    narrative_voice_context = preflight.narrative_voice_context
    if narrator_patches:
        result_state = "locally_patched"
        resolved_issue_count += len(narrator_patches)

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

    prompts = compose_reflection_prompts(
        source_chunk,
        draft_translation,
        target_language,
        novel_context,
        custom_instructions,
        glossary_block,
        deterministic_findings,
        narrative_voice_context,
        source_available,
        options,
        prompt_options,
    )
    active_novel_context = prompts.active_novel_context
    reflection_pair = prompts.reflection_pair
    reflection_fallback_pair = prompts.reflection_fallback_pair
    reflection_components.update(prompts.components)

    try:
        response = await generate_editor(
            prompt=reflection_pair.user,
            system_prompt=reflection_pair.system,
            temperature=0.2,
            max_output_tokens=editor_output_tokens,
            stage="reflection",
            structured=True,
            fallback_prompt=reflection_fallback_pair.user,
            fallback_system_prompt=reflection_fallback_pair.system,
        )
        critique = (response.content or "").strip() if response and getattr(response, "content", None) else ""
    except Exception as e:
        if log_callback:
            emit_progress_log(
                log_callback,
                "reflection_failed",
                "Senior Editor reflection request failed "
                f"({type(e).__name__}).",
                layer="senior_editor_reflection",
            )
        failure_class = _classify_editor_exception(e)
        record_attempt(
            "reflection", None, "",
            failure_class=failure_class,
            reason_codes=[f"provider_failure:{failure_class}"],
        )
        has_residue = any(
            finding.blocking for finding in residue_findings
        )
        finish_run(
            "review_required" if has_residue else "transport_failed",
            failure_class=failure_class,
            diagnostics={"reason": failure_class},
            unresolved_issue_count=len(residue_findings) if has_residue else 0,
        )
        return review_required_translation(
            draft_translation,
            {"stage": "reflection", "status": "review_required", "reason": failure_class},
        )

    reflection_result = parse_reflection_result(critique)
    # What the editor actually reported, before any of this pass filters it.
    # `issue_count` is what survived to the repair stage, and the two were the
    # same number until findings started being dropped between them: a chunk
    # where the editor named ten defects and two were repairable was stored as
    # a chunk that named two, so the record could not say whether a quiet
    # editor had found little or had been talked out of most of it.
    llm_issue_count = len(reflection_result.issues)
    initial_failure_class = ""
    if not critique:
        initial_failure_class = (
            "provider_blocked" if getattr(response, "blocked_reason", "")
            else "provider_truncated" if getattr(response, "was_truncated", False)
            else "provider_empty"
        )
    record_attempt(
        "reflection", response, critique,
        parse_status=reflection_result.parse_status,
        failure_class=initial_failure_class,
        issues=reflection_result.issues,
    )
    log_full_editor_response(
        log_callback, "reflection", critique,
        int(options.get("chunk_index", 0) or 0),
    )
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

    reflection_contract_invalid = _reflection_contract_invalid

    contract_review_required = bool(remaining_narrator_blockers)
    if remaining_narrator_blockers:
        review_issue_count += len(remaining_narrator_blockers)
    if reflection_contract_invalid(reflection_result):
        from src.core.editor.contract_retry import retry_invalid_contract

        contract_retry = await retry_invalid_contract(
            session=session,
            reflection_result=reflection_result,
            critique=critique,
            draft_translation=draft_translation,
            reflection_pair=reflection_pair,
            options=options,
            log_callback=log_callback,
        )
        reflection_result = contract_retry.reflection_result
        critique = contract_retry.critique
        if contract_retry.replaced:
            llm_issue_count = len(reflection_result.issues)

    if reflection_contract_invalid(reflection_result):
        from src.core.editor.contract_retry import salvage_partial_contract

        salvage = salvage_partial_contract(reflection_result, log_callback=log_callback)
        reflection_result = salvage.reflection_result
        if salvage.review_required is not None:
            contract_review_required = salvage.review_required
        review_issue_count += salvage.review_increment
        warning_count += salvage.warning_increment

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
        blocked = any(finding.blocking for finding in residue_findings)
        finish_run(
            "review_required",
            parse_status=reflection_result.parse_status,
            failure_class=(
                "contract_incomplete"
                if _reflection_contract_incomplete(reflection_result)
                else "contract_parse"
            ),
            issue_count=len(reflection_result.issues),
            unresolved_issue_count=(
                len(reflection_result.issues) + (1 if blocked else 0)
            ),
        )
        return review_required_translation(
            draft_translation,
            {
                "stage": "reflection_contract",
                "status": "review_required",
                "final_reason_codes": ["contract_parse"],
            },
        )

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

    def persist_final_voice(final_text: str) -> str:
        """Persist voice evidence only when it still grounds in final output."""

        if not reflection_result.voice_observations:
            return final_text
        from src.utils.narrator_voice import (
            persist_voice_observations,
            validate_voice_observations,
        )
        accepted_voice, rejected_voice = validate_voice_observations(
            reflection_result.voice_observations,
            source_text=source_chunk,
            target_text=final_text,
            target_language=target_language,
        )
        if rejected_voice and log_callback:
            emit_progress_log(
                log_callback,
                "narrator_voice_evidence_rejected",
                f"Discarded {len(rejected_voice)} ungrounded narrator observation(s).",
                level="warning",
                layer="narrator_voice",
                data={"reasons": [item["reason"] for item in rejected_voice]},
            )
        # Refinement consumes historical voice state but never rewrites it.
        if accepted_voice and str(options.get("editor_phase") or "") != "refinement":
            voice_db = options.get("_checkpoint_db")
            owns_voice_db = False
            if voice_db is None and options.get("jobs_db_path"):
                from src.persistence.database import Database
                voice_db = Database(str(options["jobs_db_path"]))
                owns_voice_db = True
            if voice_db is not None and options.get("translation_id"):
                persist_voice_observations(
                    voice_db,
                    str(options["translation_id"]),
                    int(options.get("chunk_index", 0) or 0),
                    accepted_voice,
                    chapter_index=options.get("chapter_index"),
                    scene_key=str(options.get("scene_key") or ""),
                    provenance=(
                        "bootstrap" if options.get("narrator_bootstrap")
                        else "senior_editor"
                    ),
                )
            if owns_voice_db:
                voice_db.close()
        return final_text

    if residue_findings:
        existing_spans = {
            str(issue.get("draft_quote") or "").casefold().strip()
            for issue in reflection_result.issues
        }
        mandatory_issues = [
            issue for issue in residue_findings_to_editor_issues(
                finding for finding in residue_findings if finding.blocking
            )
            if str(issue.get("draft_quote") or "").casefold().strip()
            not in existing_spans
        ]
        if mandatory_issues:
            reflection_result = ReflectionResult(
                "needs_repair",
                [*reflection_result.issues, *mandatory_issues],
                reflection_result.raw_text,
                reflection_result.parse_status,
                reflection_result.voice_observations,
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

    if narrator_conformance.get("status") == "fail":
        from src.core.editor import conformance_editor_issues

        narrator_issues = conformance_editor_issues(narrator_conformance)
        narrator_spans = {
            str(item.get("draft_quote") or "").casefold().strip()
            for item in narrator_issues
        }
        retained_issues = [
            item for item in reflection_result.issues
            if not (
                str(item.get("draft_quote") or "").casefold().strip()
                in narrator_spans
                and not item.get("deterministic")
            )
        ]
        reflection_result = ReflectionResult(
            "needs_repair",
            [*retained_issues, *narrator_issues],
            reflection_result.raw_text,
            reflection_result.parse_status,
            reflection_result.voice_observations,
        )
        if log_callback:
            emit_progress_log(
                log_callback,
                "narrator_conformance_blocker",
                (
                    "Detected deterministic narrator self-reference leakage "
                    f"in {len(narrator_issues)} narrative segment(s)."
                ),
                level="warning",
                layer="narrator_voice",
                data={"findings": narrator_conformance["violating_segments"]},
            )

    retained_issues, protected_issue_ids = filter_protected_span_editor_issues(
        draft_translation,
        reflection_result.issues,
        protected_terms=protected_terms,
        glossary_terms=glossary_terms,
    )
    if protected_issue_ids:
        # Dropped here and nowhere counted, so a chunk whose every finding
        # landed on a protected entity was recorded as having reported none.
        warning_count += len(protected_issue_ids)
        reflection_result = ReflectionResult(
            "needs_repair" if retained_issues else "no_issues",
            retained_issues,
            reflection_result.raw_text,
            reflection_result.parse_status,
            reflection_result.voice_observations,
        )
        if log_callback:
            emit_progress_log(
                log_callback,
                "editor_protected_span_issues_ignored",
                "Ignored editor changes that targeted complete protected entity spans.",
                layer="senior_editor_reflection",
                data={"issue_ids": protected_issue_ids},
            )

    retained_issues, contradicting_issue_ids = filter_source_contradicting_name_edits(
        source_chunk,
        reflection_result.issues,
    )
    if contradicting_issue_ids:
        # Counted the same way as a protected-span drop: a chunk whose only
        # finding was an edit against its own source reported none, and the
        # warning is what says otherwise.
        warning_count += len(contradicting_issue_ids)
        reflection_result = ReflectionResult(
            "needs_repair" if retained_issues else "no_issues",
            retained_issues,
            reflection_result.raw_text,
            reflection_result.parse_status,
            reflection_result.voice_observations,
        )
        if log_callback:
            emit_progress_log(
                log_callback,
                "editor_source_contradicting_issues_ignored",
                "Ignored editor edits that rewrote how the source names someone there.",
                layer="senior_editor_reflection",
                data={"issue_ids": contradicting_issue_ids},
            )

    if not reflection_result.needs_repair:
        if log_callback:
            emit_progress_log(
                log_callback,
                "editor_state_no_issues",
                "Senior Editor state: audit complete; no issues found.",
                layer="senior_editor_reflection",
            )
        finish_run(
            "locally_repaired" if narrator_patches else "no_issues",
            parse_status=reflection_result.parse_status,
            issue_count=0,
            response_hash=response_hash(critique),
        )
        return persist_final_voice(draft_translation)

    normalized_issues, normalized_locator_ids = normalize_unique_issue_locators(
        draft_translation, reflection_result.issues,
    )
    if normalized_locator_ids:
        reflection_result = ReflectionResult(
            reflection_result.status,
            normalized_issues,
            reflection_result.raw_text,
            reflection_result.parse_status,
            reflection_result.voice_observations,
        )
        if log_callback:
            emit_progress_log(
                log_callback,
                "editor_locators_grounded",
                "Grounded editor edits from unique exact draft spans.",
                layer="senior_editor_reflection",
                data={"issue_ids": normalized_locator_ids},
            )
    locator_errors = validate_issue_locators(
        draft_translation, reflection_result.issues,
    )
    review_required = contract_review_required
    if log_callback:
        emit_progress_log(
            log_callback,
            "editor_state_audit_parsed",
            (
                "Senior Editor state: audit parsed "
                f"({len(reflection_result.issues)} issue(s), "
                f"{len(locator_errors)} locator error(s))."
            ),
            layer="senior_editor_reflection",
        )
    if locator_errors:
        invalid_ids = {item.rsplit(":", 1)[-1] for item in locator_errors}
        locator_retry_prompt = _build_focused_locator_retry_prompt(
            draft_translation,
            reflection_result.issues,
            invalid_ids,
            locator_errors,
        )
        try:
            locator_response = await generate_editor(
                prompt=locator_retry_prompt,
                system_prompt=(
                    "You correct exact locators for structured translation "
                    "editor issues. Use only the supplied candidate segments, "
                    "return valid schema JSON, and do not rewrite translation text."
                ),
                temperature=0.0,
                max_output_tokens=editor_output_tokens,
                stage="locator_retry",
                structured=True,
            )
            locator_raw = str(getattr(locator_response, "content", "") or "").strip()
            locator_result = parse_reflection_result(locator_raw)
            record_attempt(
                "locator_retry", locator_response, locator_raw,
                parse_status=locator_result.parse_status,
                failure_class="locator_ambiguous",
                reason_codes=locator_errors,
                issues=locator_result.issues,
            )
            if not reflection_contract_invalid(locator_result):
                valid_original = [
                    issue for issue in reflection_result.issues
                    if str(issue.get("issue_id") or "") not in invalid_ids
                ]
                corrected = [
                    issue for issue in locator_result.issues
                    if not invalid_ids
                    or str(issue.get("issue_id") or "") in invalid_ids
                ]
                # The retry is asked to re-locate specific issues and answers
                # with whichever it managed. The rest are gone from here on,
                # and counting them only where they are dropped meant counting
                # them nowhere: one measured chunk sent two issues to the retry,
                # got neither back, and recorded no warning at all.
                returned_ids = {
                    str(issue.get("issue_id") or "") for issue in corrected
                }
                warning_count += len(invalid_ids - returned_ids)
                reflection_result = ReflectionResult(
                    "needs_repair",
                    [*valid_original, *corrected],
                    locator_raw,
                    locator_result.parse_status,
                    locator_result.voice_observations
                    or reflection_result.voice_observations,
                )
                locator_errors = validate_issue_locators(
                    draft_translation, reflection_result.issues,
                )
        except Exception as exc:
            locator_errors.append(f"locator_retry_failed:{type(exc).__name__}")

    if locator_errors:
        invalid_ids = {item.rsplit(":", 1)[-1] for item in locator_errors}
        invalid_issues = [
            issue for issue in reflection_result.issues
            if str(issue.get("issue_id") or "") in invalid_ids
        ]
        deterministic_invalid_ids = {
            str(issue.get("issue_id") or "")
            for issue in invalid_issues if issue.get("deterministic")
        }
        reflection_result = ReflectionResult(
            "needs_repair",
            [
                issue for issue in reflection_result.issues
                if str(issue.get("issue_id") or "") not in invalid_ids
            ],
            reflection_result.raw_text,
            reflection_result.parse_status,
            reflection_result.voice_observations,
        )
        review_required = review_required or bool(deterministic_invalid_ids)
        review_issue_count += len(deterministic_invalid_ids)
        warning_count += len(invalid_ids - deterministic_invalid_ids)
        if not reflection_result.issues:
            unresolved_issue_count = len(invalid_ids)
            finish_run(
                "review_required" if deterministic_invalid_ids else "warnings_only",
                parse_status=reflection_result.parse_status,
                failure_class=(
                    "locator_ambiguous"
                    if any(item.startswith("locator_ambiguous") for item in locator_errors)
                    else "locator_missing"
                    if any(item.startswith("locator_missing") for item in locator_errors)
                    else "contract_issue"
                ),
                issue_count=len(invalid_ids),
                diagnostics={"reason_codes": locator_errors},
            )
            final_text = persist_final_voice(draft_translation)
            if deterministic_invalid_ids:
                return review_required_translation(
                    final_text,
                    {
                        "stage": "locator_validation",
                        "status": "review_required",
                        "final_reason_codes": locator_errors,
                    },
                )
            return final_text

    no_op_issue_ids = {
        str(issue.get("issue_id") or "")
        for issue in reflection_result.issues
        if isinstance(issue.get("draft_replacement"), dict)
        and (
            str(issue["draft_replacement"].get("draft") or "").strip()
            == str(issue["draft_replacement"].get("replacement") or "").strip()
            # An edit written as an expansion of what it quotes, where the draft
            # already reads the expansion. Left in, it applies nowhere and costs
            # the chunk two retries and a review.
            or replacement_already_in_draft(draft_translation, issue)
        )
    }
    # Severity decides how much a defect matters, not how sure the editor is
    # that it is one. Requiring `major` here threw away every confident `minor`
    # finding the editor had already chosen to repair -- one measured chunk
    # returned twelve exact-span edits and applied none of them -- while the
    # editor's own uncertainty already has a channel: `review_only`, which it
    # is told to use, and which is still refused below. Confidence remains the
    # gate, so a hedged repair of any severity stays a warning.
    actionable_issues = [
        issue for issue in reflection_result.issues
        if str(issue.get("issue_id") or "") not in no_op_issue_ids
        and str(issue.get("repair_kind") or "rewrite").casefold() != "review_only"
        and (
            issue.get("deterministic")
            or float(issue.get("confidence", 0.8) or 0.0) >= 0.80
        )
    ]
    # Accumulate. Assigning here erased the findings already dropped upstream --
    # an incomplete contract entry, or an issue whose locator never validated --
    # so a chunk where the editor reported five defects and four were discarded
    # before this point was recorded as having reported one.
    warning_count += len(reflection_result.issues) - len(actionable_issues)
    original_draft = draft_translation
    salvaged = apply_repairs_that_hold(
        draft_translation,
        actionable_issues,
        apply_patches=lambda text, items: apply_editor_patches(
            text,
            items,
            # The alignment the editor answered against is the one built from
            # the texts it was shown, and it was shown no source when there is
            # none.
            source_text=source_chunk if source_available else "",
        ),
        validate=lambda text, items: validate_editor_repair(
            text,
            items,
            draft_text=draft_translation,
            source_text=source_chunk,
            source_language=source_language,
            target_language=target_language,
            protected_terms=protected_terms,
            glossary_terms=glossary_terms,
        ),
    )
    patched_draft = salvaged.text
    unresolved_issues = salvaged.unresolved
    patch_errors = salvaged.errors
    unresolved_ids = {
        str(issue.get("issue_id") or "") for issue in unresolved_issues
    }
    locally_resolved = [
        issue for issue in actionable_issues
        if str(issue.get("issue_id") or "") not in unresolved_ids
    ]
    resolved_issue_count = len(locally_resolved)
    unresolved_issue_count = len(unresolved_issues) + review_issue_count
    if locally_resolved and patched_draft != original_draft:
        result_state = "locally_patched"
    if log_callback:
        emit_progress_log(
            log_callback,
            "editor_state_local_patch",
            (
                "Senior Editor state: local patch pass resolved "
                f"{len(locally_resolved)} issue(s); "
                f"{len(unresolved_issues)} remain actionable."
            ),
            layer="senior_editor_repair",
        )
    if patch_errors:
        patched_draft = original_draft
        unresolved_issues = list(actionable_issues)
        result_state = "unchanged_draft"
        resolved_issue_count = 0
        unresolved_issue_count = len(actionable_issues) + review_issue_count
    elif locally_resolved:
        # Already validated against the text that is actually being kept, and
        # re-running it here would re-report the faults the draft arrived with.
        patch_validation = list(salvaged.errors)
        if not patch_validation:
            if not unresolved_issues:
                local_term_pairs = extract_term_replacements_from_critique(
                    _reflection_feedback_payload(reflection_result)
                )
                if local_term_pairs and context_session and hasattr(
                    context_session, "register_editor_terms"
                ):
                    context_session.register_editor_terms(local_term_pairs)
                finish_run(
                    (
                        "review_required" if review_required
                        else "locally_repaired" if locally_resolved
                        else "warnings_only" if warning_count
                        else "no_issues"
                    ),
                    parse_status=reflection_result.parse_status,
                    issue_count=len(reflection_result.issues),
                    diagnostics={
                        "repair_mode": "local_patch",
                        "ignored_no_op_issue_ids": sorted(no_op_issue_ids),
                        # A batch that only held once something was given up
                        # should not read as a clean one.
                        "dropped_to_apply_the_rest": sorted(
                            str(issue.get("issue_id") or "unknown")
                            for issue in salvaged.dropped
                        ),
                    },
                )
                final_text = persist_final_voice(patched_draft)
                if review_required:
                    return review_required_translation(
                        final_text,
                        {
                            "stage": "locator_validation",
                            "status": "review_required",
                            "final_reason_codes": locator_errors,
                            "repair_mode": "local_patch",
                        },
                    )
                return final_text
        else:
            patched_draft = original_draft
            unresolved_issues = list(actionable_issues)
            patch_errors.extend(patch_validation)
            result_state = "unchanged_draft"
            resolved_issue_count = 0
            unresolved_issue_count = len(actionable_issues) + review_issue_count

    local_patch_followup_attempted = False
    already_read_ids: List[str] = []
    if (
        unresolved_issues
        and all(_issue_requires_draft_replacement(issue) for issue in unresolved_issues)
    ):
        from src.core.editor.locator_retry import run_focused_locator_retry

        local_patch_followup_attempted = True
        followup = await run_focused_locator_retry(
            session=session,
            followup_base=original_draft if patch_errors else patched_draft,
            unresolved_issues=unresolved_issues,
            patch_errors=patch_errors,
            source_chunk=source_chunk,
            source_language=source_language,
            target_language=target_language,
            protected_terms=protected_terms,
            glossary_terms=glossary_terms,
        )
        already_read_ids.extend(followup.already_read_ids)
        patch_errors = followup.patch_errors
        if followup.succeeded:
            patched_draft = followup.patched_draft
            resolved_issue_count += len(followup.resolved_issues)
            unresolved_issues = []
            unresolved_issue_count = review_issue_count
            result_state = "locally_patched"
            reflection_result = ReflectionResult(
                "needs_repair",
                followup.resolved_issues,
                followup.raw,
                followup.parse_status,
                followup.voice_observations,
            )

    if not unresolved_issues:
        finish_run(
            (
                "review_required" if review_required
                else "locally_repaired" if locally_resolved
                else "warnings_only" if warning_count
                else "no_issues"
            ),
            parse_status=reflection_result.parse_status,
            failure_class="local_patch_conflict" if patch_errors else None,
            issue_count=len(reflection_result.issues),
            diagnostics={
                "reason_codes": patch_errors,
                "repair_mode": (
                    "focused_local_patch_retry"
                    if local_patch_followup_attempted else "local_patch"
                ),
                "already_read_issue_ids": sorted(set(already_read_ids)),
            },
        )
        final_text = persist_final_voice(patched_draft)
        if review_required:
            return review_required_translation(
                final_text,
                {
                    "stage": "locator_validation",
                    "status": "review_required",
                    "final_reason_codes": locator_errors,
                    "repair_mode": "local_patch",
                },
            )
        return final_text

    reflection_result = ReflectionResult(
        "needs_repair",
        unresolved_issues,
        reflection_result.raw_text,
        reflection_result.parse_status,
        reflection_result.voice_observations,
    )
    draft_translation = patched_draft

    # A failed exact-span edit is not evidence that the whole translation needs
    # rewriting.  Rewriting here greatly increases prompt size and can damage
    # unrelated text, especially with compact editor models.  Locator repair has
    # already had its one bounded retry above, so retain the best structurally
    # valid draft and surface the remaining local issues for human review.
    if all(_issue_requires_draft_replacement(issue) for issue in unresolved_issues):
        reason_codes = [
            *patch_errors,
            *[
                f"local_patch_unresolved:{issue.get('issue_id') or 'unknown'}"
                for issue in unresolved_issues
            ],
        ]
        finish_run(
            "review_required",
            parse_status=reflection_result.parse_status,
            failure_class="local_patch_unresolved",
            issue_count=len(reflection_result.issues) + review_issue_count,
            diagnostics={
                "reason_codes": reason_codes,
                "repair_mode": "bounded_local_patch",
                "focused_followup_attempted": local_patch_followup_attempted,
            },
        )
        return review_required_translation(
            persist_final_voice(draft_translation),
            {
                "stage": "local_patch",
                "status": "review_required",
                "final_reason_codes": reason_codes,
                "repair_mode": "bounded_local_patch",
                "unresolved_issues": unresolved_issues,
                "editor_run_id": recorder.run_id,
            },
        )

    from src.utils.unified_logger import LogType, get_logger
    critique_summary = format_critique_tldr(critique)
    get_logger().info(
        f"Senior Editor critique: {critique_summary}",
        log_type=LogType.GENERAL,
    )

    if log_callback:
        log_callback("reflection_critique", f"Senior Editor critique: {critique_summary}")

    critique_feedback = _reflection_feedback_payload(reflection_result)
    term_pairs = extract_term_replacements_from_critique(critique_feedback)
    if not term_pairs:
        term_pairs = extract_term_replacements_from_critique(critique)
    if log_callback:
        emit_progress_log(
            log_callback,
            "editor_state_fallback_started",
            (
                "Senior Editor state: starting bounded full-chunk repair "
                f"(up to {max_automatic_repair_attempts} attempts) for "
                f"{len(unresolved_issues)} unresolved major issue(s)."
            ),
            layer="senior_editor_repair",
        )
    from src.core.editor.full_repair import run_full_chunk_repair

    repair = await run_full_chunk_repair(
        session=session,
        source_chunk=source_chunk,
        draft_translation=draft_translation,
        critique_feedback=critique_feedback,
        reflection_issues=reflection_result.issues,
        term_pairs=term_pairs,
        context_session=context_session,
        target_language=target_language,
        source_language=source_language,
        custom_instructions=custom_instructions,
        glossary_block=glossary_block,
        active_novel_context=active_novel_context,
        source_available=source_available,
        narrative_voice_context=narrative_voice_context,
        protected_terms=protected_terms,
        glossary_terms=glossary_terms,
        options=options,
        repair_validator=repair_validator,
        max_attempts=max_automatic_repair_attempts,
        log_callback=log_callback,
    )
    # finish_run reads these, so they carry what the loop ended with.
    automatic_repair_attempts = repair.attempts
    if repair.narrator_conformance is not None:
        narrator_conformance = repair.narrator_conformance
    if repair.repaired_text:
        finish_run(
            "review_required" if review_required else "llm_repaired",
            parse_status=reflection_result.parse_status,
            issue_count=len(reflection_result.issues),
            response_hash=response_hash(repair.raw_content),
            diagnostics={"repair_attempt": repair.attempts},
            result_state="rewritten",
            resolved_issue_count=len(reflection_result.issues),
            unresolved_issue_count=review_issue_count,
        )
        return persist_final_voice(repair.repaired_text)
    validation_errors = repair.validation_errors
    repair_attempt_diagnostics = repair.attempt_diagnostics

    has_deterministic_blocker = any(
        bool(issue.get("deterministic"))
        and str(issue.get("severity") or "major").casefold() in {"blocker", "major"}
        for issue in reflection_result.issues
    )
    diagnostics = {
        "stage": "repair_validation",
        "contract_version": REFLECTION_CONTRACT_VERSION,
        "parse_status": reflection_result.parse_status,
        "issues": reflection_result.issues,
        "attempts": repair_attempt_diagnostics,
        "final_reason_codes": list(validation_errors),
        "status": "blocked" if has_deterministic_blocker else "review_required",
    }
    if has_deterministic_blocker:
        if log_callback:
            emit_progress_log(
                log_callback,
                "editor_state_blocked",
                "Senior Editor state: hard blocked by deterministic validation.",
                level="error",
                layer="senior_editor_repair",
                data={"reasons": validation_errors},
            )
        narrator_only = bool(validation_errors) and all(
            str(reason).startswith("narrator_self_reference_mismatch:")
            for reason in validation_errors
        )
        finish_run(
            "review_required",
            parse_status=reflection_result.parse_status,
            failure_class=("narrator_policy" if narrator_only else "repair_validation"),
            issue_count=len(reflection_result.issues),
            diagnostics=diagnostics,
            unresolved_issue_count=len(validation_errors),
        )
        return review_required_translation(
            persist_final_voice(draft_translation), diagnostics,
        )
    if log_callback:
        emit_progress_log(
            log_callback,
            "editor_state_review_preserved",
            "Senior Editor state: valid draft preserved with review-required findings.",
            level="warning",
            layer="senior_editor_repair",
            data={"reasons": validation_errors},
        )
    finish_run(
        "review_required",
        parse_status=reflection_result.parse_status,
        failure_class=(
            "adapter_invalid" if any(
                str(reason).startswith("adapter_") for reason in validation_errors
            ) else "repair_validation"
        ),
        issue_count=len(reflection_result.issues),
        diagnostics=diagnostics,
    )
    return review_required_translation(
        persist_final_voice(draft_translation), diagnostics,
    )


def _split_text_preserving(value: str, count: int) -> List[str]:
    """Split at paragraph boundaries while preserving exact reconstruction."""

    text = str(value or "")
    if count <= 1 or not text:
        return [text]
    paragraph_boundaries = [
        match.end() for match in re.finditer(r"\n\s*\n", text)
    ]
    line_boundaries = [match.end() for match in re.finditer(r"\n", text)]
    sentence_boundaries = [
        match.end() for match in re.finditer(r"(?<=[.!?。！？])\s+", text)
    ]
    word_boundaries = [match.end() for match in re.finditer(r"\s+", text)]
    chosen: List[int] = []
    previous = 0
    for index in range(1, count):
        target = int(len(text) * index / count)
        boundary = 0
        for boundaries in (
            paragraph_boundaries, line_boundaries, sentence_boundaries,
            word_boundaries,
        ):
            candidates = [
                item for item in boundaries if previous < item < len(text)
            ]
            if candidates:
                boundary = min(candidates, key=lambda item: abs(item - target))
                break
        if not boundary:
            boundary = max(previous + 1, min(target, len(text) - 1))
        if boundary <= previous:
            continue
        chosen.append(boundary)
        previous = boundary
    points = [0, *chosen, len(text)]
    return [text[points[i]:points[i + 1]] for i in range(len(points) - 1)]


async def run_chunk_reflection_pass(*args: Any, **kwargs: Any):
    """Compatibility wrapper with bounded, complete editor review windows."""

    from src.core.editor import EditorService, review_required_translation

    signature = inspect.signature(_run_chunk_reflection_pass_impl)
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    values = dict(bound.arguments)
    options = dict(values.get("prompt_options") or {})
    source = str(values.get("source_chunk") or "")
    draft = str(values.get("draft_translation") or "")
    file_type = str(options.get("file_type") or "txt").casefold()
    # Full-unit review is the default. Windowing is an explicit provider/model
    # capacity fallback so relevant long-range evidence is not hidden merely
    # because a unit crosses an arbitrary application threshold.
    max_input_tokens = int(
        options.get("editor_max_input_tokens")
        or options.get("editor_model_input_limit")
        or 0
    )
    # The fixed editor contract and selected context use roughly 4k tokens.
    # Reserve that space, then split source/draft content conservatively.
    estimated_tokens = 4000 + int((len(source) + len(draft)) / 4)
    can_window = file_type in {"txt", "text", "docx", "epub"}
    if (
        not options.get("_editor_windowed")
        and can_window
        and max_input_tokens > 0
        and estimated_tokens > max_input_tokens
    ):
        usable_tokens = max(2000, max_input_tokens - 4000)
        window_count = max(
            2,
            int((len(source) + len(draft) + usable_tokens * 4 - 1) / (usable_tokens * 4)),
        )
        source_windows = _split_text_preserving(source, window_count)
        draft_windows = _split_text_preserving(draft, window_count)
        window_count = max(len(source_windows), len(draft_windows))
        while len(source_windows) < window_count:
            source_windows.append("")
        while len(draft_windows) < window_count:
            draft_windows.append("")
        final_windows: List[str] = []
        review_diagnostics = []
        for index, (source_window, draft_window) in enumerate(
            zip(source_windows, draft_windows), start=1,
        ):
            window_values = dict(values)
            window_options = dict(options)
            window_options.update({
                "_editor_windowed": True,
                "editor_window_index": index,
                "editor_window_count": window_count,
                "editor_neighbor_context": "\n\n".join(
                    part for part in (
                        (
                            "Previous source/draft tail:\n"
                            + source_windows[index - 2][-800:]
                            + "\n---\n"
                            + draft_windows[index - 2][-800:]
                        ) if index > 1 else "",
                        (
                            "Next source/draft head:\n"
                            + source_windows[index][:800]
                            + "\n---\n"
                            + draft_windows[index][:800]
                        ) if index < window_count else "",
                    ) if part
                ),
            })
            window_values.update({
                "source_chunk": source_window,
                "draft_translation": draft_window,
                "prompt_options": window_options,
                "repair_validator": None,
            })
            result = await EditorService().review_chunk(**window_values)
            final_windows.append(str(result))
            if getattr(result, "quality_status", "passed") == "review_required":
                review_diagnostics.append(
                    getattr(result, "editor_validation", {}) or {
                        "window_index": index,
                        "status": "review_required",
                    }
                )
        final_text = "".join(final_windows)
        validator = values.get("repair_validator")
        structural_error = validator(final_text) if validator else None
        if structural_error:
            review_diagnostics.append({
                "stage": "window_reassembly",
                "status": "review_required",
                "reason": str(structural_error),
            })
        if review_diagnostics:
            return review_required_translation(
                final_text,
                {
                    "stage": "windowed_editor",
                    "status": "review_required",
                    "windows": review_diagnostics,
                },
            )
        return final_text
    return await EditorService().review_chunk(**values)
