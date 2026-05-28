/**
 * Progress Manager - Translation progress tracking and display
 *
 * Manages progress bar updates and statistics display for active translations.
 * Supports different file types (text, EPUB, SRT) with appropriate stat labels.
 */

import { DomHelpers } from '../ui/dom-helpers.js';
import { t } from '../i18n/i18n.js';
import { navigateToSetting } from '../ui/settings-summary.js';

// State for tracking chunk completion times (for ETA calculation)
let chunkCompletionTimes = [];
let lastCompletedChunks = 0;
let lastElapsedTime = 0;
const MAX_SAMPLES = 10; // Number of recent chunks to average for ETA

/**
 * Format elapsed time in a human-readable format
 * - Under 60s: shows seconds (e.g., "45.2s")
 * - Under 1h: shows minutes and seconds (e.g., "5m 23s")
 * - 1h+: shows hours, minutes and seconds (e.g., "1h 23m 45s")
 * @param {number} seconds - Elapsed time in seconds
 * @returns {string} Formatted time string
 */
export function formatElapsedTime(seconds) {
    if (seconds < 60) {
        return seconds.toFixed(1) + 's';
    }

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    }

    return `${minutes}m ${secs}s`;
}

/**
 * Calculate and update estimated time remaining
 * Uses a moving average of the last N chunk completion times
 * @param {number} completedChunks - Number of completed chunks
 * @param {number} totalChunks - Total number of chunks
 * @param {number} elapsedTime - Total elapsed time in seconds
 */
function updateEstimatedTimeRemaining(completedChunks, totalChunks, elapsedTime) {
    // Track time per chunk when a new chunk is completed
    if (completedChunks > lastCompletedChunks && elapsedTime > lastElapsedTime) {
        const chunksCompleted = completedChunks - lastCompletedChunks;
        const timeTaken = elapsedTime - lastElapsedTime;
        const timePerChunk = timeTaken / chunksCompleted;

        chunkCompletionTimes.push(timePerChunk);

        // Keep only the last N samples
        if (chunkCompletionTimes.length > MAX_SAMPLES) {
            chunkCompletionTimes.shift();
        }
    }

    lastCompletedChunks = completedChunks;
    lastElapsedTime = elapsedTime;

    // Calculate ETA only if we have samples
    if (chunkCompletionTimes.length === 0 || completedChunks === 0) {
        DomHelpers.setText('estimatedTimeRemaining', '--');
        return;
    }

    const remainingChunks = totalChunks - completedChunks;

    if (remainingChunks <= 0) {
        DomHelpers.setText('estimatedTimeRemaining', '0s');
        return;
    }

    // Calculate average time per chunk from recent samples
    const avgTimePerChunk = chunkCompletionTimes.reduce((a, b) => a + b, 0) / chunkCompletionTimes.length;
    const estimatedRemaining = avgTimePerChunk * remainingChunks;

    DomHelpers.setText('estimatedTimeRemaining', formatElapsedTime(estimatedRemaining));
}

/**
 * Reset ETA tracking state
 */
function resetEtaTracking() {
    chunkCompletionTimes = [];
    lastCompletedChunks = 0;
    lastElapsedTime = 0;
}

/**
 * Update progress bar
 * @param {number} percent - Progress percentage (0-100)
 */
function updateProgressBar(percent) {
    const progressBar = DomHelpers.getElement('progressBar');

    if (progressBar) {
        progressBar.style.width = percent + '%';
        progressBar.textContent = Math.round(percent) + '%';
    }
}

/**
 * Resolve the localized operation label ("Translating" / "Refining (2/2)" / etc.)
 * for the current phase of the workflow.
 *
 * @param {Object} stats - Server stats. May contain enable_refinement,
 *   current_phase, refine_only.
 * @returns {string} Label to show in the progress section title.
 */
export function resolveOperationLabel(stats) {
    if (!stats) return t('translation:translating');

    if (stats.enable_refinement) {
        return stats.current_phase === 2
            ? t('translation:refining_step', { step: 2, total: 2, defaultValue: 'Refining (2/2)' })
            : t('translation:translating_step', { step: 1, total: 2, defaultValue: 'Translating (1/2)' });
    }

    if (stats.refine_only) {
        return t('translation:refining');
    }

    return t('translation:translating');
}

/**
 * Update the operation label inside the current-file title without rebuilding
 * the surrounding DOM (icon, filename, languages). The element is created by
 * `updateTranslationTitle` in the file controllers and tagged with id
 * `progressOperationLabel`.
 *
 * @param {Object} stats - Server stats
 */
function updateOperationLabel(stats) {
    const labelEl = DomHelpers.getElement('progressOperationLabel');
    if (!labelEl) return;
    labelEl.textContent = resolveOperationLabel(stats);
}

// The Fallbacks card has a single active state — "critical" (red). Any
// fallback at all is treated as critical because placeholder loss already
// degrades the output; there is no intermediate "warning" tier any more.
//   false → nothing to surface (card neutral, panel hidden)
//   true  → red card + red panel with the full recommendation block
function isAlertActive(stats, fallbacks) {
    if (fallbacks > 0) return true;
    // Backend's one-shot quality flag and our client-side threshold check
    // are kept as additional ways to flip the card on even when fallbacks
    // are still 0 but placeholder errors are accumulating fast.
    if (stats && stats.quality_warning_fired) return true;
    return exceedsQualityThreshold(stats);
}

// Mirror of TranslationMetrics._QUALITY_* thresholds, kept here so the UI can
// detect the failure state from the cumulative stats payload even when no
// single XHTML file ever reached the per-file 5-chunk minimum (e.g. an EPUB
// split into many short chapters).
const QUALITY_MIN_PROCESSED = 5;
const QUALITY_RETRY_RATE_THRESHOLD = 0.30;
const QUALITY_FALLBACK_RATE_THRESHOLD = 0.10;
const QUALITY_AVG_ERRORS_THRESHOLD = 1.0;

function exceedsQualityThreshold(stats) {
    if (!stats) return false;
    const processed = stats.processed_chunks || 0;
    if (processed < QUALITY_MIN_PROCESSED) return false;
    const fallbackCount = (stats.token_alignment_used || 0) + (stats.fallback_used || 0);
    const notFirstTry = (stats.successful_after_retry || 0) + fallbackCount;
    const retryRate = notFirstTry / processed;
    const fallbackRate = fallbackCount / processed;
    const avgErrors = (stats.placeholder_errors || 0) / processed;
    return (
        retryRate > QUALITY_RETRY_RATE_THRESHOLD
        || fallbackRate > QUALITY_FALLBACK_RATE_THRESHOLD
        || avgErrors > QUALITY_AVG_ERRORS_THRESHOLD
    );
}

/**
 * Compute the rate metrics the critical intro line surfaces. We re-derive them
 * client-side from the cumulative stats payload so the numbers stay in sync
 * across multi-file EPUB runs (the per-file backend warning text would drift).
 */
function deriveRateContext(stats) {
    const processed = stats.processed_chunks || stats.completed_chunks || 0;
    if (processed <= 0) {
        return { retryPct: 0, fallbackPct: 0, avgErrors: '0.0', processed: 0 };
    }
    const fallbackCount = (stats.token_alignment_used || 0) + (stats.fallback_used || 0);
    const notFirstTry = (stats.successful_after_retry || 0) + fallbackCount;
    const placeholderErrors = stats.placeholder_errors || 0;
    return {
        retryPct: Math.round((notFirstTry / processed) * 100),
        fallbackPct: Math.round((fallbackCount / processed) * 100),
        avgErrors: (placeholderErrors / processed).toFixed(1),
        processed,
    };
}

/**
 * Build a list item that interpolates a "{{link}}" placeholder in the given
 * template with a clickable link that jumps to the matching setting.
 */
function buildTipWithLink(template, linkText, action) {
    const li = document.createElement('li');
    const idx = template.indexOf('{{link}}');
    if (idx === -1) {
        // Defensive: if the i18n string lost its placeholder, fall back to
        // appending the link at the end so the user still has the affordance.
        li.appendChild(document.createTextNode(template + ' '));
    } else {
        li.appendChild(document.createTextNode(template.slice(0, idx)));
    }
    const link = document.createElement('a');
    link.href = '#';
    link.className = 'recommendation-link';
    link.textContent = linkText;
    link.addEventListener('click', (event) => {
        event.preventDefault();
        navigateToSetting(action);
    });
    li.appendChild(link);
    if (idx !== -1) {
        li.appendChild(document.createTextNode(template.slice(idx + '{{link}}'.length)));
    }
    return li;
}

/**
 * Render the recommendation content into the inline panel: an intro line in
 * bold (with live retry / fallback / error rates) followed by a bulleted list
 * of mitigations, two of which carry clickable links to the matching settings
 * (chunk size and Plain Text Mode).
 *
 * @param {Object} [context] - Numbers used by the intro
 *   (retryPct, fallbackPct, avgErrors, processed).
 */
function renderRecommendationPanel(context) {
    const panel = DomHelpers.getElement('fallbackRecommendationPanel');
    if (!panel) return;

    panel.textContent = '';

    const ctx = context || { retryPct: 0, fallbackPct: 0, avgErrors: '0.0', processed: 0 };
    const intro = document.createElement('strong');
    intro.textContent = t('translation:fallback_panel_intro_critical', ctx);
    panel.appendChild(intro);

    const list = document.createElement('ul');
    list.className = 'recommendation-list';

    const llmTip = document.createElement('li');
    llmTip.textContent = t('translation:fallback_panel_tip_llm');
    list.appendChild(llmTip);

    list.appendChild(buildTipWithLink(
        t('translation:fallback_panel_tip_plain_text_mode'),
        t('translation:fallback_panel_link_plain_text_mode'),
        'plainText',
    ));

    panel.appendChild(list);
}

/**
 * Open/close the inline recommendation panel. When opening (or refreshing
 * while already open) we re-render with the latest context so the rate
 * numbers in the intro stay live.
 */
function toggleRecommendationPanel({ forceState, context } = {}) {
    const card = DomHelpers.getElement('fallbackStatCard');
    const panel = DomHelpers.getElement('fallbackRecommendationPanel');
    if (!card || !panel) return;
    const willOpen = forceState !== undefined
        ? forceState
        : panel.hasAttribute('hidden');
    if (willOpen) {
        renderRecommendationPanel(context);
        panel.removeAttribute('hidden');
        card.setAttribute('aria-expanded', 'true');
    } else {
        panel.setAttribute('hidden', '');
        card.setAttribute('aria-expanded', 'false');
    }
}

// Latest derived rate context (retry %, fallback %, avg errors), so the click
// handler can re-render the panel without re-receiving the stats payload.
let _lastSeverityContext = null;

let _fallbackCardClickBound = false;

function bindFallbackCardClick() {
    if (_fallbackCardClickBound) return;
    const card = DomHelpers.getElement('fallbackStatCard');
    if (!card) return;
    card.addEventListener('click', () => {
        // Only react when the card is in the alert state.
        if (!card.classList.contains('stat-card-critical')) return;
        toggleRecommendationPanel({ context: _lastSeverityContext });
    });
    _fallbackCardClickBound = true;
}

/**
 * Apply the alert highlight to the Fallbacks card and keep the inline panel
 * in sync. Only one active state exists ("critical", red palette):
 *  - inactive → card neutral, panel hidden
 *  - active   → red card, red panel with live retry/fallback/avg-error rates
 *
 * The panel stays open if the user already opened it; we just refresh its
 * content. When dropping back to inactive, any open panel is closed.
 */
function updateFallbackHighlight(count, stats) {
    const card = DomHelpers.getElement('fallbackStatCard');
    if (!card) return;

    const active = isAlertActive(stats, count);
    _lastSeverityContext = active ? deriveRateContext(stats || {}) : null;

    card.classList.toggle('stat-card-critical', active);
    card.title = t(active
        ? 'translation:stat_fallbacks_tooltip_critical'
        : 'translation:stat_fallbacks_tooltip');

    if (!active) {
        toggleRecommendationPanel({ forceState: false });
        return;
    }

    bindFallbackCardClick();
    const panel = DomHelpers.getElement('fallbackRecommendationPanel');
    if (panel && !panel.hasAttribute('hidden')) {
        renderRecommendationPanel(_lastSeverityContext);
    }
}

/**
 * Update statistics display based on file type
 * All file types (txt, epub, srt) show stats uniformly
 * @param {Object} stats - Statistics object from server
 * @param {string} fileType - File type ('txt', 'epub', 'srt')
 */
function updateStatistics(stats, fileType) {
    if (!stats) return;

    DomHelpers.show('statsGrid');

    DomHelpers.setText('totalChunks', stats.total_chunks || '0');
    DomHelpers.setText('completedChunks', stats.completed_chunks || '0');
    DomHelpers.setText('failedChunks', stats.failed_chunks || '0');

    if (fileType === 'srt') {
        // SRT does not use placeholders → Fallbacks stays at 0.
        DomHelpers.setText('fallbackChunks', '0');
        updateFallbackHighlight(0, stats);
    } else {
        const fallbacks = (stats.token_alignment_used || 0) + (stats.fallback_used || 0);
        DomHelpers.setText('fallbackChunks', String(fallbacks));
        updateFallbackHighlight(fallbacks, stats);
    }

    if (stats.elapsed_time !== undefined) {
        DomHelpers.setText('elapsedTime', formatElapsedTime(stats.elapsed_time));
        updateEstimatedTimeRemaining(
            stats.completed_chunks || 0,
            stats.total_chunks || 0,
            stats.elapsed_time
        );
    }
}

export const ProgressManager = {
    /**
     * Update progress display
     * @param {number} percent - Progress percentage (0-100)
     */
    updateProgress(percent) {
        updateProgressBar(percent);
    },

    /**
     * Update statistics display
     * @param {string} fileType - File type ('txt', 'epub', 'srt')
     * @param {Object} stats - Statistics object from server
     */
    updateStats(fileType, stats) {
        updateStatistics(stats, fileType);
    },

    /**
     * Update progress and statistics together
     * @param {Object} data - Update data from server
     * @param {Object} data.stats - Statistics object
     * @param {string} fileType - File type ('txt', 'epub', 'srt')
     */
    update(data, fileType) {
        if (!data.stats) return;

        const stats = data.stats;
        const completed = stats.completed_chunks || 0;
        const total = stats.total_chunks || 0;

        const phasePercent = total > 0 ? (completed / total) * 100 : 0;
        const enableRefinement = !!stats.enable_refinement;

        // Global bar:
        //   - refine_after (enable_refinement === true): the backend runs two
        //     independent progress trackers (translate then refine), so we
        //     compute the unified 0-100% client-side: phase 1 maps to 0-50%
        //     and phase 2 maps to 50-100%. This keeps the bar monotonic
        //     across the phase transition.
        //   - single-phase runs (plain translation, refine-only): trust the
        //     backend's progress_percent when present, otherwise fall back to
        //     chunk-based.
        let globalPercent;
        if (enableRefinement) {
            const phase = stats.current_phase || 1;
            globalPercent = phase === 2 ? 50 + phasePercent * 0.5 : phasePercent * 0.5;
        } else if (typeof stats.progress_percent === 'number') {
            globalPercent = stats.progress_percent;
        } else {
            globalPercent = phasePercent;
        }
        updateProgressBar(globalPercent);

        updateOperationLabel(stats);
        updateStatistics(stats, fileType);
    },

    /**
     * Reset progress display to initial state
     */
    reset() {
        updateProgressBar(0);
        DomHelpers.setText('totalChunks', '0');
        DomHelpers.setText('completedChunks', '0');
        DomHelpers.setText('failedChunks', '0');
        DomHelpers.setText('fallbackChunks', '0');
        _lastSeverityContext = null;
        updateFallbackHighlight(0);
        DomHelpers.setText('elapsedTime', '0s');
        DomHelpers.setText('estimatedTimeRemaining', '--');
        resetEtaTracking();
        DomHelpers.hide('statsGrid');
    },

    /**
     * Show progress section
     */
    show() {
        DomHelpers.show('progressSection');
    },

    /**
     * Hide progress section
     */
    hide() {
        DomHelpers.hide('progressSection');
    },

    /**
     * Set progress to complete (100%)
     */
    complete() {
        updateProgressBar(100);
    },

    /**
     * Get current progress percentage
     * @returns {number} Current progress (0-100)
     */
    getCurrentProgress() {
        const progressBar = DomHelpers.getElement('progressBar');

        if (!progressBar) return 0;

        const widthStyle = progressBar.style.width;
        const match = widthStyle.match(/(\d+(?:\.\d+)?)/);

        return match ? parseFloat(match[1]) : 0;
    }
};
