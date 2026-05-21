/**
 * Settings Summary - concise overview of LLM + active options.
 *
 * Reads form state and renders a summary under the Start Translation button:
 *  - LLM line: gray text (provider · model · src → tgt)
 *  - Options line: each active option as a small colored chip
 */

import { DomHelpers } from './dom-helpers.js';

const PROVIDER_LABELS = {
    ollama: 'Ollama',
    gemini: 'Gemini',
    openai: 'OpenAI',
    openrouter: 'OpenRouter',
    mistral: 'Mistral',
    deepseek: 'DeepSeek',
    poe: 'Poe',
    nim: 'NVIDIA NIM',
};

// Each chip carries its own tint. Background uses rgba so it stays readable
// on both light and dark themes; the text uses the same hue at full saturation.
const OPTION_STYLES = {
    polish:       { bg: 'rgba(34, 197, 94, 0.18)',  fg: '#16a34a', border: 'rgba(34, 197, 94, 0.45)' },
    bilingual:    { bg: 'rgba(59, 130, 246, 0.18)', fg: '#2563eb', border: 'rgba(59, 130, 246, 0.45)' },
    draft:        { bg: 'rgba(245, 158, 11, 0.20)', fg: '#d97706', border: 'rgba(245, 158, 11, 0.45)' },
    ocr:          { bg: 'rgba(168, 85, 247, 0.18)', fg: '#9333ea', border: 'rgba(168, 85, 247, 0.45)' },
    noPause:      { bg: 'rgba(239, 68, 68, 0.18)',  fg: '#dc2626', border: 'rgba(239, 68, 68, 0.45)' },
    glossary:     { bg: 'rgba(99, 102, 241, 0.18)', fg: '#4f46e5', border: 'rgba(99, 102, 241, 0.45)' },
    instructions: { bg: 'rgba(20, 184, 166, 0.20)', fg: '#0d9488', border: 'rgba(20, 184, 166, 0.45)' },
    refineOnly:   { bg: 'rgba(34, 197, 94, 0.20)',  fg: '#16a34a', border: 'rgba(34, 197, 94, 0.55)' },
};

function getSelectText(id) {
    const el = DomHelpers.getElement(id);
    if (!el || el.selectedIndex < 0) return '';
    const opt = el.options[el.selectedIndex];
    return opt ? (opt.textContent || opt.value || '').trim() : '';
}

function getLanguage(selectId, customId) {
    const select = DomHelpers.getElement(selectId);
    if (!select) return '';
    if (select.value === 'Other') {
        const custom = DomHelpers.getElement(customId);
        return custom ? (custom.value || '').trim() : '';
    }
    return select.value || '';
}

function isChecked(id) {
    const el = DomHelpers.getElement(id);
    return !!(el && el.checked);
}

function buildLlmLine() {
    const providerKey = (DomHelpers.getValue('llmProvider') || '').trim();
    const providerLabel = PROVIDER_LABELS[providerKey] || providerKey || '—';
    const modelLabel = getSelectText('model') || DomHelpers.getValue('model') || '—';
    const sourceLang = getLanguage('sourceLang', 'customSourceLang') || 'auto-detect';
    const targetLang = getLanguage('targetLang', 'customTargetLang') || '—';
    if (isChecked('refineOnlyMode')) {
        return [providerLabel, modelLabel, `Refining in ${targetLang}`];
    }
    return [providerLabel, modelLabel, `${sourceLang} → ${targetLang}`];
}

function buildChips() {
    const chips = [];

    // Refine-only is exclusive: show only the prominent chip plus the few
    // options that still apply (glossary, instructions, auto-pause).
    if (isChecked('refineOnlyMode')) {
        chips.push({ key: 'refineOnly', label: 'Refine Only (skips translation)', prominent: true });

        const glossaryText = getSelectText('glossarySelect');
        if (glossaryText && glossaryText !== 'None') {
            const name = glossaryText.split('·')[0].trim();
            chips.push({ key: 'glossary', label: `Glossary: ${name}` });
        }
        const instrText = getSelectText('customInstructionSelect');
        if (instrText && instrText !== 'None') {
            chips.push({ key: 'instructions', label: `Instructions: ${instrText}` });
        }
        if (isChecked('disableAutoPause')) {
            chips.push({ key: 'noPause', label: 'No auto-pause' });
        }
        return chips;
    }

    if (isChecked('refineTranslation')) chips.push({ key: 'polish', label: 'Polish (2nd pass)' });
    if (isChecked('bilingualMode'))     chips.push({ key: 'bilingual', label: 'Bilingual' });
    if (isChecked('draftMode'))         chips.push({ key: 'draft', label: 'Draft mode' });
    if (isChecked('textCleanup'))       chips.push({ key: 'ocr', label: 'OCR cleanup' });
    if (isChecked('disableAutoPause'))  chips.push({ key: 'noPause', label: 'No auto-pause' });

    const glossaryText = getSelectText('glossarySelect');
    if (glossaryText && glossaryText !== 'None') {
        const name = glossaryText.split('·')[0].trim();
        chips.push({ key: 'glossary', label: `Glossary: ${name}` });
    }

    const instrText = getSelectText('customInstructionSelect');
    if (instrText && instrText !== 'None') {
        chips.push({ key: 'instructions', label: `Instructions: ${instrText}` });
    }

    return chips;
}

function renderChip({ key, label, prominent }) {
    const s = OPTION_STYLES[key] || OPTION_STYLES.bilingual;
    const style = [
        'display: inline-flex',
        'align-items: center',
        prominent ? 'padding: 6px 18px' : 'padding: 2px 10px',
        'border-radius: 999px',
        prominent ? 'font-size: 0.8125rem' : 'font-size: 0.75rem',
        'font-weight: 600',
        'line-height: 1.6',
        `background: ${s.bg}`,
        `color: ${s.fg}`,
        `border: ${prominent ? '1.5px' : '1px'} solid ${s.border}`,
    ].join('; ');
    return `<span style="${style}">${DomHelpers.escapeHtml(label)}</span>`;
}

function render() {
    const container = DomHelpers.getElement('settingsSummary');
    if (!container) return;

    const llmParts = buildLlmLine();
    const chips = buildChips();

    const sep = '<span style="opacity: 0.5; margin: 0 6px;">·</span>';
    const llmLine = llmParts.map(DomHelpers.escapeHtml).join(sep);

    const chipsHtml = chips.length
        ? `<div style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; justify-content: center;">
               ${chips.map(renderChip).join('')}
           </div>`
        : '';

    container.innerHTML = `<div>${llmLine}</div>${chipsHtml}`;
}

const WATCHED_IDS = [
    'llmProvider', 'model',
    'sourceLang', 'customSourceLang',
    'targetLang', 'customTargetLang',
    'refineTranslation', 'refineOnlyMode', 'bilingualMode', 'draftMode',
    'textCleanup', 'disableAutoPause',
    'glossarySelect', 'customInstructionSelect',
];

export const SettingsSummary = {
    initialize() {
        for (const id of WATCHED_IDS) {
            const el = DomHelpers.getElement(id);
            if (!el) continue;
            el.addEventListener('change', render);
            if (el.tagName === 'INPUT' && el.type === 'text') {
                el.addEventListener('input', render);
            }
        }
        // Several dropdowns are populated asynchronously (model list, custom
        // instructions). Those paths don't fire native change events, so we
        // also listen to the custom signals they emit after restoring state.
        window.addEventListener('modelChanged', render);
        window.addEventListener('customInstructionsLoaded', render);
        render();
    },
    refresh: render,
};
