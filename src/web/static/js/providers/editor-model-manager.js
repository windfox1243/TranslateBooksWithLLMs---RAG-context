/**
 * Senior Editor provider/model selection.
 *
 * Empty values deliberately mean "inherit the translation setting". Model
 * inventory and rendering use the same backend and helpers as the draft model.
 */

import { ApiClient } from '../core/api-client.js';
import { SettingsManager } from '../core/settings-manager.js';
import { StateManager } from '../core/state-manager.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { SearchableSelectFactory } from '../ui/searchable-select.js';
import { t, applyToDOM } from '../i18n/i18n.js';
import { ApiKeyUtils } from '../utils/api-key-utils.js';
import {
    attachProviderSearchable,
    attachModelSearchable,
    populateModelSelectInto,
    setPlaceholderOption,
} from './provider-select-helpers.js';

function addInheritOption(select, key) {
    const option = document.createElement('option');
    option.value = '';
    option.dataset.i18n = key;
    option.textContent = t(key);
    select.insertBefore(option, select.firstChild);
}

const THINKING_MODE_KEYS = {
    auto: 'settings:thinking_mode_auto',
    off: 'settings:thinking_mode_off',
    on: 'settings:thinking_mode_on',
    minimal: 'settings:thinking_mode_minimal',
    low: 'settings:thinking_mode_low',
    medium: 'settings:thinking_mode_medium',
    high: 'settings:thinking_mode_high',
    dynamic: 'settings:thinking_mode_dynamic',
};

export const EditorModelManager = {
    initialized: false,
    loadSequence: 0,
    generationSequence: 0,
    pendingGenerationSettings: {},

    initialize() {
        if (this.initialized) return;
        const enabled = DomHelpers.getElement('enableReflection');
        const provider = DomHelpers.getElement('editorProvider');
        const model = DomHelpers.getElement('editorModel');
        if (!enabled || !provider || !model) return;

        this.initialized = true;
        attachProviderSearchable(provider, {
            placeholder: t('settings:search_providers_placeholder'),
            onChange: (value) => {
                this.rememberSelection(value || '', '', true);
                model.value = '';
                SearchableSelectFactory.get('editorModel')?.refresh();
                this.syncCredential();
                this.loadModels();
                this.syncGenerationControls();
            },
        });
        attachModelSearchable(model, {
            placeholder: t('settings:search_models_placeholder'),
            onChange: (value) => {
                this.rememberSelection(this.selectedProvider(), value || '', true);
                window.dispatchEvent(new CustomEvent('editorModelChanged'));
                this.syncGenerationControls();
            },
        });

        enabled.addEventListener('change', () => this.syncVisibility());
        DomHelpers.getElement('llmProvider')?.addEventListener('change', () => {
            this.syncCredential();
            if (!provider.value) this.loadModels();
            this.syncGenerationControls();
        });
        DomHelpers.getElement('model')?.addEventListener(
            'change', () => this.syncGenerationControls()
        );
        DomHelpers.getElement('editorOutputBudget')?.addEventListener(
            'change', () => this.syncCustomOutputVisibility()
        );
        window.addEventListener('defaultConfigLoaded', () => this.syncVisibility());
        window.addEventListener('localeChanged', () => this.refreshLocalizedDisplay());

        const pending = window.__pendingEditorSelection || {};
        this.setSelection(
            pending.provider || provider.value,
            pending.model || model.value,
            false,
            false,
        );
        delete window.__pendingEditorSelection;
        this.syncVisibility();
    },

    selectedOutputLimit(selectId) {
        const select = DomHelpers.getElement(selectId);
        const option = select?.selectedOptions?.[0];
        return parseInt(option?.dataset?.outputTokenLimit || '0', 10) || 0;
    },

    withSelectedModelMetadata(selectId, capabilities) {
        const select = DomHelpers.getElement(selectId);
        const raw = select?.selectedOptions?.[0]?.dataset?.reasoning;
        if (!raw) return capabilities;
        try {
            const reasoning = JSON.parse(raw);
            const efforts = Array.isArray(reasoning?.supported_efforts)
                ? reasoning.supported_efforts.map((value) => value === 'none' ? 'off' : value)
                : [];
            if (!efforts.length) return capabilities;
            const modes = ['auto', ...efforts.filter((value) => value !== 'off')];
            if (!reasoning.mandatory && efforts.includes('off')) modes.splice(1, 0, 'off');
            return {
                ...(capabilities || {}),
                thinking_supported: true,
                thinking_control: 'effort',
                thinking_modes: [...new Set(modes)],
                default_thinking_mode: reasoning.default_effort || 'auto',
                can_disable_thinking: !reasoning.mandatory,
            };
        } catch (_error) {
            return capabilities;
        }
    },

    renderThinkingModes(containerId, selectId, capabilities, enabled = true) {
        const container = DomHelpers.getElement(containerId);
        const select = DomHelpers.getElement(selectId);
        if (!container || !select) return;
        const supported = enabled && !!capabilities?.thinking_supported;
        container.dataset.thinkingSupported = supported ? 'true' : 'false';
        container.style.display = supported ? 'block' : 'none';
        if (!supported) {
            select.replaceChildren();
            const option = document.createElement('option');
            option.value = 'auto';
            option.dataset.i18n = THINKING_MODE_KEYS.auto;
            option.textContent = t(THINKING_MODE_KEYS.auto);
            select.appendChild(option);
            select.value = 'auto';
            return;
        }

        const previous = this.pendingGenerationSettings[selectId]
            || select.value
            || 'auto';
        select.replaceChildren();
        (capabilities.thinking_modes || ['auto']).forEach((mode) => {
            const option = document.createElement('option');
            option.value = mode;
            option.dataset.i18n = THINKING_MODE_KEYS[mode] || THINKING_MODE_KEYS.auto;
            option.textContent = t(option.dataset.i18n);
            select.appendChild(option);
        });
        select.value = Array.from(select.options).some((option) => option.value === previous)
            ? previous
            : 'auto';
        delete this.pendingGenerationSettings[selectId];
        container.dataset.outputTokenLimit = String(
            capabilities.output_token_limit || ''
        );
        applyToDOM(container);
    },

    async syncGenerationControls() {
        const sequence = ++this.generationSequence;
        const draftProvider = DomHelpers.getValue('llmProvider') || 'ollama';
        const draftModel = (DomHelpers.getValue('model') || '').trim();
        const editorProvider = this.effectiveProvider();
        const editorModel = this.effectiveModel();
        const reflectionEnabled = !!DomHelpers.getElement('enableReflection')?.checked;

        const load = async (provider, model, endpoint) => {
            if (!model) return null;
            try {
                return await ApiClient.getGenerationCapabilities(provider, model, endpoint);
            } catch (_error) {
                return null;
            }
        };
        const [draftCapabilities, editorCapabilities] = await Promise.all([
            load(draftProvider, draftModel, this.effectiveEndpoint(draftProvider)),
            reflectionEnabled
                ? load(editorProvider, editorModel, this.effectiveEndpoint(editorProvider))
                : Promise.resolve(null),
        ]);
        if (sequence !== this.generationSequence) return;

        const resolvedDraftCapabilities = this.withSelectedModelMetadata(
            'model', draftCapabilities
        );
        const resolvedEditorCapabilities = this.withSelectedModelMetadata(
            'editorModel', editorCapabilities
        );
        this.renderThinkingModes(
            'draftThinkingOptions', 'draftThinkingLevel', resolvedDraftCapabilities, true
        );
        this.renderThinkingModes(
            'editorThinkingOptions', 'editorThinkingLevel',
            resolvedEditorCapabilities, reflectionEnabled
        );
        const output = DomHelpers.getElement('editorOutputOptions');
        if (output) {
            output.style.display = reflectionEnabled ? 'block' : 'none';
            const outputLimit = (
                this.selectedOutputLimit('editorModel')
                || resolvedEditorCapabilities?.output_token_limit
                || 0
            );
            output.dataset.outputTokenLimit = String(outputLimit || '');
            const modelMax = DomHelpers.getElement('editorOutputBudget')
                ?.querySelector('option[value="model_max"]');
            if (modelMax) modelMax.disabled = !outputLimit;
            if (!outputLimit && DomHelpers.getValue('editorOutputBudget') === 'model_max') {
                DomHelpers.setValue('editorOutputBudget', 'auto');
            }
        }
        this.syncCustomOutputVisibility();
    },

    syncCustomOutputVisibility() {
        const custom = DomHelpers.getElement('editorCustomOutputContainer');
        const selected = DomHelpers.getValue('editorOutputBudget') || 'auto';
        if (custom) custom.style.display = selected === 'custom' ? 'block' : 'none';
    },

    effectiveProvider() {
        return this.selectedProvider() || DomHelpers.getValue('llmProvider') || 'ollama';
    },

    modelRequestOptions(provider) {
        const config = StateManager.getState('ui.defaultConfig') || {};
        let apiEndpoint;
        if (provider === 'ollama') {
            apiEndpoint = DomHelpers.getValue('apiEndpoint') || config.ollama_api_endpoint || config.api_endpoint;
        } else if (provider === 'openai') {
            apiEndpoint = DomHelpers.getValue('openaiEndpoint') || config.openai_api_endpoint;
        }
        const apiKey = this.usesSeparateProvider()
            ? (ApiKeyUtils.getValue('editorApiKey') || '__USE_ENV__')
            : (ApiKeyUtils.getValueForProvider(provider) || '__USE_ENV__');
        return { apiKey, apiEndpoint };
    },

    effectiveModel() {
        return (this.selectedModel() || DomHelpers.getValue('model') || '').trim();
    },

    effectiveEndpoint(provider = this.effectiveProvider()) {
        const config = StateManager.getState('ui.defaultConfig') || {};
        if (provider === 'ollama') {
            return DomHelpers.getValue('apiEndpoint') || config.ollama_api_endpoint || config.api_endpoint || '';
        }
        if (provider === 'openai') {
            return DomHelpers.getValue('openaiEndpoint') || config.openai_api_endpoint || '';
        }
        return '';
    },

    usesSeparateProvider() {
        return this.effectiveProvider() !== (DomHelpers.getValue('llmProvider') || 'ollama');
    },

    selectedProvider() {
        const select = DomHelpers.getElement('editorProvider');
        return String(
            select?.dataset?.persistedValue ?? select?.value ?? ''
        ).trim();
    },

    selectedModel() {
        const select = DomHelpers.getElement('editorModel');
        return String(
            select?.dataset?.persistedValue ?? select?.value ?? ''
        ).trim();
    },

    rememberSelection(providerValue = '', modelValue = '', persist = true) {
        const provider = DomHelpers.getElement('editorProvider');
        const model = DomHelpers.getElement('editorModel');
        const cleanProvider = String(providerValue || '').trim();
        const cleanModel = String(modelValue || '').trim();
        if (provider) provider.dataset.persistedValue = cleanProvider;
        if (model) model.dataset.persistedValue = cleanModel;
        if (persist) {
            SettingsManager.saveEditorSelection(cleanProvider, cleanModel);
        }
    },

    syncCredential() {
        const container = DomHelpers.getElement('editorKeyOptions');
        const field = DomHelpers.getElement('editorApiKey');
        const status = DomHelpers.getElement('editorKeyStatus');
        if (!container || !field) return;
        const enabled = !!DomHelpers.getElement('enableReflection')?.checked;
        const provider = this.effectiveProvider();
        const sourceFieldId = ApiKeyUtils.getFieldIdForProvider(provider);
        const visible = enabled && this.usesSeparateProvider() && !!sourceFieldId;
        container.style.display = visible ? 'block' : 'none';
        if (!visible) return;

        const source = DomHelpers.getElement(sourceFieldId);
        if (field.dataset.provider !== provider) field.value = '';
        field.dataset.provider = provider;
        field.dataset.envConfigured = source?.dataset.envConfigured || 'false';
        field.dataset.envKeyCount = source?.dataset.envKeyCount || '0';
        field.placeholder = source?.placeholder || t('settings:editor_api_key_placeholder');
        if (status) {
            const count = parseInt(field.dataset.envKeyCount || '0', 10);
            status.textContent = field.dataset.envConfigured === 'true'
                ? (count > 1
                    ? t('settings:key_status_configured_rotation', { count })
                    : t('settings:key_status_configured'))
                : t('settings:key_status_not_configured_badge');
            status.className = `key-status ${field.dataset.envConfigured === 'true' ? 'configured' : 'not-configured'}`;
        }
    },

    requestConfig() {
        const enabled = !!DomHelpers.getElement('enableReflection')?.checked;
        const promptOptions = {
            draft_thinking_level: DomHelpers.getValue('draftThinkingLevel') || 'auto',
            draft_reasoning_supported: (
                DomHelpers.getElement('draftThinkingOptions')?.dataset?.thinkingSupported
                === 'true'
            ),
        };
        if (!enabled) return { promptOptions, credentials: {} };
        const provider = this.selectedProvider();
        const effectiveProvider = this.effectiveProvider();
        const budgetMode = DomHelpers.getValue('editorOutputBudget') || 'auto';
        const outputBudget = budgetMode === 'custom'
            ? String(parseInt(DomHelpers.getValue('editorCustomOutputTokens'), 10) || 8192)
            : budgetMode;
        Object.assign(promptOptions, {
            editor_provider: provider,
            editor_model: this.selectedModel(),
            editor_api_endpoint: this.effectiveEndpoint(effectiveProvider),
            editor_thinking_level: DomHelpers.getValue('editorThinkingLevel') || 'auto',
            editor_reasoning_supported: (
                DomHelpers.getElement('editorThinkingOptions')?.dataset?.thinkingSupported
                === 'true'
            ),
            editor_max_output_tokens: outputBudget,
            editor_model_output_limit: (
                this.selectedOutputLimit('editorModel')
                || parseInt(
                    DomHelpers.getElement('editorOutputOptions')?.dataset?.outputTokenLimit || '0',
                    10,
                )
                || 0
            ),
            auto_review_repair_threshold: Math.max(
                0,
                Math.min(
                    parseInt(
                        DomHelpers.getValue('autoReviewRepairThreshold'), 10
                    ) || 0,
                    20
                )
            ),
        });
        const credentials = {};
        if (this.usesSeparateProvider() && ApiKeyUtils.getFieldIdForProvider(effectiveProvider)) {
            credentials[`${effectiveProvider}_api_key`] = ApiKeyUtils.getValue('editorApiKey');
        }
        return { promptOptions, credentials };
    },

    applyToRequest(config) {
        const runtime = this.requestConfig();
        config.prompt_options = { ...(config.prompt_options || {}), ...runtime.promptOptions };
        return Object.assign(config, runtime.credentials);
    },

    async loadModels() {
        const enabled = DomHelpers.getElement('enableReflection');
        const model = DomHelpers.getElement('editorModel');
        if (!enabled?.checked || !model) return;

        const sequence = ++this.loadSequence;
        const provider = this.effectiveProvider();
        const selected = this.selectedModel();
        setPlaceholderOption(model, 'common:loading');
        SearchableSelectFactory.get('editorModel')?.refresh();

        try {
            const data = await ApiClient.getModels(provider, this.modelRequestOptions(provider));
            if (sequence !== this.loadSequence) return;
            const models = data.models || [];
            populateModelSelectInto(model, models, selected, provider);
            addInheritOption(model, 'settings:editor_model_inherit');

            if (selected) {
                const exists = Array.from(model.options).some((option) => option.value === selected);
                if (!exists) {
                    const custom = document.createElement('option');
                    custom.value = selected;
                    custom.textContent = selected;
                    model.appendChild(custom);
                }
                model.value = selected;
            } else {
                model.value = '';
            }
            applyToDOM(model);
            SearchableSelectFactory.get('editorModel')?.refresh();
            this.syncGenerationControls();
        } catch (error) {
            if (sequence !== this.loadSequence) return;
            console.error('[EditorModelManager] Model fetch failed', error);
            setPlaceholderOption(model, 'settings:search_models_error');
            SearchableSelectFactory.get('editorModel')?.refresh();
        }
    },

    syncVisibility() {
        const enabled = !!DomHelpers.getElement('enableReflection')?.checked;
        const container = DomHelpers.getElement('editorModelOptions');
        if (container) container.style.display = enabled ? 'grid' : 'none';
        this.syncCredential();
        if (enabled) this.loadModels();
        this.syncGenerationControls();
    },

    setSelection(
        providerValue = '',
        modelValue = '',
        reload = true,
        persist = true,
    ) {
        const provider = DomHelpers.getElement('editorProvider');
        const model = DomHelpers.getElement('editorModel');
        if (!provider || !model) return;
        provider.value = providerValue || '';
        const providerPicker = SearchableSelectFactory.get('editorProvider');
        providerPicker?.refresh();

        const exists = Array.from(model.options).some((option) => option.value === modelValue);
        if (modelValue && !exists) {
            const option = document.createElement('option');
            option.value = modelValue;
            option.textContent = modelValue;
            model.appendChild(option);
        }
        model.value = modelValue || '';
        this.rememberSelection(providerValue, modelValue, persist);
        SearchableSelectFactory.get('editorModel')?.refresh();
        if (reload && DomHelpers.getElement('enableReflection')?.checked) this.loadModels();
        this.syncCredential();
        this.syncGenerationControls();
    },

    setGenerationSettings(settings = {}) {
        if (settings.draft_thinking_level) {
            this.pendingGenerationSettings.draftThinkingLevel = settings.draft_thinking_level;
        }
        if (settings.editor_thinking_level) {
            this.pendingGenerationSettings.editorThinkingLevel = settings.editor_thinking_level;
        }
        if (settings.auto_review_repair_threshold !== undefined) {
            DomHelpers.setValue(
                'autoReviewRepairThreshold',
                String(settings.auto_review_repair_threshold)
            );
        }
        const budget = String(settings.editor_max_output_tokens || 'auto');
        const standard = ['auto', '4096', '8192', '16384', 'model_max'];
        if (standard.includes(budget)) {
            DomHelpers.setValue('editorOutputBudget', budget);
        } else if (/^\d+$/.test(budget)) {
            DomHelpers.setValue('editorOutputBudget', 'custom');
            DomHelpers.setValue('editorCustomOutputTokens', budget);
        }
        this.syncCustomOutputVisibility();
        this.syncGenerationControls();
    },

    refreshLocalizedDisplay() {
        SearchableSelectFactory.get('editorProvider')?.refresh();
        SearchableSelectFactory.get('editorModel')?.refresh();
        this.syncCredential();
        this.syncGenerationControls();
    },
};
