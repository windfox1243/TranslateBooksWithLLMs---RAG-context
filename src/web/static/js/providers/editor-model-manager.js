/**
 * Senior Editor provider/model selection.
 *
 * Empty values deliberately mean "inherit the translation setting". Model
 * inventory and rendering use the same backend and helpers as the draft model.
 */

import { ApiClient } from '../core/api-client.js';
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

export const EditorModelManager = {
    initialized: false,
    loadSequence: 0,

    initialize() {
        if (this.initialized) return;
        const enabled = DomHelpers.getElement('enableReflection');
        const provider = DomHelpers.getElement('editorProvider');
        const model = DomHelpers.getElement('editorModel');
        if (!enabled || !provider || !model) return;

        this.initialized = true;
        attachProviderSearchable(provider, {
            placeholder: t('settings:search_providers_placeholder'),
            onChange: () => {
                this.syncCredential();
                this.loadModels();
            },
        });
        attachModelSearchable(model, {
            placeholder: t('settings:search_models_placeholder'),
            onChange: () => window.dispatchEvent(new CustomEvent('editorModelChanged')),
        });

        enabled.addEventListener('change', () => this.syncVisibility());
        DomHelpers.getElement('llmProvider')?.addEventListener('change', () => {
            this.syncCredential();
            if (!provider.value) this.loadModels();
        });
        window.addEventListener('defaultConfigLoaded', () => this.syncVisibility());
        window.addEventListener('localeChanged', () => this.refreshLocalizedDisplay());

        const pending = window.__pendingEditorSelection || {};
        this.setSelection(pending.provider || provider.value, pending.model || model.value, false);
        delete window.__pendingEditorSelection;
        this.syncVisibility();
    },

    effectiveProvider() {
        return DomHelpers.getValue('editorProvider') || DomHelpers.getValue('llmProvider') || 'ollama';
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
        return (DomHelpers.getValue('editorModel') || DomHelpers.getValue('model') || '').trim();
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
        if (!enabled) return { promptOptions: {}, credentials: {} };
        const provider = DomHelpers.getValue('editorProvider') || '';
        const effectiveProvider = this.effectiveProvider();
        const promptOptions = {
            editor_provider: provider,
            editor_model: (DomHelpers.getValue('editorModel') || '').trim(),
            editor_api_endpoint: this.effectiveEndpoint(effectiveProvider),
        };
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
        const selected = model.value;
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
    },

    setSelection(providerValue = '', modelValue = '', reload = true) {
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
        SearchableSelectFactory.get('editorModel')?.refresh();
        if (reload && DomHelpers.getElement('enableReflection')?.checked) this.loadModels();
        this.syncCredential();
    },

    refreshLocalizedDisplay() {
        SearchableSelectFactory.get('editorProvider')?.refresh();
        SearchableSelectFactory.get('editorModel')?.refresh();
        this.syncCredential();
    },
};
