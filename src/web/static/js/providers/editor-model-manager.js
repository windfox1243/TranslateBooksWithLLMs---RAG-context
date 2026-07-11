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
            onChange: () => this.loadModels(),
        });
        attachModelSearchable(model, {
            placeholder: t('settings:search_models_placeholder'),
            onChange: () => window.dispatchEvent(new CustomEvent('editorModelChanged')),
        });

        enabled.addEventListener('change', () => this.syncVisibility());
        DomHelpers.getElement('llmProvider')?.addEventListener('change', () => {
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
        return { apiKey: '__USE_ENV__', apiEndpoint };
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
    },

    refreshLocalizedDisplay() {
        SearchableSelectFactory.get('editorProvider')?.refresh();
        SearchableSelectFactory.get('editorModel')?.refresh();
    },
};
