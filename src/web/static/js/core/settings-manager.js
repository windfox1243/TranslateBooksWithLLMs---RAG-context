/**
 * Settings Manager - User preferences persistence
 *
 * Handles saving/loading user preferences via:
 * 1. localStorage for per-browser translation options and languages
 * 2. Server API for provider defaults and sensitive data saved to .env
 */

import { ApiClient } from './api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { MessageLogger } from '../ui/message-logger.js';
import { t } from '../i18n/i18n.js';

// Storage configuration with versioning
const STORAGE_VERSION = 1;
const STORAGE_KEY_PREFIX = 'tbl_user_preferences';
const STORAGE_KEY = `${STORAGE_KEY_PREFIX}_v${STORAGE_VERSION}`;

/**
 * Validate user preferences structure
 * @param {any} data - Data to validate
 * @returns {boolean} True if valid
 */
function validatePreferences(data) {
    if (!data || typeof data !== 'object') return false;

    // Check version
    if (!('version' in data)) return false;

    // Validate types for known fields (non-exhaustive, just critical ones)
    if ('ttsEnabled' in data && typeof data.ttsEnabled !== 'boolean') return false;
    if ('textCleanup' in data && typeof data.textCleanup !== 'boolean') return false;
    if ('chapterMode' in data && typeof data.chapterMode !== 'boolean') return false;

    return true;
}

/**
 * Flag to prevent localStorage from overriding .env default model
 * Set to true once the .env model has been applied
 */
let envModelApplied = false;

/**
 * Debounce timer for auto-save
 */
let autoSaveTimer = null;
const AUTO_SAVE_DELAY = 1000; // 1 second debounce

/**
 * Flag to prevent auto-save during initial load
 */
let isInitializing = true;

export const SettingsManager = {
    /**
     * Initialize settings manager - load saved preferences and setup auto-save
     */
    initialize() {
        // Clean up old storage versions
        this.cleanupOldStorageVersions();

        this.loadLocalPreferences();

        // Listen for custom instructions loaded event
        window.addEventListener('customInstructionsLoaded', () => {
            this.applyPendingCustomInstructionSelection();
        });

        // Listen for novel contexts loaded event
        window.addEventListener('novelContextsLoaded', () => {
            this.applyPendingNovelContextSelection();
        });

        // Setup auto-save listeners after a short delay to avoid triggering during initial load
        setTimeout(() => {
            this._setupAutoSaveListeners();
            isInitializing = false;
        }, 500);
    },

    /**
     * Clean up old localStorage versions
     */
    cleanupOldStorageVersions() {
        try {
            // Remove old non-versioned key
            const oldKey = 'tbl_user_preferences';
            if (localStorage.getItem(oldKey)) {
                // Migrate data from old key before removing
                const oldData = localStorage.getItem(oldKey);
                if (oldData) {
                    try {
                        const parsed = JSON.parse(oldData);
                        // Add version and save to new key
                        parsed.version = STORAGE_VERSION;
                        localStorage.setItem(STORAGE_KEY, JSON.stringify(parsed));
                    } catch (e) {
                        console.warn('Could not migrate old preferences:', e);
                    }
                }
                localStorage.removeItem(oldKey);
            }

            // Remove any other versions (future-proofing)
            for (let i = 0; i < STORAGE_VERSION; i++) {
                const oldVersionKey = `${STORAGE_KEY_PREFIX}_v${i}`;
                if (localStorage.getItem(oldVersionKey)) {
                    localStorage.removeItem(oldVersionKey);
                }
            }
        } catch (error) {
            console.warn('Failed to cleanup old storage versions:', error);
        }
    },

    /**
     * Setup event listeners for auto-save on all settings elements
     * @private
     */
    _setupAutoSaveListeners() {
        // Auto-save targets localStorage only. Fields that persist to .env
        // (provider, model, endpoints, API keys, naming convention) are saved
        // exclusively via the explicit "Save Settings to .env" button.
        const localAutoSaveElements = [
            { id: 'sourceLang', event: 'change' },
            { id: 'targetLang', event: 'change' },
            { id: 'customSourceLang', event: 'change' },
            { id: 'customTargetLang', event: 'change' },
            { id: 'ttsEnabled', event: 'change' },
            { id: 'textCleanup', event: 'change' },
            { id: 'bilingualMode', event: 'change' },
            { id: 'plainTextMode', event: 'change' },
            { id: 'chapterMode', event: 'change' },
            { id: 'enableReflection', event: 'change' },
            { id: 'editorProvider', event: 'change' },
            { id: 'editorModel', event: 'change' },
            { id: 'customInstructionSelect', event: 'change' },
            { id: 'novelContextSelect', event: 'change' },
            { id: 'autoUpdateContext', event: 'change' }
        ];

        localAutoSaveElements.forEach(({ id, event }) => {
            const element = DomHelpers.getElement(id);
            if (element) {
                element.addEventListener(event, () => this._triggerAutoSave());
            }
        });

        // Dirty-tracking for .env fields: any change marks the Save button as
        // having pending changes; saveAllSettings(true) clears it on success.
        const envDirtyElements = [
            { id: 'llmProvider', event: 'change' },
            { id: 'model', event: 'change' },
            { id: 'apiEndpoint', event: 'input' },
            { id: 'openaiEndpoint', event: 'input' },
            { id: 'outputFilenamePattern', event: 'input' },
            { id: 'geminiApiKey', event: 'input' },
            { id: 'openaiApiKey', event: 'input' },
            { id: 'openrouterApiKey', event: 'input' },
            { id: 'mistralApiKey', event: 'input' },
            { id: 'deepseekApiKey', event: 'input' },
            { id: 'poeApiKey', event: 'input' },
            { id: 'nimApiKey', event: 'input' },
            { id: 'disableAutoPause', event: 'change' },
            { id: 'bypassContextGating', event: 'change' },
            { id: 'enableReflection', event: 'change' },
            { id: 'parallelWorkers', event: 'input' }
        ];

        envDirtyElements.forEach(({ id, event }) => {
            const element = DomHelpers.getElement(id);
            if (element) {
                element.addEventListener(event, () => this._markEnvDirty());
            }
        });
    },

    /**
     * Enable the .env Save button — there are pending unsaved changes
     */
    _markEnvDirty() {
        if (isInitializing) return;
        const btn = DomHelpers.getElement('saveSettingsBtn');
        if (btn) btn.disabled = false;
    },

    /**
     * Disable the .env Save button — nothing to save
     */
    _clearEnvDirty() {
        const btn = DomHelpers.getElement('saveSettingsBtn');
        if (btn) btn.disabled = true;
    },

    /**
     * Trigger auto-save with debounce
     * @private
     */
    _triggerAutoSave() {
        if (isInitializing) return;

        // Clear existing timer
        if (autoSaveTimer) {
            clearTimeout(autoSaveTimer);
        }

        // Set new timer
        autoSaveTimer = setTimeout(async () => {
            await this._performAutoSave();
        }, AUTO_SAVE_DELAY);
    },

    /**
     * Perform the actual auto-save
     * @private
     */
    async _performAutoSave() {
        try {
            await this.saveAllSettings(false);
        } catch {
            // Auto-save failed silently
        }
    },

    /**
     * Get all local preferences from localStorage
     * @returns {Object} Saved preferences
     */
    getLocalPreferences() {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);

            if (!stored) return {};

            const parsed = JSON.parse(stored);

            // Validate structure
            if (!validatePreferences(parsed)) {
                console.warn('Invalid preferences structure, resetting to defaults');
                localStorage.removeItem(STORAGE_KEY);
                return {};
            }

            // Check version compatibility
            if (parsed.version !== STORAGE_VERSION) {
                console.warn(`Preferences version mismatch (found ${parsed.version}, expected ${STORAGE_VERSION})`);
                // Could implement migration here in the future
                localStorage.removeItem(STORAGE_KEY);
                return {};
            }

            return parsed;
        } catch (error) {
            console.error('Failed to load preferences from localStorage:', error);
            MessageLogger.addLog(t('translation:preferences_load_failed_log'));
            return {};
        }
    },

    /**
     * Save preferences to localStorage
     * @param {Object} prefs - Preferences to save
     */
    saveLocalPreferences(prefs) {
        try {
            const current = this.getLocalPreferences();
            const updated = {
                ...current,
                ...prefs,
                version: STORAGE_VERSION,
                timestamp: Date.now()
            };

            localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
        } catch (error) {
            console.error('Failed to save preferences to localStorage:', error);

            // Check if it's a quota exceeded error
            if (error.name === 'QuotaExceededError') {
                MessageLogger.addLog(t('translation:preferences_save_quota'));
            } else {
                MessageLogger.addLog(t('translation:preferences_save_failed_log'));
            }
        }
    },

    /**
     * Load and apply saved local preferences to the form
     */
    loadLocalPreferences() {
        const prefs = this.getLocalPreferences();

        // Apply last languages
        if (prefs.lastSourceLanguage) {
            this._setLanguage('sourceLang', 'customSourceLang', prefs.lastSourceLanguage);
        }
        if (prefs.lastTargetLanguage) {
            this._setLanguage('targetLang', 'customTargetLang', prefs.lastTargetLanguage);
        }

        // Provider, model, endpoints, and output naming are .env-backed fields.
        // Legacy localStorage copies are intentionally ignored so browser state
        // cannot hide the packaged application's current configuration.

        // Apply TTS Enabled setting
        if (prefs.ttsEnabled !== undefined) {
            const ttsEnabledCheckbox = DomHelpers.getElement('ttsEnabled');
            if (ttsEnabledCheckbox) {
                ttsEnabledCheckbox.checked = prefs.ttsEnabled;
                // Show/hide the TTS options panel based on checkbox state
                const ttsOptions = DomHelpers.getElement('ttsOptions');
                if (ttsOptions) {
                    ttsOptions.style.display = prefs.ttsEnabled ? 'block' : 'none';
                }
            }
        }

        // Apply Prompt Options settings
        if (prefs.textCleanup !== undefined) {
            const cleanupCheckbox = DomHelpers.getElement('textCleanup');
            if (cleanupCheckbox) {
                cleanupCheckbox.checked = prefs.textCleanup;
            }
        }
        if (prefs.bilingualMode !== undefined) {
            const bilingualCheckbox = DomHelpers.getElement('bilingualMode');
            if (bilingualCheckbox) {
                bilingualCheckbox.checked = prefs.bilingualMode;
            }
        }
        if (prefs.plainTextMode !== undefined) {
            const plainTextCheckbox = DomHelpers.getElement('plainTextMode');
            if (plainTextCheckbox) {
                plainTextCheckbox.checked = prefs.plainTextMode;
            }
        }
        if (prefs.chapterMode !== undefined) {
            const chapterModeCheckbox = DomHelpers.getElement('chapterMode');
            if (chapterModeCheckbox) {
                chapterModeCheckbox.checked = prefs.chapterMode;
            }
        }
        if (prefs.enableReflection !== undefined) {
            const enableReflectionCheckbox = DomHelpers.getElement('enableReflection');
            if (enableReflectionCheckbox) {
                enableReflectionCheckbox.checked = prefs.enableReflection;
            }
        }
        if (prefs.editorProvider !== undefined || prefs.editorModel !== undefined) {
            window.__pendingEditorSelection = {
                provider: prefs.editorProvider || '',
                model: prefs.editorModel || '',
            };
        }
        // Note: disableAutoPause is now loaded from .env via /api/config in FormManager,
        // not from localStorage.

        // Store custom instruction file for later application (after loadCustomInstructions completes)
        if (prefs.customInstructionFile) {
            window.__pendingCustomInstructionSelection = prefs.customInstructionFile;
        }

        // Store novel context file for later application
        if (prefs.novelContextFile) {
            window.__pendingNovelContextSelection = prefs.novelContextFile;
        }
        if (prefs.autoUpdateContext !== undefined) {
            const autoUpdateCheckbox = DomHelpers.getElement('autoUpdateContext');
            if (autoUpdateCheckbox) {
                autoUpdateCheckbox.checked = prefs.autoUpdateContext;
            }
        }

        // Keep Prompt Options section open if any option is active.
        // Note: disableAutoPause now lives in the Provider & Defaults section, not here.
        const hasAnyPromptOption = prefs.textCleanup || prefs.bilingualMode || prefs.plainTextMode || prefs.chapterMode || prefs.enableReflection || prefs.customInstructionFile || prefs.novelContextFile || prefs.autoUpdateContext;
        if (hasAnyPromptOption) {
            const promptOptionsSection = DomHelpers.getElement('promptOptionsSection');
            const promptOptionsIcon = DomHelpers.getElement('promptOptionsIcon');
            if (promptOptionsSection) {
                promptOptionsSection.classList.remove('hidden');
            }
            if (promptOptionsIcon) {
                promptOptionsIcon.style.transform = 'rotate(180deg)';
            }
        }
    },

    /**
     * Set language in select/custom input
     * @private
     */
    _setLanguage(selectId, customInputId, value) {
        const select = DomHelpers.getElement(selectId);
        const customInput = DomHelpers.getElement(customInputId);

        if (!select) return;

        // Check if value exists in options (excluding "Other" which is just a placeholder)
        let found = false;
        for (let option of select.options) {
            // Skip "Other" option - we only want to match actual language values
            if (option.value === 'Other') continue;

            if (option.value.toLowerCase() === value.toLowerCase()) {
                select.value = option.value;
                found = true;
                break;
            }
        }

        // If language is not in the predefined list, use "Other" and fill custom input
        if (!found && customInput) {
            select.value = 'Other';
            customInput.value = value;
            // Show the custom input - need both class removal AND style change
            // because HTML has inline style="display: none"
            customInput.classList.remove('hidden');
            customInput.style.display = 'block';
        }
    },

    /**
     * Save current form state to local preferences
     */
    saveCurrentState() {
        const ttsEnabledCheckbox = DomHelpers.getElement('ttsEnabled');
        const textCleanupCheckbox = DomHelpers.getElement('textCleanup');
        const bilingualModeCheckbox = DomHelpers.getElement('bilingualMode');
        const plainTextModeCheckbox = DomHelpers.getElement('plainTextMode');
        const chapterModeCheckbox = DomHelpers.getElement('chapterMode');
        const enableReflectionCheckbox = DomHelpers.getElement('enableReflection');

        const prefs = {
            lastSourceLanguage: this._getLanguageValue('sourceLang', 'customSourceLang'),
            lastTargetLanguage: this._getLanguageValue('targetLang', 'customTargetLang'),
            ttsEnabled: ttsEnabledCheckbox ? ttsEnabledCheckbox.checked : false,
            textCleanup: textCleanupCheckbox ? textCleanupCheckbox.checked : false,
            bilingualMode: bilingualModeCheckbox ? bilingualModeCheckbox.checked : false,
            plainTextMode: plainTextModeCheckbox ? plainTextModeCheckbox.checked : false,
            chapterMode: chapterModeCheckbox ? chapterModeCheckbox.checked : false,
            enableReflection: enableReflectionCheckbox ? enableReflectionCheckbox.checked : false,
            editorProvider: DomHelpers.getValue('editorProvider') || '',
            editorModel: DomHelpers.getValue('editorModel') || '',
            customInstructionFile: DomHelpers.getValue('customInstructionSelect') || '',
            novelContextFile: DomHelpers.getValue('novelContextSelect') || '',
            autoUpdateContext: DomHelpers.getElement('autoUpdateContext')?.checked || false
        };

        this.saveLocalPreferences(prefs);
    },

    /**
     * Get language value from select or custom input
     * @private
     */
    _getLanguageValue(selectId, customInputId) {
        const selectVal = DomHelpers.getValue(selectId);
        if (selectVal === 'Other') {
            return DomHelpers.getValue(customInputId) || selectVal;
        }
        return selectVal;
    },

    /**
     * Save all current settings (both local and to .env)
     * @param {boolean} includeApiKeys - Whether to save API keys to .env
     * @returns {Promise<Object>} Result with success status
     */
    async saveAllSettings(includeApiKeys = false) {
        // Save local preferences
        this.saveCurrentState();

        if (includeApiKeys) {
            // Collect API keys to save
            const envSettings = {};
            const provider = DomHelpers.getValue('llmProvider');

            if (provider === 'gemini') {
                const key = DomHelpers.getValue('geminiApiKey');
                if (key) envSettings['GEMINI_API_KEY'] = key;
            } else if (provider === 'openai') {
                const key = DomHelpers.getValue('openaiApiKey');
                if (key) envSettings['OPENAI_API_KEY'] = key;
            } else if (provider === 'openrouter') {
                const key = DomHelpers.getValue('openrouterApiKey');
                if (key) envSettings['OPENROUTER_API_KEY'] = key;
            } else if (provider === 'mistral') {
                const key = DomHelpers.getValue('mistralApiKey');
                if (key) envSettings['MISTRAL_API_KEY'] = key;
            } else if (provider === 'deepseek') {
                const key = DomHelpers.getValue('deepseekApiKey');
                if (key) envSettings['DEEPSEEK_API_KEY'] = key;
            } else if (provider === 'poe') {
                const key = DomHelpers.getValue('poeApiKey');
                if (key) envSettings['POE_API_KEY'] = key;
            } else if (provider === 'nim') {
                const key = DomHelpers.getValue('nimApiKey');
                if (key) envSettings['NIM_API_KEY'] = key;
            }

            // Save endpoints to .env
            const ollamaEndpoint = DomHelpers.getValue('apiEndpoint');
            const openaiEndpoint = DomHelpers.getValue('openaiEndpoint');
            if (ollamaEndpoint) {
                envSettings['OLLAMA_API_ENDPOINT'] = ollamaEndpoint;
            }
            if (openaiEndpoint) {
                envSettings['OPENAI_API_ENDPOINT'] = openaiEndpoint;
            }

            // Save output filename pattern (naming convention)
            const filenamePattern = DomHelpers.getValue('outputFilenamePattern');
            if (filenamePattern) {
                envSettings['OUTPUT_FILENAME_PATTERN'] = filenamePattern;
            }

            // Save disable auto-pause flag (runtime behavior default)
            const disableAutoPauseCheckbox = DomHelpers.getElement('disableAutoPause');
            envSettings['DISABLE_AUTO_PAUSE'] = (disableAutoPauseCheckbox && disableAutoPauseCheckbox.checked) ? 'true' : 'false';

            // Save bypass context gating flag (runtime behavior default)
            const bypassContextGatingCheckbox = DomHelpers.getElement('bypassContextGating');
            envSettings['BYPASS_CONTEXT_GATING'] = (bypassContextGatingCheckbox && bypassContextGatingCheckbox.checked) ? 'true' : 'false';

            // Save chunk reflection flag (runtime behavior default)
            const enableReflectionCheckbox = DomHelpers.getElement('enableReflection');
            envSettings['ENABLE_CHUNK_REFLECTION'] = (enableReflectionCheckbox && enableReflectionCheckbox.checked) ? 'true' : 'false';

            // Save parallel-requests default (the per-job request still overrides
            // this; it only seeds the input on next load). Backend clamps it.
            const parallelWorkersInput = DomHelpers.getElement('parallelWorkers');
            if (parallelWorkersInput) {
                const pw = parseInt(parallelWorkersInput.value, 10);
                envSettings['PARALLEL_TRANSLATIONS'] = String(Number.isFinite(pw) && pw > 0 ? pw : 1);
            }

            // Webhook notifications — always serialized (even empty) so the user
            // can disable notifications by clearing the URL and clicking Save.
            const notifyUrl = DomHelpers.getElement('notifyWebhookUrl');
            if (notifyUrl) {
                envSettings['NOTIFY_WEBHOOK_URL'] = notifyUrl.value.trim();
                envSettings['NOTIFY_WEBHOOK_METHOD'] = DomHelpers.getValue('notifyWebhookMethod') || 'POST';
                envSettings['NOTIFY_WEBHOOK_HEADERS'] = (DomHelpers.getValue('notifyWebhookHeaders') || '').trim();
                envSettings['NOTIFY_WEBHOOK_PAYLOAD'] = (DomHelpers.getValue('notifyWebhookPayload') || '').trim();
                const onSuccess = DomHelpers.getElement('notifyOnSuccess');
                const onFailure = DomHelpers.getElement('notifyOnFailure');
                const onInterruption = DomHelpers.getElement('notifyOnInterruption');
                envSettings['NOTIFY_ON_SUCCESS'] = (onSuccess && onSuccess.checked) ? 'true' : 'false';
                envSettings['NOTIFY_ON_FAILURE'] = (onFailure && onFailure.checked) ? 'true' : 'false';
                envSettings['NOTIFY_ON_INTERRUPTION'] = (onInterruption && onInterruption.checked) ? 'true' : 'false';
                const timeoutRaw = DomHelpers.getValue('notifyTimeoutSeconds');
                const timeoutNum = parseInt(timeoutRaw, 10);
                envSettings['NOTIFY_TIMEOUT_SECONDS'] = Number.isFinite(timeoutNum) && timeoutNum > 0 ? String(timeoutNum) : '5';
            }

            // Also save provider and model as defaults
            envSettings['LLM_PROVIDER'] = provider;
            const model = DomHelpers.getValue('model');
            if (model) {
                // Save to provider-specific model variable
                if (provider === 'openrouter') {
                    envSettings['OPENROUTER_MODEL'] = model;
                } else if (provider === 'gemini') {
                    envSettings['GEMINI_MODEL'] = model;
                } else if (provider === 'mistral') {
                    envSettings['MISTRAL_MODEL'] = model;
                } else if (provider === 'deepseek') {
                    envSettings['DEEPSEEK_MODEL'] = model;
                } else if (provider === 'poe') {
                    envSettings['POE_MODEL'] = model;
                } else if (provider === 'nim') {
                    envSettings['NIM_MODEL'] = model;
                } else {
                    // Ollama and OpenAI use DEFAULT_MODEL
                    envSettings['DEFAULT_MODEL'] = model;
                }
            }

            // Languages are no longer saved to .env - they are:
            // - Source: auto-detected from file content
            // - Target: auto-detected from browser language per session

            if (Object.keys(envSettings).length > 0) {
                try {
                    const result = await ApiClient.saveSettings(envSettings);
                    // Reset the lock since user explicitly saved their choice
                    this.resetEnvModelApplied();
                    this._clearEnvDirty();
                    return { success: true, savedToEnv: result.saved_keys };
                } catch (e) {
                    return { success: false, error: e.message };
                }
            }
        }

        return { success: true, savedToEnv: [] };
    },

    /**
     * Apply pending model selection after models are loaded
     * Called by provider-manager after loading models
     */
    applyPendingModelSelection() {
        // Don't apply localStorage preference if .env model was already applied
        if (envModelApplied) {
            delete window.__pendingModelSelection;
            return;
        }

        if (window.__pendingModelSelection) {
            const modelSelect = DomHelpers.getElement('model');
            if (modelSelect && modelSelect.options.length > 0) {
                // Check if the model exists in options
                let found = false;
                for (let option of modelSelect.options) {
                    if (option.value === window.__pendingModelSelection) {
                        modelSelect.value = window.__pendingModelSelection;
                        found = true;
                        break;
                    }
                }
                if (found) {
                    delete window.__pendingModelSelection;
                }
            }
        }
    },

    /**
     * Apply pending custom instruction selection after custom instructions are loaded
     * Called when 'customInstructionsLoaded' event is fired
     */
    applyPendingCustomInstructionSelection() {
        if (window.__pendingCustomInstructionSelection) {
            const select = DomHelpers.getElement('customInstructionSelect');
            if (select && select.options.length > 0) {
                // Check if the value exists in options
                let found = false;
                for (let option of select.options) {
                    if (option.value === window.__pendingCustomInstructionSelection) {
                        select.value = window.__pendingCustomInstructionSelection;
                        found = true;
                        console.log('[SettingsManager] Restored custom instruction:', window.__pendingCustomInstructionSelection);
                        break;
                    }
                }
                if (!found) {
                    console.warn('[SettingsManager] Custom instruction not found:', window.__pendingCustomInstructionSelection);
                }
                delete window.__pendingCustomInstructionSelection;
            }
        }
    },

    /**
     * Apply pending novel context selection after novel contexts are loaded
     * Called when 'novelContextsLoaded' event is fired
     */
    applyPendingNovelContextSelection() {
        if (window.__pendingNovelContextSelection) {
            const select = DomHelpers.getElement('novelContextSelect');
            if (select && select.options.length > 0) {
                let found = false;
                for (let option of select.options) {
                    if (option.value === window.__pendingNovelContextSelection) {
                        select.value = window.__pendingNovelContextSelection;
                        found = true;
                        console.log('[SettingsManager] Restored novel context:', window.__pendingNovelContextSelection);
                        break;
                    }
                }
                if (!found) {
                    console.warn('[SettingsManager] Novel context not found:', window.__pendingNovelContextSelection);
                }
                delete window.__pendingNovelContextSelection;
            }
        }
    },

    /**
     * Mark that the .env default model has been applied
     * This prevents localStorage from overriding it
     */
    markEnvModelApplied() {
        envModelApplied = true;
    },

    /**
     * Reset the envModelApplied flag
     * Called after user explicitly saves settings to .env
     */
    resetEnvModelApplied() {
        envModelApplied = false;
    },

    /**
     * Check if .env model was already applied
     * @returns {boolean}
     */
    isEnvModelApplied() {
        return envModelApplied;
    },

    /**
     * Mark an endpoint as customized by the user
     * @param {string} endpointType - 'ollama' or 'openai'
     */
    markEndpointCustomized(endpointType) {
        this.updateEndpointBadge(endpointType, true);
    },

    /**
     * Check if an endpoint was customized by the user
     * @param {string} endpointType - 'ollama' or 'openai'
     * @returns {boolean}
     */
    isEndpointCustomized(endpointType) {
        const badgeId = endpointType === 'openai' ? 'openaiEndpointBadge' : 'apiEndpointBadge';
        return DomHelpers.getElement(badgeId)?.style.display !== 'none';
    },

    /**
     * Reset endpoint to server default (.env value)
     * @param {string} endpointType - 'ollama' or 'openai'
     * @param {string} serverValue - The value from server config
     */
    resetEndpointToServerDefault(endpointType, serverValue) {
        const inputId = endpointType === 'openai' ? 'openaiEndpoint' : 'apiEndpoint';
        // Update input field
        DomHelpers.setValue(inputId, serverValue);
        
        // Update badge
        this.updateEndpointBadge(endpointType, false);
        
        // Reload models with new endpoint
        const currentProvider = DomHelpers.getValue('llmProvider');
        if (currentProvider === endpointType || (endpointType === 'ollama' && currentProvider === 'ollama')) {
            window.dispatchEvent(new Event('endpointReset'));
        }
        
        MessageLogger.addLog(`↺ ${t('common:endpoint_reset_log')}`);
    },

    /**
     * Update the visual badge for endpoint customization
     * @param {string} endpointType - 'ollama' or 'openai'
     * @param {boolean} isCustomized - Whether the endpoint is customized
     */
    updateEndpointBadge(endpointType, isCustomized) {
        const badgeId = endpointType === 'openai' ? 'openaiEndpointBadge' : 'apiEndpointBadge';
        const badge = DomHelpers.getElement(badgeId);
        if (badge) {
            badge.style.display = isCustomized ? 'inline-block' : 'none';
        }
        
        // Also show/hide the reset button
        const resetBtnId = endpointType === 'openai' ? 'resetOpenaiEndpointBtn' : 'resetApiEndpointBtn';
        const resetBtn = DomHelpers.getElement(resetBtnId);
        if (resetBtn) {
            resetBtn.style.display = isCustomized ? 'inline-flex' : 'none';
        }
    },

    /**
     * Initialize endpoint badges on page load
     * Call this after server config is loaded
     * @param {Object} serverConfig - The config from /api/config
     */
    initializeEndpointBadges(serverConfig) {
        void serverConfig;
        this.updateEndpointBadge('ollama', false);
        this.updateEndpointBadge('openai', false);
    }
};

// Auto-save preferences when leaving page
if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', () => {
        SettingsManager.saveCurrentState();
    });
}
