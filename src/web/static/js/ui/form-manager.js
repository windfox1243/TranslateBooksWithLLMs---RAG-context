/**
 * Form Manager - Form configuration and settings management
 *
 * Handles form state, custom language toggles, advanced settings,
 * default configuration loading, and form reset functionality.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from './dom-helpers.js';
import { MessageLogger } from './message-logger.js';
import { ApiKeyUtils } from '../utils/api-key-utils.js';
import { TranslationTracker } from '../translation/translation-tracker.js';
import { SettingsManager } from '../core/settings-manager.js';
import { ProviderManager } from '../providers/provider-manager.js';
import { EditorModelManager } from '../providers/editor-model-manager.js';
import { GlossaryManager } from '../glossary/glossary-manager.js';
import { t } from '../i18n/i18n.js';

/**
 * Set default language in select/input
 * @param {string} selectId - Select element ID
 * @param {string} customInputId - Custom input element ID
 * @param {string} defaultLanguage - Default language value
 * @param {boolean} [forceOverwrite=false] - If true, overwrite even if "Other" is selected with a value
 */
function setDefaultLanguage(selectId, customInputId, defaultLanguage, forceOverwrite = false) {
    const select = DomHelpers.getElement(selectId);
    const customInput = DomHelpers.getElement(customInputId);
    const containerId = customInputId + 'Container';
    const container = DomHelpers.getElement(containerId);

    if (!select || !customInput) return;

    // Don't overwrite if "Other" is already selected with a custom value (restored from file)
    // This preserves custom languages across page reloads
    if (!forceOverwrite && select.value === 'Other' && customInput.value.trim()) {
        // Keep the existing "Other" selection and show the container
        if (container) container.style.display = 'block';
        return;
    }

    // Check if the default language is in the dropdown options (excluding "Other")
    let languageFound = false;
    for (let option of select.options) {
        // Skip "Other" option - we only want to match actual language values
        if (option.value === 'Other') continue;

        if (option.value.toLowerCase() === defaultLanguage.toLowerCase()) {
            select.value = option.value;
            languageFound = true;
            if (container) container.style.display = 'none';
            break;
        }
    }

    // If language not found in dropdown, use "Other" and set custom input
    if (!languageFound) {
        select.value = 'Other';
        customInput.value = defaultLanguage;
        // Show the container (not just the input)
        if (container) container.style.display = 'block';
    }
}

export const FormManager = {
    /**
     * Initialize form manager
     */
    initialize() {
        this.setupEventListeners();
        this.defaultConfigPromise = this.loadDefaultConfig();
        this.loadCustomInstructions();
        this.loadNovelContexts();
        this.loadProfiles();
        return this.defaultConfigPromise;
    },

    /**
     * Set up event listeners for form elements
     */
    setupEventListeners() {
        // Source language change
        const sourceLang = DomHelpers.getElement('sourceLang');
        if (sourceLang) {
            sourceLang.addEventListener('change', (e) => {
                this.checkCustomSourceLanguage(e.target);
            });
        }

        // Target language change
        const targetLang = DomHelpers.getElement('targetLang');
        if (targetLang) {
            targetLang.addEventListener('change', (e) => {
                this.checkCustomTargetLanguage(e.target);
            });
        }

        // TTS enabled checkbox
        const ttsEnabled = DomHelpers.getElement('ttsEnabled');
        if (ttsEnabled) {
            ttsEnabled.addEventListener('change', (e) => {
                this.handleTtsToggle(e.target.checked);
            });
        }

        // Prompt options checkboxes - keep section open if any is checked
        const textCleanup = DomHelpers.getElement('textCleanup');
        const bilingualMode = DomHelpers.getElement('bilingualMode');
        const customInstructionSelect = DomHelpers.getElement('customInstructionSelect');
        const plainTextMode = DomHelpers.getElement('plainTextMode');
        const chapterMode = DomHelpers.getElement('chapterMode');

        [textCleanup, bilingualMode, plainTextMode, chapterMode].forEach(checkbox => {
            if (checkbox) {
                checkbox.addEventListener('change', () => {
                    this.handlePromptOptionChange();
                });
            }
        });

        // Custom instruction select - keep section open if a file is selected
        if (customInstructionSelect) {
            customInstructionSelect.addEventListener('change', () => {
                this.handlePromptOptionChange();
            });
        }

        // Custom instructions refresh button
        const refreshCustomInstructionsBtn = DomHelpers.getElement('refreshCustomInstructionsBtn');
        if (refreshCustomInstructionsBtn) {
            refreshCustomInstructionsBtn.addEventListener('click', () => {
                this.loadCustomInstructions();
            });
        }

        // Custom instructions open folder button
        const openCustomInstructionsFolderBtn = DomHelpers.getElement('openCustomInstructionsFolderBtn');
        if (openCustomInstructionsFolderBtn) {
            openCustomInstructionsFolderBtn.addEventListener('click', () => {
                this.openCustomInstructionsFolder();
            });
        }

        // Novel context select - keep section open if a file is selected
        const novelContextSelect = DomHelpers.getElement('novelContextSelect');
        if (novelContextSelect) {
            novelContextSelect.addEventListener('change', () => {
                this.handlePromptOptionChange();
            });
        }

        // Novel contexts refresh button
        const refreshNovelContextsBtn = DomHelpers.getElement('refreshNovelContextsBtn');
        if (refreshNovelContextsBtn) {
            refreshNovelContextsBtn.addEventListener('click', () => {
                this.loadNovelContexts();
            });
        }

        // Novel contexts open folder button
        const openNovelContextsFolderBtn = DomHelpers.getElement('openNovelContextsFolderBtn');
        if (openNovelContextsFolderBtn) {
            openNovelContextsFolderBtn.addEventListener('click', () => {
                this.openNovelContextsFolder();
            });
        }

        // Auto-update context checkbox changes
        const autoUpdateContext = DomHelpers.getElement('autoUpdateContext');
        if (autoUpdateContext) {
            autoUpdateContext.addEventListener('change', () => {
                this.handlePromptOptionChange();
            });
        }

        // Reset button
        const resetBtn = DomHelpers.getElement('resetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.resetForm();
            });
        }

        // Endpoint change listeners - detect manual modifications
        const apiEndpoint = DomHelpers.getElement('apiEndpoint');
        if (apiEndpoint) {
            apiEndpoint.addEventListener('change', () => {
                SettingsManager.markEndpointCustomized('ollama');
                console.log('[FormManager] Ollama endpoint customized by user');
            });
        }

        const openaiEndpoint = DomHelpers.getElement('openaiEndpoint');
        if (openaiEndpoint) {
            openaiEndpoint.addEventListener('change', () => {
                SettingsManager.markEndpointCustomized('openai');
                console.log('[FormManager] OpenAI endpoint customized by user');
            });
        }

        // Reset endpoint to server default buttons
        const resetApiEndpointBtn = DomHelpers.getElement('resetApiEndpointBtn');
        if (resetApiEndpointBtn) {
            resetApiEndpointBtn.addEventListener('click', () => {
                const config = StateManager.getState('ui.defaultConfig');
                const serverEndpoint = config?.ollama_api_endpoint || config?.api_endpoint;
                if (serverEndpoint) {
                    SettingsManager.resetEndpointToServerDefault('ollama', serverEndpoint);
                }
            });
        }

        const resetOpenaiEndpointBtn = DomHelpers.getElement('resetOpenaiEndpointBtn');
        if (resetOpenaiEndpointBtn) {
            resetOpenaiEndpointBtn.addEventListener('click', () => {
                const config = StateManager.getState('ui.defaultConfig');
                const serverEndpoint = config?.openai_api_endpoint;
                if (serverEndpoint) {
                    SettingsManager.resetEndpointToServerDefault('openai', serverEndpoint);
                }
            });
        }
    },

    /**
     * Check if custom source language input should be shown
     * @param {HTMLSelectElement} selectElement - Source language select element
     */
    checkCustomSourceLanguage(selectElement) {
        const container = DomHelpers.getElement('customSourceLangContainer');
        const customLangInput = DomHelpers.getElement('customSourceLang');
        if (!container || !customLangInput) return;

        if (selectElement.value === 'Other') {
            container.style.display = 'block';
            customLangInput.focus();
        } else {
            container.style.display = 'none';
        }
    },

    /**
     * Check if custom target language input should be shown
     * @param {HTMLSelectElement} selectElement - Target language select element
     */
    checkCustomTargetLanguage(selectElement) {
        const container = DomHelpers.getElement('customTargetLangContainer');
        const customLangInput = DomHelpers.getElement('customTargetLang');
        if (!container || !customLangInput) return;

        if (selectElement.value === 'Other') {
            container.style.display = 'block';
            customLangInput.focus();
        } else {
            container.style.display = 'none';
        }
    },


    /**
     * Toggle settings options panel
     */
    toggleSettingsOptions() {
        const section = DomHelpers.getElement('settingsOptionsSection');
        const icon = DomHelpers.getElement('settingsOptionsIcon');

        if (!section || !icon) return;

        const isHidden = section.classList.toggle('hidden');
        icon.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(180deg)';

        // Update state
        StateManager.setState('ui.isSettingsOptionsOpen', !isHidden);
    },

    /**
     * Toggle prompt options panel
     */
    togglePromptOptions() {
        const section = DomHelpers.getElement('promptOptionsSection');
        const icon = DomHelpers.getElement('promptOptionsIcon');

        if (!section || !icon) return;

        const isHidden = section.classList.toggle('hidden');
        icon.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(180deg)';

        // Update state
        StateManager.setState('ui.isPromptOptionsOpen', !isHidden);
    },

    /**
     * Toggle activity log panel
     */
    toggleActivityLog() {
        const section = DomHelpers.getElement('activityLogSection');
        const icon = DomHelpers.getElement('activityLogIcon');

        if (!section || !icon) return;

        const isHidden = section.classList.toggle('hidden');
        icon.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(180deg)';

        // Update state
        StateManager.setState('ui.isActivityLogOpen', !isHidden);
    },

    /**
     * Handle prompt option checkbox change - keep section open if any option is active
     */
    handlePromptOptionChange() {
        const textCleanup = DomHelpers.getElement('textCleanup');
        const bilingualMode = DomHelpers.getElement('bilingualMode');
        const plainTextMode = DomHelpers.getElement('plainTextMode');
        const chapterMode = DomHelpers.getElement('chapterMode');
        const customInstructionSelect = DomHelpers.getElement('customInstructionSelect');
        const novelContextSelect = DomHelpers.getElement('novelContextSelect');
        const autoUpdateContext = DomHelpers.getElement('autoUpdateContext');

        const anyActive = (
            textCleanup?.checked ||
            bilingualMode?.checked ||
            plainTextMode?.checked ||
            chapterMode?.checked ||
            (customInstructionSelect?.value && customInstructionSelect.value !== '') ||
            (novelContextSelect?.value && novelContextSelect.value !== '') ||
            autoUpdateContext?.checked
        );

        if (anyActive) {
            const section = DomHelpers.getElement('promptOptionsSection');
            const icon = DomHelpers.getElement('promptOptionsIcon');

            if (section && section.classList.contains('hidden')) {
                section.classList.remove('hidden');
                if (icon) {
                    icon.style.transform = 'rotate(180deg)';
                }
                StateManager.setState('ui.isPromptOptionsOpen', true);
            }
        }
    },

    /**
     * Handle TTS toggle
     * @param {boolean} isChecked - Whether TTS is enabled
     */
    handleTtsToggle(isChecked) {
        const ttsOptions = DomHelpers.getElement('ttsOptions');

        if (ttsOptions) {
            if (isChecked) {
                ttsOptions.style.display = 'block';
            } else {
                ttsOptions.style.display = 'none';
            }
        }

        // Dispatch event for other components
        window.dispatchEvent(new CustomEvent('ttsChanged', { detail: { enabled: isChecked } }));
    },

    /**
     * Detect browser language and map to full language name
     * @returns {string} Full language name (e.g., "French", "English")
     */
    detectBrowserLanguage() {
        // Get browser language (e.g., "fr-FR", "en-US", "zh-CN")
        const browserLang = navigator.language || navigator.userLanguage || 'en';
        const langCode = browserLang.split('-')[0].toLowerCase();

        // Map language codes to full names used in the UI
        const languageMap = {
            'en': 'English',
            'zh': 'Chinese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'ja': 'Japanese',
            'ko': 'Korean',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ar': 'Arabic',
            'it': 'Italian',
            'nl': 'Dutch',
            'pl': 'Polish',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'da': 'Danish',
            'fi': 'Finnish',
            'el': 'Greek',
            'hu': 'Hungarian',
            'cs': 'Czech',
            'sk': 'Slovak',
            'ro': 'Romanian',
            'bg': 'Bulgarian',
            'hr': 'Croatian',
            'sr': 'Serbian',
            'uk': 'Ukrainian',
            'ca': 'Catalan',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'ur': 'Urdu',
            'pa': 'Punjabi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi',
            'gu': 'Gujarati',
            'vi': 'Vietnamese',
            'th': 'Thai',
            'id': 'Indonesian',
            'ms': 'Malay',
            'tl': 'Tagalog',
            'my': 'Burmese',
            'fa': 'Persian',
            'tr': 'Turkish',
            'he': 'Hebrew',
            'sw': 'Swahili',
            'am': 'Amharic'
        };

        return languageMap[langCode] || 'English'; // Default to English if not found
    },

    /**
     * Load default configuration from server
     */
    async loadDefaultConfig() {
        try {
            const config = await ApiClient.getConfig();

            // Store config in state first so other modules can access it
            StateManager.setState('ui.defaultConfig', config);

            // Provider/model defaults saved to .env are authoritative. Apply
            // the provider before ProviderManager receives the loaded event so
            // it lists models and reveals the matching configured-key field.
            if (config.llm_provider) {
                DomHelpers.setValue('llmProvider', config.llm_provider);
            }
            if (config.default_model) {
                window.__pendingModelSelection = config.default_model;
            }

            // Set target language from server config if available
            // This fixes GitHub issue #108: DEFAULT_TARGET_LANGUAGE was ignored
            // Only override if server has a default target language configured
            if (config.default_target_language && config.default_target_language.trim()) {
                console.log('[FormManager] Applying DEFAULT_TARGET_LANGUAGE from server:', config.default_target_language);
                setDefaultLanguage('targetLang', 'customTargetLang', config.default_target_language);
            } else {
                console.log('[FormManager] No DEFAULT_TARGET_LANGUAGE from server, keeping current value');
            }

            // Set source language from server config if available
            const sourceLanguage = config.default_source_language && config.default_source_language.trim()
                ? config.default_source_language
                : '';  // Empty = auto-detect from file
            setDefaultLanguage('sourceLang', 'customSourceLang', sourceLanguage)

            // .env-backed endpoints are authoritative at startup. Edits remain
            // usable for the current session and become durable only after Save.
            const ollamaEndpoint = config.ollama_api_endpoint || config.api_endpoint;
            if (ollamaEndpoint) {
                DomHelpers.setValue('apiEndpoint', ollamaEndpoint);
                SettingsManager.updateEndpointBadge('ollama', false);
            }
            
            // OpenAI endpoint (for OpenAI-compatible providers like OpenAI, LM Studio)
            if (config.openai_api_endpoint) {
                DomHelpers.setValue('openaiEndpoint', config.openai_api_endpoint);
                SettingsManager.updateEndpointBadge('openai', false);
            }
            
            // Output filename pattern (naming convention)
            if (config.output_filename_pattern) {
                DomHelpers.setValue('outputFilenamePattern', config.output_filename_pattern);
            }

            // Disable auto-pause on rate limit (runtime behavior default)
            if (typeof config.disable_auto_pause === 'boolean') {
                const disableAutoPauseCheckbox = DomHelpers.getElement('disableAutoPause');
                if (disableAutoPauseCheckbox) {
                    disableAutoPauseCheckbox.checked = config.disable_auto_pause;
                }
            }

            // Bypass context validation (runtime behavior default)
            if (typeof config.bypass_context_gating === 'boolean') {
                const bypassContextGatingCheckbox = DomHelpers.getElement('bypassContextGating');
                if (bypassContextGatingCheckbox) {
                    bypassContextGatingCheckbox.checked = config.bypass_context_gating;
                }
            }

            // Enable 2-pass chunk reflection (runtime behavior default)
            if (typeof config.enable_chunk_reflection === 'boolean') {
                const enableReflectionCheckbox = DomHelpers.getElement('enableReflection');
                if (enableReflectionCheckbox) {
                    enableReflectionCheckbox.checked = config.enable_chunk_reflection;
                }
            }

            // Parallel requests default (seeds the input; per-job request overrides it)
            if (config.parallel_translations) {
                const parallelWorkersInput = DomHelpers.getElement('parallelWorkers');
                if (parallelWorkersInput) {
                    parallelWorkersInput.value = String(config.parallel_translations);
                }
                if (config.max_parallel_translations) {
                    const pwEl = DomHelpers.getElement('parallelWorkers');
                    if (pwEl) pwEl.max = String(config.max_parallel_translations);
                }
            }

            // Webhook notifications — populate fields from .env
            if ('notify_webhook_url' in config) {
                DomHelpers.setValue('notifyWebhookUrl', config.notify_webhook_url || '');
            }
            if (config.notify_webhook_method) {
                DomHelpers.setValue('notifyWebhookMethod', config.notify_webhook_method);
            }
            if ('notify_webhook_headers' in config) {
                DomHelpers.setValue('notifyWebhookHeaders', config.notify_webhook_headers || '');
            }
            if ('notify_webhook_payload' in config) {
                DomHelpers.setValue('notifyWebhookPayload', config.notify_webhook_payload || '');
            }
            const setCheckbox = (id, value) => {
                if (typeof value === 'boolean') {
                    const cb = DomHelpers.getElement(id);
                    if (cb) cb.checked = value;
                }
            };
            setCheckbox('notifyOnSuccess', config.notify_on_success);
            setCheckbox('notifyOnFailure', config.notify_on_failure);
            setCheckbox('notifyOnInterruption', config.notify_on_interruption);
            if (typeof config.notify_timeout_seconds === 'number') {
                DomHelpers.setValue('notifyTimeoutSeconds', String(config.notify_timeout_seconds));
            }
            const notifyBadge = DomHelpers.getElement('notifyStatusBadge');
            if (notifyBadge) {
                notifyBadge.style.display = config.notify_configured ? 'inline-block' : 'none';
            }

            // Handle API keys - show indicator if configured in .env, otherwise keep placeholder
            ApiKeyUtils.setupField('geminiApiKey', config.gemini_api_key_configured, config.gemini_api_key, config.gemini_api_key_count);
            ApiKeyUtils.setupField('openaiApiKey', config.openai_api_key_configured, config.openai_api_key, config.openai_api_key_count);
            ApiKeyUtils.setupField('openrouterApiKey', config.openrouter_api_key_configured, config.openrouter_api_key, config.openrouter_api_key_count);
            ApiKeyUtils.setupField('mistralApiKey', config.mistral_api_key_configured, config.mistral_api_key, config.mistral_api_key_count);
            ApiKeyUtils.setupField('deepseekApiKey', config.deepseek_api_key_configured, config.deepseek_api_key, config.deepseek_api_key_count);
            ApiKeyUtils.setupField('poeApiKey', config.poe_api_key_configured, config.poe_api_key, config.poe_api_key_count);
            ApiKeyUtils.setupField('nimApiKey', config.nim_api_key_configured, config.nim_api_key, config.nim_api_key_count);

            // After loading defaults, dispatch event to notify other modules
            console.log('[FormManager] Default config loaded, dispatching event');
            window.dispatchEvent(new CustomEvent('defaultConfigLoaded'));

        } catch (error) {
            console.error('[FormManager] Failed to load default configuration:', error);
            MessageLogger.showMessage(t('settings:default_config_load_failed'), 'warning');
            // Still dispatch event even on error so other modules aren't blocked
            console.log('[FormManager] Dispatching defaultConfigLoaded event despite error');
            window.dispatchEvent(new CustomEvent('defaultConfigLoaded'));
        }
    },

    /**
     * Load available custom instruction files
     */
    async loadCustomInstructions() {
        try {
            console.log('[CustomInstructions] Loading custom instructions...');
            const data = await ApiClient.getCustomInstructions();
            console.log('[CustomInstructions] Data received:', data);

            const select = DomHelpers.getElement('customInstructionSelect');
            if (!select) {
                console.warn('[CustomInstructions] Select element not found!');
                return;
            }

            // Save current value before resetting dropdown
            const currentValue = select.value;

            // Reset dropdown to default
            select.innerHTML = `<option value="">${t('settings:select_none')}</option>`;

            // Populate dropdown with available files
            if (data.files && data.files.length > 0) {
                console.log('[CustomInstructions] Adding', data.files.length, 'files to dropdown');
                data.files.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file.filename;
                    option.textContent = file.display_name;
                    select.appendChild(option);
                    console.log('[CustomInstructions] Added option:', file.display_name);
                });
            } else {
                console.warn('[CustomInstructions] No files found in response');
            }

            // Restore previously selected value if it still exists in the list
            if (currentValue) {
                // Check if the value exists in options
                let found = false;
                for (let option of select.options) {
                    if (option.value === currentValue) {
                        select.value = currentValue;
                        found = true;
                        console.log('[CustomInstructions] Restored selection:', currentValue);
                        break;
                    }
                }
                if (!found) {
                    console.warn('[CustomInstructions] Previously selected file not found:', currentValue);
                }
            }

            // Dispatch event to notify that custom instructions are loaded
            // This allows SettingsManager to restore the saved value
            window.dispatchEvent(new CustomEvent('customInstructionsLoaded'));
        } catch (error) {
            console.error('[CustomInstructions] Error loading custom instructions:', error);
            // Graceful degradation - dropdown will only show "None" option
            // Still dispatch event even on error
            window.dispatchEvent(new CustomEvent('customInstructionsLoaded'));
        }
    },

    /**
     * Open the Custom_Instructions folder in the system file explorer
     */
    async openCustomInstructionsFolder() {
        try {
            const response = await ApiClient.openCustomInstructionsFolder();
            if (!response.success) {
                console.error('[CustomInstructions] Failed to open folder:', response.error);
                MessageLogger.addLog(t('settings:custom_instructions_load_failed'));
            }
        } catch (error) {
            console.error('[CustomInstructions] Error opening folder:', error);
            MessageLogger.addLog(t('settings:custom_instructions_load_failed'));
        }
    },

    /**
     * Load list of available novel context files
     */
    async loadNovelContexts() {
        try {
            console.log('[NovelContexts] Loading novel contexts...');
            const data = await ApiClient.getNovelContexts();
            console.log('[NovelContexts] Data received:', data);

            const select = DomHelpers.getElement('novelContextSelect');
            if (!select) {
                console.warn('[NovelContexts] Select element not found!');
                return;
            }

            const currentValue = select.value;
            select.innerHTML = `<option value="">${t('settings:select_none')}</option>`;

            if (data.files && data.files.length > 0) {
                console.log('[NovelContexts] Adding', data.files.length, 'files to dropdown');
                data.files.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file.filename;
                    option.textContent = file.display_name;
                    select.appendChild(option);
                    console.log('[NovelContexts] Added option:', file.display_name);
                });
            } else {
                console.warn('[NovelContexts] No files found in response');
            }

            if (currentValue) {
                let found = false;
                for (let option of select.options) {
                    if (option.value === currentValue) {
                        select.value = currentValue;
                        found = true;
                        console.log('[NovelContexts] Restored selection:', currentValue);
                        break;
                    }
                }
                if (!found) {
                    console.warn('[NovelContexts] Previously selected file not found:', currentValue);
                }
            }

            window.dispatchEvent(new CustomEvent('novelContextsLoaded'));
        } catch (error) {
            console.error('[NovelContexts] Error loading novel contexts:', error);
            window.dispatchEvent(new CustomEvent('novelContextsLoaded'));
        }
    },

    /**
     * Open the Novel_Contexts folder in the system file explorer
     */
    async openNovelContextsFolder() {
        try {
            const response = await ApiClient.openNovelContextsFolder();
            if (!response.success) {
                console.error('[NovelContexts] Failed to open folder:', response.error);
                MessageLogger.addLog(t('translation:context_open_folder_failed'));
            }
        } catch (error) {
            console.error('[NovelContexts] Error opening folder:', error);
            MessageLogger.addLog(t('translation:context_open_folder_failed'));
        }
    },

    /**
     * Reset form to default state
     */
    async resetForm() {
        // Get current files to process
        const filesToProcess = StateManager.getState('files.toProcess');
        const currentJob = StateManager.getState('translation.currentJob');
        const isBatchActive = StateManager.getState('translation.isBatchActive');

        // First, interrupt current translation if active
        if (currentJob && currentJob.translationId && isBatchActive) {
            MessageLogger.addLog(t('translation:interrupt_before_clear_log'));
            try {
                await ApiClient.interruptTranslation(currentJob.translationId);
            } catch {
                // Interrupt failed
            }
        }

        // Collect file paths to delete from server
        const uploadedFilePaths = filesToProcess
            .filter(file => file.filePath)
            .map(file => file.filePath);

        // Clear client-side state
        StateManager.setState('files.toProcess', []);
        StateManager.setState('translation.currentJob', null);
        StateManager.setState('translation.isBatchActive', false);

        // Clear saved translation state from localStorage
        if (TranslationTracker && TranslationTracker.clearTranslationState) {
            TranslationTracker.clearTranslationState();
        }

        // Reset file input
        DomHelpers.setValue('fileInput', '');

        // Hide progress section
        DomHelpers.hide('progressSection');

        // Reset buttons
        DomHelpers.setText('translateBtn', t('translation:start_batch_with_icon'));
        DomHelpers.setDisabled('translateBtn', true);
        DomHelpers.hide('interruptBtn');
        DomHelpers.setDisabled('interruptBtn', false);

        // Reset language selectors
        const sourceContainer = DomHelpers.getElement('customSourceLangContainer');
        const targetContainer = DomHelpers.getElement('customTargetLangContainer');
        if (sourceContainer) sourceContainer.style.display = 'none';
        if (targetContainer) targetContainer.style.display = 'none';
        DomHelpers.getElement('sourceLang').selectedIndex = 0;
        DomHelpers.getElement('targetLang').selectedIndex = 0;

        // Reset stats and progress
        DomHelpers.show('statsGrid');
        this.updateProgress(0);
        MessageLogger.showMessage('', '');

        // Delete uploaded files from server
        if (uploadedFilePaths.length > 0) {
            MessageLogger.addLog(t('translation:delete_uploaded_log', { count: uploadedFilePaths.length }));
            try {
                const result = await ApiClient.clearUploads(uploadedFilePaths);

                MessageLogger.addLog(t('translation:delete_uploaded_success_log', { count: result.total_deleted }));
                if (result.failed && result.failed.length > 0) {
                    MessageLogger.addLog(t('translation:delete_uploaded_failed_log', { count: result.failed.length }));
                }
            } catch {
                MessageLogger.addLog(t('translation:delete_uploaded_error_log'));
            }
        }

        MessageLogger.addLog(t('translation:form_reset_log'));

        // Trigger UI update
        window.dispatchEvent(new CustomEvent('formReset'));
    },

    /**
     * Update progress bar
     * @param {number} percent - Progress percentage (0-100)
     */
    updateProgress(percent) {
        const progressBar = DomHelpers.getElement('progressBar');
        if (!progressBar) return;

        progressBar.style.width = percent + '%';
        DomHelpers.setText(progressBar, Math.round(percent) + '%');
    },

    /**
     * Get form configuration for translation
     * @returns {Object} Translation configuration object
     */
    getTranslationConfig() {
        // Get source language
        let sourceLanguageVal = DomHelpers.getValue('sourceLang');
        if (sourceLanguageVal === 'Other') {
            sourceLanguageVal = DomHelpers.getValue('customSourceLang').trim();
        }

        // Get target language
        let targetLanguageVal = DomHelpers.getValue('targetLang');
        if (targetLanguageVal === 'Other') {
            targetLanguageVal = DomHelpers.getValue('customTargetLang').trim();
        }

        // Get provider and model
        const provider = DomHelpers.getValue('llmProvider');
        const model = DomHelpers.getValue('model');

        // Get API endpoint based on provider
        let apiEndpoint;
        if (provider === 'openai') {
            apiEndpoint = DomHelpers.getValue('openaiEndpoint');
        } else {
            apiEndpoint = DomHelpers.getValue('apiEndpoint');
        }

        // Get API keys - use helper to handle .env configured keys
        const geminiApiKey = provider === 'gemini' ? ApiKeyUtils.getValue('geminiApiKey') : '';
        const openaiApiKey = provider === 'openai' ? ApiKeyUtils.getValue('openaiApiKey') : '';
        const openrouterApiKey = provider === 'openrouter' ? ApiKeyUtils.getValue('openrouterApiKey') : '';
        const mistralApiKey = provider === 'mistral' ? ApiKeyUtils.getValue('mistralApiKey') : '';
        const deepseekApiKey = provider === 'deepseek' ? ApiKeyUtils.getValue('deepseekApiKey') : '';
        const poeApiKey = provider === 'poe' ? ApiKeyUtils.getValue('poeApiKey') : '';
        const nimApiKey = provider === 'nim' ? ApiKeyUtils.getValue('nimApiKey') : '';

        // Get TTS configuration
        const ttsEnabled = DomHelpers.getElement('ttsEnabled')?.checked || false;

        return {
            source_language: sourceLanguageVal,
            target_language: targetLanguageVal,
            model: model,
            llm_api_endpoint: apiEndpoint,
            llm_provider: provider,
            gemini_api_key: geminiApiKey,
            openai_api_key: openaiApiKey,
            openrouter_api_key: openrouterApiKey,
            mistral_api_key: mistralApiKey,
            deepseek_api_key: deepseekApiKey,
            poe_api_key: poeApiKey,
            nim_api_key: nimApiKey,
            // Prompt options (optional system prompt instructions)
            // Technical content protection is always enabled
            prompt_options: {
                preserve_technical_content: true,
                text_cleanup: DomHelpers.getElement('textCleanup')?.checked || false,
                refine: false,
                plain_text_mode: DomHelpers.getElement('plainTextMode')?.checked || false,
                chapter_mode: DomHelpers.getElement('chapterMode')?.checked || false,
                custom_instruction_file: DomHelpers.getValue('customInstructionSelect') || '',
                novel_context_file: DomHelpers.getValue('novelContextSelect') || '',
                auto_update_context: DomHelpers.getElement('autoUpdateContext')?.checked || false,
                bypass_context_gating: DomHelpers.getElement('bypassContextGating')?.checked || false,
                reflection_mode: DomHelpers.getElement('enableReflection')?.checked || false,
                editor_provider: DomHelpers.getElement('enableReflection')?.checked
                    ? (DomHelpers.getValue('editorProvider') || '')
                    : '',
                editor_model: DomHelpers.getElement('enableReflection')?.checked
                    ? (DomHelpers.getValue('editorModel') || '').trim()
                    : ''
            },
            // Bilingual output (original + translation interleaved)
            bilingual_output: DomHelpers.getElement('bilingualMode')?.checked || false,
            // Disable auto-pause on rate limit (auto-resume after Retry-After)
            auto_pause_on_rate_limit: !(DomHelpers.getElement('disableAutoPause')?.checked || false),
            // Parallel chunk translation (cloud only; backend gates local to 1)
            parallel_workers: provider === 'ollama'
                ? 1
                : (parseInt(DomHelpers.getValue('parallelWorkers'), 10) || 1),
            // TTS configuration
            tts_enabled: ttsEnabled,
            tts_voice: ttsEnabled ? (DomHelpers.getValue('ttsVoice') || '') : '',
            tts_rate: ttsEnabled ? (DomHelpers.getValue('ttsRate') || '+0%') : '+0%',
            tts_format: ttsEnabled ? (DomHelpers.getValue('ttsFormat') || 'opus') : 'opus',
            tts_bitrate: ttsEnabled ? (DomHelpers.getValue('ttsBitrate') || '64k') : '64k'
        };
    },

    /**
     * Validate form configuration
     * @returns {Object} { valid: boolean, message: string }
     */
    validateConfig() {
        const config = this.getTranslationConfig();

        if (!config.source_language) {
            return { valid: false, message: t('translation:validation_source_required') };
        }

        if (!config.target_language) {
            return { valid: false, message: t('translation:validation_target_required') };
        }

        if (!config.model) {
            return { valid: false, message: t('translation:validation_model_required') };
        }

        if (!config.llm_api_endpoint) {
            return { valid: false, message: t('translation:validation_endpoint_empty') };
        }

        // Validate API keys for cloud providers using shared utility
        const apiKeyValidation = ApiKeyUtils.validateForProvider(config.llm_provider, config.llm_api_endpoint);
        if (!apiKeyValidation.valid) {
            return apiKeyValidation;
        }

        return { valid: true, message: '' };
    },
    /**
     * Load translation profiles from API
     */
    async loadProfiles() {
        try {
            const profiles = await ApiClient.getProfiles();
            const select = document.getElementById('profileSelect');
            if (!select) return;

            const currentValue = select.value;
            const customOption = document.createElement('option');
            customOption.value = '';
            customOption.textContent = t('translation:profile_custom_settings');
            customOption.setAttribute('data-i18n', 'translation:profile_custom_settings');
            select.replaceChildren(customOption);

            profiles.forEach(name => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                select.appendChild(option);
            });
            if (profiles.includes(currentValue)) {
                select.value = currentValue;
            }
        } catch (e) {
            console.error('Failed to load profiles:', e);
        }
    }
};

window.loadSelectedProfile = async function() {
    const select = document.getElementById('profileSelect');
    const name = select.value;
    if (!name) return;
    
    try {
        const data = await ApiClient.getProfile(name);
        
        // Restore settings into form
        if (data.source_language) setDefaultLanguage('sourceLang', 'customSourceLang', data.source_language, true);
        if (data.target_language) setDefaultLanguage('targetLang', 'customTargetLang', data.target_language, true);
        
        if (data.glossary !== undefined) {
            await GlossaryManager.refreshDropdown();
            const glossarySelect = document.getElementById('glossarySelect');
            if (glossarySelect) {
                const hasGlossary = Array.from(glossarySelect.options)
                    .some(option => option.value === String(data.glossary));
                glossarySelect.value = hasGlossary ? String(data.glossary) : '';
                glossarySelect.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }

        // Profiles store filenames, while these dropdowns are populated by
        // independent API requests during startup. Refresh and await both
        // lists before assigning values so a fast profile response cannot
        // silently clear a valid saved selection.
        if (data.novel_context_file !== undefined) {
            await FormManager.loadNovelContexts();
            const novelContextSelect = document.getElementById('novelContextSelect');
            if (novelContextSelect) {
                const hasNovelContext = Array.from(novelContextSelect.options)
                    .some(option => option.value === data.novel_context_file);
                novelContextSelect.value = hasNovelContext
                    ? data.novel_context_file
                    : '';
            }
        }

        if (data.custom_instruction_file !== undefined) {
            await FormManager.loadCustomInstructions();
            const customInstructionSelect = document.getElementById('customInstructionSelect');
            if (customInstructionSelect) {
                const hasCustomInstruction = Array.from(customInstructionSelect.options)
                    .some(option => option.value === data.custom_instruction_file);
                customInstructionSelect.value = hasCustomInstruction
                    ? data.custom_instruction_file
                    : '';
            }
        }

        if (data.llm_api_endpoint !== undefined) {
            const endpointId = data.llm_provider === 'openai' ? 'openaiEndpoint' : 'apiEndpoint';
            DomHelpers.setValue(endpointId, data.llm_api_endpoint);
        }

        const providerEl = document.getElementById('llmProvider');
        if (providerEl && data.llm_provider) {
            providerEl.value = data.llm_provider;
            providerEl.dispatchEvent(new Event('change'));
            await ProviderManager.waitForCurrentModelLoad();
        }

        const modelEl = document.getElementById('model');
        if (modelEl && data.model) {
            const optionExists = Array.from(modelEl.options)
                .some(option => option.value === data.model);
            if (!optionExists) {
                const option = document.createElement('option');
                option.value = data.model;
                option.textContent = data.model;
                modelEl.appendChild(option);
            }
            ProviderManager.setCurrentModel(data.model);
            modelEl.dispatchEvent(new Event('change'));
        }

        const checkboxValues = {
            bilingualMode: data.bilingual_output,
            textCleanup: data.text_cleanup,
            autoUpdateContext: data.auto_update_context,
            bypassContextGating: data.bypass_context_gating,
            enableReflection: data.reflection_mode,
            plainTextMode: data.plain_text_mode,
            chapterMode: data.chapter_mode,
            disableAutoPause: data.auto_pause_on_rate_limit === undefined
                ? undefined
                : !data.auto_pause_on_rate_limit,
            ttsEnabled: data.tts_enabled
        };
        Object.entries(checkboxValues).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element && value !== undefined) element.checked = value;
        });
        if (data.editor_provider !== undefined || data.editor_model !== undefined) {
            EditorModelManager.setSelection(
                data.editor_provider || '',
                data.editor_model || '',
            );
        }

        const valueFields = {
            parallelWorkers: data.parallel_workers,
            ttsVoice: data.tts_voice,
            ttsRate: data.tts_rate,
            ttsFormat: data.tts_format,
            ttsBitrate: data.tts_bitrate,
            outputFilenamePattern: data.output_filename_pattern
        };
        Object.entries(valueFields).forEach(([id, value]) => {
            if (value !== undefined) DomHelpers.setValue(id, String(value));
        });

        if (data.tts_enabled !== undefined) {
            FormManager.handleTtsToggle(data.tts_enabled);
        }

        // Open the prompt options section if any of the prompt options are enabled
        FormManager.handlePromptOptionChange();
        
        MessageLogger.addLog(t('translation:profile_loaded_log', { name }));
    } catch (e) {
        console.error("Failed to load profile:", e);
        MessageLogger.addLog(t('translation:profile_load_failed_log', { error: e.message }));
    }
};

window.promptSaveProfile = async function() {
    const name = prompt(t('translation:profile_enter_name'));
    if (!name) return;
    
    const formData = FormManager.getTranslationConfig();
    const glossarySelect = document.getElementById('glossarySelect');
    
    const profileData = {
        source_language: formData.source_language,
        target_language: formData.target_language,
        llm_provider: formData.llm_provider,
        model: formData.model,
        llm_api_endpoint: formData.llm_api_endpoint || '',
        novel_context_file: formData.prompt_options?.novel_context_file || '',
        glossary: glossarySelect ? glossarySelect.value : '',
        custom_instruction_file: formData.prompt_options?.custom_instruction_file || '',
        bilingual_output: formData.bilingual_output,
        text_cleanup: !!formData.prompt_options?.text_cleanup,
        auto_update_context: !!formData.prompt_options?.auto_update_context,
        bypass_context_gating: !!formData.prompt_options?.bypass_context_gating,
        reflection_mode: !!formData.prompt_options?.reflection_mode,
        editor_provider: formData.prompt_options?.editor_provider || '',
        editor_model: formData.prompt_options?.editor_model || '',
        plain_text_mode: !!formData.prompt_options?.plain_text_mode,
        chapter_mode: !!formData.prompt_options?.chapter_mode,
        auto_pause_on_rate_limit: formData.auto_pause_on_rate_limit,
        parallel_workers: formData.parallel_workers,
        tts_enabled: formData.tts_enabled,
        tts_voice: formData.tts_voice,
        tts_rate: formData.tts_rate,
        tts_format: formData.tts_format,
        tts_bitrate: formData.tts_bitrate,
        output_filename_pattern: DomHelpers.getValue('outputFilenamePattern') || ''
    };
    
    try {
        await ApiClient.saveProfile(name, profileData);
        MessageLogger.addLog(t('translation:profile_saved_log', { name }));
        await FormManager.loadProfiles();
        
        const select = document.getElementById('profileSelect');
        if (select) {
            select.value = name;
            const loadButton = document.getElementById('btnLoadProfile');
            if (loadButton) loadButton.disabled = false;
        }
    } catch (e) {
        console.error("Failed to save profile:", e);
        MessageLogger.addLog(t('translation:profile_save_failed_log', { error: e.message }));
    }
};
