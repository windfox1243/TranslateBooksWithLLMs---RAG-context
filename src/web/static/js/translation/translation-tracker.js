/**
 * Translation Tracker - Track active translations and handle WebSocket updates
 *
 * Manages active translation state, WebSocket event handling,
 * translation completion, error handling, and batch queue progression.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { MessageLogger } from '../ui/message-logger.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { StatusManager } from '../utils/status-manager.js';
import { FileUpload } from '../files/file-upload.js';
import { FileActions } from '../files/file-actions.js';
import { ProgressManager, formatElapsedTime, deriveRateContext, buildRecommendationContent } from './progress-manager.js';
import { renderTranslationTitle, getFileIcon, createGenericEPUBIcon } from './progress-title.js';
import { LifecycleManager } from '../utils/lifecycle-manager.js';
import { ApiKeyUtils } from '../utils/api-key-utils.js';
import { t } from '../i18n/i18n.js';

// Storage configuration with versioning
const STORAGE_VERSION = 1;
const STORAGE_KEY_PREFIX = 'tbl_translation_state';
const TRANSLATION_STATE_STORAGE_KEY = `${STORAGE_KEY_PREFIX}_v${STORAGE_VERSION}`;

/**
 * Validate translation state structure
 * @param {any} data - Data to validate
 * @returns {boolean} True if valid
 */
function validateTranslationState(data) {
    if (!data || typeof data !== 'object') return false;

    // Check required fields
    if (!('version' in data)) return false;
    if (!('currentJob' in data)) return false;
    if (!('isBatchActive' in data)) return false;
    if (!('activeJobs' in data)) return false;
    if (!('hasActive' in data)) return false;

    // Validate types
    if (typeof data.isBatchActive !== 'boolean') return false;
    if (typeof data.hasActive !== 'boolean') return false;
    if (!Array.isArray(data.activeJobs)) return false;

    // Validate currentJob if present
    if (data.currentJob !== null) {
        if (typeof data.currentJob !== 'object') return false;
        if (!('translationId' in data.currentJob)) return false;
        if (!('fileRef' in data.currentJob)) return false;
    }

    return true;
}

export const TranslationTracker = {
    // Debounce timer for saving state
    _saveStateTimer: null,
    _saveStateDebounceMs: 100,

    /**
     * Initialize translation tracker
     */
    async initialize() {
        // Clean up old storage versions
        this.cleanupOldStorageVersions();

        // Setup event listeners FIRST (they need to be ready before any state changes)
        this.setupEventListeners();

        // CRITICAL: Check server session BEFORE restoring state
        // This prevents restoring state from a previous server session
        try {
            const serverWasRestarted = await LifecycleManager.getServerSessionCheck();

            if (serverWasRestarted) {
                this.initializeDefaultTranslationState();
            } else {
                this.restoreTranslationStateSync();

                await Promise.all([
                    this.updateActiveTranslationsState(),
                    this.reconcileStateWithServer()
                ]);
            }
        } catch (error) {
            console.error('Failed to initialize translation state:', error);
            MessageLogger.addLog(t('translation:session_init_failed'));

            // Fallback: restore from localStorage anyway
            this.restoreTranslationStateSync();
        }

        // Mark initialization as complete
        this._initializationComplete = true;
    },

    /**
     * Check if initialization is complete
     * @returns {boolean} True if initialization is complete
     */
    isInitialized() {
        return this._initializationComplete === true;
    },

    /**
     * Clean up old localStorage versions
     */
    cleanupOldStorageVersions() {
        try {
            // Remove old non-versioned key
            const oldKey = 'tbl_translation_state';
            if (localStorage.getItem(oldKey)) {
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
     * Restore translation state from localStorage synchronously
     * This ensures the UI shows the translation state immediately on page load
     */
    restoreTranslationStateSync() {
        try {
            const stored = localStorage.getItem(TRANSLATION_STATE_STORAGE_KEY);

            if (!stored) {
                this.initializeDefaultTranslationState();
                return;
            }

            const savedState = JSON.parse(stored);

            if (!validateTranslationState(savedState)) {
                MessageLogger.addLog(t('translation:session_corrupted_log'));
                this.initializeDefaultTranslationState();
                this.clearTranslationState();
                return;
            }

            if (savedState.version !== STORAGE_VERSION) {
                this.initializeDefaultTranslationState();
                this.clearTranslationState();
                return;
            }

            if (savedState.isBatchActive && savedState.currentJob) {
                StateManager.setState('translation.currentJob', savedState.currentJob);
                StateManager.setState('translation.isBatchActive', savedState.isBatchActive);
                StateManager.setState('translation.activeJobs', savedState.activeJobs || []);
                StateManager.setState('translation.hasActive', savedState.hasActive || false);
                if (savedState.lastJobId) {
                    StateManager.setState('translation.lastJobId', savedState.lastJobId);
                }

                DomHelpers.show('progressSection');
                DomHelpers.show('interruptBtn');

                const translateBtn = DomHelpers.getElement('translateBtn');
                if (translateBtn) {
                    translateBtn.disabled = true;
                    translateBtn.innerHTML = t('translation:batch_in_progress');
                }

                MessageLogger.addLog(t('translation:session_restored_log'));
            } else {
                this.initializeDefaultTranslationState();
            }
        } catch (error) {
            console.error('Failed to restore translation state from localStorage:', error);
            MessageLogger.addLog(t('translation:session_could_not_restore'));
            this.initializeDefaultTranslationState();
        }
    },

    /**
     * Reconcile local state with server state
     * Checks if localStorage state matches server reality
     */
    async reconcileStateWithServer() {
        try {
            const currentJob = StateManager.getState('translation.currentJob');

            // If we have a local job, verify it exists on server
            if (currentJob && currentJob.translationId) {
                try {
                    const serverState = await ApiClient.getTranslationStatus(currentJob.translationId);

                    if (serverState.status === 'completed' ||
                        serverState.status === 'error' ||
                        serverState.status === 'interrupted' ||
                        serverState.status === 'rate_limited') {

                        MessageLogger.addLog(t('translation:session_sync_log', { status: serverState.status }));
                        this.resetUIToIdle();
                    } else if (serverState.status === 'running' || serverState.status === 'queued') {
                        // Calculate progress from stats if available
                        if (serverState.stats) {
                            this.updateStats(currentJob.fileRef.fileType, serverState.stats);
                        }
                    }
                } catch (error) {
                    if (error.status === 404) {
                        MessageLogger.addLog(t('translation:session_job_missing_log'));
                        this.resetUIToIdle();
                    }
                }
            }

            await this.restoreActiveTranslation();

        } catch (error) {
            console.warn('Failed to reconcile state with server:', error);
        }
    },

    /**
     * Initialize default translation state (when no saved state exists)
     */
    initializeDefaultTranslationState() {
        StateManager.setState('translation.currentJob', null);
        StateManager.setState('translation.isBatchActive', false);
        StateManager.setState('translation.activeJobs', []);
        StateManager.setState('translation.hasActive', false);
    },

    /**
     * Save translation state to localStorage (debounced)
     */
    saveTranslationState() {
        // Clear existing timer
        if (this._saveStateTimer) {
            clearTimeout(this._saveStateTimer);
        }

        // Debounce to avoid multiple rapid saves
        this._saveStateTimer = setTimeout(() => {
            this._performSaveTranslationState();
        }, this._saveStateDebounceMs);
    },

    /**
     * Perform the actual save to localStorage
     * @private
     */
    _performSaveTranslationState() {
        try {
            const state = {
                version: STORAGE_VERSION,
                currentJob: StateManager.getState('translation.currentJob'),
                isBatchActive: StateManager.getState('translation.isBatchActive'),
                activeJobs: StateManager.getState('translation.activeJobs'),
                hasActive: StateManager.getState('translation.hasActive'),
                lastJobId: StateManager.getState('translation.lastJobId') || null,
                timestamp: Date.now()
            };

            localStorage.setItem(TRANSLATION_STATE_STORAGE_KEY, JSON.stringify(state));
        } catch (error) {
            console.error('Failed to save translation state to localStorage:', error);

            // Check if it's a quota exceeded error
            if (error.name === 'QuotaExceededError') {
                MessageLogger.addLog(t('translation:session_state_save_quota'));
            } else {
                MessageLogger.addLog(t('translation:session_state_save_failed'));
            }
        }
    },

    /**
     * Clear translation state from localStorage
     */
    clearTranslationState() {
        try {
            // Clear any pending save
            if (this._saveStateTimer) {
                clearTimeout(this._saveStateTimer);
                this._saveStateTimer = null;
            }

            localStorage.removeItem(TRANSLATION_STATE_STORAGE_KEY);
        } catch (error) {
            console.error('Failed to clear translation state from localStorage:', error);
        }
    },

    /**
     * Restore active translation state if there's one running on the server
     */
    async restoreActiveTranslation() {
        try {
            const response = await ApiClient.getActiveTranslations();
            const activeJobs = (response.translations || []).filter(
                t => t.status === 'running' || t.status === 'queued'
            );

            if (activeJobs.length === 0) return;

            // Find matching file in our queue
            const filesToProcess = StateManager.getState('files.toProcess') || [];

            for (const job of activeJobs) {
                let matchingFile = filesToProcess.find(f =>
                    f.translationId === job.translation_id ||
                    f.filePath === job.input_file ||
                    f.name === job.input_file?.split('/').pop()
                );

                // If no matching file found, create a virtual file reference from server data
                // This allows restoration after browser refresh even if filesToProcess is empty
                if (!matchingFile && (job.input_filename || job.output_filename)) {
                    matchingFile = {
                        // output_filename keeps older jobs restorable when
                        // their config predates input-filename persistence.
                        name: job.input_filename || job.output_filename,
                        translationId: job.translation_id,
                        status: 'Processing',
                        type: job.file_type || 'txt',
                        fileType: job.file_type || 'txt',
                        isVirtual: true
                    };
                }

                if (matchingFile) {
                    StateManager.setState('translation.currentJob', {
                        fileRef: matchingFile,
                        translationId: job.translation_id
                    });
                    StateManager.setState('translation.isBatchActive', true);

                    DomHelpers.show('progressSection');
                    this.updateTranslationTitle(matchingFile);

                    // Check if restored job uses a novel context and update preview visibility
                    try {
                        const jobStatus = await ApiClient.getTranslationStatus(job.translation_id);
                        const hasNovelContext = jobStatus.config?.prompt_options?.novel_context_file || jobStatus.config?.prompt_options?.auto_update_context;
                        const contextSection = DomHelpers.getElement('novelContextPreviewSection');
                        if (contextSection) {
                            contextSection.style.display = hasNovelContext ? 'block' : 'none';
                        }
                    } catch (err) {
                        console.warn('[Context] Failed to fetch restored job config:', err);
                    }

                    // Calculate progress from stats (job contains total_chunks, completed_chunks, etc.)
                    if (job.total_chunks > 0) {
                        const stats = {
                            total_chunks: job.total_chunks,
                            completed_chunks: job.completed_chunks || 0,
                            failed_chunks: job.failed_chunks || 0,
                            elapsed_time: job.elapsed_time,
                            progress_percent: job.progress_percent,
                            current_phase: job.current_phase,
                            enable_refinement: job.enable_refinement || false
                        };
                        this.updateStats(matchingFile.fileType, stats);
                    }

                    if (job.last_translation) {
                        MessageLogger.updateTranslationPreview(job.last_translation);
                    }

                    const translateBtn = DomHelpers.getElement('translateBtn');
                    if (translateBtn) {
                        translateBtn.disabled = true;
                        translateBtn.innerHTML = t('translation:batch_in_progress');
                    }
                    DomHelpers.show('interruptBtn');

                    if (!matchingFile.isVirtual) {
                        this.updateFileStatusInList(matchingFile.name, 'Processing', job.translation_id);
                    }

                    break;
                }
            }
        } catch (error) {
            console.warn('Failed to restore active translation:', error);
        }
    },

    setupEventListeners() {
        StateManager.subscribe('translation.currentJob', () => {
            this.saveTranslationState();
        });

        StateManager.subscribe('translation.isBatchActive', () => {
            this.saveTranslationState();
        });

        StateManager.subscribe('translation.hasActive', () => {
            this.updateResumeButtonsState();
            this.saveTranslationState();
        });

        StateManager.subscribe('translation.activeJobs', () => {
            this.saveTranslationState();
        });

        StateManager.subscribe('translation.lastJobId', () => {
            this.saveTranslationState();
        });
    },

    /**
     * Handle translation update from WebSocket
     * @param {Object} data - Translation update data
     */
    handleTranslationUpdate(data) {
        const currentJob = StateManager.getState('translation.currentJob');

        if (!currentJob || data.translation_id !== currentJob.translationId) {
            const isResyncStep = data.log_entry?.data?.ui_step === 'context_resync';
            const lastJobId = StateManager.getState('translation.lastJobId');
            if (isResyncStep && data.translation_id === lastJobId && data.log) {
                MessageLogger.addStepLog(data.log);
                updateContextResyncControls(
                    data.log_entry?.data?.resync_state
                );
            }
            if (data.translation_id) {
                if (data.status === 'completed' || data.status === 'partial' || data.status === 'error' || data.status === 'interrupted' || data.status === 'rate_limited') {
                    if (!currentJob) {
                        this.resetUIToIdle();
                    }
                } else if (data.status === 'running' || data.status === 'queued') {
                    // We received an update for a running job that the UI is not showing.
                    // This happens when auto-resuming after context resync, or if another tab started a job.
                    // We must restore the active translation state.
                    this.restoreActiveTranslation().then(() => {
                        // After restoring, re-handle the update so we don't lose this event's payload
                        this.handleTranslationUpdate(data);
                    });
                }
            }
            return;
        }

        const currentFile = currentJob.fileRef;

        if (data.log) {
            const isWorkflowStep = !!data.log_entry?.data?.ui_step;
            const formattedLog = `[${currentFile.name}] ${data.log}`;
            if (isWorkflowStep) {
                MessageLogger.addStepLog(formattedLog);
            } else {
                MessageLogger.addLog(formattedLog);
            }
            if (data.log_entry?.data?.ui_step === 'context_resync') {
                updateContextResyncControls(
                    data.log_entry?.data?.resync_state
                );
            }
        }

        // Progress is now calculated from stats in ProgressManager.update()
        // No need to call updateProgress() separately
        if (data.stats) {
            this.updateStats(currentFile.fileType, data.stats);
        }

        if (data.log_entry
            && (data.log_entry.type === 'llm_response' || data.log_entry.type === 'refinement_response')
            && data.log_entry.data && data.log_entry.data.response) {
            MessageLogger.updateTranslationPreview(data.log_entry.data.response);
        }

        if (data.log_entry && data.log_entry.type === 'novel_context_state' && data.log_entry.data) {
            const contextSection = DomHelpers.getElement('novelContextPreviewSection');
            const filenameSpan = DomHelpers.getElement('contextPreviewFilename');
            if (contextSection) {
                contextSection.style.display = 'block';
            }
            if (filenameSpan && data.log_entry.data.filename) {
                filenameSpan.textContent = `(${data.log_entry.data.filename})`;
            }
            
            const currentStats = StateManager.getState('translation.stats') || {};
            if (window.NovelContextUI) {
                const hasContextContent = typeof data.log_entry.data.content === 'string';
                if (hasContextContent) {
                    // Keep track of the latest content
                    window.NovelContextUI.latestContent = data.log_entry.data.content || '';
                    
                    // In-memory context rebuilt during standalone refinement has
                    // no persisted snapshots, so do not advertise chunk options
                    // that the snapshot endpoint cannot load.
                    if (!data.log_entry.data.ephemeral) {
                        window.NovelContextUI.updateChunkSelector(
                            currentStats.context_chunk_indices || []
                        );
                    }
                    
                    const selector = document.getElementById('contextChunkSelector');
                    // Only update the display if the user is NOT currently editing the context
                    if (!window.NovelContextUI.isEditing) {
                        if (!selector || selector.value === 'latest') {
                            window.NovelContextUI.renderContextTabs(window.NovelContextUI.latestContent, false);
                        } else {
                            // Do not overwrite display if a specific chunk is selected, but refresh the loaded snapshot
                            window.loadContextSnapshot(selector.value);
                        }
                    }
                }
            }
        }

        if (data.status === 'completed') {
            MessageLogger.resetProgressTracking();
            const completionKey = currentFile.operation === 'refine'
                ? 'translation:refinement_completed_msg'
                : 'translation:translation_completed_msg';
            this.finishCurrentFileTranslation(
                t(completionKey, { name: currentFile.name }),
                'success',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'partial') {
            // Finished, but some units stayed failed after the automatic
            // retries. The output file exists (best effort) and the job is
            // resumable; the completion card explains and gives advice.
            MessageLogger.resetProgressTracking();
            this.finishCurrentFileTranslation(
                t('translation:translation_partial_msg', { name: currentFile.name }),
                'info',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'interrupted') {
            MessageLogger.resetProgressTracking();
            this.finishCurrentFileTranslation(
                t('translation:translation_interrupted_msg', { name: currentFile.name }),
                'info',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'rate_limited') {
            MessageLogger.resetProgressTracking();
            this.finishCurrentFileTranslation(
                t('translation:translation_rate_limited_msg', { name: currentFile.name }),
                'info',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'error') {
            MessageLogger.resetProgressTracking();
            this.finishCurrentFileTranslation(
                t('translation:translation_error_msg', { name: currentFile.name, error: data.error || t('translation:translation_unknown_error') }),
                'error',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'running') {
            MessageLogger.resetProgressTracking();
            
            // Wake up the UI controls from idle/paused state if needed
            StateManager.setState('translation.isBatchActive', true);
            const translateBtn = DomHelpers.getElement('translateBtn');
            if (translateBtn) {
                translateBtn.disabled = true;
                translateBtn.innerHTML = t('translation:batch_in_progress');
            }
            DomHelpers.show('interruptBtn');
            DomHelpers.setDisabled('interruptBtn', false);
            DomHelpers.setText('interruptBtn', t('translation:interrupt_batch_with_icon'));

            DomHelpers.show('progressSection');
            DomHelpers.show('statsGrid');
            this.updateTranslationTitle(currentFile);
            this.resetOpenRouterCostDisplay();

            // Hide context preview section until first state update
            const contextSection = DomHelpers.getElement('novelContextPreviewSection');
            if (contextSection) {
                contextSection.style.display = 'none';
            }

            MessageLogger.showMessage(t('translation:translation_in_progress', { name: currentFile.name }), 'info');
            this.updateFileStatusInList(currentFile.name, 'Processing');
            this.updateActiveTranslationsState();
        }
    },

    /**
     * Update translation title with file icon/thumbnail and name
     * @param {Object} file - File object
     */
    updateTranslationTitle(file) {
        renderTranslationTitle(file);
    },

    /**
     * Update statistics display
     * @param {string} fileType - File type (txt, epub, srt)
     * @param {Object} stats - Statistics object
     */
    updateStats(fileType, stats) {
        StateManager.setState('translation.stats', stats);
        ProgressManager.update({ stats: stats }, fileType);
        this.updateOpenRouterCost(stats);
        if (window.NovelContextUI && stats) {
            window.NovelContextUI.updateChunkSelector(
                stats.context_chunk_indices || []
            );
        }
    },

    /**
     * Update OpenRouter cost display
     * @param {Object} stats - Statistics object containing cost data
     */
    updateOpenRouterCost(stats) {
        const costGrid = DomHelpers.getElement('openrouterCostGrid');
        if (!costGrid) return;

        const cost = stats.openrouter_cost || 0;
        const promptTokens = stats.openrouter_prompt_tokens || 0;
        const completionTokens = stats.openrouter_completion_tokens || 0;
        const totalTokens = promptTokens + completionTokens;

        // Show cost grid if there's any cost or token data
        if (cost > 0 || totalTokens > 0) {
            DomHelpers.show('openrouterCostGrid');
            DomHelpers.setText('openrouterCost', '$' + cost.toFixed(4));
            DomHelpers.setText('openrouterTokens', totalTokens.toLocaleString());
        }
    },

    /**
     * Reset OpenRouter cost display for a new translation
     */
    resetOpenRouterCostDisplay() {
        DomHelpers.hide('openrouterCostGrid');
        DomHelpers.setText('openrouterCost', '$0.0000');
        DomHelpers.setText('openrouterTokens', '0');
    },

    /**
     * Update file status in UI list
     * @param {string} fileName - File name
     * @param {string} newStatus - New status text
     * @param {string} [translationId] - Translation ID
     */
    updateFileStatusInList(fileName, newStatus, translationId = null) {
        const fileListItem = DomHelpers.getOne(`#fileListContainer li[data-filename="${fileName}"] .file-status`);
        if (fileListItem) {
            DomHelpers.setText(fileListItem, `(${newStatus})`);
        }

        // Update in state
        const filesToProcess = StateManager.getState('files.toProcess');
        const fileObj = filesToProcess.find(f => f.name === fileName);
        if (fileObj) {
            fileObj.status = newStatus;
            if (translationId) {
                fileObj.translationId = translationId;
            }
            StateManager.setState('files.toProcess', filesToProcess);
            // Persist to localStorage
            FileUpload.notifyFileListChanged();
        }
    },

    /**
     * Finish current file translation and update UI
     * @param {string} statusMessage - Status message to display
     * @param {string} messageType - Message type (success, error, info)
     * @param {Object} resultData - Translation result data
     */
    finishCurrentFileTranslation(statusMessage, messageType, resultData) {
        const currentJob = StateManager.getState('translation.currentJob');
        if (!currentJob) return;

        // Store lastJobId so we can load context snapshots after it finishes
        StateManager.setState('translation.lastJobId', currentJob.translationId);

        const currentFile = currentJob.fileRef;
        currentFile.status = resultData.status || 'unknown_error';
        currentFile.result = resultData.result;

        MessageLogger.showMessage(statusMessage, messageType);
        this.updateFileStatusInList(
            currentFile.name,
            resultData.status === 'completed' ? 'Completed' :
            resultData.status === 'partial' ? 'Partial' :
            resultData.status === 'interrupted' ? 'Interrupted' :
            resultData.status === 'rate_limited' ? 'Rate Limited' : 'Error'
        );

        if (resultData.status === 'completed' || resultData.status === 'partial') {
            // Partial jobs still produced a best-effort output file; the card
            // surfaces it together with the warning block and its advice.
            this.renderCompletionCard(currentFile, resultData);
        }

        StateManager.setState('translation.currentJob', null);

        if (resultData.status === 'completed') {
            this.processNextFileInQueue();
        } else if (resultData.status === 'interrupted') {
            MessageLogger.addLog(t('translation:batch_stopped_user_log'));
            this.resetUIToIdle();
        } else if (resultData.status === 'rate_limited') {
            MessageLogger.addLog(t('translation:batch_paused_log'));
            this.resetUIToIdle();
        } else {
            this.processNextFileInQueue();
        }
    },

    /**
     * Render a persistent success card for a completed file, with quick actions
     * to locate it on disk.
     * @param {Object} file - The file that just finished
     * @param {Object} resultData - Final payload from the server (output_filename, output_dir)
     */
    renderCompletionCard(file, resultData) {
        const container = DomHelpers.getElement('completionCardsContainer');
        if (!container) return;

        const card = document.createElement('div');
        card.className = 'completion-card';
        this._populateCompletionCard(card, file, resultData);
        container.appendChild(card);
        this._ensureCompletionCardsLocaleListener();

        DomHelpers.hide('progressSection');
    },

    /**
     * Fill (or rebuild) an existing completion card with localized content.
     * Pulled out of `renderCompletionCard` so the same DOM tree can be
     * re-rendered on `localeChanged` without dropping the card from the page.
     *
     * Stashes the source payload on the element itself so the locale listener
     * can rebuild without coordinating extra storage.
     */
    _populateCompletionCard(card, file, resultData) {
        card._tblPayload = { file, resultData };

        const outputFilename = resultData.output_filename || file.outputFilename || file.name;
        const safeFilename = DomHelpers.escapeHtml(outputFilename);
        const statsHtml = this._buildCompletionStatsHtml(file, resultData);
        const dismissLabel = t('translation:completion_card_dismiss');
        const isPartial = resultData.status === 'partial';
        const titleIcon = isPartial ? 'warning' : 'check_circle';
        const titleText = t(isPartial
            ? 'translation:translation_partial_card_title'
            : 'translation:translation_completed_card_title');

        card.innerHTML = '';

        const topRow = document.createElement('div');
        topRow.className = 'completion-card__top';
        topRow.appendChild(this._buildCompletionThumb(file));

        const main = document.createElement('div');
        main.className = 'completion-card__main';
        main.innerHTML = `
            <div class="completion-card__header">
                <h3 class="completion-card__title">
                    <span class="material-symbols-outlined">${titleIcon}</span>
                    <span>${titleText}${statsHtml}</span>
                </h3>
                <button type="button" class="completion-card__close" title="${dismissLabel}" aria-label="${dismissLabel}">
                    <span class="material-symbols-outlined">close</span>
                </button>
            </div>
            <div class="completion-card__filename" title="${safeFilename}">${safeFilename}</div>
        `;
        topRow.appendChild(main);
        card.appendChild(topRow);

        const warningBlock = this._buildCompletionWarningBlock(file, resultData);
        if (warningBlock) {
            card.appendChild(warningBlock);
        }

        const actionsGroup = FileActions.createActionGroup({
            actions: ['download', 'open', 'reveal', 'files-tab'],
            filename: outputFilename,
            variant: 'labeled'
        });
        actionsGroup.classList.add('completion-card__actions');
        card.appendChild(actionsGroup);

        card.querySelector('.completion-card__close').addEventListener('click', () => card.remove());
    },

    /**
     * Re-render every visible completion card whenever the user switches
     * locale, so the dynamically interpolated strings (title, stat badges,
     * warning block, action labels) stay in sync with the rest of the UI.
     * Bound once, lazily, the first time a card is rendered.
     */
    _ensureCompletionCardsLocaleListener() {
        if (this._completionLocaleListenerBound) return;
        this._completionLocaleListenerBound = true;
        window.addEventListener('localeChanged', () => {
            const container = DomHelpers.getElement('completionCardsContainer');
            if (!container) return;
            container.querySelectorAll('.completion-card').forEach((card) => {
                if (card._tblPayload) {
                    this._populateCompletionCard(card, card._tblPayload.file, card._tblPayload.resultData);
                }
            });
        });
    },

    /**
     * Build the thumbnail element for the completion card.
     * Uses the book cover for EPUBs (with SVG fallback), generic icon otherwise.
     * @param {Object} file - File object (fileType, thumbnail)
     * @returns {HTMLElement} Thumb wrapper element
     */
    _buildCompletionThumb(file) {
        const wrap = document.createElement('div');
        wrap.className = 'completion-card__thumb';

        if (file.fileType === 'epub' && file.thumbnail) {
            const img = document.createElement('img');
            img.src = `/api/thumbnails/${encodeURIComponent(file.thumbnail)}`;
            img.alt = 'Cover';
            img.onerror = () => {
                wrap.innerHTML = createGenericEPUBIcon();
            };
            wrap.appendChild(img);
        } else {
            wrap.innerHTML = getFileIcon(file.fileType);
        }

        return wrap;
    },

    /**
     * Build the stats block HTML for the completion card.
     * @param {Object} file - File object (for fileType)
     * @param {Object} resultData - Final payload (contains stats)
     * @returns {string} HTML for the stats block (empty string if no stats)
     */
    _buildCompletionStatsHtml(file, resultData) {
        const stats = resultData.stats || {};

        const failed = stats.failed_chunks || 0;
        const elapsed = stats.elapsed_time;
        const fallbacks = (file && file.fileType === 'srt')
            ? 0
            : (stats.token_alignment_used || 0) + (stats.fallback_used || 0);
        const placeholderErrors = (file && file.fileType === 'srt')
            ? 0
            : (stats.placeholder_errors || 0);

        const cost = stats.openrouter_cost || 0;
        const promptTokens = stats.openrouter_prompt_tokens || 0;
        const completionTokens = stats.openrouter_completion_tokens || 0;
        const totalTokens = promptTokens + completionTokens;

        const items = [];

        if (typeof elapsed === 'number' && elapsed > 0) {
            items.push(formatElapsedTime(elapsed));
        }

        if (failed > 0) {
            items.push(`<span class="completion-card__stat--error">${t('translation:completion_failed_chunks', { count: failed })}</span>`);
        }

        if (fallbacks > 0) {
            items.push(`<span class="completion-card__stat--warn">${t('translation:completion_fallback_chunks', { count: fallbacks })}</span>`);
        }

        if (placeholderErrors > 0) {
            items.push(`<span class="completion-card__stat--warn">${t('translation:completion_placeholder_errors', { count: placeholderErrors })}</span>`);
        }

        if (cost > 0 || totalTokens > 0) {
            items.push(`$${cost.toFixed(4)} · ${totalTokens.toLocaleString()} tokens`);
        }

        if (items.length === 0) return '';

        return `<span class="completion-card__stats"> - ${items.join(' · ')}</span>`;
    },

    /**
     * Build the warning block surfaced beneath the title when the run produced
     * fallbacks, placeholder errors, or failed chunks. Mirrors the live
     * recommendation panel from progress-manager so the post-translation
     * advice stays in sync with what was shown during the run.
     *
     * @param {Object} file - File object (used to gate by file type)
     * @param {Object} resultData - Final payload (contains stats)
     * @returns {HTMLElement|null} Warning block element, or null when there is
     *   nothing worth surfacing.
     */
    _buildCompletionWarningBlock(file, resultData) {
        const stats = resultData.stats || {};
        if (file && file.fileType === 'srt') {
            return this._buildSrtCompletionWarningBlock(stats);
        }

        const fallbacks = (stats.token_alignment_used || 0) + (stats.fallback_used || 0);
        const placeholderErrors = stats.placeholder_errors || 0;
        const failed = stats.failed_chunks || 0;
        const tokenAlignment = stats.token_alignment_used || 0;
        const untranslated = stats.fallback_used || 0;

        if (fallbacks === 0 && placeholderErrors === 0 && failed === 0) {
            return null;
        }

        const block = document.createElement('div');
        block.className = 'completion-card__warning';

        const heading = document.createElement('div');
        heading.className = 'completion-card__warning-heading';
        const icon = document.createElement('span');
        icon.className = 'material-symbols-outlined';
        icon.textContent = 'warning';
        heading.appendChild(icon);
        const headingText = document.createElement('span');
        // When chunks were left in the source language (Phase 3 fallback) or
        // outright failed, the optimistic "translations are correct" heading is
        // misleading — surface the missing-content message instead.
        const hasUntranslatedContent = untranslated > 0 || failed > 0;
        headingText.textContent = t(hasUntranslatedContent
            ? 'translation:completion_warning_heading_untranslated'
            : 'translation:completion_warning_heading');
        heading.appendChild(headingText);
        block.appendChild(heading);

        const breakdownItems = [];
        if (tokenAlignment > 0) {
            breakdownItems.push(t('translation:completion_warning_token_alignment', { count: tokenAlignment }));
        }
        if (untranslated > 0) {
            breakdownItems.push(t('translation:completion_warning_untranslated', { count: untranslated }));
        }
        if (placeholderErrors > 0) {
            breakdownItems.push(t('translation:completion_warning_placeholder_errors', { count: placeholderErrors }));
        }
        if (failed > 0) {
            breakdownItems.push(t('translation:completion_warning_failed', { count: failed }));
        }
        if (breakdownItems.length > 0) {
            const breakdown = document.createElement('div');
            breakdown.className = 'completion-card__warning-breakdown';
            breakdown.textContent = breakdownItems.join(' · ');
            block.appendChild(breakdown);
        }

        // Only renew the rate-based recommendations when there were actual
        // fallbacks or placeholder issues — a run with only `failed_chunks`
        // (e.g. provider errors) is not really a "tune the LLM" situation.
        if (fallbacks > 0 || placeholderErrors > 0) {
            const recommendations = document.createElement('div');
            recommendations.className = 'completion-card__warning-recommendations';
            buildRecommendationContent(
                recommendations,
                deriveRateContext(stats),
                'translation:completion_warning_intro',
            );
            block.appendChild(recommendations);
        }

        return block;
    },

    /**
     * SRT variant of the completion warning block. Shown when subtitle
     * blocks still failed after the automatic marker-validation retries:
     * the affected cues kept the source-language text. Mirrors the EPUB
     * fallback panel structure (heading + breakdown + advice list) with
     * SRT-specific recommendations.
     *
     * @param {Object} stats - Final stats payload
     * @returns {HTMLElement|null} Warning block, or null when nothing failed
     */
    _buildSrtCompletionWarningBlock(stats) {
        const failed = stats.failed_chunks || 0;
        if (failed === 0) {
            return null;
        }

        const block = document.createElement('div');
        block.className = 'completion-card__warning';

        const heading = document.createElement('div');
        heading.className = 'completion-card__warning-heading';
        const icon = document.createElement('span');
        icon.className = 'material-symbols-outlined';
        icon.textContent = 'warning';
        heading.appendChild(icon);
        const headingText = document.createElement('span');
        headingText.textContent = t('translation:srt_completion_warning_heading');
        heading.appendChild(headingText);
        block.appendChild(heading);

        const breakdown = document.createElement('div');
        breakdown.className = 'completion-card__warning-breakdown';
        breakdown.textContent = t('translation:srt_completion_warning_blocks', { count: failed });
        block.appendChild(breakdown);

        const recommendations = document.createElement('div');
        recommendations.className = 'completion-card__warning-recommendations';
        const intro = document.createElement('strong');
        intro.textContent = t('translation:srt_completion_warning_intro');
        recommendations.appendChild(intro);

        const list = document.createElement('ul');
        list.className = 'recommendation-list';
        const llmTip = document.createElement('li');
        llmTip.textContent = t('translation:fallback_panel_tip_llm');
        list.appendChild(llmTip);
        const blockSizeTip = document.createElement('li');
        blockSizeTip.textContent = t('translation:srt_completion_tip_block_size');
        list.appendChild(blockSizeTip);
        recommendations.appendChild(list);
        block.appendChild(recommendations);

        return block;
    },

    /**
     * Process next file in queue (delegates to batch-controller when available)
     */
    processNextFileInQueue() {
        // Trigger event for batch controller to handle
        window.dispatchEvent(new CustomEvent('processNextFile'));
    },

    /**
     * Check and update active translations state
     */
    async updateActiveTranslationsState() {
        try {
            const response = await ApiClient.getActiveTranslations();
            const activeJobs = (response.translations || []).filter(
                t => t.status === 'running' || t.status === 'queued'
            );

            const wasActive = StateManager.getState('translation.hasActive');
            const hasActive = activeJobs.length > 0;

            StateManager.setState('translation.hasActive', hasActive);
            StateManager.setState('translation.activeJobs', activeJobs);

            // If state changed, update UI
            if (wasActive !== hasActive) {
                this.updateResumeButtonsState();
            }

            return { hasActive, activeJobs };
        } catch {
            return {
                hasActive: StateManager.getState('translation.hasActive'),
                activeJobs: StateManager.getState('translation.activeJobs')
            };
        }
    },

    /**
     * Update the state of all resume buttons based on active translations
     */
    updateResumeButtonsState() {
        const resumeButtons = DomHelpers.getElements('button[onclick^="resumeJob"]');
        const hasActive = StateManager.getState('translation.hasActive');

        resumeButtons.forEach(button => {
            if (hasActive) {
                button.disabled = true;
                button.style.opacity = '0.5';
                button.style.cursor = 'not-allowed';
                button.title = t('translation:cannot_resume_in_progress_title');
            } else {
                button.disabled = false;
                button.style.opacity = '1';
                button.style.cursor = 'pointer';
                button.title = t('translation:resume_btn_title');
            }
        });

        // Update warning banner
        this.updateResumableJobsWarningBanner();
    },

    /**
     * Update or create the warning banner in resumable jobs section
     */
    updateResumableJobsWarningBanner() {
        const listContainer = DomHelpers.getElement('resumableJobsList');
        if (!listContainer) return;

        const existingBanner = listContainer.querySelector('.active-translation-warning');
        const hasActive = StateManager.getState('translation.hasActive');
        const activeJobs = StateManager.getState('translation.activeJobs');

        if (hasActive) {
            const activeNames = activeJobs.map(job => job.output_filename || t('translation:job_card_unknown')).join(', ');
            const bannerHtml = `
                <div class="active-translation-warning" style="background: #fef3c7; border: 1px solid #f59e0b; padding: 12px; margin-bottom: 15px; border-radius: 6px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 20px;">⚠️</span>
                        <div style="flex: 1;">
                            <strong style="color: #92400e;">${t('translation:active_translation_warning_title')}</strong>
                            <p style="margin: 5px 0 0 0; font-size: 13px; color: #78350f;">
                                ${t('translation:active_translation_warning_desc', { names: DomHelpers.escapeHtml(activeNames) })}
                            </p>
                        </div>
                    </div>
                </div>
            `;

            if (existingBanner) {
                existingBanner.outerHTML = bannerHtml;
            } else {
                // Insert at the beginning of the container
                listContainer.insertAdjacentHTML('afterbegin', bannerHtml);
            }
        } else if (existingBanner) {
            // Remove banner if no active translations
            existingBanner.remove();
        }
    },

    resetUIToIdle() {
        StateManager.setState('translation.isBatchActive', false);
        StateManager.setState('translation.currentJob', null);

        this.clearTranslationState();

        DomHelpers.hide('interruptBtn');
        DomHelpers.setDisabled('interruptBtn', false);
        DomHelpers.setText('interruptBtn', t('translation:interrupt_batch_with_icon'));

        const filesToProcess = StateManager.getState('files.toProcess');
        DomHelpers.setDisabled('translateBtn', filesToProcess.length === 0 || !StatusManager.isConnected());
        DomHelpers.setText('translateBtn', t('translation:start_batch_with_icon'));

        if (filesToProcess.length === 0) {
            DomHelpers.hide('progressSection');
        }

        this.updateActiveTranslationsState();

        if (window.loadResumableJobs) {
            window.loadResumableJobs();
        }
    }
};

window.NovelContextUI = {
    latestContent: '',
    displayedContent: '',
    availableChunkIndices: [],
    globalAnchorChunkIndex: null,
    globalAnchorFullContent: '',
    activeTabIndex: 0,
    localizedView: null,
    lastResyncState: null,
    _localeListenerBound: false,

    initializeLocaleListener: function() {
        if (this._localeListenerBound) return;
        this._localeListenerBound = true;

        window.addEventListener('localeChanged', () => {
            // data-i18n markers update selector and tab labels automatically.
            // Rebuild only synthetic localized content, and never replace an
            // active edit form because that would discard unsaved changes.
            if (this.localizedView && !this.isEditing) {
                this._renderLocalizedView();
            }
            updateContextResyncControls(this.lastResyncState);
        });
    },

    renderLocalizedView: function(bodyKey, {
        titleKey = null,
        params = {},
        isSnapshot = true
    } = {}) {
        this.localizedView = { bodyKey, titleKey, params, isSnapshot };
        this._renderLocalizedView();
    },

    _renderLocalizedView: function() {
        if (!this.localizedView) return;

        const { bodyKey, titleKey, params, isSnapshot } = this.localizedView;
        const body = t(bodyKey, params);
        const content = titleKey
            ? `# ${t(titleKey, params)}\n${body}`
            : body;
        this.renderContextTabs(content, isSnapshot, true);
    },
    
    renderContextTabs: function(content, isSnapshot = false, preserveLocalizedView = false) {
        if (!preserveLocalizedView) {
            this.localizedView = null;
        }
        if (!isSnapshot) {
            this.latestContent = content;
        }
        this.displayedContent = content;
        const header = document.getElementById('contextTabsHeader');
        const body = document.getElementById('contextTabsBody');
        if (!header || !body) return;

        // Ensure header has a scrollbar if tabs overflow
        header.style.overflowX = 'visible';
        header.style.overflowY = 'visible';
        header.style.whiteSpace = 'normal';
        header.style.flexWrap = 'wrap';
        header.style.display = 'flex';
        header.style.gap = '0.5rem';
        header.style.paddingBottom = '0.5rem';

        // Split by markdown headers # or ##
        const sections = [];
        const regex = /^(#{1,3})\s*(.+)$/gm;
        let match;
        let lastIndex = 0;
        let currentTitle = t('translation:context_general_tab');
        let currentTitleKey = 'translation:context_general_tab';
        
        while ((match = regex.exec(content)) !== null) {
            const textBefore = content.substring(lastIndex, match.index).trim();
            const cleanText = textBefore
                .replace(/---DYNAMIC_STATE_START---|---DYNAMIC_STATE_END---/g, '')
                .replace(/^\s*\(.*?\)\s*$/gm, '')
                .trim();
            
            // Push section only if it has content, or if it's the very first section and we have no other choice
            if (cleanText) {
                sections.push({
                    title: currentTitle,
                    titleKey: currentTitleKey,
                    content: cleanText
                });
            }
            
            currentTitle = match[2].trim();
            currentTitleKey = null;
            lastIndex = match.index + match[0].length;
        }
        
        const lastText = content.substring(lastIndex).trim();
        const cleanLast = lastText
            .replace(/---DYNAMIC_STATE_START---|---DYNAMIC_STATE_END---/g, '')
            .replace(/^\s*\(.*?\)\s*$/gm, '')
            .trim();
        if (cleanLast || sections.length === 0) {
            sections.push({
                title: currentTitle,
                titleKey: currentTitleKey,
                content: cleanLast
            });
        }

        // Always add a raw view tab.
        sections.push({
            title: t('translation:context_raw_view_tab'),
            titleKey: 'translation:context_raw_view_tab',
            content
        });

        header.innerHTML = '';
        body.innerHTML = '';

        // Determine which tab should be active
        const activeIdx = Number.isInteger(this.activeTabIndex)
            && this.activeTabIndex >= 0
            && this.activeTabIndex < sections.length
            ? this.activeTabIndex
            : 0;
        this.activeTabIndex = activeIdx;

        sections.forEach((sec, idx) => {
            const btn = document.createElement('button');
            const isActive = idx === activeIdx;
            btn.className = `btn btn-sm ${isActive ? 'btn-primary' : 'btn-secondary'}`;
            btn.style.whiteSpace = 'nowrap';
            btn.style.flexShrink = '0';
            btn.style.borderRadius = '4px';
            btn.style.padding = '0.25rem 0.75rem';
            btn.textContent = sec.title;
            if (sec.titleKey) {
                btn.setAttribute('data-i18n', sec.titleKey);
            }
            
            const pane = document.createElement('div');
            pane.style.display = isActive ? 'block' : 'none';
            pane.textContent = sec.content;
            
            btn.onclick = () => {
                this.activeTabIndex = idx;
                Array.from(header.children).forEach(c => {
                    c.className = 'btn btn-sm btn-secondary';
                    c.style.background = 'transparent';
                    c.style.color = 'var(--text-dark)';
                    c.style.border = '1px solid var(--border-color)';
                });
                btn.className = 'btn btn-sm btn-primary';
                btn.style.background = 'var(--primary-color)';
                btn.style.color = '#fff';
                btn.style.border = '1px solid var(--primary-color)';
                
                Array.from(body.children).forEach(p => { p.style.display = 'none'; });
                pane.style.display = 'block';
            };
            
            // Set initial style for active tab
            if (isActive) {
                btn.style.background = 'var(--primary-color)';
                btn.style.color = '#fff';
                btn.style.border = '1px solid var(--primary-color)';
            } else {
                btn.style.background = 'transparent';
                btn.style.color = 'var(--text-dark)';
                btn.style.border = '1px solid var(--border-color)';
            }
            
            header.appendChild(btn);
            body.appendChild(pane);
        });
    },

    updateChunkSelector: function(chunkIndices) {
        const selector = document.getElementById('contextChunkSelector');
        if (!selector) return;
        
        // Preserve current selection if possible
        const currentVal = selector.value;
        
        const availableIndices = Array.isArray(chunkIndices)
            ? [...new Set(chunkIndices)]
                .filter(index => Number.isInteger(index) && index >= 0)
                .sort((a, b) => a - b)
            : [];
        this.availableChunkIndices = availableIndices;
        selector.innerHTML = '';

        if (availableIndices.length > 0) {
            const globalOption = document.createElement('option');
            globalOption.value = 'global';
            globalOption.textContent = t('translation:context_global_state');
            globalOption.setAttribute(
                'data-i18n',
                'translation:context_global_state'
            );
            selector.appendChild(globalOption);
        }

        const latestOption = document.createElement('option');
        latestOption.value = 'latest';
        latestOption.textContent = t('translation:context_latest_state');
        latestOption.setAttribute(
            'data-i18n',
            'translation:context_latest_state'
        );
        selector.appendChild(latestOption);

        availableIndices.forEach(index => {
            const opt = document.createElement('option');
            opt.value = index;
            opt.textContent = t('translation:context_chunk_option', { number: index + 1 });
            opt.setAttribute('data-i18n', 'translation:context_chunk_option');
            opt.setAttribute('data-i18n-params', JSON.stringify({ number: index + 1 }));
            selector.appendChild(opt);
        });
        
        if (currentVal && Array.from(selector.options).some(o => o.value === currentVal)) {
            selector.value = currentVal;
        }
    }
};

window.NovelContextUI.initializeLocaleListener();

function setContextEditButtonMode(isGlobal) {
    const btnEdit = document.getElementById('btnEditResync');
    const btnSave = document.getElementById('btnSaveResync');
    const editLabel = btnEdit?.querySelector('[data-context-action-label]');
    const saveLabel = btnSave?.querySelector('[data-context-action-label]');
    const editTitleKey = isGlobal
        ? 'translation:context_edit_global_title'
        : 'translation:context_edit_resync_title';
    const editLabelKey = isGlobal
        ? 'translation:context_edit_global_btn'
        : 'translation:context_edit_resync_btn';
    const saveLabelKey = isGlobal
        ? 'translation:context_save_global_btn'
        : 'translation:context_save_resync_btn';

    if (btnEdit) {
        btnEdit.setAttribute('data-i18n-attr', `title:${editTitleKey}`);
        btnEdit.title = t(editTitleKey);
    }
    if (editLabel) {
        editLabel.setAttribute('data-i18n', editLabelKey);
        editLabel.textContent = t(editLabelKey);
    }
    if (saveLabel) {
        saveLabel.setAttribute('data-i18n', saveLabelKey);
        saveLabel.textContent = t(saveLabelKey);
    }
}

function logContextResyncFailure(errorKey, params = {}) {
    MessageLogger.addLog(
        t('translation:context_resync_failed_log', {
            error: t(errorKey, params)
        })
    );
}

function currentTranslationIdForContext() {
    const currentJob = StateManager.getState('translation.currentJob');
    if (currentJob && currentJob.translationId) {
        return currentJob.translationId;
    }
    return StateManager.getState('translation.lastJobId');
}

let contextResyncStatusPollTimer = null;

function updateContextResyncPolling(status) {
    const shouldPoll = ['running', 'pause_requested', 'paused'].includes(status);
    if (shouldPoll && !contextResyncStatusPollTimer) {
        contextResyncStatusPollTimer = window.setInterval(
            refreshContextResyncStatus,
            3000
        );
    } else if (!shouldPoll && contextResyncStatusPollTimer) {
        window.clearInterval(contextResyncStatusPollTimer);
        contextResyncStatusPollTimer = null;
    }
}

function updateContextResyncControls(resyncState = null) {
    const btnPause = document.getElementById('btnPauseResync');
    const btnResume = document.getElementById('btnResumeResync');
    const badge = document.getElementById('contextResyncStatusBadge');
    
    if (!resyncState && window.NovelContextUI?.lastResyncState) {
        resyncState = window.NovelContextUI.lastResyncState;
    }
    if (window.NovelContextUI) {
        window.NovelContextUI.lastResyncState = resyncState;
    }

    const status = resyncState?.status || '';
    const isRunning = status === 'running' || status === 'pause_requested';
    const isPaused = status === 'paused';
    const hasResyncState = Boolean(status);

    updateContextResyncPolling(status);

    if (hasResyncState && (isRunning || isPaused)) {
        const contextSection = document.getElementById('novelContextPreviewSection');
        if (contextSection) {
            contextSection.style.display = 'block';
        }
    }

    if (!btnPause || !btnResume) return;

    btnPause.style.display = isRunning ? 'inline-flex' : 'none';
    btnPause.disabled = status === 'pause_requested';
    btnResume.style.display = isPaused ? 'inline-flex' : 'none';
    btnResume.disabled = false;

    if (badge) {
        const statusKey = {
            running: 'translation:context_resync_status_running',
            pause_requested: 'translation:context_resync_status_pause_requested',
            paused: 'translation:context_resync_status_paused',
            completed: 'translation:context_resync_status_completed',
            failed: 'translation:context_resync_status_failed'
        }[status] || 'translation:context_resync_status_idle';
        badge.setAttribute('data-i18n', statusKey);
        badge.textContent = t(statusKey);
        badge.style.display = hasResyncState ? 'inline-flex' : 'none';
        badge.style.alignItems = 'center';
        badge.style.background = isRunning
            ? 'rgba(59, 130, 246, 0.12)'
            : isPaused
                ? 'rgba(245, 158, 11, 0.12)'
                : status === 'failed'
                    ? 'rgba(239, 68, 68, 0.12)'
                    : 'var(--bg-light)';
        badge.style.color = isRunning
            ? '#1d4ed8'
            : isPaused
                ? '#92400e'
                : status === 'failed'
                    ? '#b91c1c'
                    : 'var(--text-muted-light)';
    }
}

function collectContextResyncOverrides() {
    const provider = DomHelpers.getValue('llmProvider');
    const model = DomHelpers.getValue('model');
    const endpoint = provider === 'openai'
        ? DomHelpers.getValue('openaiEndpoint')
        : DomHelpers.getValue('apiEndpoint');
    const overrides = {
        llm_provider: provider,
        model,
        llm_api_endpoint: endpoint
    };
    const apiKey = ApiKeyUtils.getValueForProvider(provider);
    if (apiKey) {
        overrides.api_key = apiKey;
    }
    return overrides;
}

async function refreshContextResyncStatus() {
    const translationId = currentTranslationIdForContext();
    if (!translationId) {
        updateContextResyncControls(null);
        return;
    }
    try {
        const result = await ApiClient.getContextResyncStatus(translationId);
        updateContextResyncControls(result?.resync_state || null);
    } catch (_e) {
        updateContextResyncControls(null);
    }
}

window.loadContextSnapshot = async function(chunkValue) {
    const btnEdit = document.getElementById('btnEditResync');
    const btnSave = document.getElementById('btnSaveResync');
    const btnCancel = document.getElementById('btnCancelResync');
    if (btnEdit) btnEdit.style.display = 'none';
    if (btnSave) btnSave.style.display = 'none';
    if (btnCancel) btnCancel.style.display = 'none';
    refreshContextResyncStatus();

    if (chunkValue === 'latest') {
        setContextEditButtonMode(false);
        if (window.NovelContextUI.latestContent) {
            window.NovelContextUI.renderContextTabs(window.NovelContextUI.latestContent, false);
        }
        return;
    }

    const isGlobal = chunkValue === 'global';
    const resolvedChunkValue = isGlobal
        ? window.NovelContextUI.availableChunkIndices[0]
        : parseInt(chunkValue);
    if (!Number.isInteger(resolvedChunkValue)) {
        window.NovelContextUI.renderLocalizedView(
            'translation:context_no_context_body',
            { titleKey: 'translation:context_no_context_title' }
        );
        return;
    }
    
    // Resolve translationId from multiple sources
    let translationId = null;
    const currentJob = StateManager.getState('translation.currentJob');
    if (currentJob && currentJob.translationId) {
        translationId = currentJob.translationId;
    }
    if (!translationId) {
        translationId = StateManager.getState('translation.lastJobId');
    }
    
    if (!translationId) {
        console.warn('[Context] No translationId available for loading chunk snapshot');
        window.NovelContextUI.renderLocalizedView(
            'translation:context_no_job_body',
            { titleKey: 'translation:context_no_job_title' }
        );
        return;
    }
    
    try {
        console.log(`[Context] Loading snapshot for job=${translationId}, chunk=${resolvedChunkValue}`);
        const result = await ApiClient.getContextSnapshot(
            translationId,
            resolvedChunkValue,
            isGlobal ? { scope: 'global_lore' } : {}
        );
        if (result && (result.context_content !== undefined && result.context_content !== null)) {
            // Check if novel context is configured
            if (result.has_novel_context === false && !result.context_content) {
                // Hide context preview section if no context is configured
                const contextSection = document.getElementById('novelContextPreviewSection');
                if (contextSection) {
                    contextSection.style.display = 'none';
                }
                return;
            }
            
            // Render context tabs or show empty snapshot if empty but configured
            if (result.context_content === "") {
                window.NovelContextUI.renderLocalizedView(
                    'translation:context_empty_snapshot'
                );
            } else if (isGlobal) {
                window.NovelContextUI.globalAnchorChunkIndex = resolvedChunkValue;
                window.NovelContextUI.globalAnchorFullContent = result.context_content;
                const dynamicMarker = result.context_content.indexOf(
                    '---DYNAMIC_STATE_START---'
                );
                const globalContent = dynamicMarker >= 0
                    ? result.context_content.substring(0, dynamicMarker).trim()
                    : result.context_content.trim();
                window.NovelContextUI.renderContextTabs(globalContent, true);
            } else {
                window.NovelContextUI.renderContextTabs(result.context_content, true);
            }
            setContextEditButtonMode(isGlobal);
            if (btnEdit) btnEdit.style.display = 'inline-flex';
        } else {
            window.NovelContextUI.renderLocalizedView(
                'translation:context_no_context_body',
                { titleKey: 'translation:context_no_context_title' }
            );
        }
    } catch (e) {
        console.error(`[Context] Failed to load snapshot for job=${translationId}, chunk=${resolvedChunkValue}:`, e);
        window.NovelContextUI.renderLocalizedView(
            'translation:context_load_error_body',
            {
                titleKey: 'translation:context_load_error_title',
                params: { error: e.message }
            }
        );
    }
};

window.pauseContextResync = async function(targetTranslationId = null) {
    const translationId = (typeof targetTranslationId === 'string' && targetTranslationId.trim())
        ? targetTranslationId.trim()
        : currentTranslationIdForContext();
    if (!translationId) {
        logContextResyncFailure('translation:context_no_job_body');
        return;
    }
    const btnPause = document.getElementById('btnPauseResync');
    try {
        if (btnPause) btnPause.disabled = true;
        const result = await ApiClient.pauseContextResync(translationId);
        updateContextResyncControls(result?.resync_state || {
            status: 'pause_requested'
        });
        MessageLogger.addLog(t('translation:context_resync_pause_requested_log'));
        if (window.ResumeManager && typeof window.ResumeManager.loadResumableJobs === 'function') {
            window.ResumeManager.loadResumableJobs();
        }
    } catch (e) {
        logContextResyncFailure('translation:context_resync_pause_failed', {
            error: e.message
        });
        if (btnPause) btnPause.disabled = false;
    }
};

window.resumeContextResync = async function(targetTranslationId = null, overrides = null) {
    const translationId = (typeof targetTranslationId === 'string' && targetTranslationId.trim())
        ? targetTranslationId.trim()
        : currentTranslationIdForContext();
    if (!translationId) {
        logContextResyncFailure('translation:context_no_job_body');
        return;
    }
    const btnResume = document.getElementById('btnResumeResync');
    try {
        if (btnResume) btnResume.disabled = true;
        StateManager.setState('translation.lastJobId', translationId);
        const reqOverrides = overrides || (
            (!targetTranslationId || targetTranslationId === currentTranslationIdForContext())
                ? collectContextResyncOverrides()
                : null
        );
        const result = await ApiClient.resumeContextResync(
            translationId,
            reqOverrides
        );
        updateContextResyncControls(result?.resync_state || {
            status: 'running'
        });
        MessageLogger.addLog(t('translation:context_resync_resumed_log'));
        if (window.ResumeManager && typeof window.ResumeManager.loadResumableJobs === 'function') {
            window.ResumeManager.loadResumableJobs();
        }
    } catch (e) {
        logContextResyncFailure('translation:context_resync_resume_failed', {
            error: e.message
        });
        if (btnResume) btnResume.disabled = false;
    }
};

window.enableContextEdit = function() {
    const btnEdit = document.getElementById('btnEditResync');
    const btnSave = document.getElementById('btnSaveResync');
    const btnCancel = document.getElementById('btnCancelResync');
    const header = document.getElementById('contextTabsHeader');
    const body = document.getElementById('contextTabsBody');
    if (!header || !body) return;
    
    let content = window.NovelContextUI.latestContent;
    const selector = document.getElementById('contextChunkSelector');
    if (selector && selector.value !== 'latest') {
        content = window.NovelContextUI.displayedContent || content;
    } else if (window.NovelContextUI.latestContent) {
        content = window.NovelContextUI.latestContent;
    } else {
        content = window.NovelContextUI.displayedContent || "";
    }
    
    // Parse the content into exact chunks so we can reconstruct it losslessly
    const sections = [];
    const regex = /^(#{1,3})\s*(.+)$/gm;
    let match;
    let lastIndex = 0;
    let currentHeaderFull = ""; 
    let currentTitle = t('translation:context_general_tab');
    let currentTitleKey = 'translation:context_general_tab';
    
    while ((match = regex.exec(content)) !== null) {
        sections.push({
            title: currentTitle,
            titleKey: currentTitleKey,
            fullHeader: currentHeaderFull,
            rawContent: content.substring(lastIndex, match.index)
        });
        currentHeaderFull = match[0];
        currentTitle = match[2].trim();
        currentTitleKey = null;
        lastIndex = match.index + match[0].length;
    }
    sections.push({
        title: currentTitle,
        titleKey: currentTitleKey,
        fullHeader: currentHeaderFull,
        rawContent: content.substring(lastIndex)
    });
    
    // Group empty structural headers with the following section to hide empty tabs
    const groupedSections = [];
    let pendingHeader = "";
    sections.forEach((sec, idx) => {
        const cleanContent = sec.rawContent
            .replace(/---DYNAMIC_STATE_START---|---DYNAMIC_STATE_END---/g, '')
            .replace(/^\s*\(.*?\)\s*$/gm, '')
            .trim();
        if (!cleanContent && idx < sections.length - 1) {
            pendingHeader += sec.fullHeader + sec.rawContent;
        } else {
            groupedSections.push({
                title: sec.title,
                titleKey: sec.titleKey,
                fullHeader: pendingHeader + sec.fullHeader,
                rawContent: sec.rawContent,
                id: 'edit-textarea-' + idx
            });
            pendingHeader = "";
        }
    });
    
    window.NovelContextUI.editSections = groupedSections;
    window.NovelContextUI.isEditing = true;
    
    // Render the edit UI while keeping the tabs
    header.style.display = 'flex';
    header.style.flexWrap = 'wrap';
    header.style.gap = '0.5rem';
    header.innerHTML = '';
    body.innerHTML = '';
    
    const activeIdx = Number.isInteger(window.NovelContextUI.activeTabIndex)
        && window.NovelContextUI.activeTabIndex >= 0
        && window.NovelContextUI.activeTabIndex < groupedSections.length
        ? window.NovelContextUI.activeTabIndex
        : 0;
    window.NovelContextUI.activeTabIndex = activeIdx;
    
    function checkForChanges() {
        let isChanged = false;
        groupedSections.forEach(sec => {
            const textarea = document.getElementById(sec.id);
            if (textarea && textarea.value !== sec.rawContent) {
                isChanged = true;
            }
        });
        if (btnSave) btnSave.disabled = !isChanged;
    }
    
    groupedSections.forEach((sec, idx) => {
        const btn = document.createElement('button');
        const isActive = idx === activeIdx;
        btn.className = `btn btn-sm ${isActive ? 'btn-primary' : 'btn-secondary'}`;
        btn.style.whiteSpace = 'nowrap';
        btn.style.flexShrink = '0';
        btn.style.borderRadius = '4px';
        btn.style.padding = '0.25rem 0.75rem';
        btn.textContent = sec.title;
        if (sec.titleKey) {
            btn.setAttribute('data-i18n', sec.titleKey);
        }
        
        const pane = document.createElement('div');
        pane.style.display = isActive ? 'block' : 'none';
        pane.style.height = '100%';
        
        const textarea = document.createElement('textarea');
        textarea.id = sec.id;
        textarea.className = 'form-control';
        textarea.style.width = '100%';
        textarea.style.height = '60vh';
        textarea.style.minHeight = '400px';
        textarea.style.resize = 'vertical';
        textarea.style.fontFamily = 'monospace';
        textarea.style.fontSize = '0.85rem';
        textarea.style.padding = '15px';
        textarea.style.border = '2px solid var(--primary-light)';
        textarea.style.borderRadius = '8px';
        textarea.style.boxShadow = 'inset 0 2px 8px rgba(0,0,0,0.05)';
        textarea.style.backgroundColor = 'var(--bg-light)';
        textarea.value = sec.rawContent;
        textarea.addEventListener('input', checkForChanges);
        
        pane.appendChild(textarea);
        
        btn.onclick = () => {
            window.NovelContextUI.activeTabIndex = idx;
            Array.from(header.children).forEach(c => {
                c.className = 'btn btn-sm btn-secondary';
                c.style.background = 'transparent';
                c.style.color = 'var(--text-dark)';
                c.style.border = '1px solid var(--border-color)';
            });
            btn.className = 'btn btn-sm btn-primary';
            btn.style.background = 'var(--primary-color)';
            btn.style.color = '#fff';
            btn.style.border = '1px solid var(--primary-color)';
            
            Array.from(body.children).forEach(p => { p.style.display = 'none'; });
            pane.style.display = 'block';
        };
        
        if (isActive) {
            btn.style.background = 'var(--primary-color)';
            btn.style.color = '#fff';
            btn.style.border = '1px solid var(--primary-color)';
        } else {
            btn.style.background = 'transparent';
            btn.style.color = 'var(--text-dark)';
            btn.style.border = '1px solid var(--border-color)';
        }
        
        header.appendChild(btn);
        body.appendChild(pane);
    });
    
    // Swap buttons
    if (btnEdit) btnEdit.style.display = 'none';
    if (btnSave) {
        btnSave.style.display = 'inline-flex';
        btnSave.disabled = true; // disabled until changed
    }
    if (btnCancel) btnCancel.style.display = 'inline-flex';
};

window.cancelContextEdit = function() {
    window.NovelContextUI.isEditing = false;
    window.NovelContextUI.editSections = null;
    const btnEdit = document.getElementById('btnEditResync');
    const btnSave = document.getElementById('btnSaveResync');
    const btnCancel = document.getElementById('btnCancelResync');
    
    if (btnEdit) btnEdit.style.display = 'inline-flex';
    if (btnSave) btnSave.style.display = 'none';
    if (btnCancel) btnCancel.style.display = 'none';
    
    if (window.NovelContextUI.localizedView) {
        window.NovelContextUI._renderLocalizedView();
    } else if (window.NovelContextUI.displayedContent) {
        window.NovelContextUI.renderContextTabs(window.NovelContextUI.displayedContent, true);
    }
};

window.saveContextResync = async function() {
    const selector = document.getElementById('contextChunkSelector');
    if (!selector || selector.value === 'latest') {
        logContextResyncFailure('translation:context_resync_unavailable');
        return;
    }
    
    let newContent = "";
    if (window.NovelContextUI.editSections) {
        window.NovelContextUI.editSections.forEach(sec => {
            const textarea = document.getElementById(sec.id);
            if (textarea) {
                newContent += sec.fullHeader + textarea.value;
            } else {
                newContent += sec.fullHeader + sec.rawContent;
            }
        });
    } else {
        const textarea = document.getElementById('contextResyncEditor');
        if (!textarea) {
            logContextResyncFailure('translation:context_resync_unavailable');
            return;
        }
        newContent = textarea.value;
    }
    
    const isGlobal = selector.value === 'global';
    const chunkIndex = isGlobal
        ? window.NovelContextUI.globalAnchorChunkIndex
        : parseInt(selector.value);
    if (!Number.isInteger(chunkIndex)) {
        logContextResyncFailure('translation:context_resync_unavailable');
        return;
    }

    let submittedContent = newContent;
    if (isGlobal) {
        const anchorContent = window.NovelContextUI.globalAnchorFullContent || '';
        const dynamicMarker = anchorContent.indexOf('---DYNAMIC_STATE_START---');
        if (dynamicMarker < 0) {
            logContextResyncFailure('translation:context_global_anchor_missing');
            return;
        }
        submittedContent = (
            `${newContent.trim()}\n\n${anchorContent.substring(dynamicMarker)}`
        );
    }
    
    let translationId = null;
    const currentJob = StateManager.getState('translation.currentJob');
    if (currentJob && currentJob.translationId) {
        translationId = currentJob.translationId;
    }
    if (!translationId) {
        translationId = StateManager.getState('translation.lastJobId');
    }
    
    if (!translationId) {
        console.warn('[Context] No translationId available for resync');
        logContextResyncFailure('translation:context_no_job_body');
        return;
    }
    
    try {
        const btnSave = document.getElementById('btnSaveResync');
        if (btnSave) btnSave.disabled = true;
        
        const resyncResult = await ApiClient.resyncContextSnapshot(
            translationId,
            chunkIndex,
            submittedContent,
            {
                scope: isGlobal ? 'global_lore' : 'snapshot'
            }
        );
        updateContextResyncControls(resyncResult?.resync_state || {
            status: 'running'
        });
        MessageLogger.addLog(
            isGlobal
                ? t('translation:context_global_resync_started_log')
                : t('translation:context_resync_started_log', {
                    chunk: chunkIndex + 1
                })
        );
        
        // Reload tabs
        window.NovelContextUI.isEditing = false;
        window.NovelContextUI.editSections = null;
        const header = document.getElementById('contextTabsHeader');
        if (header) header.style.display = 'flex';
        // Keep displaying the edited historical snapshot, but do not replace
        // the separately tracked latest state. The resync worker will emit the
        // new canonical latest context when its forward pass completes.
        if (isGlobal) {
            window.NovelContextUI.globalAnchorFullContent = submittedContent;
        }
        window.NovelContextUI.renderContextTabs(newContent, true);
        
        const btnEdit = document.getElementById('btnEditResync');
        if (btnEdit) btnEdit.style.display = 'inline-flex';
        if (btnSave) {
            btnSave.style.display = 'none';
            btnSave.disabled = true;
        }
        const btnCancel = document.getElementById('btnCancelResync');
        if (btnCancel) btnCancel.style.display = 'none';
        
    } catch (e) {
        console.error("Failed to start resync:", e);
        MessageLogger.addLog(t('translation:context_resync_failed_log', { error: e.message }));
        const btnSave = document.getElementById('btnSaveResync');
        if (btnSave) btnSave.disabled = false;
    }
};
