/**
 * Resume Manager - Resumable jobs management
 *
 * Handles loading, resuming, and deleting interrupted translation checkpoints.
 * Manages resumable jobs UI and state synchronization.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { MessageLogger } from '../ui/message-logger.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { ProgressManager } from './progress-manager.js';
import { t, getCurrentLocale } from '../i18n/i18n.js';

/**
 * Format resumable job card HTML
 * @param {Object} job - Job data
 * @param {boolean} hasActiveTranslation - Whether there's an active translation
 * @param {string} activeNames - Names of active translations
 * @returns {string} HTML for job card
 */
function formatJobCard(job, hasActiveTranslation, activeNames) {
    const progress = job.progress || {};
    const completedChunks = progress.completed_chunks || 0;
    const totalChunks = progress.total_chunks || 0;
    const failedChunks = progress.failed_chunks || 0;
    const progressPercent = job.progress_percentage || 0;
    const fileType = (job.file_type || 'txt').toUpperCase();
    const isPartial = job.status === 'partial';

    const failedBadgeLabel = t('translation:job_card_failed_badge', { count: failedChunks });
    const statusBadge = isPartial
        ? `<span style="display: inline-block; margin-left: 8px; padding: 2px 8px; font-size: 11px; font-weight: 600; color: #92400e; background: #fef3c7; border: 1px solid #f59e0b; border-radius: 4px;" title="${t('translation:job_card_partial_title', { count: failedChunks })}">${failedBadgeLabel}</span>`
        : (failedChunks > 0
            ? `<span style="display: inline-block; margin-left: 8px; padding: 2px 8px; font-size: 11px; font-weight: 600; color: #92400e; background: #fef3c7; border: 1px solid #f59e0b; border-radius: 4px;" title="${t('translation:job_card_failed_title', { count: failedChunks })}">${failedBadgeLabel}</span>`
            : '');

    const naText = t('translation:job_card_na');
    const dateLocale = getCurrentLocale();
    const createdDate = job.created_at ? new Date(job.created_at).toLocaleString(dateLocale) : naText;
    const pausedDate = job.paused_at ? new Date(job.paused_at).toLocaleString(dateLocale) :
                       job.updated_at ? new Date(job.updated_at).toLocaleString(dateLocale) : naText;

    // Extract original filename (remove 16-char hash prefix + underscore)
    const unknownText = t('translation:job_card_unknown');
    const inputFilename = job.input_filename || unknownText;
    const outputFilename = job.output_filename || unknownText;

    // Extract hash and original name from input filename
    const inputMatch = inputFilename.match(/^([a-f0-9]{16})_(.+)$/);
    const inputHash = inputMatch ? inputMatch[1] : null;
    const inputOriginalName = inputMatch ? inputMatch[2] : inputFilename;

    // Format the display name (capitalize first letter, remove extension for display)
    const displayName = inputOriginalName.replace(/\.[^.]+$/, '');
    const displayNameFormatted = displayName.charAt(0).toUpperCase() + displayName.slice(1);

    const idValue = inputHash || job.translation_id.replace('trans_', '');
    const typeIdLine = t('translation:job_card_type_id', { type: fileType, id: idValue });
    const resumeTitle = hasActiveTranslation
        ? t('translation:cannot_resume_in_progress_title')
        : t('translation:resume_btn_title');

    return `
        <div class="resumable-job-card" style="border: 1px solid #e5e7eb; padding: 20px; margin-bottom: 15px; border-radius: 8px; background: #f9fafb;">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px; gap: 15px;">
                <div style="flex: 1; min-width: 0;">
                    <div style="font-size: 18px; font-weight: 600; color: #1f2937; margin-bottom: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${DomHelpers.escapeHtml(displayNameFormatted)}">
                        ${DomHelpers.escapeHtml(displayNameFormatted)}
                    </div>
                    <div style="font-size: 14px; color: #6b7280; margin-bottom: 5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="→ ${DomHelpers.escapeHtml(outputFilename)}">
                        → ${DomHelpers.escapeHtml(outputFilename)}
                    </div>
                    <div style="font-size: 12px; color: #9ca3af; margin-top: 8px;">
                        ${typeIdLine}${statusBadge}
                    </div>
                </div>

                <div style="display: flex; gap: 10px; flex-shrink: 0;">
                    <button class="btn btn-primary" onclick="resumeJob('${job.translation_id}')"
                            title="${resumeTitle}"
                            ${hasActiveTranslation ? 'disabled style="opacity: 0.5; cursor: not-allowed;"' : ''}>
                        ${t('translation:job_card_resume_btn')}
                    </button>
                    <button class="btn btn-danger" onclick="deleteCheckpoint('${job.translation_id}')" title="${t('translation:job_card_delete_title')}">
                        ${t('translation:job_card_delete_btn')}
                    </button>
                </div>
            </div>

            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 13px; color: #6b7280; margin-bottom: 5px;">
                    <span>${t('translation:job_card_progress', { completed: completedChunks, total: totalChunks, percent: progressPercent })}</span>
                </div>
                <div style="width: 100%; background: #e5e7eb; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="width: ${progressPercent}%; background: #3b82f6; height: 100%; transition: width 0.3s;"></div>
                </div>
            </div>

            <div style="display: flex; gap: 20px; font-size: 12px; color: #9ca3af;">
                <span>${t('translation:job_card_created', { date: createdDate })}</span>
                <span>${t('translation:job_card_paused', { date: pausedDate })}</span>
            </div>
        </div>
    `;
}

/**
 * Create warning banner HTML if active translations exist
 * @param {Array} activeJobs - Active translation jobs
 * @returns {string} Warning banner HTML or empty string
 */
function createWarningBanner(activeJobs) {
    if (!activeJobs || activeJobs.length === 0) return '';

    const activeNames = activeJobs.map(job => job.output_filename || t('translation:job_card_unknown')).join(', ');

    return `
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
}

export const ResumeManager = {
    /**
     * Load and display resumable jobs
     */
    async loadResumableJobs() {
        const section = DomHelpers.getElement('resumableJobsSection');
        const loading = DomHelpers.getElement('resumableJobsLoading');
        const listContainer = DomHelpers.getElement('resumableJobsList');
        const emptyMessage = DomHelpers.getElement('resumableJobsEmpty');

        // Show loading, hide list and empty message (use inline style to override)
        if (loading) loading.style.display = 'block';
        if (listContainer) listContainer.style.display = 'none';
        if (emptyMessage) emptyMessage.style.display = 'none';

        try {
            const data = await ApiClient.getResumableJobs();
            const jobs = data.resumable_jobs || [];

            // Get active translation state
            const hasActiveTranslation = StateManager.getState('translation.hasActive') || false;
            const activeJobs = StateManager.getState('translation.activeJobs') || [];

            // Hide loading
            if (loading) loading.style.display = 'none';

            if (jobs.length === 0) {
                // Hide section if no jobs (use inline style to override)
                if (section) section.style.display = 'none';
                if (emptyMessage) emptyMessage.style.display = 'block';
                return;
            }

            // Show section and populate jobs (use inline style to override)
            if (section) section.style.display = 'block';
            if (listContainer) listContainer.style.display = 'block';

            // Build warning banner if active translation exists
            const warningBanner = createWarningBanner(hasActiveTranslation ? activeJobs : null);

            // Build jobs HTML
            const jobsHtml = jobs.map(job => formatJobCard(job, hasActiveTranslation, activeJobs)).join('');

            if (!listContainer) {
                console.error('Error: resumableJobsList element not found');
                return;
            }

            listContainer.innerHTML = warningBanner + jobsHtml;

            MessageLogger.addLog(t('translation:paused_count_log', { count: jobs.length }));

        } catch (error) {
            // Hide loading, show error message
            if (loading) loading.style.display = 'none';
            if (emptyMessage) {
                emptyMessage.style.display = 'block';
                emptyMessage.innerHTML = `<p style="color: #ef4444;">${t('translation:paused_load_error', { error: DomHelpers.escapeHtml(error.message) })}</p>`;
            }
            // Hide section on error
            if (section) section.style.display = 'none';
            console.error('Error loading resumable jobs:', error);
        }
    },

    /**
     * Resume a paused translation job
     * @param {string} translationId - Translation ID to resume
     */
    async resumeJob(translationId) {
        // Check if there's an active translation
        const hasActive = StateManager.getState('translation.hasActive') || false;
        const activeJobs = StateManager.getState('translation.activeJobs') || [];

        if (hasActive) {
            const activeNames = activeJobs.map(job => job.output_filename || t('translation:job_card_unknown')).join(', ');
            MessageLogger.showMessage(
                t('translation:cannot_resume_active', { names: activeNames }),
                'error'
            );
            return;
        }

        if (!confirm(t('translation:confirm_resume'))) {
            return;
        }

        try {
            MessageLogger.addLog(t('translation:resuming_log', { id: translationId }));
            MessageLogger.showMessage(t('translation:resuming_msg'), 'info');

            const data = await ApiClient.resumeJob(translationId);

            MessageLogger.showMessage(
                t('translation:resume_success', { chunk: data.resume_from_chunk }),
                'success'
            );
            MessageLogger.addLog(t('translation:resume_success_log', { id: translationId, chunk: data.resume_from_chunk }));

            // Fetch job details to get filename and file type
            const jobData = await ApiClient.getTranslationStatus(translationId);

            // Set up current processing job in state
            StateManager.setState('translation.currentJob', {
                translationId: translationId,
                fileRef: {
                    name: jobData.config?.output_filename || t('translation:resumed_translation_default'),
                    fileType: jobData.config?.file_type || 'txt'
                }
            });

            // Mark as batch active
            StateManager.setState('translation.isBatchActive', true);

            // Show progress section
            ProgressManager.show();
            const progressSection = DomHelpers.getElement('progressSection');
            if (progressSection) {
                progressSection.scrollIntoView({ behavior: 'smooth' });
            }

            // Update title with actual filename
            const fileName = jobData.config?.output_filename || t('translation:resumed_translation_default');
            DomHelpers.setText('currentFileProgressTitle', t('translation:resuming_file', { name: fileName }));

            // Show stats grid
            DomHelpers.show('statsGrid');

            // Show interrupt button
            const interruptBtn = DomHelpers.getElement('interruptBtn');
            if (interruptBtn) {
                DomHelpers.show('interruptBtn');
                interruptBtn.disabled = false;
            }

            // Initialize progress
            ProgressManager.updateProgress(jobData.progress || 0);

            // Emit event for translation started
            const event = new CustomEvent('translationResumed', { detail: { translationId, jobData } });
            window.dispatchEvent(event);

            // Refresh resumable jobs list after a delay
            setTimeout(() => {
                this.loadResumableJobs();
            }, 1000);

        } catch (error) {
            // Enhanced error message for active translation conflicts
            if (error.status === 409 && error.data?.active_translations) {
                const activeList = error.data.active_translations
                    .map(item => `• ${item.output_filename} (${item.status})`)
                    .join('\n');
                MessageLogger.showMessage(
                    t('translation:cannot_resume_with_list', { list: activeList }),
                    'error'
                );
                MessageLogger.addLog(`⚠️ ${error.data.message}`);
            } else {
                MessageLogger.showMessage(t('translation:resume_error', { error: error.message }), 'error');
                MessageLogger.addLog(t('translation:resume_network_error_log', { error: error.message }));
            }
            console.error('Error resuming job:', error);
        }
    },

    /**
     * Delete a checkpoint
     * @param {string} translationId - Translation ID to delete
     */
    async deleteCheckpoint(translationId) {
        if (!confirm(t('translation:confirm_delete_checkpoint'))) {
            return;
        }

        try {
            MessageLogger.addLog(t('translation:deleting_checkpoint_log', { id: translationId }));

            await ApiClient.deleteCheckpoint(translationId);

            MessageLogger.showMessage(t('translation:checkpoint_deleted'), 'success');
            MessageLogger.addLog(t('translation:checkpoint_deleted_log', { id: translationId }));

            // Refresh resumable jobs list
            this.loadResumableJobs();

        } catch (error) {
            MessageLogger.showMessage(t('translation:checkpoint_delete_error', { error: error.message }), 'error');
            MessageLogger.addLog(t('translation:resume_network_error_log', { error: error.message }));
            console.error('Error deleting checkpoint:', error);
        }
    },

    /**
     * Initialize resume manager
     */
    initialize() {
        // Load resumable jobs on initialization
        this.loadResumableJobs();

        // Listen for translation state changes
        StateManager.subscribe('translation.hasActive', (hasActive) => {
            // Refresh job list when active state changes
            this.loadResumableJobs();
        });
    }
};
