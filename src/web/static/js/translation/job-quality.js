/** Presentation helpers for the quality state of a completed translation. */

import { t } from '../i18n/i18n.js';

export function deriveQualityStatus(resultData = {}) {
    if (resultData.quality_status) return resultData.quality_status;
    if ((resultData.stats?.review_required_chunks || 0) > 0) return 'review_required';
    return resultData.status === 'completed' ? 'passed' : 'not_checked';
}

export function buildReviewRequiredNotice(resultData = {}) {
    if (deriveQualityStatus(resultData) !== 'review_required') return null;

    const count = resultData.stats?.review_required_chunks || 0;
    const block = document.createElement('div');
    block.className = 'completion-card__warning';

    const heading = document.createElement('div');
    heading.className = 'completion-card__warning-heading';
    const icon = document.createElement('span');
    icon.className = 'material-symbols-outlined';
    icon.textContent = 'rate_review';
    heading.appendChild(icon);

    const text = document.createElement('span');
    text.textContent = t('translation:completion_review_required_notice', { count });
    heading.appendChild(text);
    block.appendChild(heading);
    return block;
}
