let pollInterval;
let currentTheme = 'light';

// Theme Management
const themeToggle = document.getElementById('themeToggle');
const body = document.body;

function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
}

function setTheme(theme) {
    currentTheme = theme;
    body.setAttribute('data-theme', theme);
    body.className = theme + '-mode';
    localStorage.setItem('theme', theme);
}

// Initialize theme on load
initTheme();

// Desktop theme toggle
if (themeToggle) {
    themeToggle.addEventListener('click', () => {
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
    });
}

// Mobile theme toggle
document.querySelector('.theme-toggle-mobile').addEventListener('click', () => {
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
});

// Side Navigation Functions
function toggleSideNav() {
    const sideNav = document.getElementById('sideNav');
    const overlay = document.getElementById('sideNavOverlay');
    sideNav.classList.toggle('active');
    overlay.classList.toggle('active');
}

function closeSideNav() {
    const sideNav = document.getElementById('sideNav');
    const overlay = document.getElementById('sideNavOverlay');
    sideNav.classList.remove('active');
    overlay.classList.remove('active');
}

// Event listeners for side nav
document.getElementById('burgerMenu').addEventListener('click', toggleSideNav);
document.getElementById('sideNavOverlay').addEventListener('click', closeSideNav);
document.querySelector('.close-side-nav').addEventListener('click', closeSideNav);

// Validation and Error Handling
function validateWalmartURL(url) {
    const pattern = /walmart\.com/i;
    return pattern.test(url);
}

function showError(message) {
    const input = document.getElementById('urlInput');
    const errorDiv = document.getElementById('errorMessage');
    input.classList.add('error');
    errorDiv.textContent = message;
}

function clearError() {
    const input = document.getElementById('urlInput');
    const errorDiv = document.getElementById('errorMessage');
    input.classList.remove('error');
    errorDiv.textContent = '';
}

// Modal Functions
function openAboutModal() {
    document.getElementById('aboutModal').style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeAboutModal() {
    document.getElementById('aboutModal').style.display = 'none';
    document.body.style.overflow = 'auto';
}

function openResultsModal() {
    document.getElementById('resultsModal').style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeResultsModal() {
    document.getElementById('resultsModal').style.display = 'none';
    document.body.style.overflow = 'auto';
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
    resetResultsModal();
}

function resetResultsModal() {
    const resultsContent = document.getElementById('resultsContent');
    resultsContent.innerHTML = `
        <div class="loading-state">
            <div class="spinner"></div>
            <p id="loadingMessage">Initializing browser...</p>
        </div>
    `;
}

window.onclick = function(event) {
    const aboutModal = document.getElementById('aboutModal');
    const resultsModal = document.getElementById('resultsModal');
    if (event.target === aboutModal) {
        closeAboutModal();
    }
    if (event.target === resultsModal) {
        closeResultsModal();
    }
}

document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        closeAboutModal();
        closeResultsModal();
        closeSideNav();
    }
});

// Analyze Reviews Function
async function analyzeReviews() {
    clearError();
    
    const url = document.getElementById('urlInput').value.trim();
    
    if (!url) {
        showError('Please enter a Walmart product URL');
        return;
    }

    if (!validateWalmartURL(url)) {
        showError('Please enter a valid Walmart URL (must contain "walmart.com")');
        return;
    }

    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Analyzing...</span>';

    openResultsModal();
    showLoading('Initializing browser...');

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url, max_reviews: 50 })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Analysis failed');
        }

        const data = await response.json();
        const sessionId = data.session_id;

        let pollAttempts = 0;
        const maxPollAttempts = 150;
        
        await new Promise(resolve => setTimeout(resolve, 1000));

        pollInterval = setInterval(async () => {
            pollAttempts++;

            if (pollAttempts > maxPollAttempts) {
                clearInterval(pollInterval);
                pollInterval = null;
                showLoadingError('Analysis timed out. Please try again.');
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="fas fa-rocket"></i> <span>Analyze Reviews</span>';
                return;
            }

            try {
                const statusResponse = await fetch(`/api/status/${sessionId}`);
                
                if (!statusResponse.ok) {
                    throw new Error('Failed to check status');
                }

                const status = await statusResponse.json();

                if (status.status === 'loading') {
                    showLoading(status.message);
                } else if (status.status === 'complete') {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    displayResults(status.data);
                    analyzeBtn.disabled = false;
                    analyzeBtn.innerHTML = '<i class="fas fa-rocket"></i> <span>Analyze Reviews</span>';
                } else if (status.status === 'error') {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    showLoadingError(status.message || 'An error occurred during analysis');
                    analyzeBtn.disabled = false;
                    analyzeBtn.innerHTML = '<i class="fas fa-rocket"></i> <span>Analyze Reviews</span>';
                } else if (status.status === 'not_found') {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    showLoadingError('Session not found. Please try again.');
                    analyzeBtn.disabled = false;
                    analyzeBtn.innerHTML = '<i class="fas fa-rocket"></i> <span>Analyze Reviews</span>';
                }
            } catch (error) {
                clearInterval(pollInterval);
                pollInterval = null;
                showLoadingError('Error checking status: ' + error.message);
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="fas fa-rocket"></i> <span>Analyze Reviews</span>';
            }
        }, 2000);

    } catch (error) {
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
        showLoadingError('Error: ' + error.message);
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-rocket"></i> <span>Analyze Reviews</span>';
    }
}

function showLoading(message) {
    const loadingMessageEl = document.getElementById('loadingMessage');
    if (loadingMessageEl) {
        loadingMessageEl.textContent = message;
    }
}

function showLoadingError(message) {
    const resultsContent = document.getElementById('resultsContent');
    resultsContent.innerHTML = `
        <div style="text-align: center; padding: 3rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1.5rem; color: #ef4444;">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <h3 style="color: var(--text-primary); margin-bottom: 1rem; font-weight: 700;">Error</h3>
            <p style="color: var(--text-secondary); line-height: 1.6; max-width: 500px; margin: 0 auto;">${message}</p>
            <button 
                onclick="closeResultsModal()" 
                class="btn-primary"
                style="margin-top: 2rem;"
            >
                <i class="fas fa-times"></i>
                Close
            </button>
        </div>
    `;
}

function displayResults(data) {
    const { metadata, samples } = data;
    const resultsContent = document.getElementById('resultsContent');

    const positivePercent = (metadata.positive_count / metadata.total_reviews * 100).toFixed(1);
    const negativePercent = (metadata.negative_count / metadata.total_reviews * 100).toFixed(1);
    const neutralPercent = (metadata.neutral_count / metadata.total_reviews * 100).toFixed(1);

    let overallSentiment = 'Neutral';
    if (metadata.positive_count > metadata.negative_count && metadata.positive_count > metadata.neutral_count) {
        overallSentiment = 'Positive';
    } else if (metadata.negative_count > metadata.positive_count && metadata.negative_count > metadata.neutral_count) {
        overallSentiment = 'Negative';
    }

    let html = `
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${metadata.total_reviews.toLocaleString()}</div>
                <div class="stat-label">Total Reviews</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #10b981;">${metadata.positive_count}</div>
                <div class="stat-label">Positive</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #ef4444;">${metadata.negative_count}</div>
                <div class="stat-label">Negative</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #f59e0b;">${metadata.neutral_count}</div>
                <div class="stat-label">Neutral</div>
            </div>
        </div>

        <div class="sentiment-bars">
            <div class="sentiment-bar">
                <div class="sentiment-bar-label">
                    <span>Positive</span>
                    <span>${positivePercent}%</span>
                </div>
                <div class="sentiment-bar-track">
                    <div class="sentiment-bar-fill positive-bar" style="width: ${positivePercent}%">
                        ${metadata.positive_count}
                    </div>
                </div>
            </div>
            <div class="sentiment-bar">
                <div class="sentiment-bar-label">
                    <span>Negative</span>
                    <span>${negativePercent}%</span>
                </div>
                <div class="sentiment-bar-track">
                    <div class="sentiment-bar-fill negative-bar" style="width: ${negativePercent}%">
                        ${metadata.negative_count}
                    </div>
                </div>
            </div>
            <div class="sentiment-bar">
                <div class="sentiment-bar-label">
                    <span>Neutral</span>
                    <span>${neutralPercent}%</span>
                </div>
                <div class="sentiment-bar-track">
                    <div class="sentiment-bar-fill neutral-bar" style="width: ${neutralPercent}%">
                        ${metadata.neutral_count}
                    </div>
                </div>
            </div>
        </div>

        ${metadata.average_rating ? `
        <div class="stat-card" style="margin-bottom: 2.5rem; grid-column: 1 / -1;">
            <div class="stat-value" style="color: #fbbf24;">⭐ ${metadata.average_rating.toFixed(1)}</div>
            <div class="stat-label">Average Rating</div>
        </div>
        ` : ''}

        <div class="conclusion">
            <h3>Overall Sentiment</h3>
            <div class="conclusion-sentiment">${overallSentiment}</div>
            <p>Analysis of ${metadata.total_reviews} reviews</p>
            <p style="opacity: 0.9; font-size: 0.95rem;">Average confidence: ${(metadata.average_confidence * 100).toFixed(1)}%</p>
        </div>
    `;

    if (samples.positive && samples.positive.length > 0) {
        html += '<div class="reviews-section"><h3><i class="fas fa-arrow-trend-up"></i> Top Positive Reviews</h3>';
        samples.positive.forEach(review => {
            html += createReviewCard(review, 'positive');
        });
        html += '</div>';
    }

    if (samples.negative && samples.negative.length > 0) {
        html += '<div class="reviews-section"><h3><i class="fas fa-arrow-trend-down"></i> Top Negative Reviews</h3>';
        samples.negative.forEach(review => {
            html += createReviewCard(review, 'negative');
        });
        html += '</div>';
    }

    if (samples.neutral && samples.neutral.length > 0) {
        html += '<div class="reviews-section"><h3><i class="fas fa-minus"></i> Top Neutral Reviews</h3>';
        samples.neutral.forEach(review => {
            html += createReviewCard(review, 'neutral');
        });
        html += '</div>';
    }

    resultsContent.innerHTML = html;
}

function createReviewCard(review, sentimentClass) {
    const confidencePercent = (review.confidence * 100).toFixed(1);
    
    let reviewText = review.review_text;
    if (reviewText.length > 300) {
        reviewText = reviewText.substring(0, 300) + '...';
    }

    return `
        <div class="review-card ${sentimentClass}">
            <div class="review-header">
                <div class="review-confidence">${confidencePercent}% confidence</div>
            </div>
            ${review.title ? `<div class="review-title">${review.title}</div>` : ''}
            <div class="review-text">${reviewText}</div>
            <div class="review-meta">
                ${review.reviewer_name ? `<span><i class="fas fa-user"></i> ${review.reviewer_name}</span>` : ''}
                ${review.date ? `<span><i class="fas fa-calendar"></i> ${review.date}</span>` : ''}
                ${review.verified_purchase ? '<span><i class="fas fa-check-circle"></i> Verified Purchase</span>' : ''}
                ${review.rating ? `<span><i class="fas fa-star"></i> ${review.rating}/5</span>` : ''}
            </div>
        </div>
    `;
}

// Enter key support
document.getElementById('urlInput').addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        analyzeReviews();
    }
});

// Analyze button click
document.getElementById('analyzeBtn').addEventListener('click', analyzeReviews);