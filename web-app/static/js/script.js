let pollInterval;

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

        function openAboutModal() {
            document.getElementById('aboutModal').style.display = 'block';
        }

        function closeAboutModal() {
            document.getElementById('aboutModal').style.display = 'none';
        }

        function openResultsModal() {
            document.getElementById('resultsModal').style.display = 'block';
        }

        function closeResultsModal() {
            document.getElementById('resultsModal').style.display = 'none';
            if (pollInterval) {
                clearInterval(pollInterval);
            }
        }

        // Close modals when clicking outside
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
            analyzeBtn.textContent = 'Analyzing...';

            openResultsModal();
            showLoading('Starting analysis...');

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

                // Poll for results
                pollInterval = setInterval(async () => {
                    try {
                        const statusResponse = await fetch(`/api/status/${sessionId}`);
                        const status = await statusResponse.json();

                        if (status.status === 'loading') {
                            showLoading(status.message);
                        } else if (status.status === 'complete') {
                            clearInterval(pollInterval);
                            displayResults(status.data);
                            analyzeBtn.disabled = false;
                            analyzeBtn.textContent = 'Analyze Reviews';
                        } else if (status.status === 'error') {
                            clearInterval(pollInterval);
                            showError(status.message);
                            analyzeBtn.disabled = false;
                            analyzeBtn.textContent = 'Analyze Reviews';
                        }
                    } catch (error) {
                        clearInterval(pollInterval);
                        showLoadingError('Error checking status: ' + error.message);
                        analyzeBtn.disabled = false;
                        analyzeBtn.textContent = 'Analyze Reviews';
                    }
                }, 2000);

            } catch (error) {
                showLoadingError('Error: ' + error.message);
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analyze Reviews';
            }
        }

        function showLoading(message) {
            document.getElementById('loadingMessage').textContent = message;
        }

        function showLoadingError(message) {
            const resultsContent = document.getElementById('resultsContent');
            resultsContent.innerHTML = `
                <div style="text-align: center; padding: 2rem; color: #f44336;">
                    <h3>❌ Error</h3>
                    <p>${message}</p>
                </div>
            `;
        }

        function displayResults(data) {
            const { metadata, samples } = data;
            const resultsContent = document.getElementById('resultsContent');

            const positivePercent = (metadata.positive_count / metadata.total_reviews * 100).toFixed(1);
            const negativePercent = (metadata.negative_count / metadata.total_reviews * 100).toFixed(1);
            const neutralPercent = (metadata.neutral_count / metadata.total_reviews * 100).toFixed(1);

            // Determine overall sentiment
            let overallSentiment = 'Neutral';
            let sentimentEmoji = '😐';
            if (metadata.positive_count > metadata.negative_count && metadata.positive_count > metadata.neutral_count) {
                overallSentiment = 'Positive';
                sentimentEmoji = '😊';
            } else if (metadata.negative_count > metadata.positive_count && metadata.negative_count > metadata.neutral_count) {
                overallSentiment = 'Negative';
                sentimentEmoji = '😞';
            }

            let html = `
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">${metadata.total_reviews}</div>
                        <div class="stat-label">Total Reviews</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #4caf50;">${metadata.positive_count}</div>
                        <div class="stat-label">Positive</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #f44336;">${metadata.negative_count}</div>
                        <div class="stat-label">Negative</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #ff9800;">${metadata.neutral_count}</div>
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
                <div class="stat-card" style="margin-bottom: 2rem;">
                    <div class="stat-value">⭐ ${metadata.average_rating}</div>
                    <div class="stat-label">Average Rating</div>
                </div>
                ` : ''}

                <div class="conclusion">
                    <h3>Overall Sentiment</h3>
                    <div class="conclusion-sentiment">${sentimentEmoji} ${overallSentiment}</div>
                    <p>Based on analysis of ${metadata.total_reviews} reviews with ${(metadata.average_confidence * 100).toFixed(1)}% average confidence</p>
                </div>
            `;

            // Add sample reviews
            if (samples.positive.length > 0) {
                html += '<div class="reviews-section"><h3>📈 Sample Positive Reviews</h3>';
                samples.positive.forEach(review => {
                    html += createReviewCard(review, 'positive');
                });
                html += '</div>';
            }

            if (samples.negative.length > 0) {
                html += '<div class="reviews-section"><h3>📉 Sample Negative Reviews</h3>';
                samples.negative.forEach(review => {
                    html += createReviewCard(review, 'negative');
                });
                html += '</div>';
            }

            if (samples.neutral.length > 0) {
                html += '<div class="reviews-section"><h3>➖ Sample Neutral Reviews</h3>';
                samples.neutral.forEach(review => {
                    html += createReviewCard(review, 'neutral');
                });
                html += '</div>';
            }

            resultsContent.innerHTML = html;
        }

        function createReviewCard(review, sentimentClass) {
            const ratingStars = review.rating ? '⭐'.repeat(Math.round(review.rating)) : 'N/A';
            const confidencePercent = (review.confidence * 100).toFixed(1);
            
            let reviewText = review.review_text;
            if (reviewText.length > 300) {
                reviewText = reviewText.substring(0, 300) + '...';
            }

            return `
                <div class="review-card ${sentimentClass}">
                    <div class="review-header">
                        <div class="review-rating">${ratingStars}</div>
                        <div class="review-confidence">${confidencePercent}% confident</div>
                    </div>
                    ${review.title ? `<div class="review-title">${review.title}</div>` : ''}
                    <div class="review-text">${reviewText}</div>
                    <div class="review-meta">
                        ${review.reviewer_name ? `<span>👤 ${review.reviewer_name}</span>` : ''}
                        ${review.date ? `<span>📅 ${review.date}</span>` : ''}
                        ${review.verified_purchase ? '<span>✓ Verified Purchase</span>' : ''}
                    </div>
                </div>
            `;
        }