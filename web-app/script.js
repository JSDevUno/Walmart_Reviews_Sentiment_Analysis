// DOM Elements
const urlInput = document.getElementById('urlInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const errorMessage = document.getElementById('errorMessage');
const inputSection = document.getElementById('inputSection');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
const aboutBtn = document.getElementById('aboutBtn');
const aboutModal = document.getElementById('aboutModal');
const closeModal = document.querySelector('.close');

// Modal functionality
aboutBtn.addEventListener('click', () => {
    aboutModal.style.display = 'block';
});

closeModal.addEventListener('click', () => {
    aboutModal.style.display = 'none';
});

window.addEventListener('click', (e) => {
    if (e.target === aboutModal) {
        aboutModal.style.display = 'none';
    }
});

// URL Validation
function validateWalmartURL(url) {
    const patterns = [
        /walmart\.com\/ip\/.+\/\d+/,
        /walmart\.com\/.+\/\d+/
    ];
    
    return patterns.some(pattern => pattern.test(url));
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('show');
}

function hideError() {
    errorMessage.classList.remove('show');
}

// Analyze button handler
analyzeBtn.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    
    hideError();
    
    if (!url) {
        showError('Please enter a Walmart product URL');
        return;
    }
    
    if (!validateWalmartURL(url)) {
        showError('Invalid Walmart URL. Please use format: walmart.com/ip/product-name/123456');
        return;
    }
    
    // Show loading
    inputSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    loadingSection.classList.remove('hidden');
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        // Display results
        displayResults(data);
        
        loadingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        
    } catch (error) {
        loadingSection.classList.add('hidden');
        inputSection.classList.remove('hidden');
        showError(error.message || 'An error occurred during analysis');
    }
});

// Display results
function displayResults(data) {
    // Update statistics
    document.getElementById('totalReviews').textContent = data.total_reviews;
    document.getElementById('positiveCount').textContent = data.sentiment_distribution.positive;
    document.getElementById('positivePercent').textContent = data.sentiment_percentages.positive + '%';
    document.getElementById('negativeCount').textContent = data.sentiment_distribution.negative;
    document.getElementById('negativePercent').textContent = data.sentiment_percentages.negative + '%';
    document.getElementById('neutralCount').textContent = data.sentiment_distribution.neutral;
    document.getElementById('neutralPercent').textContent = data.sentiment_percentages.neutral + '%';
    
    // Update rating
    document.getElementById('avgRating').textContent = data.average_rating.toFixed(1);
    
    // Display rating distribution
    displayRatingBars(data.rating_distribution, data.total_reviews);
    
    // Display sentiment chart
    displaySentimentChart(data.sentiment_distribution);
    
    // Display sample reviews
    displaySampleReviews('positiveSamples', data.sample_reviews.positive);
    displaySampleReviews('negativeSamples', data.sample_reviews.negative);
    displaySampleReviews('neutralSamples', data.sample_reviews.neutral);
}

// Display rating bars
function displayRatingBars(distribution, total) {
    const container = document.getElementById('ratingBars');
    container.innerHTML = '';
    
    for (let star = 5; star >= 1; star--) {
        const count = distribution[star.toString()] || 0;
        const percentage = total > 0 ? (count / total) * 100 : 0;
        
        const barDiv = document.createElement('div');
        barDiv.className = 'rating-bar';
        barDiv.innerHTML = `
            <span class="rating-bar-label">${star} ⭐</span>
            <div class="rating-bar-bg">
                <div class="rating-bar-fill" style="width: ${percentage}%"></div>
            </div>
            <span class="rating-bar-count">${count}</span>
        `;
        container.appendChild(barDiv);
    }
}

// Display sentiment chart (simple bar chart)
function displaySentimentChart(distribution) {
    const canvas = document.getElementById('sentimentChart');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = 400;
    canvas.height = 300;
    
    const data = [
        { label: 'Positive', value: distribution.positive, color: '#84fab0' },
        { label: 'Negative', value: distribution.negative, color: '#fa709a' },
        { label: 'Neutral', value: distribution.neutral, color: '#a1c4fd' }
    ];
    
    const maxValue = Math.max(...data.map(d => d.value));
    const barWidth = 80;
    const barSpacing = 60;
    const startX = 50;
    const chartHeight = 200;
    const startY = 250;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw bars
    data.forEach((item, index) => {
        const barHeight = maxValue > 0 ? (item.value / maxValue) * chartHeight : 0;
        const x = startX + index * (barWidth + barSpacing);
        const y = startY - barHeight;
        
        // Draw bar
        ctx.fillStyle = item.color;
        ctx.fillRect(x, y, barWidth, barHeight);
        
        // Draw value on top
        ctx.fillStyle = '#333';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(item.value, x + barWidth / 2, y - 10);
        
        // Draw label
        ctx.font = '14px Arial';
        ctx.fillText(item.label, x + barWidth / 2, startY + 20);
    });
}

// Display sample reviews
function displaySampleReviews(containerId, reviews) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    if (!reviews || reviews.length === 0) {
        container.innerHTML = '<div class="no-reviews">No reviews available for this sentiment</div>';
        return;
    }
    
    reviews.forEach(review => {
        const reviewCard = document.createElement('div');
        reviewCard.className = 'review-card';
        
        const stars = '⭐'.repeat(Math.floor(review.rating || 0));
        const verifiedBadge = review.verified_purchase 
            ? '<span class="verified-badge">✓ Verified</span>' 
            : '';
        
        reviewCard.innerHTML = `
            <div class="review-header">
                <span class="reviewer-name">${escapeHtml(review.reviewer_name || 'Anonymous')}</span>
                <span class="review-rating">${stars}</span>
            </div>
            ${review.title ? `<div class="review-title">${escapeHtml(review.title)}</div>` : ''}
            <div class="review-text">${escapeHtml(review.review_text || '')}</div>
            <div class="review-footer">
                <span class="review-date">${escapeHtml(review.date || 'No date')}</span>
                ${verifiedBadge}
            </div>
        `;
        
        container.appendChild(reviewCard);
    });
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Analyze another product
analyzeAnotherBtn.addEventListener('click', () => {
    urlInput.value = '';
    hideError();
    resultsSection.classList.add('hidden');
    inputSection.classList.remove('hidden');
});

// Allow Enter key to submit
urlInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        analyzeBtn.click();
    }
});