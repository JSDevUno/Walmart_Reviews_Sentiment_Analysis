import React, { useEffect } from 'react'
import SentimentChart from '../analysis/SentimentChart'
import ReviewCard from '../analysis/ReviewCard'

function ResultsModal({ isOpen, onClose, data }) {
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }

    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = 'unset'
    }

    return () => {
      document.removeEventListener('keydown', handleEscape)
      document.body.style.overflow = 'unset'
    }
  }, [isOpen, onClose])

  if (!isOpen || !data) return null

  const { metadata, samples } = data
  const { positive, negative, neutral } = samples

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal results-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Analysis Results</h2>
          <button 
            className="modal-close" 
            onClick={onClose}
            aria-label="Close modal"
          >
            <i className="fas fa-times"></i>
          </button>
        </div>
        
        <div className="modal-content">
          {/* Summary Statistics */}
          <div className="results-summary">
            <div className="stat-cards">
              <div className="stat-card">
                <div className="stat-icon positive">
                  <i className="fas fa-thumbs-up"></i>
                </div>
                <div className="stat-content">
                  <div className="stat-number">{metadata.positive_count}</div>
                  <div className="stat-label">Positive</div>
                </div>
              </div>
              
              <div className="stat-card">
                <div className="stat-icon negative">
                  <i className="fas fa-thumbs-down"></i>
                </div>
                <div className="stat-content">
                  <div className="stat-number">{metadata.negative_count}</div>
                  <div className="stat-label">Negative</div>
                </div>
              </div>
              
              <div className="stat-card">
                <div className="stat-icon neutral">
                  <i className="fas fa-minus"></i>
                </div>
                <div className="stat-content">
                  <div className="stat-number">{metadata.neutral_count}</div>
                  <div className="stat-label">Neutral</div>
                </div>
              </div>
              
            </div>
            
            <div className="confidence-info">
              <p>Average Confidence: <strong>{(metadata.average_confidence * 100).toFixed(1)}%</strong></p>
            </div>
          </div>

          {/* Sentiment Distribution Chart */}
          <div className="chart-section">
            <h3>Sentiment Distribution</h3>
            <SentimentChart 
              positive={metadata.positive_count}
              negative={metadata.negative_count}
              neutral={metadata.neutral_count}
            />
          </div>

          {/* Sample Reviews */}
          <div className="samples-section">
            <h3>Sample Reviews</h3>
            
            {positive.length > 0 && (
              <div className="sample-category">
                <h4 className="category-title positive">
                  <i className="fas fa-thumbs-up"></i>
                  Positive Reviews ({positive.length} total)
                </h4>
                <div className="review-samples">
                  {positive.slice(0, 3).map((review, index) => (
                    <ReviewCard key={index} review={review} />
                  ))}
                </div>
              </div>
            )}
            
            {negative.length > 0 && (
              <div className="sample-category">
                <h4 className="category-title negative">
                  <i className="fas fa-thumbs-down"></i>
                  Negative Reviews ({negative.length} total)
                </h4>
                <div className="review-samples">
                  {negative.slice(0, 3).map((review, index) => (
                    <ReviewCard key={index} review={review} />
                  ))}
                </div>
              </div>
            )}
            
            {neutral.length > 0 && (
              <div className="sample-category">
                <h4 className="category-title neutral">
                  <i className="fas fa-minus"></i>
                  Neutral Reviews ({neutral.length} total)
                </h4>
                <div className="review-samples">
                  {neutral.slice(0, 3).map((review, index) => (
                    <ReviewCard key={index} review={review} />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ResultsModal
