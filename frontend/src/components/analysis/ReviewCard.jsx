import React from 'react'

function ReviewCard({ review }) {
  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return 'fas fa-thumbs-up'
      case 'negative':
        return 'fas fa-thumbs-down'
      default:
        return 'fas fa-minus'
    }
  }

  const getSentimentClass = (sentiment) => {
    return `sentiment-${sentiment}`
  }

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date'
    return new Date(dateString).toLocaleDateString()
  }

  const renderStars = (rating) => {
    if (!rating) return null
    
    const stars = []
    const fullStars = Math.floor(rating)
    const hasHalfStar = rating % 1 !== 0
    
    for (let i = 0; i < fullStars; i++) {
      stars.push(<i key={i} className="fas fa-star"></i>)
    }
    
    if (hasHalfStar) {
      stars.push(<i key="half" className="fas fa-star-half-alt"></i>)
    }
    
    const emptyStars = 5 - Math.ceil(rating)
    for (let i = 0; i < emptyStars; i++) {
      stars.push(<i key={`empty-${i}`} className="far fa-star"></i>)
    }
    
    return stars
  }

  return (
    <div className="review-card">
      <div className="review-header">
        <div className="reviewer-info">
          {review.verified_purchase && (
            <span className="verified-badge">
              <i className="fas fa-check-circle"></i>
              Verified Purchase
            </span>
          )}
        </div>
        
        <div className="review-meta">
          {review.rating && (
            <div className="rating">
              {renderStars(review.rating)}
              <span className="rating-value">{review.rating}/5</span>
            </div>
          )}
          
          <div className="sentiment-badge">
            <i className={getSentimentIcon(review.sentiment)}></i>
            <span className={getSentimentClass(review.sentiment)}>
              {review.sentiment.charAt(0).toUpperCase() + review.sentiment.slice(1)}
            </span>
            <span className="confidence">
              ({(review.confidence * 100).toFixed(1)}%)
            </span>
          </div>
        </div>
      </div>
      
      <h4 className="review-title">{review.title || 'No title'}</h4>
      
      <p className="review-text">{review.review_text}</p>
      
    </div>
  )
}

export default ReviewCard
