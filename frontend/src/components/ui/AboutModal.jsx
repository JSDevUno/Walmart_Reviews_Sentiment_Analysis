import React, { useEffect } from 'react'

function AboutModal({ isOpen, onClose }) {
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

  if (!isOpen) return null

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>About Walmart Sentiment Analyzer</h2>
          <button 
            className="modal-close" 
            onClick={onClose}
            aria-label="Close modal"
          >
            <i className="fas fa-times"></i>
          </button>
        </div>
        
        <div className="modal-content">
          <div className="about-section">
            <h3>What is this tool?</h3>
            <p>
              The Walmart Sentiment Analyzer is an AI-powered tool that analyzes customer reviews 
              from Walmart products to determine overall sentiment. It uses advanced machine learning 
              models to classify reviews as positive, negative, or neutral.
            </p>
          </div>
          
          <div className="about-section">
            <h3>How does it work?</h3>
            <ol>
              <li>Enter a Walmart product URL</li>
              <li>Our system scrapes all available reviews</li>
              <li>AI analyzes each review's sentiment</li>
              <li>Get comprehensive insights and statistics</li>
            </ol>
          </div>
          
          <div className="about-section">
            <h3>Features</h3>
            <ul>
              <li>Real-time sentiment analysis</li>
              <li>Comprehensive review statistics</li>
              <li>Confidence scoring for each prediction</li>
              <li>Sample reviews for each sentiment category</li>
              <li>Average rating calculations</li>
            </ul>
          </div>
          
          <div className="about-section">
            <h3>Technology</h3>
            <p>
              Built with React frontend, Flask backend, and advanced NLP models including 
              RoBERTa and TF-IDF with LinearSVC for sentiment classification.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AboutModal
