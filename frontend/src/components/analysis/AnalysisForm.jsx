import React, { useState } from 'react'

function AnalysisForm({ onAnalyze, isLoading, error, onClearError }) {
  const [url, setUrl] = useState('')
  const [maxReviews, setMaxReviews] = useState(50)

  const validateWalmartURL = (url) => {
    const walmartPattern = /^https?:\/\/(www\.)?walmart\.com\/.*$/
    return walmartPattern.test(url)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    
    if (!url.trim()) {
      onClearError()
      return
    }

    if (!validateWalmartURL(url)) {
      onClearError()
      return
    }

    onAnalyze(url.trim(), maxReviews)
  }

  const handleUrlChange = (e) => {
    setUrl(e.target.value)
    if (error) {
      onClearError()
    }
  }

  return (
    <div className="input-section">
      <form onSubmit={handleSubmit} className="analysis-form">
        <div className="input-group">
          <input
            type="text"
            value={url}
            onChange={handleUrlChange}
            placeholder="Paste Walmart product URL here..."
            className={`url-input ${error ? 'error' : ''}`}
            disabled={isLoading}
            required
          />
          <button
            type="submit"
            className="btn-primary"
            disabled={isLoading || !url.trim()}
          >
            {isLoading ? (
              <>
                <i className="fas fa-spinner fa-spin"></i>
                Analyzing...
              </>
            ) : (
              <>
                <i className="fas fa-chart-line"></i>
                Analyze Reviews
              </>
            )}
          </button>
        </div>
        
        <div className="form-options">
          <label className="option-label">
            <span>Max Reviews:</span>
            <select 
              value={maxReviews} 
              onChange={(e) => setMaxReviews(parseInt(e.target.value))}
              disabled={isLoading}
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
            </select>
          </label>
        </div>
      </form>
      
      {error && (
        <div className="error-message">
          <i className="fas fa-exclamation-triangle"></i>
          {error}
        </div>
      )}
    </div>
  )
}

export default AnalysisForm
