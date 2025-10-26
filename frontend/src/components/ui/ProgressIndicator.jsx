import React from 'react'

function ProgressIndicator({ progress, message, stage, isVisible }) {
  if (!isVisible) return null

  const getStageIcon = (stage) => {
    switch (stage) {
      case 'initializing':
        return 'fas fa-cog fa-spin'
      case 'browser_started':
        return 'fas fa-check-circle'
      case 'extracting_id':
        return 'fas fa-link'
      case 'loading_page':
        return 'fas fa-globe'
      case 'navigating_reviews':
        return 'fas fa-arrow-right'
      case 'extracting_reviews':
        return 'fas fa-list'
      case 'analyzing_sentiment':
        return 'fas fa-brain'
      case 'finalizing':
        return 'fas fa-cogs'
      case 'complete':
        return 'fas fa-check-circle'
      default:
        return 'fas fa-spinner fa-spin'
    }
  }

  const getStageColor = (stage) => {
    switch (stage) {
      case 'initializing':
      case 'browser_started':
        return '#3b82f6' // blue
      case 'extracting_id':
      case 'loading_page':
      case 'navigating_reviews':
        return '#8b5cf6' // purple
      case 'extracting_reviews':
        return '#f59e0b' // amber
      case 'analyzing_sentiment':
        return '#10b981' // emerald
      case 'finalizing':
        return '#06b6d4' // cyan
      case 'complete':
        return '#22c55e' // green
      default:
        return '#6b7280' // gray
    }
  }

  return (
    <div className="progress-indicator">
      <div className="progress-header">
        <div className="progress-icon">
          <i 
            className={getStageIcon(stage)} 
            style={{ color: getStageColor(stage) }}
          ></i>
        </div>
        <div className="progress-content">
          <div className="progress-message">{message}</div>
          <div className="progress-percentage">{Math.round(progress || 0)}%</div>
        </div>
      </div>
      
      <div className="progress-bar-container">
        <div 
          className="progress-bar"
          style={{ 
            width: `${progress || 0}%`,
            backgroundColor: getStageColor(stage)
          }}
        ></div>
      </div>
      
      <div className="progress-details">
        <div className="progress-stage">
          <span className="stage-label">Stage:</span>
          <span className="stage-name">{stage?.replace('_', ' ') || 'Processing'}</span>
        </div>
      </div>
    </div>
  )
}


export default ProgressIndicator
