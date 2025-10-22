import React from 'react'

function SentimentChart({ positive, negative, neutral }) {
  const total = positive + negative + neutral
  
  if (total === 0) {
    return (
      <div className="sentiment-chart">
        <p>No data available</p>
      </div>
    )
  }

  const positivePercent = (positive / total) * 100
  const negativePercent = (negative / total) * 100
  const neutralPercent = (neutral / total) * 100

  return (
    <div className="sentiment-chart">
      <div className="chart-bars">
        <div className="sentiment-bar positive" style={{ width: `${positivePercent}%` }}>
          {positivePercent >= 15 && (
            <span className="bar-label">Positive {positivePercent.toFixed(1)}%</span>
          )}
        </div>
        <div className="sentiment-bar neutral" style={{ width: `${neutralPercent}%` }}>
          {neutralPercent >= 15 && (
            <span className="bar-label">Neutral {neutralPercent.toFixed(1)}%</span>
          )}
        </div>
        <div className="sentiment-bar negative" style={{ width: `${negativePercent}%` }}>
          {negativePercent >= 15 && (
            <span className="bar-label">Negative {negativePercent.toFixed(1)}%</span>
          )}
        </div>
      </div>
      
      <div className="chart-legend">
        <div className="legend-item">
          <div className="legend-color positive"></div>
          <span>Positive ({positive})</span>
        </div>
        <div className="legend-item">
          <div className="legend-color neutral"></div>
          <span>Neutral ({neutral})</span>
        </div>
        <div className="legend-item">
          <div className="legend-color negative"></div>
          <span>Negative ({negative})</span>
        </div>
      </div>
    </div>
  )
}

export default SentimentChart
