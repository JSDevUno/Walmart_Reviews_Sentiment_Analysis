import React, { useState } from 'react'
import AnalysisForm from '../components/analysis/AnalysisForm'

function HomePage({ onShowResults }) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState('')

  const handleAnalyze = async (url, maxReviews) => {
    setIsAnalyzing(true)
    setError('')

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: url,
          max_reviews: maxReviews
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Analysis failed')
      }

      const { session_id } = await response.json()
      
      // Poll for results
      const pollResults = async () => {
        try {
          const statusResponse = await fetch(`/api/status/${session_id}`)
          const status = await statusResponse.json()

          if (status.status === 'complete') {
            setIsAnalyzing(false)
            onShowResults(status.data)
          } else if (status.status === 'error') {
            setIsAnalyzing(false)
            setError(status.message)
          } else if (status.status === 'loading') {
            // Continue polling
            setTimeout(pollResults, 2000)
          }
        } catch (err) {
          setIsAnalyzing(false)
          setError('Failed to get analysis status')
        }
      }

      // Start polling after a short delay
      setTimeout(pollResults, 1000)

    } catch (err) {
      setIsAnalyzing(false)
      setError(err.message)
    }
  }

  return (
    <div className="hero-section">
      <div className="hero-content">
        <h1>Analyze Product Reviews</h1>
        <p className="hero-subtitle">
          Get instant sentiment analysis of Walmart product reviews using advanced AI
        </p>
        
        <AnalysisForm 
          onAnalyze={handleAnalyze}
          isLoading={isAnalyzing}
          error={error}
          onClearError={() => setError('')}
        />
      </div>
    </div>
  )
}

export default HomePage
