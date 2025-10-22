import React, { useState, useEffect } from 'react'
import Navbar from './components/layout/Navbar'
import Sidebar from './components/layout/Sidebar'
import HomePage from './pages/HomePage'
import AboutModal from './components/ui/AboutModal'
import ResultsModal from './components/ui/ResultsModal'

function App() {
  const [theme, setTheme] = useState('light')

  useEffect(() => {
    // Apply theme to document body as well
    document.body.setAttribute('data-theme', theme)
  }, [theme])
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [aboutModalOpen, setAboutModalOpen] = useState(false)
  const [resultsModalOpen, setResultsModalOpen] = useState(false)
  const [analysisData, setAnalysisData] = useState(null)

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light')
  }

  const toggleSidebar = () => {
    setSidebarOpen(prev => !prev)
  }

  const closeSidebar = () => {
    setSidebarOpen(false)
  }

  const openAboutModal = () => {
    setAboutModalOpen(true)
    closeSidebar()
  }

  const closeAboutModal = () => {
    setAboutModalOpen(false)
  }

  const openResultsModal = (data) => {
    setAnalysisData(data)
    setResultsModalOpen(true)
  }

  const closeResultsModal = () => {
    setResultsModalOpen(false)
    setAnalysisData(null)
  }

  return (
    <div className="app" data-theme={theme}>
      <Navbar 
        onToggleSidebar={toggleSidebar}
        onToggleTheme={toggleTheme}
        theme={theme}
      />
      
      <Sidebar 
        isOpen={sidebarOpen}
        onClose={closeSidebar}
        onOpenAbout={openAboutModal}
      />
      
      <main className="main-container">
        <HomePage onShowResults={openResultsModal} />
      </main>
      
      <AboutModal 
        isOpen={aboutModalOpen}
        onClose={closeAboutModal}
      />
      
      <ResultsModal 
        isOpen={resultsModalOpen}
        onClose={closeResultsModal}
        data={analysisData}
      />
    </div>
  )
}

export default App
