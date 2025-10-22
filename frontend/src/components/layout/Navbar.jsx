import React from 'react'

function Navbar({ onToggleSidebar, onToggleTheme, theme }) {
  return (
    <nav className="navbar">
      <div className="nav-brand">
        <button 
          className="nav-toggle" 
          onClick={onToggleSidebar}
          aria-label="Toggle navigation"
        >
          <i className="fas fa-bars"></i>
        </button>
        <div className="nav-icon">
          <i className="fas fa-store"></i>
        </div>
        <span>Walmart Sentiment Analyzer</span>
      </div>
      
      <div className="nav-actions">
        <button 
          className="theme-toggle" 
          onClick={onToggleTheme}
          aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
        >
          <i className={`fas fa-${theme === 'light' ? 'moon' : 'sun'}`}></i>
        </button>
      </div>
    </nav>
  )
}

export default Navbar
