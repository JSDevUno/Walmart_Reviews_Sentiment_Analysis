import React, { useEffect } from 'react'

function Sidebar({ isOpen, onClose, onOpenAbout }) {
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
    <>
      <div className="sidebar-overlay" onClick={onClose}></div>
      <div className="sidebar">
        <div className="sidebar-header">
          <h3>Navigation</h3>
          <button 
            className="sidebar-close" 
            onClick={onClose}
            aria-label="Close sidebar"
          >
            <i className="fas fa-times"></i>
          </button>
        </div>
        
        <div className="sidebar-content">
          <button 
            className="sidebar-item" 
            onClick={onOpenAbout}
          >
            <i className="fas fa-info-circle"></i>
            About
          </button>
        </div>
      </div>
    </>
  )
}

export default Sidebar
