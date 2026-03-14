import React from 'react'

export default function LabelBadge({ label, size = 'md' }) {
  const fs = size === 'lg' ? '0.95rem' : size === 'sm' ? '0.7rem' : '0.8rem'
  const px = size === 'lg' ? '12px' : '8px'
  const py = size === 'lg' ? '4px' : '3px'
  return (
    <span style={{
      display: 'inline-block',
      background: 'var(--bg3)',
      color: 'var(--text)',
      border: '1px solid var(--border)',
      borderRadius: '4px',
      padding: `${py} ${px}`,
      fontSize: fs,
      fontWeight: 600,
      whiteSpace: 'nowrap',
      letterSpacing: '0.01em',
    }}>
      {label}
    </span>
  )
}
