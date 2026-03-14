import React from 'react'

const LABELS = ['Sincere', 'Sarcastic', 'Passive-Aggressive', 'Gaslighting']

export default function ScoreBar({ scores }) {
  const sorted = LABELS.map(l => [l, scores[l] ?? 0]).sort((a, b) => b[1] - a[1])
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      {sorted.map(([label, score]) => (
        <div key={label}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '0.8rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>{label}</span>
            <span style={{ color: 'var(--text)', fontWeight: 600 }}>{(score * 100).toFixed(1)}%</span>
          </div>
          <div style={{ height: '5px', background: 'var(--border)', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${score * 100}%`,
              background: 'var(--text-muted)',
              borderRadius: '2px',
              transition: 'width 0.5s ease',
            }} />
          </div>
        </div>
      ))}
    </div>
  )
}
