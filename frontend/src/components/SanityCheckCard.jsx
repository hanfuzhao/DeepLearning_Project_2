import React from 'react'

const SEVERITY_LABELS = { low: 'Low Risk', medium: 'Medium Risk', high: 'High Risk' }

export default function SanityCheckCard({ sanityCheck, copingStrategies, suggestedResponses, severity, patternHighlights }) {
  return (
    <div style={{
      background: 'var(--bg3)',
      border: '1px solid var(--border)',
      borderRadius: '8px',
      padding: '24px',
      display: 'flex',
      flexDirection: 'column',
      gap: '20px',
    }}>
      {/* Severity */}
      <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        {SEVERITY_LABELS[severity] || 'Low Risk'}
      </div>

      {/* Sanity check */}
      <div>
        <h3 style={{ fontWeight: 700, marginBottom: '8px', fontSize: '0.82rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Sanity Check
        </h3>
        <p style={{ color: 'var(--text)', lineHeight: 1.7, fontSize: '0.9rem' }}>{sanityCheck}</p>
      </div>

      {/* Pattern highlights */}
      {patternHighlights?.length > 0 && (
        <div>
          <h3 style={{ fontWeight: 700, marginBottom: '8px', fontSize: '0.82rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Flagged Phrases
          </h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {patternHighlights.map((p, i) => (
              <span key={i} style={{
                background: 'var(--bg2)',
                border: '1px solid var(--border)',
                borderRadius: '4px', padding: '2px 8px',
                fontSize: '0.8rem', fontFamily: 'monospace', color: 'var(--text)',
              }}>"{p}"</span>
            ))}
          </div>
        </div>
      )}

      {/* Coping strategies */}
      <div>
        <h3 style={{ fontWeight: 700, marginBottom: '8px', fontSize: '0.82rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Coping Strategies
        </h3>
        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {copingStrategies.map((s, i) => (
            <li key={i} style={{ display: 'flex', gap: '10px', alignItems: 'flex-start', color: 'var(--text)', fontSize: '0.88rem', lineHeight: 1.6 }}>
              <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>—</span>
              {s}
            </li>
          ))}
        </ul>
      </div>

      {/* Suggested responses */}
      <div>
        <h3 style={{ fontWeight: 700, marginBottom: '8px', fontSize: '0.82rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Suggested Responses
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {suggestedResponses.map((r, i) => (
            <div key={i} style={{
              background: 'var(--bg2)', border: '1px solid var(--border)',
              borderRadius: '6px', padding: '10px 14px',
              color: 'var(--text)', fontSize: '0.88rem', lineHeight: 1.6,
            }}>
              {r}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
