import React, { useState } from 'react'
import { analyzeSingle } from '../api'
import LabelBadge from './LabelBadge'
import ScoreBar from './ScoreBar'
import SanityCheckCard from './SanityCheckCard'

const EXAMPLES = {
  Gaslighting: "You're way too sensitive. I never said that — you're imagining things. No one else thinks there's a problem here.",
  Sarcastic: "Oh wow, what a brilliant idea! I'm sure completely ignoring everyone's feedback will totally work out great for you.",
  'Passive-Aggressive': "Fine, do whatever you want. I'm sure you know best. Don't worry about me at all.",
  Sincere: "I appreciate you sharing that with me. It sounds really difficult, and I'm here if you want to talk.",
}

const btn = {
  base: {
    background: 'var(--bg3)', border: '1px solid var(--border)',
    borderRadius: '4px', padding: '6px 14px', cursor: 'pointer',
    color: 'var(--text-muted)', fontSize: '0.8rem', fontFamily: 'inherit',
  },
  primary: (disabled) => ({
    background: disabled ? 'var(--bg3)' : 'var(--bg3)',
    color: disabled ? 'var(--text-muted)' : 'var(--text)',
    border: '1px solid var(--border)',
    borderRadius: '6px', padding: '12px',
    fontSize: '0.9rem', fontWeight: 600,
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontFamily: 'inherit', width: '100%',
  }),
}

export default function SingleAnalyzer() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleAnalyze = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await analyzeSingle(text.trim())
      setResult(data)
    } catch (e) {
      setError(e?.response?.data?.detail || 'Analysis failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Quick examples */}
      <div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '8px' }}>Examples:</p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
          {Object.entries(EXAMPLES).map(([label, ex]) => (
            <button key={label} onClick={() => { setText(ex); setResult(null) }} style={btn.base}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Text input */}
      <div style={{ position: 'relative' }}>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') handleAnalyze() }}
          placeholder="Paste or type a message to analyze... (Ctrl+Enter to run)"
          rows={5}
          maxLength={2000}
          style={{
            width: '100%', background: 'var(--bg3)',
            border: '1px solid var(--border)', borderRadius: '6px',
            color: 'var(--text)', padding: '14px', fontSize: '0.9rem',
            resize: 'vertical', outline: 'none', fontFamily: 'inherit', lineHeight: 1.6,
          }}
        />
        <span style={{ position: 'absolute', bottom: '10px', right: '12px', color: 'var(--text-muted)', fontSize: '0.72rem' }}>
          {text.length}/2000
        </span>
      </div>

      <button onClick={handleAnalyze} disabled={loading || !text.trim()} style={btn.primary(loading || !text.trim())}>
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>

      {error && (
        <div style={{ border: '1px solid var(--border)', borderRadius: '6px', padding: '12px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {/* Result header */}
          <div style={{
            background: 'var(--bg3)', border: '1px solid var(--border)',
            borderRadius: '8px', padding: '18px',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px',
          }}>
            <div>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Result</p>
              <LabelBadge label={result.label} size="lg" />
            </div>
            <div style={{ textAlign: 'right' }}>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginBottom: '2px' }}>Confidence</p>
              <p style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--text)' }}>
                {(result.confidence * 100).toFixed(0)}<span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>%</span>
              </p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>{result.elapsed_seconds}s</p>
            </div>
          </div>

          {/* Score breakdown */}
          <div style={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: '8px', padding: '18px' }}>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '12px' }}>Score Breakdown</p>
            <ScoreBar scores={result.all_scores} />
          </div>

          <SanityCheckCard
            sanityCheck={result.sanity_check}
            copingStrategies={result.coping_strategies}
            suggestedResponses={result.suggested_responses}
            severity={result.severity}
            patternHighlights={result.pattern_highlights}
          />
        </div>
      )}
    </div>
  )
}
