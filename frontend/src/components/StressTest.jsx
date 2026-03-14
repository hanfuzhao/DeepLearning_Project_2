import React, { useState } from 'react'
import { runStressTest } from '../api'

export default function StressTest() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleRun = async () => {
    setLoading(true); setError(null)
    try {
      const data = await runStressTest()
      setResult(data)
    } catch (e) {
      setError(e?.response?.data?.detail || 'Test failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: '8px', padding: '18px' }}>
        <h3 style={{ fontWeight: 700, marginBottom: '8px', color: 'var(--text)', fontSize: '0.95rem' }}>Sarcastic Sentiment Flip Test</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', lineHeight: 1.7 }}>
          Runs "positive-worded" insults through the model to check whether it correctly labels them as Sarcastic rather than Sincere.
          Pass threshold: 70% accuracy.
        </p>
      </div>

      <button
        onClick={handleRun}
        disabled={loading}
        style={{
          background: 'var(--bg3)', color: loading ? 'var(--text-muted)' : 'var(--text)',
          border: '1px solid var(--border)', borderRadius: '6px',
          padding: '12px', fontSize: '0.9rem', fontWeight: 600,
          cursor: loading ? 'not-allowed' : 'pointer', fontFamily: 'inherit',
        }}
      >
        {loading ? 'Running...' : 'Run Stress Test'}
      </button>

      {error && (
        <div style={{ border: '1px solid var(--border)', borderRadius: '6px', padding: '12px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {/* Overall result */}
          <div style={{
            background: 'var(--bg3)', border: '1px solid var(--border)',
            borderRadius: '8px', padding: '18px',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px',
          }}>
            <div>
              <p style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--text)', marginBottom: '4px' }}>
                {result.verdict}
              </p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                {result.correct}/{result.total} correctly identified as Sarcastic
              </p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <p style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text)' }}>
                {(result.accuracy * 100).toFixed(0)}<span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>%</span>
              </p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>threshold: 70%</p>
            </div>
          </div>

          {/* Individual results */}
          <div style={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: '8px', padding: '16px 20px' }}>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '12px' }}>
              Sample Results
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {result.results.map((r, i) => (
                <div key={i} style={{
                  display: 'flex', gap: '10px', alignItems: 'center',
                  background: 'var(--bg2)', borderRadius: '6px', padding: '10px 14px',
                  borderLeft: `3px solid ${r.passed ? '#555' : 'var(--border)'}`,
                }}>
                  <span style={{ color: r.passed ? 'var(--text)' : 'var(--text-muted)', fontSize: '0.75rem', width: '14px', flexShrink: 0 }}>
                    {r.passed ? '+' : '-'}
                  </span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ color: 'var(--text)', fontSize: '0.83rem', lineHeight: 1.5 }}>{r.text}</p>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.73rem', marginTop: '2px' }}>
                      Predicted: <strong style={{ color: 'var(--text)' }}>{r.predicted}</strong>
                      {' '}({(r.confidence * 100).toFixed(0)}%) — Expected: {r.expected}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
