import React, { useState } from 'react'
import { analyzeConversation } from '../api'
import LabelBadge from './LabelBadge'
import SanityCheckCard from './SanityCheckCard'

export default function ConversationAnalyzer() {
  const [messages, setMessages] = useState(['', '', ''])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const addMessage = () => setMessages([...messages, ''])
  const removeMessage = (i) => setMessages(messages.filter((_, idx) => idx !== i))
  const updateMessage = (i, val) => {
    const copy = [...messages]; copy[i] = val; setMessages(copy)
  }

  const handleAnalyze = async () => {
    const valid = messages.filter(m => m.trim())
    if (valid.length === 0) return
    setLoading(true); setError(null)
    try {
      const data = await analyzeConversation(valid)
      setResult(data)
    } catch (e) {
      setError(e?.response?.data?.detail || 'Analysis failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Paste up to 5 messages in order. The model analyzes the full context window.
          </p>
          <button onClick={addMessage} disabled={messages.length >= 10}
            style={{ background: 'var(--bg3)', border: '1px solid var(--border)', color: 'var(--text)', borderRadius: '4px', padding: '5px 12px', cursor: 'pointer', fontSize: '0.8rem', whiteSpace: 'nowrap', fontFamily: 'inherit' }}>
            + Add
          </button>
        </div>
        {messages.map((msg, i) => (
          <div key={i} style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '10px', width: '18px', textAlign: 'right', flexShrink: 0 }}>{i + 1}</span>
            <textarea
              value={msg}
              onChange={e => updateMessage(i, e.target.value)}
              placeholder={`Message ${i + 1}`}
              rows={2}
              style={{
                flex: 1, background: 'var(--bg3)', border: '1px solid var(--border)',
                borderRadius: '6px', color: 'var(--text)', padding: '8px 12px',
                fontSize: '0.88rem', resize: 'vertical', outline: 'none', fontFamily: 'inherit',
              }}
            />
            {messages.length > 1 && (
              <button onClick={() => removeMessage(i)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.1rem', marginTop: '6px', padding: '0 4px' }}>
                x
              </button>
            )}
          </div>
        ))}
      </div>

      <button
        onClick={handleAnalyze}
        disabled={loading || messages.every(m => !m.trim())}
        style={{
          background: 'var(--bg3)', color: loading || messages.every(m => !m.trim()) ? 'var(--text-muted)' : 'var(--text)',
          border: '1px solid var(--border)', borderRadius: '6px',
          padding: '12px', fontSize: '0.9rem', fontWeight: 600,
          cursor: loading || messages.every(m => !m.trim()) ? 'not-allowed' : 'pointer',
          fontFamily: 'inherit',
        }}
      >
        {loading ? 'Analyzing...' : 'Analyze Conversation'}
      </button>

      {error && (
        <div style={{ border: '1px solid var(--border)', borderRadius: '6px', padding: '12px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Summary banner */}
          <div style={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: '8px', padding: '16px 20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <span style={{ fontWeight: 700, fontSize: '0.9rem', color: 'var(--text)' }}>
                {result.escalation_detected ? 'Escalating Pattern Detected' : result.pattern_detected ? 'Toxic Pattern Detected' : 'No Persistent Pattern'}
              </span>
              <LabelBadge label={result.dominant_pattern} size="sm" />
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', lineHeight: 1.7 }}
              dangerouslySetInnerHTML={{ __html: result.context_note.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
          </div>

          {/* Per-message breakdown */}
          <div style={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: '8px', padding: '16px 20px' }}>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '12px' }}>Message Breakdown</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {result.individual_results.map((item, i) => (
                <div key={i} style={{
                  display: 'flex', gap: '10px', alignItems: 'flex-start',
                  background: 'var(--bg2)', borderRadius: '6px', padding: '10px 14px',
                  borderLeft: '3px solid var(--border)',
                }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', width: '16px', flexShrink: 0, paddingTop: '2px' }}>{i + 1}</span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ color: 'var(--text)', fontSize: '0.85rem', lineHeight: 1.6, marginBottom: '5px' }}>{item.message}</p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                      <LabelBadge label={item.label} size="sm" />
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                        {(item.confidence * 100).toFixed(0)}% confidence
                      </span>
                      {item.pattern_highlights?.map((p, j) => (
                        <span key={j} style={{
                          background: 'var(--bg2)', color: 'var(--text-muted)',
                          border: '1px solid var(--border)', borderRadius: '3px',
                          padding: '1px 6px', fontSize: '0.72rem', fontFamily: 'monospace',
                        }}>"{p}"</span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {result.pattern_detected && (
            <SanityCheckCard
              sanityCheck={result.sanity_check}
              copingStrategies={result.coping_strategies}
              suggestedResponses={result.suggested_responses}
              severity={result.overall_severity}
              patternHighlights={[]}
            />
          )}
        </div>
      )}
    </div>
  )
}
