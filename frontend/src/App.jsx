import React, { useState } from 'react'
import SingleAnalyzer from './components/SingleAnalyzer'
import ConversationAnalyzer from './components/ConversationAnalyzer'
import StressTest from './components/StressTest'
import Resources from './components/Resources'

const TABS = [
  { id: 'single',       label: 'Single Message' },
  { id: 'conversation', label: 'Conversation'   },
  { id: 'stress',       label: 'Stress Test'    },
  { id: 'resources',    label: 'Get Help'        },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('single')

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
      {/* Header */}
      <header style={{
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg2)',
        padding: '0 24px',
        position: 'sticky', top: 0, zIndex: 100,
      }}>
        <div style={{ maxWidth: '960px', margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: '56px' }}>
          <div>
            <span style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text)' }}>
              NLP Product
            </span>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginLeft: '12px' }}>
              Sarcasm &amp; Gaslighting Detector
            </span>
          </div>
          <span style={{
            border: '1px solid var(--border)',
            borderRadius: '4px', padding: '2px 10px',
            fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 500,
          }}>
            RoBERTa-base
          </span>
        </div>
      </header>

      {/* Tab nav */}
      <div style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg2)', padding: '0 24px' }}>
        <div style={{ maxWidth: '960px', margin: '0 auto', display: 'flex', overflowX: 'auto' }}>
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                padding: '14px 20px', fontSize: '0.85rem', fontWeight: 500,
                color: activeTab === tab.id ? 'var(--text)' : 'var(--text-muted)',
                borderBottom: activeTab === tab.id ? '2px solid var(--text)' : '2px solid transparent',
                whiteSpace: 'nowrap', fontFamily: 'inherit',
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main content */}
      <main style={{ maxWidth: '960px', margin: '0 auto', padding: '32px 24px' }}>
        {activeTab === 'single'       && <SingleAnalyzer />}
        {activeTab === 'conversation' && <ConversationAnalyzer />}
        {activeTab === 'stress'       && <StressTest />}
        {activeTab === 'resources'    && <Resources />}
      </main>

      {/* Footer */}
      <footer style={{ borderTop: '1px solid var(--border)', padding: '20px 24px', textAlign: 'center', marginTop: '40px' }}>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.78rem', lineHeight: 1.7 }}>
          For educational purposes only. Does not replace professional mental health support.
          <br />NLP Product · 540 Hackathon 2026 · RoBERTa-base · SARC + Cyberbullying Classification
        </p>
      </footer>
    </div>
  )
}
