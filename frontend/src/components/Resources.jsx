import React, { useState, useEffect } from 'react'
import { getResources } from '../api'

const SECTION_LABELS = {
  crisis:    'Crisis Support',
  self_help: 'Self-Help Resources',
  education: 'Educational',
}

export default function Resources() {
  const [data, setData] = useState(null)

  useEffect(() => {
    getResources().then(setData).catch(() => setData(null))
  }, [])

  const staticData = {
    crisis: [
      { name: 'Crisis Text Line', detail: 'Text HOME to 741741', url: 'https://www.crisistextline.org' },
      { name: 'National Domestic Violence Hotline', detail: '1-800-799-7233', url: 'https://www.thehotline.org' },
      { name: 'SAMHSA Helpline', detail: '1-800-662-4357 (free, 24/7)', url: 'https://www.samhsa.gov/find-help/national-helpline' },
    ],
    self_help: [
      { name: 'Loveisrespect', detail: 'Chat, call, or text for relationship help', url: 'https://www.loveisrespect.org' },
      { name: '7 Cups', detail: 'Free online chat with trained listeners', url: 'https://www.7cups.com' },
      { name: 'Gaslighting Recovery Workbook', detail: 'Book by Amy Marlow-MaCoy', url: 'https://www.amazon.com/dp/1648481043' },
    ],
    education: [
      { name: 'Psychology Today — Gaslighting', detail: 'Articles on recognizing gaslighting', url: 'https://www.psychologytoday.com/us/basics/gaslighting' },
      { name: 'NAMI', detail: 'National Alliance on Mental Illness', url: 'https://www.nami.org' },
    ],
  }

  const resources = data || staticData

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', lineHeight: 1.7 }}>
        If you or someone you know is experiencing verbal abuse, gaslighting, or emotional harm — help is available.
      </p>
      {Object.entries(resources).map(([section, items]) => (
        <div key={section}>
          <h3 style={{ fontWeight: 600, marginBottom: '10px', fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
            {SECTION_LABELS[section]}
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {items.map((item, i) => (
              <a key={i} href={item.url} target="_blank" rel="noopener noreferrer"
                style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  background: 'var(--bg3)', border: '1px solid var(--border)',
                  borderRadius: '6px', padding: '12px 16px', textDecoration: 'none',
                }}
                onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--text-muted)'}
                onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
              >
                <div>
                  <p style={{ color: 'var(--text)', fontWeight: 600, fontSize: '0.88rem', marginBottom: '2px' }}>{item.name}</p>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>{item.detail}</p>
                </div>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem', flexShrink: 0 }}>-&gt;</span>
              </a>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
