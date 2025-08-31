import React, { useState } from 'react'
import { i18n } from '../i18n'

const slides = [
  {
    title: i18n.t('slides.title'),
    text: 'Slide 1: Collect 20–30 pairs; take one sock from each. Only socks on the ground.',
  },
  { text: 'Slide 2: Lay socks flat, stretched. Do not move between photos.' },
  { text: 'Slide 3: Flip socks, same arrangement.' },
  { text: 'Slide 4: Crumple slightly; take photo.' },
  { text: 'Slide 5: Crumple differently; take photo.' },
  { text: 'Slide 6: Optional extra configurations.' },
]

export default function Slides(props: { onDone: () => void }) {
  const [idx, setIdx] = useState(0)
  const s = slides[idx]
  const next = () => (idx < slides.length - 1 ? setIdx(idx + 1) : props.onDone())
  const prev = () => idx > 0 && setIdx(idx - 1)
  return (
    <>
      <div style={{ padding: 16 }}>
        <h2>{i18n.t('slides.title')}</h2>
        <div style={{ fontSize: 13, color: '#374151', marginTop: 4, marginBottom: 8 }}>
          For best results, keep about 20 socks per photo with some spacing.
        </div>
        <div style={{ border: '1px dashed #aaa', height: 200, marginBottom: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <span>Slide image placeholder</span>
        </div>
        <p>{s.text}</p>
      </div>
      <div aria-hidden style={{ height: 72 }} />
      <div style={{ position: 'fixed', left: 0, right: 0, bottom: 48, zIndex: 40 }}>
        <div style={{ background: 'rgba(255,255,255,0.9)', backdropFilter: 'blur(4px)', borderTop: '1px solid #e5e7eb', boxShadow: '0 -1px 4px rgba(0,0,0,0.04)' }}>
          <div style={{ maxWidth: '48rem', margin: '0 auto', paddingLeft: 16, paddingRight: 16, paddingTop: 8, paddingBottom: 8 }}>
            <div style={{ display: 'flex', gap: 8, width: '100%' }}>
              <button style={{ flex: 1, padding: '12px 16px', borderRadius: 8, border: '1px solid #d1d5db', background: '#fff' }} onClick={prev} disabled={idx === 0}>{i18n.t('slides.prev')}</button>
              <button style={{ flex: 1, padding: '12px 16px', borderRadius: 8, border: '1px solid transparent', background: '#2563eb', color: '#fff' }} onClick={next}>{i18n.t('slides.next')}</button>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
