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
    <div style={{ padding: 16 }}>
      <h2>{i18n.t('slides.title')}</h2>
      <div style={{ border: '1px dashed #aaa', height: 200, marginBottom: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span>Slide image placeholder</span>
      </div>
      <p>{s.text}</p>
      <div style={{ display: 'flex', gap: 8 }}>
        <button onClick={prev} disabled={idx === 0}>{i18n.t('slides.prev')}</button>
        <button onClick={next}>{i18n.t('slides.next')}</button>
      </div>
    </div>
  )
}

