import React from 'react'

export default function Recap(props: { id: string; onRestart: () => void }) {
  const { id, onRestart } = props
  return (
    <div style={{ padding: 16, display:'grid', gap:12, maxWidth: 640, margin: '0 auto' }}>
      <h2>Thank you!</h2>
      <div>Your contribution has been submitted.</div>
      <div style={{display:'flex', alignItems:'center', gap:8}}>
        <code style={{background:'#f5f5f5', padding:'4px 6px', borderRadius:4}}>{id}</code>
        <button onClick={()=>{navigator.clipboard.writeText(id)}} title="Copy"><i className="bi bi-clipboard"></i></button>
      </div>
      <div style={{display:'flex', gap:8}}>
        <button onClick={onRestart}>Start another contribution</button>
      </div>
    </div>
  )
}
