import React, { useEffect, useState } from 'react'
import { i18n } from './i18n'
import Interlaced from './pages/Interlaced'
import Admin from './pages/Admin'
import Terms from './pages/Terms'
import Privacy from './pages/Privacy'

export default function App() {
  const [hash, setHash] = useState<string>(typeof window !== 'undefined' ? window.location.hash : '')
  useEffect(() => {
    const onHash = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const isHome = hash === ''
  const isAdmin = hash === '#admin'
  const isTerms = hash === '#terms'
  const isPrivacy = hash === '#privacy'

  return (
    <div className="container my-3">
      <header style={{display:'flex', alignItems:'center', justifyContent:'space-between', gap:12, marginBottom:12}}>
        <div style={{display:'flex', alignItems:'center', gap:8}}>
          <img src="/opensock-icon.svg" alt="OpenSock" width={40} height={40} />
          <div>
            <div style={{fontWeight:700, lineHeight:1}}>OpenSock</div>
            <div className="muted" style={{fontSize:12, lineHeight:1.2}}>Contribute sock photos for research</div>
          </div>
        </div>
        <nav style={{display:'flex', gap:12, fontSize:14}}>
          {isHome ? <></> : <a href="">Home</a>}
          {isTerms ? <></> : <a href="#terms">Terms</a>}
          {isPrivacy ? <></> : <a href="#privacy">Privacy</a>}
        </nav>
      </header>
      {isAdmin ? <Admin /> : isTerms ? <Terms /> : isPrivacy ? <Privacy /> : <Interlaced /> }
    </div>
  )
}
