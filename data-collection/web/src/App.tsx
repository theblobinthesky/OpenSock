import React, { useEffect, useState } from 'react'
import { i18n } from './i18n'
import Interlaced from './pages/Interlaced'
import Admin from './pages/Admin'
import Terms from './pages/Terms'
import Privacy from './pages/Privacy'
import Impressum from './pages/Impressum'

export default function App() {
  const [hash, setHash] = useState<string>(typeof window !== 'undefined' ? window.location.hash : '')
  const [homeNonce, setHomeNonce] = useState(0)
  useEffect(() => {
    const onHash = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  useEffect(() => {
    if (hash === '') setHomeNonce(n => n + 1)
  }, [hash])

  const isHome = hash === ''
  const isAdmin = hash === '#admin'
  const isTerms = hash === '#terms'
  const isPrivacy = hash === '#privacy'
  const isImpressum = hash === '#impressum'

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 pt-6 pb-16">
      <header className="flex items-center justify-between gap-3 mb-2">
        <a href="#" onClick={(e)=>{ e.preventDefault(); setHomeNonce(n=>n+1); window.location.hash = '' }} className="flex items-center gap-3 hover:opacity-90 transition-opacity">
          <img src="/opensock-icon.svg" alt="OpenSock" width={22} height={22} />
          <div>
            <div className="font-bold leading-none">OpenSock</div>
          </div>
        </a>
        <div />
      </header>
      {isAdmin ? <Admin /> : isTerms ? <Terms /> : isPrivacy ? <Privacy /> : isImpressum ? <Impressum /> : <Interlaced key={homeNonce} /> }

      <footer className="fixed bottom-0 left-0 right-0 border-t bg-white/95">
        <div className="max-w-3xl mx-auto px-4 h-12 flex items-center justify-between text-sm text-gray-600">
          <div className="flex items-center gap-4">
            <a href="#" className="hover:underline">{i18n.t('nav.home')}</a>
          </div>
          <div className="flex items-center gap-4">
            <a href="#terms" className="hover:underline">{i18n.t('nav.terms')}</a>
            <a href="#privacy" className="hover:underline">{i18n.t('nav.privacy')}</a>
            <a href="#impressum" className="hover:underline">{i18n.t('nav.impressum')}</a>
          </div>
        </div>
      </footer>
    </div>
  )
}
