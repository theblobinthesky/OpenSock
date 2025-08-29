import React, { useMemo, useRef, useState } from 'react'
import { i18n } from '../i18n'
import { createSession, uploadImages, finalize, getConfig } from '../api'
import Recap from '../components/Recap'

export default function Collect() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [remaining, setRemaining] = useState<number | null>(null)
  const [uploaded, setUploaded] = useState<number>(0)
  const [mode, setMode] = useState<'MIXED_UNIQUES' | 'SAME_TYPE'>('MIXED_UNIQUES')
  const [busy, setBusy] = useState(false)
  const [consent, setConsent] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [recap, setRecap] = useState<{ id: string } | null>(null)
  const [cap, setCap] = useState<number>(20)
  const [minMap, setMinMap] = useState<{MIXED_UNIQUES:number; SAME_TYPE:number}>({MIXED_UNIQUES:4, SAME_TYPE:3})

  React.useEffect(()=>{
    getConfig().then(cfg=>{
      setCap(cfg.session_image_cap)
      setMinMap(cfg.min_required as any)
    }).catch(()=>{})
  },[])

  const onCreate = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!consent) { setError('Please accept consent'); return }
    setError(null)
    const fd = new FormData(e.currentTarget)
    const payload = {
      name: String(fd.get('name') || ''),
      handle: String(fd.get('handle') || ''),
      email: String(fd.get('email') || ''),
      notify_opt_in: !!fd.get('notify'),
      language: i18n.lang,
      mode: String(fd.get('mode') || 'MIXED_UNIQUES'),
    }
    setBusy(true)
    try {
      const res = await createSession(payload)
      setSessionId(res.session_id)
      const m = String(payload.mode || 'MIXED_UNIQUES')
      setMode(m === 'SAME_TYPE' ? 'SAME_TYPE' : 'MIXED_UNIQUES')
    } catch (e:any) {
      setError(e.message)
    } finally { setBusy(false) }
  }

  const onUpload = async (files: FileList | null) => {
    if (!sessionId || !files || files.length === 0) return
    setBusy(true)
    try {
      const resp = await uploadImages(sessionId, Array.from(files))
      setUploaded(u => u + resp.uploaded)
      setRemaining(r => {
        const used = (typeof r === 'number' && r >= 0) ? (cap - r) : uploaded
        const after = Math.max(0, cap - (used + resp.uploaded))
        return after
      })
    } catch (e:any) {
      setError(e.message)
    } finally { setBusy(false); if(inputRef.current) inputRef.current.value = '' }
  }

  const onFinalize = async () => {
    if (!sessionId) return
    setBusy(true)
    try { await finalize(sessionId); setRecap({ id: sessionId }) }
    catch (e:any) {
      try {
        const parsed = JSON.parse(String(e.message))
        if (parsed.error_key === 'session.min_images_unmet') {
          setError(`Need at least ${parsed.params?.required} images before finalizing.`)
        } else if (parsed.error_key) {
          setError(parsed.error_key)
        } else {
          setError(e.message)
        }
      } catch {
        setError(e.message)
      }
    }
    finally { setBusy(false) }
  }

  if (!sessionId) {
    return (
      <form onSubmit={onCreate} style={{ display: 'grid', gap: 8, padding: 16 }}>
        <h2>{i18n.t('collect.title')}</h2>
        <input name="name" placeholder={i18n.t('collect.name')} />
        <input name="handle" placeholder={i18n.t('collect.handle')} />
        <input name="email" placeholder={i18n.t('collect.email')} type="email" required />
        <div>
          <label><input type="radio" name="mode" value="MIXED_UNIQUES" defaultChecked /> Mixed Uniques</label>
          <label style={{marginLeft:12}}><input type="radio" name="mode" value="SAME_TYPE" /> Same Type</label>
        </div>
        <label><input name="notify" type="checkbox" /> {i18n.t('collect.notify')}</label>
        <label><input type="checkbox" checked={consent} onChange={e=>setConsent(e.target.checked)} /> {i18n.t('collect.consent')}</label>
        <small>{i18n.t('note.privacy')}</small>
        <button disabled={busy || !consent} type="submit">{i18n.t('collect.start')}</button>
        {error && <div style={{color:'crimson'}}>{error}</div>}
      </form>
    )
  }

  if (recap) {
    return <Recap id={recap.id} onRestart={()=>{ setSessionId(null); setUploaded(0); setRemaining(null); setError(null); setRecap(null) }} />
  }
  const minRequired = mode === 'SAME_TYPE' ? minMap.SAME_TYPE : minMap.MIXED_UNIQUES
  const canFinalize = uploaded >= minRequired
  return (
    <div style={{ padding: 16, display:'grid', gap:8 }}>
      <div><strong>Session:</strong> {sessionId}</div>
      <div>
        <input ref={inputRef} type="file" accept="image/*" multiple capture="environment" onChange={e=>onUpload(e.target.files)} />
      </div>
      {remaining !== null && <div>{i18n.t('upload.remaining')}: {remaining} / cap {cap} (uploaded ~ {uploaded})</div>}
      {!canFinalize && <div style={{color:'#b36b00'}}>Add at least {minRequired} images before finalizing.</div>}
      <button disabled={busy || !canFinalize} onClick={onFinalize}>{i18n.t('finalize')}</button>
      {error && <div style={{color:'crimson'}}>{error}</div>}
    </div>
  )
}
