import React, { useMemo, useRef, useState } from 'react'
import { i18n } from '../i18n'
import { createSession, finalize, uploadImages, getConfig } from '../api'

type Mode = 'MIXED_UNIQUES' | 'SAME_TYPE'
type Step = { key: string; title: string; required: boolean }

function stepsFor(mode: Mode): Step[] {
  if (mode === 'SAME_TYPE') {
    return [
      { key: 'same.1', title: i18n.t('slides.same.1.title'), required: true },
      { key: 'same.2', title: i18n.t('slides.same.2.title'), required: true },
      { key: 'same.3', title: i18n.t('slides.same.3.title'), required: true },
      { key: 'same.4', title: i18n.t('slides.same.4.title'), required: false },
    ]
  }
  return [
    { key: 'mixed.1', title: i18n.t('slides.mixed.1.title'), required: true },
    { key: 'mixed.2', title: i18n.t('slides.mixed.2.title'), required: true },
    { key: 'mixed.3', title: i18n.t('slides.mixed.3.title'), required: true },
    { key: 'mixed.4', title: i18n.t('slides.mixed.4.title'), required: false },
    { key: 'mixed.5', title: i18n.t('slides.mixed.5.title'), required: false },
  ]
}

export default function Interlaced() {
  const [mode, setMode] = useState<Mode>('MIXED_UNIQUES')
  const [consent, setConsent] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [cap, setCap] = useState<number>(20)
  const [maxMB, setMaxMB] = useState<number>(25)
  // removed: minimum images logic
  const [stepIdx, setStepIdx] = useState(0)
  const [remaining, setRemaining] = useState<number | null>(null)
  const [uploadedByStep, setUploadedByStep] = useState<Record<number, number>>({})
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [recap, setRecap] = useState<{ id: string, at: string } | null>(null)
  const [warnLowRes, setWarnLowRes] = useState<string | null>(null)
  // removed: 4:3 framing overlay toggle and 3x3 grid overlay
  const [tilt, setTilt] = useState<{beta:number, gamma:number} | null>(null)
  const [needsPerm, setNeedsPerm] = useState<boolean>(false)

  const steps = useMemo(() => stepsFor(mode), [mode])
  const totalUploaded = Object.values(uploadedByStep).reduce((a, b) => a + b, 0)

  React.useEffect(()=>{
    // load config
    getConfig().then(cfg=>{
      setCap(cfg.session_image_cap)
      // @ts-ignore: file_size_limit_mb may be present
      if ((cfg as any).file_size_limit_mb) setMaxMB((cfg as any).file_size_limit_mb)
    }).catch(()=>{})
  },[])

  // Device orientation hints
  React.useEffect(()=>{
    const handler = (e: DeviceOrientationEvent) => {
      if (e.beta!=null && e.gamma!=null) setTilt({beta: e.beta, gamma: e.gamma})
    }
    if (typeof (window as any).DeviceOrientationEvent !== 'undefined') {
      const anyWin = window as any
      if (typeof anyWin.DeviceOrientationEvent.requestPermission === 'function') {
        setNeedsPerm(true)
      } else {
        window.addEventListener('deviceorientation', handler)
      }
    }
    return () => { window.removeEventListener('deviceorientation', handler) }
  },[])

  const requestOrientationPermission = async () => {
    try {
      const anyWin = window as any
      if (typeof anyWin.DeviceOrientationEvent?.requestPermission === 'function') {
        const res = await anyWin.DeviceOrientationEvent.requestPermission()
        if (res === 'granted') {
          setNeedsPerm(false)
          const handler = (e: DeviceOrientationEvent) => { if (e.beta!=null && e.gamma!=null) setTilt({beta: e.beta, gamma: e.gamma}) }
          window.addEventListener('deviceorientation', handler)
        }
      }
    } catch {}
  }

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
      mode: mode,
    }
    setBusy(true)
    try {
      const res = await createSession(payload)
      setSessionId(res.session_id)
      setStepIdx(0)
      setError(null)
    } catch (e:any) {
      try {
        const parsed = JSON.parse(String(e.message))
        if (parsed.error_key === 'session.email_required') {
          setError('Email is required to continue.')
        } else {
          setError(e.message)
        }
      } catch { setError(e.message) }
    } finally { setBusy(false) }
  }

  const onUpload = async (files: FileList | null) => {
    if (!sessionId || !files || files.length === 0) return
    // soft low-res nudge
    setWarnLowRes(null)
    try {
      const f0 = files[0]
      const url = URL.createObjectURL(f0)
      const img = new Image()
      img.src = url
      await img.decode()
      if (img.width < 1024) setWarnLowRes('For better results, use higher resolution if possible.')
      URL.revokeObjectURL(url)
    } catch {}
    setBusy(true)
    try {
      const resp = await uploadImages(sessionId, Array.from(files))
      const newUploadedThisStep = (uploadedByStep[stepIdx] || 0) + resp.uploaded
      const currentTotal = totalUploaded
      const remainingLocal = Math.max(0, cap - (currentTotal + resp.uploaded))
      setRemaining(remainingLocal)
      setUploadedByStep((prev) => ({ ...prev, [stepIdx]: newUploadedThisStep }))
      if (resp.uploaded === 0) {
        setError('No new images uploaded. If you re-used the same photo, please take a fresh one (duplicates are skipped).')
      } else {
        setError(null)
      }
      if (inputRef.current) inputRef.current.value = ''
    } catch (e:any) {
      const msg = String(e?.message||'')
      if (msg.includes('unsupported')) setError(i18n.t('errors.upload.unsupported_format'))
      else if (msg.includes('too large')) setError(i18n.t('errors.upload.too_large').replace('{limit}', String(maxMB)))
      else setError(msg)
    } finally { setBusy(false) }
  }

  const onNext = () => { if (stepIdx < steps.length - 1) setStepIdx(stepIdx + 1) }
  const onPrev = () => { if (stepIdx > 0) setStepIdx(stepIdx - 1) }

  const onFinalize = async () => {
    if (!sessionId) return
    setBusy(true)
    try {
      await finalize(sessionId)
      setError(null)
      setRecap({ id: sessionId, at: new Date().toISOString() })
    } catch (e:any) {
      setError(String(e?.message || 'Could not finalize'))
    } finally { setBusy(false) }
  }

  if (!sessionId) {
    return (
      <form onSubmit={onCreate} className="container" style={{ display: 'grid', gap: 12, padding: 16, maxWidth: 680 }}>
        <h2>{i18n.t('collect.title')}</h2>
        {error && <div className="alert alert-danger" style={{margin:0}}>{error}</div>}
        <label>
          <input className="form-control" name="name" placeholder={i18n.t('collect.name')} />
          <div className="text-muted" style={{fontSize:12, marginTop:4}}>Optional. If you want, we can list this name in credits later.</div>
        </label>
        <label>
          <input className="form-control" name="handle" placeholder={i18n.t('collect.handle')} />
          <div className="text-muted" style={{fontSize:12, marginTop:4}}>Optional. Social handle for credits (e.g. @name). Leave blank to stay anonymous.</div>
        </label>
        <input className="form-control" name="email" placeholder={i18n.t('collect.email')} type="email" required />
        <div>
          <div className="text-muted" style={{fontSize:12}}>Pick what you’re photographing:</div>
          <div style={{display:'grid', gap:8, gridTemplateColumns:'repeat(2, minmax(0, 1fr))'}}>
            <button type="button" onClick={()=>setMode('MIXED_UNIQUES')} className={"mode-card" + (mode==='MIXED_UNIQUES'?' selected':'')} aria-pressed={mode==='MIXED_UNIQUES'}>
              <img src="/slides/mixed/1.svg" alt="Mixed Uniques example" />
              <div className="overlay"><div className="title">Mixed Uniques</div><div className="subtitle">Different single socks</div></div>
            </button>
            <button type="button" onClick={()=>setMode('SAME_TYPE')} className={"mode-card" + (mode==='SAME_TYPE'?' selected':'')} aria-pressed={mode==='SAME_TYPE'}>
              <img src="/slides/same/1.svg" alt="Same Type example" />
              <div className="overlay"><div className="title">Same Type</div><div className="subtitle">Multiple identical socks</div></div>
            </button>
          </div>
        </div>
        <div className="form-check">
          <input className="form-check-input" id="notify" name="notify" type="checkbox" />
          <label className="form-check-label" htmlFor="notify">{i18n.t('collect.notify')}</label>
        </div>
        <div className="form-check">
          <input className="form-check-input" id="consent" type="checkbox" checked={consent} onChange={e=>setConsent(e.target.checked)} />
          <label className="form-check-label" htmlFor="consent">{i18n.t('collect.consent')}</label>
        </div>
        <small className="text-muted">{i18n.t('note.privacy')} <a href="#terms">Terms</a> · <a href="#privacy">Privacy</a></small>
        <button className="btn btn-primary" disabled={busy || !consent} type="submit">{i18n.t('collect.start')}</button>
      </form>
    )
  }

  if (recap) {
    return (
      <div style={{ padding: 16, display:'grid', gap:12, maxWidth: 640, margin: '0 auto' }}>
        <h2>{i18n.t('recap.title')}</h2>
        <div className="text-muted">{new Date(recap.at).toLocaleString()}</div>
        <div style={{display:'flex', alignItems:'center', gap:8}}>
          <span>{i18n.t('recap.id')}:</span>
          <code style={{background:'#f5f5f5', padding:'4px 6px', borderRadius:4}}>{recap.id}</code>
          <button className="btn btn-light btn-sm" onClick={()=>{navigator.clipboard.writeText(recap.id)}} title="Copy">Copy</button>
        </div>
        <div className="small text-muted">{i18n.t('recap.nudge')}</div>
        <div style={{display:'flex', gap:8}}>
          <button onClick={()=>{ setSessionId(null); setUploadedByStep({}); setRemaining(null); setError(null); setWarnLowRes(null); setStepIdx(0); setRecap(null); }}>Start another contribution</button>
        </div>
        {/* bubble level indicator */}
        {tilt && (
          <div style={{ position:'absolute', left:8, top:8, width:64, height:64, borderRadius:32, background:'rgba(255,255,255,0.7)', border:'2px solid rgba(0,0,0,0.2)', display:'flex', alignItems:'center', justifyContent:'center' }}>
            {(() => {
              const clamp = (v:number, lo:number, hi:number)=> Math.max(lo, Math.min(hi, v))
              const x = clamp((tilt.gamma||0)/30, -1, 1) * 22
              const y = clamp((tilt.beta||0)/30, -1, 1) * 22
              return <div style={{ width:14, height:14, borderRadius:7, background:'#1976d2', transform:`translate(${x}px, ${y}px)` }} />
            })()}
          </div>
        )}
      </div>
    )
  }

  const step = steps[stepIdx]
  const uploadedThisStep = uploadedByStep[stepIdx] || 0
  const imgSrc = mode === 'SAME_TYPE' ? `/slides/same/${stepIdx+1}.svg` : `/slides/mixed/${stepIdx+1}.svg`
  const alt = step.title

  return (
    <div style={{ padding: 16, display:'grid', gap:12, maxWidth: 640, margin: '0 auto' }}>
      {error && <div className="alert alert-danger" style={{margin:0}}>{error}</div>}
      <div style={{display:'flex', gap:6}}>
        {steps.map((s, i) => (
          <div key={s.key} title={s.title} style={{width:12, height:12, borderRadius:6, background: i<stepIdx? '#4caf50' : i===stepIdx? '#1976d2' : '#ccc'}} />
        ))}
      </div>
      <h3 style={{display:'flex', alignItems:'center', gap:8}}>
        {step.title}
        <span className="text-muted" title="Do not move socks between photos. Flip or crumple in place."><i className="bi bi-info-circle"></i></span>
      </h3>
      <div style={{ position:'relative', border: '1px solid #dee2e6', height: 220, marginBottom: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow:'hidden', borderRadius: 6 }}>
        <img src={imgSrc} alt={alt} style={{ maxWidth:'100%', maxHeight:'100%', objectFit:'contain' }} />
        {/* bubble level indicator */}
        {tilt && (
          <div style={{ position:'absolute', left:8, top:8, width:64, height:64, borderRadius:32, background:'rgba(255,255,255,0.7)', border:'2px solid rgba(0,0,0,0.2)', display:'flex', alignItems:'center', justifyContent:'center' }}>
            {(() => {
              const clamp = (v:number, lo:number, hi:number)=> Math.max(lo, Math.min(hi, v))
              const x = clamp((tilt.gamma||0)/30, -1, 1) * 22
              const y = clamp((tilt.beta||0)/30, -1, 1) * 22
              return <div style={{ width:14, height:14, borderRadius:7, background:'#1976d2', transform:`translate(${x}px, ${y}px)` }} />
            })()}
          </div>
        )}
      </div>
      <div className="d-flex align-items-center gap-2">
        <input className="form-control" ref={inputRef} type="file" accept="image/*" multiple capture="environment" onChange={e=>onUpload(e.target.files)} />
        <span className="text-muted">{uploadedThisStep} uploaded for this step</span>
      </div>
      {needsPerm && <div className="alert alert-secondary py-2">On iOS Safari, enable motion sensors to show level/tilt hints. <button className="btn btn-sm btn-primary ms-2" onClick={requestOrientationPermission}>Enable guidance</button></div>}
      {warnLowRes && <div className="alert alert-warning py-1 mb-0">{warnLowRes}</div>}
      {remaining !== null && <div className="small text-muted">{i18n.t('upload.remaining')}: {remaining} / cap {cap} (total uploaded ~ {totalUploaded})</div>}

      <div className="d-flex gap-2">
        <button className="btn btn-secondary" onClick={onPrev} disabled={stepIdx===0 || busy}>Back</button>
        <button className="btn btn-primary" onClick={onNext} disabled={stepIdx===steps.length-1 || busy}>Next</button>
      </div>
      {stepIdx===steps.length-1 && (
        <div>
          <hr />
          <button className="btn btn-success mt-1" onClick={onFinalize} disabled={busy}>{i18n.t('finalize')}</button>
        </div>
      )}
    </div>
  )
}
