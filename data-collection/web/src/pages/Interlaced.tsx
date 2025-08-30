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
  // Mixed flow: Flat, Flip, Crumple, then Extras (optional, supports multiple images)
  return [
    { key: 'mixed.1', title: i18n.t('slides.mixed.1.title'), required: true },
    { key: 'mixed.2', title: i18n.t('slides.mixed.2.title'), required: true },
    { key: 'mixed.3', title: i18n.t('slides.mixed.3.title'), required: true },
    { key: 'mixed.4', title: i18n.t('slides.mixed.4.title'), required: true },
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
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const cameraInputRef = useRef<HTMLInputElement | null>(null)
  const progressRef = useRef<HTMLDivElement | null>(null)
  const currentStepRef = useRef<HTMLDivElement | null>(null)
  const [previewsByStep, setPreviewsByStep] = useState<Record<number, string[]>>({})
  const [replacedByStep, setReplacedByStep] = useState<Record<number, boolean>>({})
  const [recap, setRecap] = useState<{ id: string, at: string } | null>(null)
  const [warnLowRes, setWarnLowRes] = useState<string | null>(null)
  // removed: 4:3 framing overlay toggle and 3x3 grid overlay
  const [tilt, setTilt] = useState<{beta:number, gamma:number} | null>(null)
  const [needsPerm, setNeedsPerm] = useState<boolean>(false)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const pendingRef = useRef<null | {
    name: string
    handle: string
    email: string
    notify_opt_in: boolean
    language: string
    mode: Mode
  }>(null)
  const [copied, setCopied] = useState<boolean>(false)

  const steps = useMemo(() => stepsFor(mode), [mode])
  const totalUploaded = Object.values(uploadedByStep).reduce((a, b) => a + b, 0)

  React.useEffect(()=>{
    // load config
    getConfig().then(cfg=>{
      const capVal = Number((cfg as any).session_image_cap)
      if (!Number.isFinite(capVal) || capVal <= 0) setCap(20); else setCap(capVal)
      // @ts-ignore: file_size_limit_mb may be present
      if ((cfg as any).file_size_limit_mb) setMaxMB((cfg as any).file_size_limit_mb)
    }).catch(()=>{})
  },[])

  // Auto-scroll progress bar to active step
  React.useEffect(() => {
    const el = currentStepRef.current
    if (!el) return
    // Use scrollIntoView to center the active blob horizontally
    try {
      el.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' })
    } catch {
      // Fallback: manual scroll calculation
      const container = progressRef.current
      if (!container) return
      const rect = el.getBoundingClientRect()
      const crect = container.getBoundingClientRect()
      const offset = (rect.left + rect.right) / 2 - (crect.left + crect.right) / 2
      container.scrollLeft += offset
    }
  }, [stepIdx])

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
    if (!consent) { setError(i18n.t('errors.consent_required')); return }
    setError(null)
    const fd = new FormData(e.currentTarget)
    pendingRef.current = {
      name: String(fd.get('name') || ''),
      handle: String(fd.get('handle') || ''),
      email: String(fd.get('email') || ''),
      notify_opt_in: !!fd.get('notify'),
      language: i18n.lang,
      mode: mode,
    }
    setConfirmOpen(true)
  }

  const proceedCreate = async () => {
    if (!pendingRef.current) return
    const payload = pendingRef.current
    setConfirmOpen(false)
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
          setError(i18n.t('errors.email_required'))
        } else {
          setError(e.message)
        }
      } catch { setError(e.message) }
    } finally { setBusy(false) }
  }

  const onUpload = async (files: FileList | null) => {
    if (!sessionId || !files || files.length === 0) return
    // If cap reached, nudge user to submit
    if (totalUploaded >= cap) {
      setError(i18n.t('errors.upload.at_limit'))
      return
    }
    // soft low-res nudge
    setWarnLowRes(null)
    try {
      const f0 = files[0]
      const url = URL.createObjectURL(f0)
      const img = new Image()
      img.src = url
      await img.decode()
      if (img.width < 1024) setWarnLowRes(i18n.t('upload.low_res_hint'))
      URL.revokeObjectURL(url)
    } catch {}
    setBusy(true)
    try {
      const arr = Array.from(files)
      const prevLen = (previewsByStep[stepIdx] || []).length
      const resp = await uploadImages(sessionId, arr)
      const newUploadedThisStep = (uploadedByStep[stepIdx] || 0) + resp.uploaded
      const currentTotal = totalUploaded
      const remainingLocal = Math.max(0, cap - (currentTotal + resp.uploaded))
      setRemaining(remainingLocal)
      setUploadedByStep((prev) => ({ ...prev, [stepIdx]: newUploadedThisStep }))
      if (resp.uploaded === 0) {
        setError(i18n.t('errors.upload.duplicate'))
      } else {
        setError(null)
        // Only add previews for actually-uploaded files
        const acc = arr.slice(0, resp.uploaded).map(f=>URL.createObjectURL(f))
        setPreviewsByStep(prev=>{
          const prevList = prev[stepIdx] || []
          if (step.required) {
            return { ...prev, [stepIdx]: acc.length ? [acc[0]] : prevList }
          }
          const next = prevList.concat(acc).slice(-8)
          return { ...prev, [stepIdx]: next }
        })
        if (step.required) {
          setReplacedByStep(prev => ({ ...prev, [stepIdx]: prevLen > 0 && resp.uploaded > 0 }))
        }
        if (remainingLocal === 0) {
          setError(i18n.t('errors.upload.at_limit'))
        }
      }
      if (fileInputRef.current) fileInputRef.current.value = ''
      if (cameraInputRef.current) cameraInputRef.current.value = ''
    } catch (e:any) {
      const msg = String(e?.message||'')
      if (msg.includes('unsupported')) setError(i18n.t('errors.upload.unsupported_format'))
      else if (msg.includes('too large')) setError(i18n.t('errors.upload.too_large').replace('{limit}', String(maxMB)))
      else setError(msg)
    } finally { setBusy(false) }
  }

  const onNext = async () => {
    if (stepIdx < steps.length - 1) {
      setStepIdx(stepIdx + 1)
      setError(null); setWarnLowRes(null)
      try { window.scrollTo({ top: 0, behavior: 'smooth' }) } catch {}
    } else {
      await onFinalize()
    }
  }
  const onPrev = () => { if (stepIdx > 0) { setStepIdx(stepIdx - 1); setError(null); setWarnLowRes(null); try { window.scrollTo({ top: 0, behavior: 'smooth' }) } catch {} } }

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
      <form onSubmit={onCreate} className="grid gap-4 p-4 max-w-2xl mx-auto overflow-x-hidden relative">
        {confirmOpen && (
          <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center px-4">
            <div className="w-full max-w-md bg-white rounded-xl shadow-xl ring-1 ring-gray-200 p-4 grid gap-3">
              <div className="text-lg font-semibold">{i18n.t('confirm.title')}</div>
              <div className="text-sm text-gray-700">
                <p className="mb-2">{i18n.t('confirm.note')}</p>
                <ul className="list-disc pl-5 space-y-1">
                  <li>{i18n.t('rules.shared')}</li>
                  <li>{mode==='MIXED_UNIQUES' ? i18n.t('rules.mixed') : i18n.t('rules.same')}</li>
                </ul>
              </div>
              <div className="flex gap-2 justify-end">
                <button type="button" className="px-3 py-2 rounded border border-gray-300 hover:bg-gray-50 text-sm" onClick={()=>setConfirmOpen(false)}>{i18n.t('confirm.cancel')}</button>
                <button type="button" className="px-3 py-2 rounded bg-brand-600 text-white hover:bg-brand-700 text-sm" onClick={proceedCreate}>{i18n.t('confirm.accept')}</button>
              </div>
            </div>
          </div>
        )}
        {/* About OpenSock */}
        <div className="text-gray-800 bg-white rounded-xl ring-1 ring-gray-200 p-4">
          <div className="text-lg font-semibold mb-2">{i18n.t('about.title')}</div>
          <div className="text-[15px] leading-6 space-y-2 mb-3">
            <p>{i18n.t('about.purpose')}</p>
            <p>{i18n.t('about.family')}</p>
            <p>{i18n.t('about.why')}</p>
          </div>
          <div>
            <div className="text-sm font-medium text-gray-700 mb-2">{i18n.t('about.modes.title')}</div>
            <ul className="list-disc pl-5 text-[15px] leading-6 space-y-1.5">
              <li>{i18n.t('about.modes.mixed')}</li>
              <li>{i18n.t('about.modes.same')}</li>
            </ul>
          </div>
          <div className="mt-4">
            <button type="button" className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-brand-600 text-white hover:bg-brand-700"
              onClick={()=>{ const el = document.getElementById('contribute-form-start'); if (el) el.scrollIntoView({behavior:'smooth', block:'start'}); }}>
              {i18n.t('contribute.scroll_cta')}
            </button>
          </div>
        </div>
        <h2 className="text-xl font-semibold">{i18n.t('contribute.title')}</h2>
        {error && <div className="flex items-center gap-2 text-red-700 bg-red-50 border border-red-200 px-3 py-2 rounded">{error}</div>}
        <div id="contribute-form-start" />
        <label className="grid gap-1">
          <span className="text-sm text-gray-700">{i18n.t('collect.email_label')} <span className="text-red-600">{i18n.t('collect.required')}</span></span>
          <input className="w-full rounded border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-500" name="email" placeholder={i18n.t('collect.email')} type="email" required />
        </label>
        <label className="grid gap-1">
          <span className="text-sm text-gray-700">{i18n.t('collect.name_label')} <span className="text-gray-400">{i18n.t('collect.optional')}</span></span>
          <input className="w-full rounded border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-500" name="name" placeholder={i18n.t('collect.name')} />
          <div className="text-xs text-gray-500">{i18n.t('collect.name_help')}</div>
        </label>
        <label className="grid gap-1">
          <span className="text-sm text-gray-700">{i18n.t('collect.handle_label')} <span className="text-gray-400">{i18n.t('collect.optional')}</span></span>
          <input className="w-full rounded border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-500" name="handle" placeholder={i18n.t('collect.handle')} />
          <div className="text-xs text-gray-500">{i18n.t('collect.handle_help')}</div>
        </label>
        <div>
          <div className="text-xs text-gray-500 mb-2">{i18n.t('collect.pick')}</div>
          <div className="grid grid-cols-2 gap-2">
            <button type="button" onClick={()=>setMode('MIXED_UNIQUES')} className={`relative overflow-hidden rounded-lg ring-2 ${mode==='MIXED_UNIQUES'?'ring-brand-500':'ring-gray-200'} hover:ring-brand-400 transition`}
              aria-pressed={mode==='MIXED_UNIQUES'}>
              <img src="/slides/mixed/1.svg" alt={`${i18n.t('mode.mixed.title')} example`} className="w-full h-32 object-cover" />
              <div className="absolute inset-x-0 bottom-0 px-2 pt-8 pb-2 text-white bg-gradient-to-t from-black/70 via-black/40 to-transparent min-h-20">
                <div className="font-semibold text-sm">{i18n.t('mode.mixed.title')}</div>
                <div className="text-[11px] opacity-90">{i18n.t('mode.mixed.sub')}</div>
              </div>
            </button>
            <button type="button" onClick={()=>setMode('SAME_TYPE')} className={`relative overflow-hidden rounded-lg ring-2 ${mode==='SAME_TYPE'?'ring-brand-500':'ring-gray-200'} hover:ring-brand-400 transition`}
              aria-pressed={mode==='SAME_TYPE'}>
              <img src="/slides/same/1.svg" alt={`${i18n.t('mode.same.title')} example`} className="w-full h-32 object-cover" />
              <div className="absolute inset-x-0 bottom-0 px-2 pt-8 pb-2 text-white bg-gradient-to-t from-black/70 via-black/40 to-transparent min-h-20">
                <div className="font-semibold text-sm">{i18n.t('mode.same.title')}</div>
                <div className="text-[11px] opacity-90">{i18n.t('mode.same.sub')}</div>
              </div>
            </button>
          </div>
        </div>
        <label className="flex items-center gap-2">
          <input className="w-4 h-4" id="notify" name="notify" type="checkbox" />
          <span className="text-sm">{i18n.t('collect.notify')}</span>
        </label>
        <label className="flex items-center gap-2">
          <input className="w-4 h-4" id="consent" type="checkbox" checked={consent} onChange={e=>setConsent(e.target.checked)} />
          <span className="text-sm">{i18n.t('collect.consent')}</span>
        </label>
        <small className="text-xs text-gray-500">{i18n.t('note.privacy')} <a className="underline" href="#terms">Terms</a> · <a className="underline" href="#privacy">Privacy</a></small>
        <button className="inline-flex items-center justify-center gap-2 rounded-md bg-brand-600 text-white px-4 py-2 hover:bg-brand-700 disabled:opacity-50" disabled={busy || !consent} type="submit">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5"><path d="M12 2.25c.414 0 .75.336.75.75v5.19l1.72-1.72a.75.75 0 1 1 1.06 1.06l-3 3a.75.75 0 0 1-1.06 0l-3-3a.75.75 0 1 1 1.06-1.06l1.72 1.72V3a.75.75 0 0 1 .75-.75ZM3 13.5a.75.75 0 0 1 .75-.75h16.5a.75.75 0 0 1 .75.75v5.25A2.25 2.25 0 0 1 18.75 21H5.25A2.25 2.25 0 0 1 3 18.75V13.5Z"/></svg>
          {i18n.t('collect.start')}
        </button>
      </form>
    )
  }

  if (recap) {
    return (
      <div className="px-4 py-8">
        <div className="max-w-md mx-auto bg-white rounded-2xl shadow ring-1 ring-gray-200 p-6 text-center grid gap-4">
          <div className="mx-auto w-12 h-12 rounded-full bg-green-100 text-green-700 flex items-center justify-center">
            <img src="/icons/check.svg" alt="Success" className="w-8 h-8" />
          </div>
          <div>
            <h2 className="text-xl font-semibold">{i18n.t('recap.title') || 'Contribution submitted'}</h2>
            <div className="text-sm text-gray-600">{new Date(recap.at).toLocaleString()}</div>
          </div>
          <div className="flex items-center justify-center gap-2 flex-wrap">
            <span className="text-sm text-gray-600">{i18n.t('recap.id') || 'ID'}:</span>
            <div className="relative inline-block max-w-[240px] sm:max-w-[320px] w-full">
              <code className="block text-sm bg-gray-50 px-2 py-1 rounded border border-gray-200 whitespace-nowrap overflow-hidden">{recap.id}</code>
              <div className="pointer-events-none absolute right-0 top-0 h-full w-10 bg-gradient-to-l from-gray-50 to-transparent rounded-r"></div>
            </div>
            <button
              className={`px-2 py-1 rounded border text-sm transition-all active:scale-95 ${copied ? 'bg-green-50 text-green-700 border-green-300' : 'border-gray-300 hover:bg-gray-50'}`}
              onClick={()=>{ navigator.clipboard.writeText(recap.id); setCopied(true); setTimeout(()=>setCopied(false), 1200) }}
              title="Copy"
            >
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>
          <div className="text-sm text-gray-600">{i18n.t('recap.nudge')}</div>
          <div>
            <button className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-brand-600 text-white hover:bg-brand-700" onClick={()=>{ setSessionId(null); setUploadedByStep({}); setRemaining(null); setError(null); setWarnLowRes(null); setStepIdx(0); setRecap(null); }}>
              {i18n.t('recap.start_another')}
            </button>
          </div>
        </div>
      </div>
    )
  }

  const step = steps[stepIdx]
  const uploadedThisStep = uploadedByStep[stepIdx] || 0
  function imgSrcFor(stepKey: string): string {
    const [kind, index] = stepKey.split('.')
    if (kind === 'mixed') return `/slides/mixed/${index}.svg`
    return `/slides/same/${index}.svg`
  }
  const imgSrc = imgSrcFor(step.key)
  const alt = step.title

  // slide descriptions via i18n
  const desc = mode === 'SAME_TYPE' ? [
    i18n.t('slides.same.1.desc'),
    i18n.t('slides.same.2.desc'),
    i18n.t('slides.same.3.desc'),
    i18n.t('slides.same.4.desc'),
  ] : [
    i18n.t('slides.mixed.1.desc'),
    i18n.t('slides.mixed.2.desc'),
    i18n.t('slides.mixed.3.desc'),
    i18n.t('slides.mixed.4.desc'),
    i18n.t('slides.mixed.5.desc'),
  ]

  const infoText = i18n.t('mode.preview.note')
  // Progress UI: simple dots with state icons
  const canProceed = step.required ? uploadedThisStep > 0 : true

  return (
    <div className="grid gap-4 p-4 max-w-3xl mx-auto w-full overflow-x-hidden bg-slate-50 rounded-2xl shadow-inner ring-1 ring-slate-200">
      {/* Progress indicator (track with markers) */}
      {(() => {
        return (
          <div className="px-1 pt-2 pb-3">
            <div className="mt-2 text-xs text-gray-700 text-center">{i18n.t('progress.step').replace('{x}', String(stepIdx+1)).replace('{y}', String(steps.length))}</div>
            <div className="overflow-x-auto no-scrollbar" ref={progressRef}>
              <div className="flex justify-center items-center gap-2 pr-2 mb-1 mt-1">
                {steps.map((s, i) => {
                  const state = i < stepIdx ? 'done' : i === stepIdx ? 'current' : 'todo'
                  const needs = s.required && (uploadedByStep[i]||0)===0
                  return (
                    <div key={s.key} className="flex items-center gap-2 shrink-0" ref={state==='current' ? currentStepRef : undefined}>
                      <div className={`relative w-6 h-6 rounded-full flex items-center justify-center ${state==='done'?'bg-green-500':state==='current'?'bg-brand-600':'bg-gray-300'}`} title={s.title}>
                        {state==='done' && <img src="/icons/check.svg" alt="done" className="w-3 h-3" />}
                        {state==='current' && <span className="w-1.5 h-1.5 rounded-full bg-white"></span>}
                        {needs && state!=='done' && <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-red-500 rounded-full ring-2 ring-white" aria-hidden="true"></span>}
                      </div>
                      {i < steps.length - 1 && (
                        <div className="relative w-5 h-1 rounded bg-gray-200 overflow-hidden">
                          <div className={`absolute left-0 top-0 h-full bg-green-400 transition-all duration-500 ${i<stepIdx? 'w-full' : i===stepIdx ? 'w-1/2' : 'w-0'}`}></div>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )
      })()}

      {/* Info panel (subtle) */}
      <div className="grid gap-1">
        <div className="text-sm text-gray-700 flex items-start gap-3">
          <span className="w-7 h-7 rounded-full bg-gray-200 flex items-center justify-center mt-0.5"><img src="/icons/info.svg" alt="info" className="w-4 h-4" /></span>
          <span className="leading-5">
            <div className="font-medium text-gray-800">{infoText}</div>
            <div className="text-[12px] mt-0.5 text-gray-600">{i18n.t('note.no_zoom')}</div>
            <div className="text-[12px] mt-0.5 text-gray-600">{mode==='MIXED_UNIQUES' ? i18n.t('rules.mixed') : i18n.t('rules.same')}</div>
          </span>
        </div>
        {warnLowRes && (
          <div className="text-sm text-amber-900 bg-amber-50 border border-amber-200 rounded px-3 py-2 flex items-start gap-2">
            <span className="w-5 h-5 rounded-full bg-amber-100 flex items-center justify-center mt-0.5"><img src="/icons/warning.svg" alt="warning" className="w-3 h-3" /></span>
            <span>{warnLowRes}</span>
          </div>
        )}
        {error && (
          <div className="text-sm text-red-900 bg-red-50 border border-red-200 rounded px-3 py-2 flex items-start gap-2">
            <span className="w-5 h-5 rounded-full bg-red-100 flex items-center justify-center mt-0.5"><img src="/icons/error.svg" alt="error" className="w-3 h-3" /></span>
            <span>{error}</span>
          </div>
        )}
      </div>
      {/* deduplicated: warn/error already shown above */}

      {/* Title + instruction (prominent) */}
      <div>
        <h3 className="text-xl font-semibold text-gray-900">{step.title}</h3>
        <p className="text-base font-medium text-gray-800">{desc[stepIdx]}</p>
      </div>

      {/* Illustration with bubble level */}
      <div className="relative h-56 rounded-xl flex items-center justify-center overflow-hidden bg-white">
        <div className="absolute bottom-5 right-3 text-[11px] px-3 py-1 rounded-full bg-gray-800/85 text-white shadow-sm">{i18n.t('example.caption')}</div>
        <img src={imgSrc} alt={alt} className="w-full h-full object-cover" />
        {tilt && (
          <div className="absolute left-2 top-2 w-16 h-16 rounded-full bg-white/70 border-2 border-black/20 flex items-center justify-center">
            {(() => {
              const clamp = (v:number, lo:number, hi:number)=> Math.max(lo, Math.min(hi, v))
              const x = clamp((tilt.gamma||0)/30, -1, 1) * 22
              const y = clamp((tilt.beta||0)/30, -1, 1) * 22
              return <div style={{ transform:`translate(${x}px, ${y}px)` }} className="w-3.5 h-3.5 rounded-full bg-brand-600" />
            })()}
          </div>
        )}
      </div>

      {/* Upload control */}
      <div className="grid gap-2">
        <input ref={cameraInputRef} className="hidden" type="file" accept="image/*" multiple={!step.required} capture="environment" onChange={e=>onUpload(e.target.files)} />
        <input ref={fileInputRef} className="hidden" type="file" accept="image/*" multiple={!step.required} onChange={e=>onUpload(e.target.files)} />
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
          <button className="inline-flex items-center justify-center gap-2 px-3 py-2 h-11 rounded-md text-white bg-brand-600 hover:bg-brand-700 disabled:opacity-60 w-full sm:w-auto" onClick={()=>cameraInputRef.current?.click()} disabled={busy}>
            <img src="/icons/camera.svg" alt="Camera" className="w-5 h-5" />
            {i18n.t('upload.take_photo')}
          </button>
          <button className="inline-flex items-center justify-center gap-2 px-3 py-2 h-11 rounded-md border border-gray-300 hover:bg-gray-50 disabled:opacity-60 w-full sm:w-auto" onClick={()=>fileInputRef.current?.click()} disabled={busy}>
            <img src="/icons/image.svg" alt="Library" className="w-5 h-5" />
            {i18n.t('upload.choose_photos')}
          </button>
        </div>
        {busy && (
          <div className="flex items-center justify-start gap-2 text-sm text-gray-600">
            <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path>
            </svg>
            {i18n.t('upload.uploading')}
          </div>
        )}
        <div className="text-sm text-gray-600">
          {step.required ? (
            uploadedThisStep === 0 ? i18n.t('upload.none_yet') : (replacedByStep[stepIdx] ? i18n.t('upload.replaced') : i18n.t('upload.added_ok'))
          ) : (
            (uploadedThisStep===1 ? i18n.t('upload.count.singular') : i18n.t('upload.count.plural')).replace('{count}', String(uploadedThisStep))
          )}
        </div>
      </div>
      {previewsByStep[stepIdx]?.length ? (
        <div className="bg-gray-50 rounded-lg p-2 ring-1 ring-gray-100">
          <div className="text-xs text-gray-500 mb-2 px-1">{i18n.t('upload.your_photos')}</div>
          <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
            {previewsByStep[stepIdx].slice(-8).map((u, i)=>(
            <img key={i} src={u} alt="preview" className="w-full h-20 object-cover rounded-md ring-1 ring-gray-200" />
            ))}
          </div>
        </div>
      ) : null}

      {needsPerm && (
        <div className="flex items-center justify-between text-gray-700 bg-gray-50 border border-gray-200 px-3 py-2 rounded">
          <span className="text-sm">{i18n.t('ios.motion.explain')}</span>
          <button className="text-sm px-2 py-1 rounded bg-brand-600 text-white" onClick={requestOrientationPermission}>{i18n.t('ios.motion.enable')}</button>
        </div>
      )}
      {/* Remaining count removed per UX request */}

      {/* Nav buttons */}
      <div className="flex gap-2">
        <button className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-gray-300 hover:bg-gray-50 disabled:opacity-50" onClick={onPrev} disabled={stepIdx===0 || busy}>
          <img src="/icons/chevron-left.svg" alt="Back" className="w-5 h-5" />
          {i18n.t('slides.prev')}
        </button>
        <button className="inline-flex items-center gap-2 px-4 py-2 rounded-md text-white bg-brand-600 hover:bg-brand-700 disabled:opacity-50"
          onClick={onNext} disabled={busy || (!canProceed)}>
          {stepIdx===steps.length-1 ? (
            <>
              <img src="/icons/send.svg" alt="Submit" className="w-5 h-5" />
              {i18n.t('finalize')}
            </>
          ) : (
            <>
              {i18n.t('slides.next')}
              <img src="/icons/chevron-right.svg" alt="Next" className="w-5 h-5" />
            </>
          )}
        </button>
      </div>
    </div>
  )
}
