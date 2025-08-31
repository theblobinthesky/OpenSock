import React, { useEffect, useState } from 'react'
import { i18n } from '../i18n'
import { adminListSessions, adminDeleteSession, adminDeleteUser, AdminSession, adminGetConfig, adminUpdateConfig } from '../api'

type Session = AdminSession

export default function Admin() {
  const [password, setPassword] = useState('')
  const [email, setEmail] = useState('')
  const [name, setName] = useState('')
  const [mode, setMode] = useState('')
  const [sessions, setSessions] = useState<Session[]>([])
  const [page, setPage] = useState(1)
  const pageSize = 10
  const [error, setError] = useState<string | null>(null)
  const [cfg, setCfg] = useState<{cap:number; minMixed:number; minSame:number; maxMB?:number}>({cap:20, minMixed:4, minSame:3, maxMB:25})
  const [status, setStatus] = useState('')
  const [copiedId, setCopiedId] = useState<string | null>(null)

  const fetchSessions = async () => {
    setError(null)
    try {
      const results = await adminListSessions(password, { email, name, mode })
      setPage(1)
      setSessions(results)
    } catch (e:any) { setError(e.message) }
  }

  const deleteSession = async (id: string) => {
    if (!confirm('Delete this session?')) return
    try { await adminDeleteSession(password, id); await fetchSessions() } catch (e:any) { setError(e.message) }
  }

  const deleteByEmail = async () => {
    
    if (!email) { setError('Email required to delete by email'); return }
    if (!confirm(`Delete all sessions for ${email}?`)) return
    try { await adminDeleteUser(password, email); await fetchSessions() } catch (e:any) { setError(e.message) }
  }

  const loadCfg = async () => {
    try { const r = await adminGetConfig(password); setCfg({cap:r.session_image_cap, minMixed:r.min_required_mixed, minSame:r.min_required_same, maxMB: r.file_size_limit_mb ?? cfg.maxMB}) } catch(e:any){ setError(e.message) }
  }
  const saveCfg = async () => {
    try { await adminUpdateConfig(password, {session_image_cap: cfg.cap, min_required_mixed: cfg.minMixed, min_required_same: cfg.minSame, file_size_limit_mb: cfg.maxMB}); await loadCfg() } catch(e:any){ setError(e.message) }
  }
  const filtered = sessions.filter(s=> (status==='' ? true : status==='finalized' ? !!s.finalized_at : !s.finalized_at))
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize))
  const disabled = password.trim().length === 0

  return (
    <div className="max-w-6xl mx-auto p-4 grid gap-4">
      <h2 className="text-2xl font-semibold">{i18n.t('admin.title')}</h2>

      {/* Access & Filters */}
      <div className="bg-white rounded-xl ring-1 ring-gray-200 p-3 sm:p-4 grid gap-3">
        <div className="text-sm text-gray-600">{i18n.t('admin.help') || 'Enter the admin password, then search or manage settings.'}</div>
        <div className="grid grid-cols-1 md:grid-cols-6 gap-2 items-end">
          <div className="md:col-span-1"><input className="w-full rounded border border-gray-300 px-3 py-2" placeholder={i18n.t('admin.password')} type="password" value={password} onChange={e=>setPassword(e.target.value)} /></div>
          <div className="md:col-span-2"><input className="w-full rounded border border-gray-300 px-3 py-2" placeholder={i18n.t('admin.filter_email')} value={email} onChange={e=>setEmail(e.target.value)} /></div>
          <div className="md:col-span-2"><input className="w-full rounded border border-gray-300 px-3 py-2" placeholder={i18n.t('admin.filter_name')} value={name} onChange={e=>setName(e.target.value)} /></div>
          <div>
            <select className="w-full rounded border border-gray-300 px-3 py-2" value={mode} onChange={e=>setMode(e.target.value)}>
              <option value="">{i18n.t('admin.mode.any')}</option>
              <option value="MIXED_UNIQUES">{i18n.t('admin.mode.mixed')}</option>
              <option value="SAME_TYPE">{i18n.t('admin.mode.same')}</option>
            </select>
          </div>
          <div>
            <select className="w-full rounded border border-gray-300 px-3 py-2" value={status} onChange={e=>setStatus(e.target.value)}>
              <option value="">{i18n.t('admin.status.any')}</option>
              <option value="finalized">{i18n.t('admin.status.finalized')}</option>
              <option value="open">{i18n.t('admin.status.open')}</option>
            </select>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <button className="inline-flex items-center justify-center px-3 py-2 rounded bg-brand-600 text-white hover:bg-brand-700 disabled:opacity-50" onClick={fetchSessions} disabled={disabled}>{i18n.t('admin.actions.search')}</button>
          <button className="px-3 py-2 rounded border border-gray-300 hover:bg-gray-50 disabled:opacity-50" disabled={disabled || sessions.length===0} onClick={()=>{
            const rows = [["id","email","name","mode","image_count","created_at","finalized_at"]].concat(
              filtered.map(s=>[s.id,s.email,s.name||"",s.mode,String(s.image_count), new Date(s.created_at).toISOString(), s.finalized_at||""])
            )
            const csv = rows.map(r=>r.map(x=>`"${String(x).replace(/"/g,'""')}"`).join(',')).join('\n')
            const blob = new Blob([csv], {type:'text/csv'})
            const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'sessions.csv'; a.click(); URL.revokeObjectURL(a.href)
          }}>{i18n.t('admin.actions.export_csv')}</button>
          <button className="px-3 py-2 rounded border border-gray-300 hover:bg-gray-50" onClick={()=>{ setEmail(''); setName(''); setMode(''); setStatus(''); }}>{i18n.t('admin.actions.reset_filters')}</button>
          <button className="ml-auto px-3 py-2 rounded border border-red-300 text-red-700 hover:bg-red-50 disabled:opacity-50" disabled={disabled || !email} onClick={deleteByEmail}>{i18n.t('admin.actions.delete_by_email')}</button>
        </div>
      </div>

      {/* Config */}
      <div className="bg-white rounded-xl ring-1 ring-gray-200 p-3 sm:p-4">
        <div className="flex items-center justify-between mb-3">
          <h5 className="font-semibold">{i18n.t('admin.config.title')}</h5>
          <div className="text-xs text-gray-500">{i18n.t('admin.config.subtitle')}</div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <label className="text-sm grid gap-1">{i18n.t('admin.config.session_cap')}<input className="w-full rounded border border-gray-300 px-2 py-1" type="number" value={cfg.cap} onChange={e=>setCfg({...cfg, cap:parseInt(e.target.value||'0')||0})} /></label>
          <label className="text-sm grid gap-1">{i18n.t('admin.config.min_mixed')}<input className="w-full rounded border border-gray-300 px-2 py-1" type="number" value={cfg.minMixed} onChange={e=>setCfg({...cfg, minMixed:parseInt(e.target.value||'0')||0})} /></label>
          <label className="text-sm grid gap-1">{i18n.t('admin.config.min_same')}<input className="w-full rounded border border-gray-300 px-2 py-1" type="number" value={cfg.minSame} onChange={e=>setCfg({...cfg, minSame:parseInt(e.target.value||'0')||0})} /></label>
          <label className="text-sm grid gap-1">{i18n.t('admin.config.max_file')}<input className="w-full rounded border border-gray-300 px-2 py-1" type="number" value={cfg.maxMB||25} onChange={e=>setCfg({...cfg, maxMB:parseInt(e.target.value||'0')||0})} /></label>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <button className="px-3 py-1.5 rounded bg-brand-600 text-white hover:bg-brand-700 text-sm disabled:opacity-50" onClick={saveCfg} disabled={disabled}>{i18n.t('admin.actions.save')}</button>
          <button className="px-3 py-1.5 rounded border border-gray-300 hover:bg-gray-50 text-sm disabled:opacity-50" onClick={loadCfg} disabled={disabled}>{i18n.t('admin.actions.reload')}</button>
        </div>
      </div>

      {error && <div className="text-sm text-red-900 bg-red-50 border border-red-200 px-3 py-2 rounded">{error}</div>}

      {/* Sessions */}
      <div className="bg-white rounded-xl ring-1 ring-gray-200 p-3 sm:p-4 grid gap-3">
        <div className="flex items-center justify-between">
          <div className="font-semibold">{i18n.t('admin.sessions.title')}</div>
          <div className="text-xs text-gray-500">{i18n.t('admin.sessions.summary').replace('{shown}', String(filtered.length)).replace('{total}', String(sessions.length))}</div>
        </div>
        <div className="overflow-auto rounded-lg ring-1 ring-gray-100">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50/70">
              <tr className="text-left">
                <th className="p-2">{i18n.t('admin.table.id')}</th><th className="p-2">{i18n.t('admin.table.email')}</th><th className="p-2">{i18n.t('admin.table.name')}</th><th className="p-2">{i18n.t('admin.table.mode')}</th><th className="p-2">{i18n.t('admin.table.images')}</th><th className="p-2">{i18n.t('admin.table.created')}</th><th className="p-2">{i18n.t('admin.table.finalized')}</th><th className="p-2">{i18n.t('admin.table.actions')}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
            {filtered
              .slice((page-1)*pageSize, page*pageSize).map(s => (
              <tr key={s.id} className="even:bg-gray-50/40 hover:bg-gray-50">
                  <td className="p-2 align-top">
                    <div className="flex items-center gap-1">
                      <code className="text-xs">{s.id}</code>
                      <button className={`ml-1 px-2 py-0.5 rounded border text-xs transition-all active:scale-95 ${copiedId===s.id ? 'bg-green-50 text-green-700 border-green-300' : 'border-gray-300 hover:bg-gray-50'}`} onClick={()=>{navigator.clipboard.writeText(s.id); setCopiedId(s.id); setTimeout(()=>setCopiedId(null), 1200)}} title={i18n.t('common.copy')}>{copiedId===s.id? i18n.t('common.copied') : i18n.t('common.copy')}</button>
                    </div>
                  </td>
                  <td className="p-2 align-top">{s.email}</td>
                  <td className="p-2 align-top">{s.name || ''}</td>
                  <td className="p-2 align-top">{s.mode}</td>
                  <td className="p-2 align-top">{s.image_count}</td>
                  <td className="p-2 align-top">{new Date(s.created_at).toLocaleString()}</td>
                  <td className="p-2 align-top">{s.finalized_at ? new Date(s.finalized_at).toLocaleString() : <span className="inline-flex items-center px-2 py-0.5 rounded bg-amber-100 text-amber-800 text-xs">Not finalized</span>}</td>
                  <td className="p-2 align-top text-right"><button className="px-2 py-1 rounded border border-red-300 text-red-700 hover:bg-red-50 text-xs" onClick={()=>deleteSession(s.id)}>{i18n.t('common.delete')}</button></td>
              </tr>
            ))}
            </tbody>
          </table>
        </div>
        <div className="flex gap-2 items-center">
          <button className="px-2 py-1 rounded border border-gray-300 hover:bg-gray-50 text-sm disabled:opacity-50" disabled={page<=1} onClick={()=>setPage(p=>Math.max(1,p-1))}>{i18n.t('common.prev')}</button>
          <span className="text-sm">Page {page} / {pageCount}</span>
          <button className="px-2 py-1 rounded border border-gray-300 hover:bg-gray-50 text-sm disabled:opacity-50" disabled={page>=pageCount} onClick={()=>setPage(p=>Math.min(pageCount, p+1))}>{i18n.t('common.next')}</button>
        </div>
      </div>
    </div>
  )
}
