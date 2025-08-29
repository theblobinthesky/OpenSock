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
  return (
    <div className="container my-3">
      <h2 className="mb-3">Admin</h2>
      <div className="row g-2 align-items-end">
        <div className="col-md-2"><input className="form-control" placeholder="Password" type="password" value={password} onChange={e=>setPassword(e.target.value)} /></div>
        <div className="col-md-3"><input className="form-control" placeholder="Filter email" value={email} onChange={e=>setEmail(e.target.value)} /></div>
        <div className="col-md-3"><input className="form-control" placeholder="Filter name" value={name} onChange={e=>setName(e.target.value)} /></div>
        <div className="col-md-2">
          <select className="form-select" value={mode} onChange={e=>setMode(e.target.value)}>
            <option value="">Any mode</option>
            <option value="MIXED_UNIQUES">Mixed Uniques</option>
            <option value="SAME_TYPE">Same Type</option>
          </select>
        </div>
        <div className="col-md-2">
          <select className="form-select" value={status} onChange={e=>setStatus(e.target.value)}>
            <option value="">Any status</option>
            <option value="finalized">Finalized</option>
            <option value="open">Not finalized</option>
          </select>
        </div>
        <div className="col-md-2 d-flex gap-1">
          <button className="btn btn-primary w-100" onClick={fetchSessions}>Search</button>
          <button className="btn btn-outline-secondary" onClick={()=>{
            const rows = [["id","email","name","handle","mode","image_count","created_at","finalized_at"]].concat(
              sessions.map(s=>[s.id,s.email,s.name||"",s.handle||"",s.mode,String(s.image_count), new Date(s.created_at).toISOString(), s.finalized_at||""])
            )
            const csv = rows.map(r=>r.map(x=>`"${String(x).replace(/"/g,'""')}"`).join(',')).join('\n')
            const blob = new Blob([csv], {type:'text/csv'})
            const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'sessions.csv'; a.click(); URL.revokeObjectURL(a.href)
          }}>CSV</button>
        </div>
      </div>
      <div className="mt-2 d-flex gap-2">
        <button className="btn btn-outline-danger btn-sm" onClick={deleteByEmail}>Delete all for email</button>
      </div>
      <div className="card mt-2">
        <div className="card-body">
          <h5 className="card-title">Config</h5>
          <div className="row g-2">
            <div className="col-4"><label className="form-label">Session cap</label><input className="form-control" type="number" value={cfg.cap} onChange={e=>setCfg({...cfg, cap:parseInt(e.target.value||'0')||0})} /></div>
            <div className="col-4"><label className="form-label">Min (Mixed)</label><input className="form-control" type="number" value={cfg.minMixed} onChange={e=>setCfg({...cfg, minMixed:parseInt(e.target.value||'0')||0})} /></div>
            <div className="col-4"><label className="form-label">Min (Same)</label><input className="form-control" type="number" value={cfg.minSame} onChange={e=>setCfg({...cfg, minSame:parseInt(e.target.value||'0')||0})} /></div>
            <div className="col-4"><label className="form-label">Max file size (MB)</label><input className="form-control" type="number" value={cfg.maxMB||25} onChange={e=>setCfg({...cfg, maxMB:parseInt(e.target.value||'0')||0})} /></div>
          </div>
          <div className="mt-2"><button className="btn btn-primary btn-sm" onClick={saveCfg}>Save</button> <button className="btn btn-outline-secondary btn-sm" onClick={loadCfg}>Reload</button></div>
        </div>
      </div>
      {error && <div className="alert alert-danger mt-2">{error}</div>}
      <div className="table-responsive mt-3">
        <table className="table table-striped table-hover table-sm align-middle">
          <thead>
            <tr>
              <th>ID</th><th>Email</th><th>Name</th><th>Handle</th><th>Mode</th><th>Images</th><th>Created</th><th>Finalized</th><th></th>
            </tr>
          </thead>
          <tbody>
          {sessions.filter(s=> status==='' ? true : status==='finalized' ? !!s.finalized_at : !s.finalized_at)
            .slice((page-1)*pageSize, page*pageSize).map(s => (
            <tr key={s.id}>
                <td><code className="small">{s.id}</code> <button className="btn btn-light btn-sm" onClick={()=>navigator.clipboard.writeText(s.id)} title="Copy">Copy</button></td>
                <td>{s.email}</td>
                <td>{s.name || ''}</td>
                <td>{s.handle || ''}</td>
                <td>{s.mode}</td>
                <td>{s.image_count}</td>
                <td>{new Date(s.created_at).toLocaleString()}</td>
                <td>{s.finalized_at ? new Date(s.finalized_at).toLocaleString() : <span className="badge text-bg-warning">Not finalized</span>}</td>
                <td className="text-end"><button className="btn btn-outline-danger btn-sm" onClick={()=>deleteSession(s.id)}>Delete</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="d-flex gap-2 align-items-center">
        <button className="btn btn-outline-secondary btn-sm" disabled={page<=1} onClick={()=>setPage(p=>Math.max(1,p-1))}>Prev</button>
        <span>Page {page} / {Math.max(1, Math.ceil(sessions.length/pageSize))}</span>
        <button className="btn btn-outline-secondary btn-sm" disabled={page>=Math.ceil(sessions.length/pageSize)} onClick={()=>setPage(p=>Math.min(Math.ceil(sessions.length/pageSize), p+1))}>Next</button>
      </div>
    </div>
  )
}
