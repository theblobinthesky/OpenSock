// Prefer explicit env, otherwise derive from current host with port fallback
const GUESSED_API_BASE = (() => {
  try {
    const { protocol, hostname } = window.location
    const port = (import.meta.env.VITE_API_PORT as string) || '8080'
    return `${protocol}//${hostname}:${port}`
  } catch {
    return 'http://localhost:8080'
  }
})()
function isLocalHost(name: string) {
  return name === 'localhost' || name === '127.0.0.1' || name === '::1'
}

let API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) || GUESSED_API_BASE
try {
  const url = new URL(API_BASE)
  const pageHost = window.location.hostname
  if (isLocalHost(url.hostname) && !isLocalHost(pageHost)) {
    const port = (import.meta.env.VITE_API_PORT as string) || (url.port || '8080')
    API_BASE = `${window.location.protocol}//${pageHost}:${port}`
  }
} catch {}
const CONNECT_BASE = `${API_BASE}/opensock.dc.v1.DataCollectionService`

export async function createSession(payload: {
  name?: string
  handle?: string
  email?: string
  notify_opt_in?: boolean
  language?: string
  mode?: "MIXED_UNIQUES" | "SAME_TYPE"
}) {
  const res = await fetch(`${CONNECT_BASE}/CreateSession`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: payload.name || "",
      handle: payload.handle || "",
      email: payload.email || "",
      notifyOptIn: !!payload.notify_opt_in,
      language: payload.language || "",
      mode: payload.mode || "MIXED_UNIQUES",
    }),
  })
  if (!res.ok) throw new Error('create session failed')
  const data = await res.json() as { sessionId: string }
  return { session_id: data.sessionId }
}

// Upload via UploadService (chunked)
const UPLOAD_BASE = `${API_BASE}/opensock.dc.v1.UploadService`
const CHUNK_SIZE = 5 * 1024 * 1024

async function initUpload(sessionId: string, filename: string, mime: string, totalSize: number): Promise<string> {
  const res = await fetch(`${UPLOAD_BASE}/InitUpload`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId, filename, mimeType: mime, totalSize }),
  })
  if (!res.ok) { throw new Error(await res.text() || 'init failed') }
  const data = await res.json() as any
  return (data.uploadId || data.upload_id)
}

function toBase64(chunk: Uint8Array): string {
  // Build a binary string in chunks (ensures code points 0–255)
  let binary = ''
  const STEP = 0x8000
  for (let i = 0; i < chunk.length; i += STEP) {
    binary += String.fromCharCode(...chunk.subarray(i, Math.min(i + STEP, chunk.length)))
  }
  return btoa(binary)
}

async function uploadChunk(uploadId: string, index: number, chunk: Uint8Array): Promise<void> {
  const b64 = toBase64(chunk)
  const res = await fetch(`${UPLOAD_BASE}/UploadChunk`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uploadId, index, chunk: b64 }),
  })
  if (!res.ok) throw new Error(await res.text() || 'chunk failed')
}

async function completeUpload(uploadId: string): Promise<void> {
  const res = await fetch(`${UPLOAD_BASE}/CompleteUpload`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uploadId }),
  })
  if (!res.ok) throw new Error(await res.text() || 'complete failed')
}

export async function uploadImages(sessionId: string, files: File[]) {
  let uploaded = 0
  let rejected_duplicate = 0
  for (const f of files) {
    try {
      const uploadId = await initUpload(sessionId, f.name, f.type || 'application/octet-stream', f.size)
      const buf = new Uint8Array(await f.arrayBuffer())
      let offset = 0
      let index = 0
      while (offset < buf.length) {
        const end = Math.min(offset + CHUNK_SIZE, buf.length)
        await uploadChunk(uploadId, index++, buf.subarray(offset, end))
        offset = end
      }
      await completeUpload(uploadId)
      uploaded += 1
    } catch (e:any) {
      const msg = String(e?.message || e)
      if (msg.toLowerCase().includes('duplicate')) {
        rejected_duplicate += 1
      } else if (msg.toLowerCase().includes('cap reached')) {
        break
      } else {
        throw e
      }
    }
  }
  return { uploaded, rejected_duplicate, remaining: -1 }
}

export async function finalize(sessionId: string) {
  const res = await fetch(`${CONNECT_BASE}/FinalizeSession`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId }),
  })
  if (!res.ok) {
    try { const err = await res.json(); throw new Error(JSON.stringify(err)) } catch { throw new Error('finalize failed') }
  }
  return res.json()
}


export async function getConfig() {
  const res = await fetch(`${API_BASE}/opensock.dc.v1.ConfigService/Get`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({})
  })
  if (!res.ok) throw new Error('config failed')
  return res.json() as Promise<{ session_image_cap: number, min_required: { MIXED_UNIQUES: number, SAME_TYPE: number } }>
}

// Admin (Connect-style JSON endpoints)
const ADMIN_BASE = `${API_BASE}/opensock.dc.v1.AdminService`

function adminAuthHeader(password: string) {
  return { Authorization: 'Basic ' + btoa('admin:' + password) }
}

export type AdminSession = {
  id: string
  email: string
  name?: string
  handle?: string
  mode: string
  image_count: number
  created_at: string
  finalized_at?: string
}

export async function adminListSessions(password: string, filters: { email?: string; name?: string; mode?: string }) {
  const res = await fetch(`${ADMIN_BASE}/ListSessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...adminAuthHeader(password) },
    body: JSON.stringify({ email: filters.email || '', name: filters.name || '', mode: filters.mode || '' }),
  })
  if (!res.ok) throw new Error('admin list failed')
  const data = await res.json() as { sessions: AdminSession[] }
  return data.sessions
}

export async function adminDeleteSession(password: string, sessionId: string) {
  const res = await fetch(`${ADMIN_BASE}/DeleteSession`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...adminAuthHeader(password) },
    body: JSON.stringify({ sessionId }),
  })
  if (!res.ok) throw new Error('delete failed')
  return res.json() as Promise<{ ok: boolean }>
}

export async function adminDeleteUser(password: string, email: string) {
  const res = await fetch(`${ADMIN_BASE}/DeleteUser`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...adminAuthHeader(password) },
    body: JSON.stringify({ email }),
  })
  if (!res.ok) throw new Error('delete failed')
  return res.json() as Promise<{ ok: boolean }>
}


// Admin Config RPCs
export async function adminGetConfig(password: string) {
  const res = await fetch(`${ADMIN_BASE}/GetConfig`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...adminAuthHeader(password) }, body: JSON.stringify({}) })
  if (!res.ok) throw new Error('get config failed')
  return res.json() as Promise<{ session_image_cap: number, min_required_mixed: number, min_required_same: number, file_size_limit_mb?: number }>
}
export async function adminUpdateConfig(password: string, cfg: { session_image_cap: number, min_required_mixed: number, min_required_same: number, file_size_limit_mb?: number }) {
  const res = await fetch(`${ADMIN_BASE}/UpdateConfig`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...adminAuthHeader(password) }, body: JSON.stringify(cfg) })
  if (!res.ok) throw new Error('update config failed')
  return res.json() as Promise<{ session_image_cap: number, min_required_mixed: number, min_required_same: number, file_size_limit_mb?: number }>
}
