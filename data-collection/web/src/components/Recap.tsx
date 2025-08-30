import React from 'react'

export default function Recap(props: { id: string; onRestart: () => void }) {
  const { id, onRestart } = props
  return (
    <div className="p-4 max-w-2xl mx-auto">
      <div className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-5 grid gap-4">
        <div className="flex items-center gap-2">
          <img src="/icons/check.svg" alt="done" className="w-6 h-6" />
          <h2 className="text-2xl font-semibold">Contribution submitted</h2>
        </div>
        <p className="text-gray-700">Thank you! Your photos have been received.</p>
        <div className="grid gap-2">
          <div className="text-sm text-gray-500">Contribution ID</div>
          <div className="relative">
            <code className="block bg-gray-50 rounded px-3 py-2 text-sm overflow-hidden">
              <span className="inline-block max-w-full whitespace-nowrap overflow-hidden text-ellipsis align-bottom">{id}</span>
            </code>
            <div className="pointer-events-none absolute right-0 top-0 bottom-0 w-12 bg-gradient-to-l from-gray-50 to-transparent"></div>
          </div>
          <div>
            <button className="inline-flex items-center gap-1 px-2 py-1 rounded border border-gray-300 hover:bg-gray-50 text-sm" onClick={()=>{navigator.clipboard.writeText(id)}} title="Copy">
              <img src="/icons/copy.svg" alt="Copy" className="w-4 h-4" />
              Copy ID
            </button>
          </div>
        </div>
        <div className="flex gap-2">
          <button className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-brand-600 text-white hover:bg-brand-700" onClick={onRestart}>
            Start another contribution
          </button>
          <a href="#" className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-gray-300 hover:bg-gray-50">Go home</a>
        </div>
      </div>
    </div>
  )
}
