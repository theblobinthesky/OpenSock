import React from 'react'
import { i18n } from '../i18n'

export default function Privacy() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-3">
      <div className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-5">
        <h2 className="text-2xl font-semibold mb-3">{i18n.t('nav.privacy')}</h2>
        <div className="space-y-3 text-gray-800 leading-7">
          <p>{i18n.t('privacy.p1') || 'We store only the images you submit, plus optional name and email. Numeric focal length may be parsed if present. No trackers or analytics are used.'}</p>
          <p>{i18n.t('privacy.p2') || 'You may request deletion of your contributions any time by contacting us from your provided email.'}</p>
        </div>
      </div>
    </div>
  )
}
