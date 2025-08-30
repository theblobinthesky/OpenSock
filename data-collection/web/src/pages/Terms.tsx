import React from 'react'
import { i18n } from '../i18n'

export default function Terms() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-3">
      <div className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-5">
        <h2 className="text-2xl font-semibold mb-3">{i18n.t('nav.terms')}</h2>
        <div className="space-y-3 text-gray-800 leading-7">
          <p>{i18n.t('terms.p1') || 'By contributing images of socks, you grant OpenSock permission to use them for research, training, and evaluation. You can request deletion at any time by contacting us from the email you provided.'}</p>
          <p>{i18n.t('terms.p2') || 'No analytics or trackers are used. Do not submit images with people or private scenes.'}</p>
          <p>{i18n.t('terms.p3') || 'Controller: Erik Stern, Gutmaninger Straße 26, 93413 Cham, Germany. Contact: erik.stern@outlook.de'}</p>
        </div>
      </div>
    </div>
  )
}
