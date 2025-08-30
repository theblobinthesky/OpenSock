import React from 'react'
import { i18n } from '../i18n'

export default function Impressum() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-3">
      <div className="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-5">
        <h2 className="text-2xl font-semibold mb-3">{i18n.t('nav.impressum')}</h2>
        <div className="space-y-3 text-gray-800 leading-7">
          <p>{i18n.t('impressum.p1') || 'OpenSock — Data Collection'}</p>
          <p>{i18n.t('impressum.p2') || 'Responsible person: Erik Stern'}</p>
          <p>{i18n.t('impressum.p3') || 'Address: Gutmaninger Straße 26, 93413 Cham, Germany'}</p>
          <p>{i18n.t('impressum.p4') || 'Contact: erik.stern@outlook.de'}</p>
        </div>
      </div>
    </div>
  )
}
