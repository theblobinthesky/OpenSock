type Dict = Record<string, string>

import en from './en.json'
import de from './de.json'

const catalogs: Record<string, Dict> = {
  en,
  de,
}

function detectLang(): string {
  const urlLang = new URLSearchParams(location.search).get('lang')
  if (urlLang && catalogs[urlLang]) return urlLang
  const nav = navigator.language.toLowerCase()
  if (nav.startsWith('de')) return 'de'
  return 'en'
}

export const i18n = {
  lang: detectLang(),
  t(key: string): string {
    const cat = catalogs[this.lang] || catalogs['en']
    return cat[key] || catalogs['en'][key] || key
  },
}

