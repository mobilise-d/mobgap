import { readdirSync } from 'node:fs'
import type { Plugin } from 'vite'

// These flat, static route files also have real documents on GitHub Pages.
export function staticPages(routesDirectory: URL): Plugin {
  return {
    name: 'mobgap-static-page-entries',
    apply: 'build',
    enforce: 'post',
    generateBundle(_options, bundle) {
      const entry = bundle['index.html']
      if (!entry || entry.type !== 'asset') this.error('The frontend HTML entry was not generated.')
      for (const file of readdirSync(routesDirectory)) {
        if (!file.endsWith('.tsx') || file.startsWith('_') || file === 'index.tsx') continue
        this.emitFile({
          type: 'asset',
          fileName: `${file.slice(0, -4)}/index.html`,
          source: entry.source
        })
      }
    }
  }
}
