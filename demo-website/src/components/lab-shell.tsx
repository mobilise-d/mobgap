import { Link, Outlet } from '@tanstack/react-router'
import { LockKeyhole } from 'lucide-react'
import { Badge } from './ui/badge'
import { LabProvider } from './lab-provider'
import { validateLabSearch } from '@/lib/lab-search'
import { useLab } from '@/lib/lab-controller'

export function LabShell() {
  return (
    <LabProvider>
      <LabFrame />
    </LabProvider>
  )
}

function LabFrame() {
  const lab = useLab()
  return (
    <div className="app-shell">
      <a className="skip-link" href="#lab-page">
        Skip to page content
      </a>
      <header className="app-header">
        <Link
          to="/upload"
          search={(previous) => validateLabSearch(previous)}
          className="brand"
          aria-label="mobgap WASM home"
        >
          mobgap WASM
        </Link>
        <Badge variant="outline">
          <LockKeyhole data-icon="inline-start" />
          Files stay on your device
        </Badge>
      </header>
      <nav className="step-navigation" aria-label="Analysis steps">
        <Link
          to="/upload"
          search={(previous) => validateLabSearch(previous)}
          activeProps={{ 'aria-current': 'page' }}
        >
          <span>01</span>Upload
        </Link>
        <Link
          to="/dataset"
          search={(previous) => validateLabSearch(previous)}
          activeProps={{ 'aria-current': 'page' }}
        >
          <span>02</span>Dataset
        </Link>
        <Link
          to="/progress"
          search={(previous) => validateLabSearch(previous)}
          activeProps={{ 'aria-current': 'page' }}
        >
          <span>03</span>Progress
        </Link>
        {lab.busy ? (
          <span className="step-unavailable" aria-disabled="true">
            <span>04</span>Results
          </span>
        ) : (
          <Link
            to="/results"
            search={(previous) => validateLabSearch(previous)}
            activeProps={{ 'aria-current': 'page' }}
          >
            <span>04</span>Results
          </Link>
        )}
      </nav>
      <main id="lab-page" className="lab-page" tabIndex={-1}>
        <Outlet />
      </main>
      <footer className="app-footer">
        <span>mobgap · Mobilise-D gait analysis</span>
        <span>Computed locally. Research use.</span>
      </footer>
    </div>
  )
}
