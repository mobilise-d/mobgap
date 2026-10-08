import { createRootRoute } from '@tanstack/react-router'
import { LabShell } from '@/components/lab-shell'
import { validateLabSearch } from '@/lib/lab-search'

export const Route = createRootRoute({
  component: LabShell,
  validateSearch: validateLabSearch
})
