import type { ReactNode } from 'react'
import { LabContext, useLabController } from '@/lib/lab-controller'

export function LabProvider({ children }: { children: ReactNode }) {
  const controller = useLabController()
  return <LabContext value={controller}>{children}</LabContext>
}
