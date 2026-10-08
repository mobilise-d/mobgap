import { createFileRoute } from '@tanstack/react-router'
import { ProgressPage } from '@/pages/browser-lab'

export const Route = createFileRoute('/progress')({ component: ProgressPage })
