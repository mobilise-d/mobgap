import { createFileRoute } from '@tanstack/react-router'
import { ResultsPage } from '@/pages/browser-lab'

export const Route = createFileRoute('/results')({ component: ResultsPage })
