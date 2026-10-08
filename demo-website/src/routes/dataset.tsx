import { createFileRoute } from '@tanstack/react-router'
import { DatasetPage } from '@/pages/browser-lab'

export const Route = createFileRoute('/dataset')({ component: DatasetPage })
