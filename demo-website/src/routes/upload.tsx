import { createFileRoute } from '@tanstack/react-router'
import { UploadPage } from '@/pages/browser-lab'

export const Route = createFileRoute('/upload')({ component: UploadPage })
