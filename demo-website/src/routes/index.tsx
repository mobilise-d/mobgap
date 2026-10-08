import { createFileRoute, redirect } from '@tanstack/react-router'

export const Route = createFileRoute('/')({
  beforeLoad: ({ search }) => {
    throw redirect({ to: '/upload', search })
  }
})
