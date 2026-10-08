import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import {
  createRootRoute,
  createRoute,
  createRouter,
  RouterProvider,
  redirect
} from '@tanstack/react-router'
import { LabShell } from './components/lab-shell'
import {
  UploadPage,
  DatasetPage,
  ProgressPage,
  ResultsPage
} from './pages/browser-lab'
import { validateLabSearch } from './lib/lab-search'
import './index.css'

const rootRoute = createRootRoute({
  component: LabShell,
  validateSearch: validateLabSearch
})
const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  beforeLoad: ({ search }) => {
    throw redirect({ to: '/upload', search })
  }
})
const uploadRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/upload',
  component: UploadPage
})
const datasetRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/dataset',
  component: DatasetPage
})
const progressRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/progress',
  component: ProgressPage
})
const resultsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/results',
  component: ResultsPage
})
const queryClient = new QueryClient()
const router = createRouter({
  basepath: import.meta.env.BASE_URL,
  routeTree: rootRoute.addChildren([
    indexRoute,
    uploadRoute,
    datasetRoute,
    progressRoute,
    resultsRoute
  ])
})

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>
)
