import { skipToken, useQuery, useQueryClient } from '@tanstack/react-query'
import type { Dispatch, SetStateAction } from 'react'
import type { DatasetRow, RowOutcome } from '@/components/dataset-model'
import type { Recording } from './contracts'

interface LabResources {
  file: File | null
  infoFile: File | null
  datasetId: string
  builtCwaSettings: string
  cwaRecording: Recording | null
  rows: DatasetRow[]
  outcomes: Record<string, RowOutcome>
}
const resourceKey = ['lab', 'session'] as const
const emptyResources = (): LabResources => ({
  file: null,
  infoFile: null,
  datasetId: '',
  builtCwaSettings: '',
  cwaRecording: null,
  rows: [],
  outcomes: {}
})

// Browser handles and large result tables are manually supplied, never refetched or persisted.
export function useLabResources() {
  const client = useQueryClient()
  const { data } = useQuery({
    queryKey: resourceKey,
    queryFn: skipToken,
    initialData: emptyResources,
    gcTime: Infinity,
    staleTime: Infinity,
    structuralSharing: false
  })
  function setter<K extends keyof LabResources>(
    key: K
  ): Dispatch<SetStateAction<LabResources[K]>> {
    return (value) =>
      client.setQueryData<LabResources>(resourceKey, (previous) => {
        const current = previous ?? emptyResources()
        const next =
          typeof value === 'function'
            ? (value as (old: LabResources[K]) => LabResources[K])(current[key])
            : value
        return { ...current, [key]: next }
      })
  }
  // Replacing the session releases old File handles, indexes and result payloads together.
  const reset = () => client.setQueryData(resourceKey, emptyResources())
  return { resources: data!, setter, reset }
}
