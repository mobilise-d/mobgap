import type { PipelinePreset } from './contracts'

export const resultTables = [
  'walking_bouts',
  'gait_sequences',
  'initial_contacts',
  'turns',
  'per_second_parameters',
  'raw_per_stride_parameters',
  'per_stride_parameters',
  'aggregated_parameters'
] as const
export interface LabSearch {
  pipeline: PipelinePreset
  split?: 'days' | 'file'
  timezone?: string
  rows?: string
  result?: number
  table?: string
  page?: number
  dataset?: string
}

export function validateLabSearch(search: Record<string, unknown>): LabSearch {
  return {
    pipeline:
      search.pipeline === 'healthy' || search.pipeline === 'impaired'
        ? search.pipeline
        : 'auto',
    split:
      search.split === 'days' || search.split === 'file'
        ? search.split
        : undefined,
    timezone: typeof search.timezone === 'string' ? search.timezone : undefined,
    rows:
      typeof search.rows === 'string' && /^(?:\d+(?:,\d+)*)?$/.test(search.rows)
        ? search.rows
        : undefined,
    result:
      typeof search.result === 'number' &&
      Number.isSafeInteger(search.result) &&
      search.result >= 0
        ? search.result
        : undefined,
    table:
      typeof search.table === 'string' &&
      resultTables.some((name) => name === search.table)
        ? search.table
        : undefined,
    page:
      typeof search.page === 'number' &&
      Number.isSafeInteger(search.page) &&
      search.page >= 0
        ? search.page
        : undefined,
    dataset: typeof search.dataset === 'string' ? search.dataset : undefined
  }
}
