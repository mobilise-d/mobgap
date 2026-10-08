import type { AnalysisResult, CwaDayWindow, Recording } from '@/lib/contracts'

export interface Sample {
  id: string
  label: string
  description: string
  dataFiles: string[]
  metadataFiles: string[]
  preset: 'healthy' | 'impaired'
  cohort: string
}
export interface ParticipantConfiguration {
  mode: 'choose' | 'file' | 'manual'
  height: string
  sensorHeight: string
  cohort: string
  condition: 'laboratory' | 'free_living'
}
export interface CwaConfiguration {
  scope: 'days' | 'window'
  timezone: string
  start: string
  duration: string
}
export interface DatasetRow {
  id: string
  recording: Recording
  label: string
  indexValues: Record<string, string>
  durationSeconds: number
  day?: CwaDayWindow
  window?: { startSeconds: number; durationSeconds: number; timezone: string }
}
export interface RowOutcome {
  status: 'queued' | 'running' | 'complete' | 'error' | 'cancelled'
  message?: string
  result?: AnalysisResult
}
export const duration = (seconds: number) => seconds >= 3600
  ? `${Math.floor(seconds / 3600)} h ${Math.floor(seconds % 3600 / 60)} min`
  : `${Math.floor(seconds / 60)} min ${(seconds % 60).toFixed(1)} s`
export const recordingLabel = (recording: Recording) => recording.label === recording.fileName
  ? recording.fileName : `${recording.fileName} · ${recording.label}`
export const rowLabel = (row: DatasetRow) => row.day
  ? `${row.recording.fileName} · ${row.day.label}` : recordingLabel(row.recording)
export function matlabRow(recording: Recording): DatasetRow {
  return {
    id: recording.id,
    recording,
    label: recording.label,
    indexValues: recording.datasetIndex ?? Object.fromEntries(recording.testName.map((value, index) => [
      ['time_measure', 'test', 'trial'][index] ?? `level_${index + 1}`, value,
    ])),
    durationSeconds: recording.durationSeconds,
  }
}
export function rowProblem(row: DatasetRow): string | undefined {
  const { recording } = row
  if (recording.cwa && !recording.cwa.hasGyroscope) return 'Gyroscope channels required'
  const height = recording.metadata.heightM
  const sensorHeight = recording.metadata.sensorHeightM
  if (!height || !sensorHeight || sensorHeight > height) return 'Participant metadata missing or invalid'
  return undefined
}
