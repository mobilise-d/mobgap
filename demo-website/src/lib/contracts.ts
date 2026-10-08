/** Browser/Python bridge types. Tables include index columns in their rows. */
export type PipelinePreset = 'healthy' | 'impaired' | 'auto'
export type ProgressHandler = (progress: RuntimeProgress) => void
export interface RuntimeProgress { stage: string; message: string; percent?: number }
export interface Recording {
  id: string
  sourceFormat?: 'mat' | 'cwa'
  cwa?: { startTimeRaw: string; endTimeRaw: string; hasGyroscope: boolean; clockTimezoneRequired: true }
  fileName: string
  label: string
  testName: string[]
  datasetIndex?: Record<string, string>
  samples: number | null
  samplingRateHz: number
  durationSeconds: number
  channels: string[]
  sensorPosition: string
  metadata: { heightM?: number; sensorHeightM?: number }
  warnings: string[]
}
export interface FileInspectionError { fileName: string; code: string; message: string }
export interface InspectionResult { recordings: Recording[]; errors: FileInspectionError[]; warnings: string[] }
export interface DatasetConfiguration {
  cohort: string
  heightM?: number
  sensorHeightM?: number
  measurementCondition?: 'laboratory' | 'free_living'
  timezone?: string
}
export interface RunPipelineOptions {
  recordingId: string
  pipeline: PipelinePreset
  heightM: number
  sensorHeightM: number
  cohort: string
  measurementCondition?: 'laboratory' | 'free_living'
  cwaFile?: { timezone: string }
  cwaDay?: { index: number; timezone: string }
}
export type CellValue = string | number | boolean | null
export interface DataTable { columns: string[]; rows: CellValue[][] }
export interface AnalysisResult {
  recordingId: string
  preset: PipelinePreset
  summary: {
    samples: number; durationSeconds: number; samplingRateHz: number
    gaitSequences: number; initialContacts: number; walkingBouts: number; strides: number
    processingSeconds: number
  }
  tables: Record<string, DataTable>
  warnings: string[]
  versions?: Record<string, string>
}

export interface CwaDayWindow { index: number; label: string; startTime: string; endTime: string; startSeconds: number; durationSeconds: number }
export interface CwaDayWindowsResult { windows: CwaDayWindow[]; timezone: string }

export interface RunDaysOptions extends Omit<RunPipelineOptions, 'cwaFile' | 'cwaDay'> { dayIndices: number[]; timezone: string }
export interface DayAnalysisEvent { day: CwaDayWindow; result?: AnalysisResult; error?: string; fatal?: boolean }
