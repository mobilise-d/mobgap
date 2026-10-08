import { createContext, useContext, useEffect, useRef, useState } from 'react'
import { useNavigate, useRouter, useSearch } from '@tanstack/react-router'
import {
  cwaMode,
  matlabRow,
  rowLabel,
  rowProblem
} from '@/components/dataset-model'
import type {
  CwaConfiguration,
  DatasetFieldErrors,
  DatasetRow,
  ParticipantConfiguration,
  RowOutcome,
  Sample
} from '@/components/dataset-model'
import { getRuntime, PythonExecutionError } from './runtime'
import type {
  DatasetConfiguration,
  PipelinePreset,
  RuntimeProgress
} from './contracts'
import { validateLabSearch } from './lab-search'
import { useLabResources } from './lab-resources'
import type { LabSearch } from './lab-search'

export const appUrl = (path: string) =>
  new URL(
    path.replace(/^\/+/, ''),
    new URL(import.meta.env.BASE_URL, document.baseURI)
  ).href
const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error)
const initialParticipant: ParticipantConfiguration = {
  mode: 'choose',
  height: '',
  sensorHeight: '',
  cohort: '',
  condition: 'laboratory'
}
const validTimezone = (timezone: string) => {
  if (!timezone.trim()) return false
  try {
    new Intl.DateTimeFormat('en', { timeZone: timezone.trim() })
    return true
  } catch {
    return false
  }
}

export function useLabController() {
  const requestVersion = useRef(0)
  const sampleFetch = useRef<AbortController | null>(null)
  const [samples, setSamples] = useState<Sample[]>([])
  const [sampleError, setSampleError] = useState(false)
  const { resources, setter, reset: resetResources } = useLabResources()
  const {
    file,
    infoFile,
    datasetId,
    builtCwaSettings,
    cwaRecording,
    rows,
    outcomes
  } = resources
  const setFile = setter('file')
  const setInfoFile = setter('infoFile')
  const setDatasetId = setter('datasetId')
  const setBuiltCwaSettings = setter('builtCwaSettings')
  const setCwaRecording = setter('cwaRecording')
  const setRows = setter('rows')
  const setOutcomes = setter('outcomes')
  const [participant, setParticipant] = useState(initialParticipant)
  const search = useSearch({ strict: false }) as LabSearch
  const navigate = useNavigate()
  const router = useRouter()
  const updateSearch = (
    value: Partial<LabSearch>,
    replace = true,
    page?: '/upload' | '/dataset' | '/progress' | '/results'
  ) => {
    const current = router.state.matches.at(-1)?.routeId
    const to =
      page ??
      (current === '/dataset' ||
      current === '/progress' ||
      current === '/results'
        ? current
        : '/upload')
    void navigate({
      to,
      search: (previous) => validateLabSearch({ ...previous, ...value }),
      replace
    })
  }
  const cwa: CwaConfiguration = {
    timezone: search.timezone ?? '',
    mode: search.split
  }
  const pipeline = search.pipeline
  const settingsKey = `${cwa.mode ?? ''}:${cwa.timezone}`
  const [jobRowIds, setJobRowIds] = useState<string[]>([])
  const [mounted, setMounted] = useState(false)
  const [operation, setOperation] = useState<'sample' | 'build' | 'run' | null>(
    null
  )
  const [progress, setProgress] = useState<RuntimeProgress | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [fileErrors, setFileErrors] = useState<string[]>([])
  const [warnings, setWarnings] = useState<string[]>([])
  const busy = operation !== null
  const hasCwa = file?.name.toLowerCase().endsWith('.cwa') ?? false
  const height = Number(participant.height)
  const sensorHeight = Number(participant.sensorHeight)
  const manualValid =
    participant.height !== '' &&
    participant.sensorHeight !== '' &&
    Number.isFinite(height) &&
    Number.isFinite(sensorHeight) &&
    height > 0 &&
    sensorHeight > 0 &&
    sensorHeight <= height
  const metadataReady =
    participant.mode === 'manual'
      ? manualValid
      : participant.mode === 'file' && infoFile !== null
  const fieldErrors: DatasetFieldErrors = {
    height:
      participant.height !== '' && (!Number.isFinite(height) || height <= 0)
        ? 'Enter a positive participant height in metres.'
        : undefined,
    sensorHeight:
      participant.sensorHeight !== '' &&
      (!Number.isFinite(sensorHeight) || sensorHeight <= 0)
        ? 'Enter a positive sensor height in metres.'
        : participant.sensorHeight !== '' && height > 0 && sensorHeight > height
          ? 'Sensor height cannot exceed participant height.'
          : undefined,
    timezone:
      cwa.timezone.trim() !== '' && !validTimezone(cwa.timezone)
        ? 'Enter a valid IANA timezone, such as Europe/Berlin or UTC.'
        : undefined
  }
  const canBuild =
    file !== null &&
    participant.cohort !== '' &&
    metadataReady &&
    (!hasCwa || (participant.mode === 'manual' && validTimezone(cwa.timezone)))
  const sessionMatches = search.dataset === datasetId && datasetId !== ''
  const configurationMatches = !hasCwa || settingsKey === builtCwaSettings
  const selectedIndexes =
    search.rows === undefined
      ? rows.map((_, index) => index)
      : search.rows === ''
        ? []
        : search.rows.split(',').map(Number)
  const selected = new Set(
    sessionMatches && configurationMatches
      ? selectedIndexes
          .filter((index) => index < rows.length)
          .map((index) => rows[index].id)
      : []
  )
  const selectedRows = rows.filter((row) => selected.has(row.id))
  const selectedProblems = selectedRows.filter((row) => rowProblem(row))
  const canRun =
    mounted &&
    sessionMatches &&
    configurationMatches &&
    selectedRows.length > 0 &&
    selectedProblems.length === 0 &&
    !busy
  const completedRows = rows.filter((row) => outcomes[row.id]?.result)
  const jobIds = new Set(jobRowIds)
  const jobRows = rows.filter((row) => jobIds.has(row.id))
  const processedCount = jobRowIds.filter(
    (id) =>
      outcomes[id]?.status === 'complete' || outcomes[id]?.status === 'error'
  ).length
  const resultRows = completedRows

  useEffect(() => {
    const controller = new AbortController()
    fetch(appUrl('samples/manifest.json'), { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error('Samples unavailable')
        return response.json() as Promise<{ samples: Sample[] }>
      })
      .then((manifest) => setSamples(manifest.samples))
      .catch(() => {
        if (!controller.signal.aborted) setSampleError(true)
      })
    return () => controller.abort()
  }, [])

  function invalidateDataset(
    patch: Partial<LabSearch> = {},
    page?: '/dataset'
  ) {
    setRows([])
    setDatasetId('')
    setMounted(false)
    setFileErrors([])
    setWarnings([])
    setJobRowIds([])
    setOutcomes({})
    setError('')
    setNotice('')
    updateSearch(
      {
        dataset: undefined,
        rows: undefined,
        result: undefined,
        table: undefined,
        page: undefined,
        ...patch
      },
      true,
      page
    )
  }

  function stageFiles(nextFiles: File[]) {
    if (busy || nextFiles.length === 0) return
    if (nextFiles.length !== 1) {
      setError(
        'Choose one recording file. Trials or calendar days from that file form the dataset rows.'
      )
      return
    }
    resetResources()
    invalidateDataset({ split: undefined, timezone: undefined })
    setFile(nextFiles[0])
    const cwaInput = nextFiles[0].name.toLowerCase().endsWith('.cwa')
    setParticipant({
      ...initialParticipant,
      mode: cwaInput ? 'manual' : 'choose',
      condition: cwaInput ? 'free_living' : 'laboratory'
    })
  }

  function updateParticipant(value: ParticipantConfiguration) {
    invalidateDataset()
    setParticipant(value)
  }
  function updateCwa(value: CwaConfiguration) {
    invalidateDataset({ split: value.mode, timezone: value.timezone })
  }
  function stageInfoFile(nextFile: File) {
    if (busy) return
    invalidateDataset()
    setInfoFile(nextFile)
    setParticipant((previous) => ({ ...previous, mode: 'file' }))
  }

  function beginOperation(kind: 'sample' | 'build' | 'run') {
    const version = ++requestVersion.current
    setOperation(kind)
    setError('')
    setNotice('')
    return version
  }

  function reportProgress(version: number, prefix = '') {
    return (update: RuntimeProgress) => {
      if (version === requestVersion.current)
        setProgress({
          ...update,
          message: prefix ? `${prefix}: ${update.message}` : update.message
        })
    }
  }

  function loseMountedDataset() {
    setMounted(false)
    updateSearch({ rows: '' })
  }

  function handleOperationError(reason: unknown) {
    if (getRuntime().hasActiveKernel) setError(errorMessage(reason))
    else {
      loseMountedDataset()
      setError(
        `The Python worker stopped. Build the dataset again to reload your selected files.\n\n${errorMessage(reason)}`
      )
    }
  }

  async function loadSample(sample: Sample) {
    if (busy) return
    resetResources()
    invalidateDataset({ split: undefined, timezone: undefined })
    const version = beginOperation('sample')
    const controller = new AbortController()
    sampleFetch.current = controller
    setProgress({ stage: 'sample', message: 'Loading example files…' })
    const fetchFile = async (path: string) => {
      const response = await fetch(appUrl(path), { signal: controller.signal })
      if (!response.ok)
        throw new Error(
          'The example files could not be loaded. Choose a local recording instead.'
        )
      return new File(
        [await response.blob()],
        path.split('/').at(-1) ?? 'sample.mat'
      )
    }
    try {
      const [data, metadata] = await Promise.all([
        fetchFile(sample.dataFiles[0]),
        fetchFile(sample.metadataFiles[0])
      ])
      if (version !== requestVersion.current) return
      setFile(data)
      setInfoFile(metadata)
      setParticipant({
        ...initialParticipant,
        mode: 'file',
        cohort: sample.cohort
      })
      updateSearch({ pipeline: sample.preset })
    } catch (reason) {
      if (version === requestVersion.current) setError(errorMessage(reason))
    } finally {
      if (version === requestVersion.current) {
        setOperation(null)
        setProgress(null)
        sampleFetch.current = null
      }
    }
  }

  async function buildDataset() {
    if (!canBuild || busy || !file) return
    invalidateDataset({}, '/dataset')
    const version = beginOperation('build')
    const configuration: DatasetConfiguration = {
      cohort: participant.cohort,
      measurementCondition: participant.condition,
      ...(participant.mode === 'manual'
        ? { heightM: height, sensorHeightM: sensorHeight }
        : {}),
      ...(hasCwa ? { timezone: cwa.timezone.trim() } : {})
    }
    const input =
      participant.mode === 'file' && infoFile ? [file, infoFile] : [file]
    try {
      const runtime = getRuntime()
      const inspection = await runtime.inspectFiles(
        input,
        reportProgress(version),
        configuration
      )
      if (version !== requestVersion.current) return
      setCwaRecording(
        inspection.recordings.find(
          (recording) => recording.sourceFormat === 'cwa'
        ) ?? null
      )
      const failures = inspection.errors.map(
        (value) => `${value.fileName}: ${value.message}`
      )
      const nextRows: DatasetRow[] = []
      for (const recording of inspection.recordings) {
        if (version !== requestVersion.current) return
        if (recording.sourceFormat !== 'cwa') {
          nextRows.push(matlabRow(recording))
          continue
        }
        if (cwaMode(recording, cwa) === 'days') {
          try {
            const planned = await runtime.getCwaDayWindows(
              recording.id,
              cwa.timezone.trim(),
              reportProgress(version, recording.fileName)
            )
            if (version !== requestVersion.current) return
            nextRows.push(
              ...planned.windows.map((day) => ({
                id: `${recording.id}:day:${day.index}`,
                recording,
                label: day.label,
                indexValues: {
                  day: day.label,
                  start_time: day.startTime,
                  end_time: day.endTime
                },
                durationSeconds: day.durationSeconds,
                day
              }))
            )
          } catch (reason) {
            if (!runtime.hasActiveKernel) throw reason
            failures.push(`${recording.fileName}: ${errorMessage(reason)}`)
          }
        } else {
          nextRows.push({
            id: `${recording.id}:file`,
            recording,
            label: 'Single file',
            indexValues: {
              recording: 'Single file',
              start_time: recording.cwa!.startTimeRaw,
              end_time: recording.cwa!.endTimeRaw
            },
            durationSeconds: recording.durationSeconds,
            wholeFile: true
          })
        }
      }
      if (version !== requestVersion.current) return
      setRows(nextRows)
      setMounted(nextRows.length > 0)
      setBuiltCwaSettings(settingsKey)
      if (nextRows.length > 0) {
        const id = crypto.randomUUID()
        setDatasetId(id)
        updateSearch({
          dataset: id,
          rows: undefined,
          result: undefined,
          table: undefined,
          page: undefined
        })
      }
      setFileErrors(failures)
      setWarnings([
        ...new Set([
          ...inspection.warnings,
          ...inspection.recordings.flatMap((recording) => recording.warnings)
        ])
      ])
      if (nextRows.length === 0 && failures.length === 0)
        setError('No dataset rows were found in these recordings.')
    } catch (reason) {
      if (version === requestVersion.current) handleOperationError(reason)
    } finally {
      if (version === requestVersion.current) {
        setOperation(null)
        setProgress(null)
      }
    }
  }

  function toggleRow(id: string) {
    const next = new Set(selected)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    updateSearch(
      {
        rows: rows
          .flatMap((row, index) => (next.has(row.id) ? [index] : []))
          .join(',')
      },
      false
    )
  }
  function toggleAll() {
    updateSearch(
      { rows: selected.size === rows.length ? '' : undefined },
      false
    )
  }
  function selectPipeline(value: PipelinePreset) {
    updateSearch({ pipeline: value }, false)
  }
  function viewResult(id: string) {
    if (busy) return
    void navigate({
      to: '/results',
      search: (previous) =>
        validateLabSearch({
          ...previous,
          result: rows.findIndex((row) => row.id === id),
          table: undefined,
          page: undefined
        })
    })
  }

  async function runSelected() {
    if (!canRun) return
    const version = beginOperation('run')
    setJobRowIds(selectedRows.map((row) => row.id))
    let completedNormally = false
    let lastResultIndex: number | undefined
    void navigate({
      to: '/progress',
      search: (previous) => validateLabSearch(previous)
    })
    setOutcomes((previous) => ({
      ...previous,
      ...Object.fromEntries(
        selectedRows.map((row) => [row.id, { status: 'queued' as const }])
      )
    }))
    const groups = new Map<string, DatasetRow[]>()
    for (const row of selectedRows)
      groups.set(row.recording.id, [
        ...(groups.get(row.recording.id) ?? []),
        row
      ])
    const updateOutcome = (id: string, outcome: RowOutcome) => {
      if (version === requestVersion.current)
        setOutcomes((previous) => ({ ...previous, [id]: outcome }))
    }
    try {
      for (const group of groups.values()) {
        if (version !== requestVersion.current) return
        const recording = group[0].recording
        const options = {
          recordingId: recording.id,
          pipeline,
          heightM: recording.metadata.heightM!,
          sensorHeightM: recording.metadata.sensorHeightM!,
          cohort: participant.cohort,
          measurementCondition: participant.condition
        }
        if (group[0].day) {
          let position = 0
          try {
            await getRuntime().runDays(
              {
                ...options,
                dayIndices: group.map((row) => row.day!.index),
                timezone: cwa.timezone.trim()
              },
              (event) => {
                if (version !== requestVersion.current) return
                const row = group.find(
                  (value) => value.day?.index === event.day.index
                )!
                if (event.result) {
                  updateOutcome(row.id, {
                    status: 'complete',
                    result: event.result
                  })
                  lastResultIndex = rows.indexOf(row)
                } else
                  updateOutcome(row.id, {
                    status: 'error',
                    message: event.error
                  })
                position++
                if (event.fatal) loseMountedDataset()
              },
              (update) => {
                if (version !== requestVersion.current) return
                reportProgress(version, recording.fileName)(update)
                if (group[position])
                  updateOutcome(group[position].id, {
                    status: 'running',
                    message: update.message
                  })
              }
            )
          } catch (reason) {
            if (version !== requestVersion.current) return
            if (
              !(reason instanceof PythonExecutionError) ||
              !getRuntime().hasActiveKernel
            )
              throw reason
            for (const row of group.slice(position))
              updateOutcome(row.id, {
                status: 'error',
                message: errorMessage(reason)
              })
          }
        } else {
          for (const row of group) {
            if (version !== requestVersion.current) return
            updateOutcome(row.id, {
              status: 'running',
              message: 'Running pipeline…'
            })
            try {
              const calculated = await getRuntime().runPipeline(
                {
                  ...options,
                  ...(row.wholeFile
                    ? { cwaFile: { timezone: cwa.timezone.trim() } }
                    : {})
                },
                (update) => {
                  if (version !== requestVersion.current) return
                  reportProgress(version, rowLabel(row))(update)
                  updateOutcome(row.id, {
                    status: 'running',
                    message: update.message
                  })
                }
              )
              if (version !== requestVersion.current) return
              updateOutcome(row.id, { status: 'complete', result: calculated })
              lastResultIndex = rows.indexOf(row)
            } catch (reason) {
              if (version !== requestVersion.current) return
              updateOutcome(row.id, {
                status: 'error',
                message: errorMessage(reason)
              })
              if (
                !(reason instanceof PythonExecutionError) ||
                !getRuntime().hasActiveKernel
              )
                throw reason
            }
          }
        }
      }
      completedNormally = true
    } catch (reason) {
      if (version === requestVersion.current) {
        handleOperationError(reason)
        setOutcomes((previous) =>
          Object.fromEntries(
            Object.entries(previous).map(([id, outcome]) => [
              id,
              outcome.status === 'running' || outcome.status === 'queued'
                ? {
                    status: 'cancelled',
                    message: 'Stopped after worker failure'
                  }
                : outcome
            ])
          )
        )
      }
    } finally {
      if (version === requestVersion.current) {
        setOperation(null)
        setProgress(null)
        if (
          completedNormally &&
          router.state.matches.some(
            (match) =>
              match.routeId === '/progress' || match.routeId === '/results'
          )
        )
          void navigate({
            to: '/results',
            search: (previous) =>
              validateLabSearch({
                ...previous,
                result: lastResultIndex,
                table: undefined,
                page: undefined
              })
          })
      }
    }
  }

  function cancelOperation() {
    requestVersion.current++
    sampleFetch.current?.abort()
    sampleFetch.current = null
    getRuntime().cancel()
    loseMountedDataset()
    setOperation(null)
    setProgress(null)
    setError('')
    setOutcomes((previous) =>
      Object.fromEntries(
        Object.entries(previous).map(([id, outcome]) => [
          id,
          outcome.status === 'running' || outcome.status === 'queued'
            ? {
                ...outcome,
                status: 'cancelled',
                message:
                  outcome.status === 'running'
                    ? `Cancelled during ${outcome.message ?? 'analysis'}`
                    : 'Not run (cancelled)'
              }
            : outcome
        ])
      )
    )
    setNotice(
      operation === 'sample'
        ? ''
        : `Cancelled.${completedRows.length > 0 ? ' Completed row results remain available.' : ''} Build the dataset again to run more rows.`
    )
  }

  return {
    search,
    updateSearch,
    file,
    infoFile,
    samples,
    sampleError,
    participant,
    cwa,
    cwaRecording,
    pipeline,
    rows,
    selected,
    outcomes,
    mounted,
    operation,
    progress,
    error,
    notice,
    fileErrors,
    warnings,
    busy,
    hasCwa,
    canBuild,
    fieldErrors,
    canRun,
    selectedProblems,
    completedRows,
    processedCount,
    jobRows,
    resultRows,
    sessionMatches,
    configurationMatches,
    stageFiles,
    stageInfoFile,
    updateParticipant,
    updateCwa,
    loadSample,
    buildDataset,
    toggleRow,
    toggleAll,
    runSelected,
    cancelOperation,
    selectPipeline,
    viewResult
  }
}

export const LabContext = createContext<ReturnType<
  typeof useLabController
> | null>(null)
export function useLab() {
  const controller = useContext(LabContext)
  if (!controller) throw new Error('Lab pages require LabProvider')
  return controller
}
