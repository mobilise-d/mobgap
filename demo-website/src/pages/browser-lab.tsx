import { useEffect, useRef, useState } from 'react'
import { FlaskConical, LoaderCircle, LockKeyhole, Play, TriangleAlert } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Field, FieldTitle } from '@/components/ui/field'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { ResultsPanel } from '@/components/results-panel'
import { DatasetIndex } from '@/components/dataset-index'
import { DatasetSidebar } from '@/components/dataset-sidebar'
import { matlabRow, rowLabel, rowProblem } from '@/components/dataset-model'
import type { CwaConfiguration, DatasetRow, ParticipantConfiguration, RowOutcome, Sample } from '@/components/dataset-model'
import { getRuntime, PythonExecutionError } from '@/lib/runtime'
import type { DatasetConfiguration, PipelinePreset, RuntimeProgress } from '@/lib/contracts'

const appUrl = (path: string) => new URL(path.replace(/^\/+/, ''), new URL(import.meta.env.BASE_URL, document.baseURI)).href
const errorMessage = (error: unknown) => error instanceof Error ? error.message : String(error)
const initialParticipant: ParticipantConfiguration = { mode: 'choose', height: '', sensorHeight: '', cohort: '', condition: 'laboratory' }
const initialCwa: CwaConfiguration = { scope: 'days', timezone: '', start: '0', duration: '60' }
const validTimezone = (timezone: string) => {
  if (!timezone.trim()) return false
  try { new Intl.DateTimeFormat('en', { timeZone: timezone.trim() }); return true } catch { return false }
}

export function BrowserLab() {
  const requestVersion = useRef(0)
  const sampleFetch = useRef<AbortController | null>(null)
  const [samples, setSamples] = useState<Sample[]>([])
  const [sampleError, setSampleError] = useState(false)
  const [files, setFiles] = useState<File[]>([])
  const [infoFiles, setInfoFiles] = useState<File[]>([])
  const [participant, setParticipant] = useState(initialParticipant)
  const [cwa, setCwa] = useState(initialCwa)
  const [pipeline, setPipeline] = useState<PipelinePreset>('auto')
  const [rows, setRows] = useState<DatasetRow[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [outcomes, setOutcomes] = useState<Record<string, RowOutcome>>({})
  const [resultRow, setResultRow] = useState('')
  const [mounted, setMounted] = useState(false)
  const [operation, setOperation] = useState<'sample' | 'build' | 'run' | null>(null)
  const [progress, setProgress] = useState<RuntimeProgress | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [fileErrors, setFileErrors] = useState<string[]>([])
  const [warnings, setWarnings] = useState<string[]>([])
  const busy = operation !== null
  const hasCwa = files.some(file => file.name.toLowerCase().endsWith('.cwa'))
  const height = Number(participant.height)
  const sensorHeight = Number(participant.sensorHeight)
  const manualValid = participant.height !== '' && participant.sensorHeight !== '' && Number.isFinite(height)
    && Number.isFinite(sensorHeight) && height > 0 && sensorHeight > 0 && sensorHeight <= height
  const metadataReady = participant.mode === 'manual' ? manualValid : participant.mode === 'file' && infoFiles.length > 0
  const start = Number(cwa.start)
  const windowDuration = Number(cwa.duration)
  const windowValid = cwa.start !== '' && cwa.duration !== '' && Number.isFinite(start) && Number.isFinite(windowDuration)
    && start >= 0 && windowDuration > 0 && windowDuration <= 3600
  const canBuild = files.length > 0 && participant.cohort !== '' && metadataReady
    && (!hasCwa || (participant.mode === 'manual' && validTimezone(cwa.timezone) && (cwa.scope === 'days' || windowValid)))
  const selectedRows = rows.filter(row => selected.has(row.id))
  const selectedProblems = selectedRows.filter(row => rowProblem(row))
  const canRun = mounted && selectedRows.length > 0 && selectedProblems.length === 0 && !busy
  const completedRows = rows.filter(row => outcomes[row.id]?.result)
  const activeRow = rows.find(row => row.id === resultRow)
  const activeResult = activeRow ? outcomes[activeRow.id]?.result : undefined
  const processedCount = Object.values(outcomes).filter(outcome => outcome.status === 'complete' || outcome.status === 'error').length

  useEffect(() => {
    const controller = new AbortController()
    fetch(appUrl('samples/manifest.json'), { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error('Samples unavailable'); return response.json() as Promise<{ samples: Sample[] }> })
      .then(manifest => setSamples(manifest.samples))
      .catch(() => { if (!controller.signal.aborted) setSampleError(true) })
    return () => controller.abort()
  }, [])

  function clearRunOutputs() { setOutcomes({}); setResultRow(''); setError(''); setNotice('') }

  function invalidateDataset() {
    setRows([]); setSelected(new Set()); setMounted(false); setFileErrors([]); setWarnings([])
    clearRunOutputs()
  }

  function stageFiles(nextFiles: File[]) {
    if (busy || nextFiles.length === 0) return
    invalidateDataset()
    setFiles(nextFiles); setInfoFiles([]); setCwa(initialCwa)
    const cwaInput = nextFiles.some(file => file.name.toLowerCase().endsWith('.cwa'))
    setParticipant({ ...initialParticipant, mode: cwaInput ? 'manual' : 'choose', condition: cwaInput ? 'free_living' : 'laboratory' })
  }

  function updateParticipant(value: ParticipantConfiguration) { invalidateDataset(); setParticipant(value) }
  function updateCwa(value: CwaConfiguration) { invalidateDataset(); setCwa(value) }
  function stageInfoFiles(nextFiles: File[]) {
    if (busy || nextFiles.length === 0) return
    invalidateDataset(); setInfoFiles(nextFiles); setParticipant(previous => ({ ...previous, mode: 'file' }))
  }

  function beginOperation(kind: 'sample' | 'build' | 'run') {
    const version = ++requestVersion.current
    setOperation(kind); setError(''); setNotice('')
    return version
  }

  function reportProgress(version: number, prefix = '') {
    return (update: RuntimeProgress) => {
      if (version === requestVersion.current) setProgress({ ...update, message: prefix ? `${prefix}: ${update.message}` : update.message })
    }
  }

  function loseMountedDataset() { setMounted(false); setSelected(new Set()) }

  function handleOperationError(reason: unknown) {
    if (getRuntime().hasActiveKernel) setError(errorMessage(reason))
    else {
      loseMountedDataset()
      setError(`The Python worker stopped. Build the dataset again to reload your selected files.\n\n${errorMessage(reason)}`)
    }
  }

  async function loadSample(sample: Sample) {
    if (busy) return
    invalidateDataset(); setFiles([]); setInfoFiles([])
    const version = beginOperation('sample')
    const controller = new AbortController()
    sampleFetch.current = controller
    setProgress({ stage: 'sample', message: 'Loading example files…' })
    const fetchFiles = (paths: string[]) => Promise.all(paths.map(async path => {
      const response = await fetch(appUrl(path), { signal: controller.signal })
      if (!response.ok) throw new Error('The example files could not be loaded. Choose local files instead.')
      return new File([await response.blob()], path.split('/').at(-1) ?? 'sample.mat')
    }))
    try {
      const [data, metadata] = await Promise.all([fetchFiles(sample.dataFiles), fetchFiles(sample.metadataFiles)])
      if (version !== requestVersion.current) return
      setFiles(data); setInfoFiles(metadata); setCwa(initialCwa)
      setParticipant({ ...initialParticipant, mode: 'file', cohort: sample.cohort }); setPipeline(sample.preset)
    } catch (reason) { if (version === requestVersion.current) setError(errorMessage(reason)) }
    finally { if (version === requestVersion.current) { setOperation(null); setProgress(null); sampleFetch.current = null } }
  }

  async function buildDataset() {
    if (!canBuild || busy) return
    invalidateDataset()
    const version = beginOperation('build')
    const configuration: DatasetConfiguration = {
      cohort: participant.cohort, measurementCondition: participant.condition,
      ...(participant.mode === 'manual' ? { heightM: height, sensorHeightM: sensorHeight } : {}),
      ...(hasCwa ? { timezone: cwa.timezone.trim() } : {}),
    }
    const input = participant.mode === 'file' ? [...files, ...infoFiles] : files
    try {
      const runtime = getRuntime()
      const inspection = await runtime.inspectFiles(input, reportProgress(version), configuration)
      if (version !== requestVersion.current) return
      const failures = inspection.errors.map(value => `${value.fileName}: ${value.message}`)
      const nextRows: DatasetRow[] = []
      for (const recording of inspection.recordings) {
        if (version !== requestVersion.current) return
        if (recording.sourceFormat !== 'cwa') { nextRows.push(matlabRow(recording)); continue }
        if (cwa.scope === 'days') {
          try {
            const planned = await runtime.getCwaDayWindows(recording.id, cwa.timezone.trim(), reportProgress(version, recording.fileName))
            if (version !== requestVersion.current) return
            nextRows.push(...planned.windows.map(day => ({
              id: `${recording.id}:day:${day.index}`, recording, label: day.label,
              indexValues: { day: day.label, start_time: day.startTime, end_time: day.endTime },
              durationSeconds: day.durationSeconds, day,
            })))
          } catch (reason) {
            if (!runtime.hasActiveKernel) throw reason
            failures.push(`${recording.fileName}: ${errorMessage(reason)}`)
          }
        } else if (start + windowDuration <= recording.durationSeconds) {
          nextRows.push({
            id: `${recording.id}:window:${start}:${windowDuration}`, recording, label: `Window ${start}–${start + windowDuration} s`,
            indexValues: { start_offset_s: String(start), duration_s: String(windowDuration) }, durationSeconds: windowDuration,
            window: { startSeconds: start, durationSeconds: windowDuration, timezone: cwa.timezone.trim() },
          })
        } else failures.push(`${recording.fileName}: The selected time window extends beyond this recording.`)
      }
      if (version !== requestVersion.current) return
      setRows(nextRows); setSelected(new Set(nextRows.map(row => row.id))); setMounted(nextRows.length > 0)
      setFileErrors(failures); setWarnings([...new Set([...inspection.warnings, ...inspection.recordings.flatMap(recording => recording.warnings)])])
      if (nextRows.length === 0 && failures.length === 0) setError('No dataset rows were found in these recordings.')
    } catch (reason) { if (version === requestVersion.current) handleOperationError(reason) }
    finally { if (version === requestVersion.current) { setOperation(null); setProgress(null) } }
  }

  function toggleRow(id: string) {
    setSelected(previous => { const next = new Set(previous); if (next.has(id)) next.delete(id); else next.add(id); return next })
  }
  function toggleAll() { setSelected(selected.size === rows.length ? new Set() : new Set(rows.map(row => row.id))) }

  async function runSelected() {
    if (!canRun) return
    const version = beginOperation('run')
    setOutcomes(previous => ({ ...previous, ...Object.fromEntries(selectedRows.map(row => [row.id, { status: 'queued' as const }])) }))
    const groups = new Map<string, DatasetRow[]>()
    for (const row of selectedRows) groups.set(row.recording.id, [...(groups.get(row.recording.id) ?? []), row])
    const updateOutcome = (id: string, outcome: RowOutcome) => {
      if (version === requestVersion.current) setOutcomes(previous => ({ ...previous, [id]: outcome }))
    }
    try {
      for (const group of groups.values()) {
        if (version !== requestVersion.current) return
        const recording = group[0].recording
        const options = {
          recordingId: recording.id, pipeline, heightM: recording.metadata.heightM!, sensorHeightM: recording.metadata.sensorHeightM!,
          cohort: participant.cohort, measurementCondition: participant.condition,
        }
        if (group[0].day) {
          let position = 0
          try {
            await getRuntime().runDays({ ...options, dayIndices: group.map(row => row.day!.index), timezone: cwa.timezone.trim() }, event => {
              if (version !== requestVersion.current) return
              const row = group.find(value => value.day?.index === event.day.index)!
              if (event.result) { updateOutcome(row.id, { status: 'complete', result: event.result }); setResultRow(row.id) }
              else updateOutcome(row.id, { status: 'error', message: event.error })
              position++
              if (event.fatal) loseMountedDataset()
            }, update => {
              if (version !== requestVersion.current) return
              reportProgress(version, recording.fileName)(update)
              if (group[position]) updateOutcome(group[position].id, { status: 'running', message: update.message })
            })
          } catch (reason) {
            if (version !== requestVersion.current) return
            if (!(reason instanceof PythonExecutionError) || !getRuntime().hasActiveKernel) throw reason
            for (const row of group.slice(position)) updateOutcome(row.id, { status: 'error', message: errorMessage(reason) })
          }
        } else {
          for (const row of group) {
            if (version !== requestVersion.current) return
            updateOutcome(row.id, { status: 'running', message: 'Running pipeline…' })
            try {
              const calculated = await getRuntime().runPipeline({ ...options, ...(row.window ? { cwaWindow: row.window } : {}) }, update => {
                if (version !== requestVersion.current) return
                reportProgress(version, rowLabel(row))(update); updateOutcome(row.id, { status: 'running', message: update.message })
              })
              if (version !== requestVersion.current) return
              updateOutcome(row.id, { status: 'complete', result: calculated }); setResultRow(row.id)
            } catch (reason) {
              if (version !== requestVersion.current) return
              updateOutcome(row.id, { status: 'error', message: errorMessage(reason) })
              if (!(reason instanceof PythonExecutionError) || !getRuntime().hasActiveKernel) throw reason
            }
          }
        }
      }
    } catch (reason) {
      if (version === requestVersion.current) {
        handleOperationError(reason)
        setOutcomes(previous => Object.fromEntries(Object.entries(previous).map(([id, outcome]) => [id,
          outcome.status === 'running' || outcome.status === 'queued' ? { status: 'cancelled', message: 'Stopped after worker failure' } : outcome,
        ])))
      }
    } finally { if (version === requestVersion.current) { setOperation(null); setProgress(null) } }
  }

  function cancelOperation() {
    requestVersion.current++
    sampleFetch.current?.abort(); sampleFetch.current = null
    getRuntime().cancel(); loseMountedDataset(); setOperation(null); setProgress(null); setError('')
    setOutcomes(previous => Object.fromEntries(Object.entries(previous).map(([id, outcome]) => [id,
      outcome.status === 'running' || outcome.status === 'queued' ? { ...outcome, status: 'cancelled', message: outcome.status === 'running' ? `Cancelled during ${outcome.message ?? 'analysis'}` : 'Not run (cancelled)' } : outcome,
    ])))
    setNotice(operation === 'sample' ? '' : `Cancelled.${completedRows.length > 0 ? ' Completed row results remain available.' : ''} Build the dataset again to run more rows.`)
  }

  return <div className="app-shell">
    <a className="skip-link" href="#dataset-workspace">Skip to dataset workspace</a>
    <header className="app-header"><a href={appUrl('')} className="brand" aria-label="Mobilise-D mobgap browser lab home"><img className="brand-logo" src={appUrl('brand/mobilise-d-logo.png')} alt="Mobilise-D" width={370} height={89} /><span className="brand-divider">/</span><span className="brand-label">mobgap Browser lab</span></a><Badge variant="outline"><LockKeyhole data-icon="inline-start" />Files stay on your device</Badge></header>
    <main>
      <div className="page-intro"><div><p className="eyebrow">Gait analysis, locally</p><h1>Configure a dataset. Analyze selected rows.</h1><p className="intro-copy">Choose recordings and participant information on the left. Build the dataset, then select trials or days from its index.</p></div><p className="prototype-note"><FlaskConical aria-hidden="true" />Research prototype</p></div>
      <div id="dataset-workspace" className="workspace">
        <div className="flex min-w-0 flex-col gap-3">
          <DatasetSidebar files={files} infoFiles={infoFiles} samples={samples} sampleError={sampleError} participant={participant} cwa={cwa} busy={busy} building={operation === 'build'} sampleLoading={operation === 'sample'} canBuild={canBuild} hasCwa={hasCwa} indexBuilt={rows.length > 0} onFiles={stageFiles} onInfoFiles={stageInfoFiles} onSample={sample => void loadSample(sample)} onParticipant={updateParticipant} onCwa={updateCwa} onBuild={() => void buildDataset()} />
          {operation === 'sample' ? <Button variant="outline" onClick={cancelOperation}>Cancel example loading</Button> : null}
        </div>
        <div className="flex min-w-0 flex-col gap-5">
          {notice ? <Alert><AlertTitle>Operation stopped</AlertTitle><AlertDescription>{notice}</AlertDescription></Alert> : null}
          {error || fileErrors.length > 0 ? <Alert variant="destructive"><TriangleAlert /><AlertTitle>Could not complete this step</AlertTitle><AlertDescription>{error ? <p className="break-words whitespace-pre-wrap">{error}</p> : null}{fileErrors.map(message => <p key={message} className="break-words">{message}</p>)}</AlertDescription></Alert> : null}
          {busy && operation !== 'sample' && progress ? <div className="flex flex-col gap-3 rounded-lg border bg-card p-5" role="status" aria-live="polite"><div className="flex items-center gap-2 text-sm"><LoaderCircle className="size-4 shrink-0 animate-spin text-primary" aria-hidden="true" /><span className="break-words">{progress.message}</span></div>{progress.percent !== undefined ? <Progress value={progress.percent} aria-label="Dataset operation progress" /> : null}{operation === 'run' ? <p className="text-xs text-muted-foreground">{processedCount} rows processed. Results appear as each row finishes.</p> : null}<Button variant="outline" size="sm" className="self-start" onClick={cancelOperation}>Cancel</Button></div> : null}
          {rows.length > 0 ? <section className="flex min-w-0 flex-col gap-4 rounded-xl border bg-card p-5" aria-labelledby="index-heading">
            <div className="flex flex-wrap items-start justify-between gap-3"><div><h2 id="index-heading" className="text-lg font-semibold tracking-tight">Dataset index</h2><p className="mt-1 text-xs text-muted-foreground">{selected.size} of {rows.length} rows selected · all rows start selected</p></div><Badge variant="outline">{new Set(rows.map(row => row.recording.fileName)).size} recording files</Badge></div>
            <div className="flex flex-wrap items-end justify-between gap-3">
              <Field className="w-full max-w-xs"><FieldTitle id="walking-preset-label">Walking preset</FieldTitle><ToggleGroup aria-labelledby="walking-preset-label" type="single" variant="outline" value={pipeline} onValueChange={value => { if (value) { clearRunOutputs(); setPipeline(value as PipelinePreset) } }} disabled={busy} className="w-full"><ToggleGroupItem className="flex-1" value="healthy">Healthy</ToggleGroupItem><ToggleGroupItem className="flex-1" value="impaired">Impaired</ToggleGroupItem><ToggleGroupItem className="flex-1" value="auto">Auto</ToggleGroupItem></ToggleGroup></Field>
              <Button disabled={!canRun} onClick={() => void runSelected()}><Play data-icon="inline-start" />Run selected ({selected.size})</Button>
            </div>
            <p className="text-xs text-muted-foreground">{pipeline === 'auto' ? 'Auto selects the preset by cohort: healthy for HA, COPD and CHF; impaired for PD, MS and PFF.' : pipeline === 'healthy' ? 'Healthy preset: recommended for HA, COPD and CHF.' : 'Impaired preset: recommended for PD, MS and PFF.'} Selected rows run sequentially.</p>
            {!mounted ? <p className="text-xs text-destructive">The worker is no longer available. Rebuild the dataset before selecting rows.</p> : selectedProblems.length > 0 ? <p className="text-xs text-destructive">{selectedProblems.length} selected rows cannot run. Check their status or deselect them.</p> : null}
            <DatasetIndex rows={rows} selected={selected} outcomes={outcomes} disabled={busy || !mounted} onToggle={toggleRow} onToggleAll={toggleAll} onViewResult={setResultRow} />
          </section> : null}
          {warnings.length > 0 ? <Alert><AlertTitle>Dataset notes</AlertTitle><AlertDescription>{warnings.map(message => <p key={message} className="break-words">{message}</p>)}</AlertDescription></Alert> : null}
          {activeRow && activeResult ? <section className="results-panel" aria-label="Selected row results">
            <Field className="mb-6"><FieldTitle id="result-row-label">Computed rows · {completedRows.length}</FieldTitle><Select value={resultRow} onValueChange={setResultRow}><SelectTrigger id="result-row" aria-labelledby="result-row-label" className="w-full"><SelectValue /></SelectTrigger><SelectContent><SelectGroup>{completedRows.map(row => <SelectItem key={row.id} value={row.id}>{rowLabel(row)}</SelectItem>)}</SelectGroup></SelectContent></Select></Field>
            <ResultsPanel key={activeRow.id + activeResult.preset + activeResult.summary.processingSeconds} result={activeResult} recordingLabel={rowLabel(activeRow)} downloadPrefix={`mobgap-${activeRow.recording.fileName.replace(/\.(mat|cwa)$/i, '')}-${activeRow.day?.label ?? activeRow.label.replace(/[^a-zA-Z0-9-]+/g, '-')}`} />
          </section> : null}
        </div>
      </div>
    </main>
    <footer className="app-footer"><span>mobgap · Mobilise-D gait analysis</span><span>Computed locally. Research use.</span></footer>
  </div>
}
