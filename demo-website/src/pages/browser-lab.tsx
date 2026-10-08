import { useEffect, useRef, useState } from 'react'
import { Activity, ArrowRight, FileCheck2, FileUp, FlaskConical, LoaderCircle, LockKeyhole, Play, TriangleAlert } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Field, FieldDescription, FieldGroup, FieldLabel, FieldLegend, FieldSet } from '@/components/ui/field'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { ResultsPanel } from '@/components/results-panel'
import { getRuntime } from '@/lib/runtime'
import { cn } from '@/lib/utils'
import type { AnalysisResult, CwaDayWindow, PipelinePreset, Recording, RuntimeProgress } from '@/lib/contracts'

interface Sample {
  id: string; label: string; description: string; files: string[]
  preset: PipelinePreset; cohort: string; sensorHeightM: number; participantHeightM: number
}
const cohortOptions = [
  ['HA', 'Healthy adults'], ['COPD', 'COPD'], ['CHF', 'Chronic heart failure'],
  ['PD', 'Parkinson’s disease'], ['MS', 'Multiple sclerosis'], ['PFF', 'Proximal femoral fracture'],
] as const
const errorMessage = (error: unknown) => error instanceof Error ? error.message : String(error)
const appUrl = (path: string) => new URL(path.replace(/^\/+/, ''), new URL(import.meta.env.BASE_URL, document.baseURI)).href
const recordingLabel = (recording: Recording) => recording.label === recording.fileName ? recording.fileName : `${recording.fileName} · ${recording.label}`
const duration = (seconds: number) => seconds >= 3600 ? `${Math.floor(seconds / 3600)} h ${Math.floor(seconds % 3600 / 60)} min` : `${Math.floor(seconds / 60)} min ${(seconds % 60).toFixed(1)} s`

export function BrowserLab() {
  const fileInput = useRef<HTMLInputElement>(null)
  const requestVersion = useRef(0)
  const sampleFetch = useRef<AbortController | null>(null)
  const [samples, setSamples] = useState<Sample[]>([])
  const [sampleError, setSampleError] = useState('')
  const [recordings, setRecordings] = useState<Recording[]>([])
  const [recordingId, setRecordingId] = useState('')
  const [pipeline, setPipeline] = useState<PipelinePreset>('healthy')
  const [height, setHeight] = useState('')
  const [sensorHeight, setSensorHeight] = useState('')
  const [cohort, setCohort] = useState('')
  const [windowStart, setWindowStart] = useState('0')
  const [windowDuration, setWindowDuration] = useState('60')
  const [clockTimezone, setClockTimezone] = useState('')
  const [cwaMode, setCwaMode] = useState<'days' | 'window'>('days')
  const [dayWindows, setDayWindows] = useState<CwaDayWindow[]>([])
  const [selectedDay, setSelectedDay] = useState('all')
  const [dayResults, setDayResults] = useState<{ day: CwaDayWindow; result: AnalysisResult }[]>([])
  const [resultDay, setResultDay] = useState('')
  const [dayErrors, setDayErrors] = useState<{ day: CwaDayWindow; message: string }[]>([])
  const [resultSourceLabel, setResultSourceLabel] = useState('')
  const [condition, setCondition] = useState<'laboratory' | 'free_living'>('laboratory')
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<RuntimeProgress | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [fileErrors, setFileErrors] = useState<string[]>([])
  const [dragging, setDragging] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const recording = recordings.find(value => value.id === recordingId)
  const selectedDayResult = dayResults.find(entry => String(entry.day.index) === resultDay)
  const heightNumber = Number(height)
  const sensorHeightNumber = Number(sensorHeight)
  const heightValid = heightNumber > 0 && Number.isFinite(heightNumber)
  const sensorHeightValid = sensorHeightNumber > 0 && Number.isFinite(sensorHeightNumber) && (!heightValid || sensorHeightNumber <= heightNumber)
  const cwa = recording?.sourceFormat === 'cwa' ? recording.cwa : undefined
  const windowStartNumber = Number(windowStart)
  const windowDurationNumber = Number(windowDuration)
  const windowValid = windowStart !== '' && windowDuration !== '' && Number.isFinite(windowStartNumber) && Number.isFinite(windowDurationNumber)
    && windowStartNumber >= 0 && windowDurationNumber > 0 && windowDurationNumber <= 3600
    && Boolean(recording && windowStartNumber + windowDurationNumber <= recording.durationSeconds)
  let timezoneValid = false
  if (clockTimezone.trim()) {
    try { new Intl.DateTimeFormat('en', { timeZone: clockTimezone.trim() }); timezoneValid = true } catch { /* The entered timezone is unsupported. */ }
  }
  const daysToRun = selectedDay === 'all' ? dayWindows : dayWindows.filter(day => String(day.index) === selectedDay)
  const cwaReady = !cwa || (cwa.hasGyroscope && timezoneValid && (cwaMode === 'days' ? daysToRun.length > 0 : windowValid))
  const ready = Boolean(recording && cohort && heightValid && sensorHeightValid && cwaReady && !busy)

  useEffect(() => {
    const controller = new AbortController()
    fetch(appUrl('samples/manifest.json'), { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error('Samples unavailable'); return response.json() as Promise<{ samples: Sample[] }> })
      .then(manifest => setSamples(manifest.samples))
      .catch(reason => { if (!controller.signal.aborted) setSampleError(errorMessage(reason)) })
    return () => controller.abort()
  }, [])

  function clearResults() {
    setResult(null); setDayResults([]); setResultDay(''); setDayErrors([]); setResultSourceLabel('')
  }

  function selectRecording(id: string, candidates = recordings) {
    setRecordingId(id)
    clearResults()
    const selected = candidates.find(value => value.id === id)
    setHeight(selected?.metadata.heightM === undefined ? '' : String(selected.metadata.heightM))
    setSensorHeight(selected?.metadata.sensorHeightM === undefined ? '' : String(selected.metadata.sensorHeightM))
    setWindowStart('0')
    setWindowDuration(String(Math.min(60, selected?.durationSeconds ?? 60)))
    setClockTimezone(''); setDayWindows([]); setSelectedDay('all'); setCwaMode('days')
    setCondition(selected?.sourceFormat === 'cwa' ? 'free_living' : 'laboratory')
  }

  function beginOperation(clearData: boolean) {
    const version = ++requestVersion.current
    setBusy(true); setError(''); setNotice(''); setFileErrors([]); clearResults()
    if (clearData) { setDayWindows([]); setClockTimezone(''); setRecordings([]); setRecordingId(''); setHeight(''); setSensorHeight(''); setCohort('') }
    return version
  }

  function onProgress(version: number) {
    return (update: RuntimeProgress) => {
      if (version === requestVersion.current) setProgress(update)
    }
  }

  async function inspectFiles(files: File[], version: number, sample?: Sample) {
    const runtime = getRuntime()
    await runtime.initialize(onProgress(version))
    if (version !== requestVersion.current) return
    const inspection = await runtime.inspectFiles(files, onProgress(version))
    if (version !== requestVersion.current) return
    setRecordings(inspection.recordings)
    setFileErrors(inspection.errors.map(value => `${value.fileName}: ${value.message}`))
    if (inspection.recordings.length > 0) {
      selectRecording(inspection.recordings[0].id, inspection.recordings)
      if (sample) {
        setPipeline(sample.preset); setCondition('laboratory'); setCohort(sample.cohort)
        setHeight(String(sample.participantHeightM)); setSensorHeight(String(sample.sensorHeightM))
      }
    } else if (inspection.errors.length === 0) {
      setError('No lower-back recordings were found in these files.')
    }
  }

  async function loadFiles(files: File[]) {
    if (busy || files.length === 0) return
    const version = beginOperation(true)
    setProgress({ stage: 'initialize', message: 'Starting the browser analysis engine…' })
    try { await inspectFiles(files, version) }
    catch (reason) { if (version === requestVersion.current) setError(errorMessage(reason)) }
    finally { if (version === requestVersion.current) { setBusy(false); setProgress(null) } }
  }

  async function loadSample(sample: Sample) {
    if (busy) return
    const version = beginOperation(true)
    const controller = new AbortController()
    sampleFetch.current = controller
    setProgress({ stage: 'sample', message: 'Loading the example recording…' })
    try {
      const files = await Promise.all(sample.files.map(async url => {
        const response = await fetch(appUrl(url), { signal: controller.signal })
        if (!response.ok) throw new Error('The example recording could not be loaded. Please try a local file.')
        return new File([await response.blob()], url.split('/').at(-1) ?? 'sample.mat')
      }))
      if (version !== requestVersion.current) return
      await inspectFiles(files, version, sample)
    } catch (reason) { if (version === requestVersion.current) setError(errorMessage(reason)) }
    finally { if (version === requestVersion.current) { setBusy(false); setProgress(null); sampleFetch.current = null } }
  }

  function clearLoadedRecording() {
    setRecordings([]); setRecordingId(''); setDayWindows([]); setSelectedDay('all')
    setHeight(''); setSensorHeight(''); setCohort(''); setClockTimezone('')
    setWindowStart('0'); setWindowDuration('60')
  }

  function handleOperationError(reason: unknown) {
    if (getRuntime().hasActiveKernel) {
      setError(errorMessage(reason))
    } else {
      clearLoadedRecording()
      setError(`The Python worker stopped. Select your files again to retry.\n\n${errorMessage(reason)}`)
    }
  }

  async function prepareDayWindows() {
    if (!recording || !cwa || !timezoneValid || busy) return
    const version = beginOperation(false)
    setProgress({ stage: 'inspect', message: 'Calculating calendar-day boundaries…' })
    try {
      const planned = await getRuntime().getCwaDayWindows(recordingId, clockTimezone.trim(), onProgress(version))
      if (version === requestVersion.current) { setDayWindows(planned.windows); setSelectedDay('all') }
    } catch (reason) { if (version === requestVersion.current) handleOperationError(reason) }
    finally { if (version === requestVersion.current) { setBusy(false); setProgress(null) } }
  }

  async function runAnalysis() {
    if (!ready || !recording) return
    const version = beginOperation(false)
    setResultSourceLabel(`${recordingLabel(recording)}${cwa ? ` (${clockTimezone.trim()})` : ''}`)
    const options = { recordingId, pipeline, heightM: heightNumber, sensorHeightM: sensorHeightNumber, cohort, measurementCondition: condition }
    setProgress({ stage: 'run', message: 'Running the gait analysis pipeline…' })
    try {
      if (cwa && cwaMode === 'days') {
        await getRuntime().runDays({ ...options, dayIndices: daysToRun.map(day => day.index), timezone: clockTimezone.trim() }, event => {
          if (version !== requestVersion.current) return
          if (event.fatal) clearLoadedRecording()
          if (event.result) {
            const calculated = event.result
            setDayResults(previous => [...previous, { day: event.day, result: calculated }])
            setResultDay(String(event.day.index)); setResult(calculated)
          } else if (event.error) {
            const message = event.error
            setDayErrors(previous => [...previous, { day: event.day, message }])
          }
        }, onProgress(version))
      } else {
        const calculated = await getRuntime().runPipeline({ ...options, ...(cwa ? { cwaWindow: { startSeconds: windowStartNumber, durationSeconds: windowDurationNumber, timezone: clockTimezone.trim() } } : {}) }, onProgress(version))
        if (version === requestVersion.current) setResult(calculated)
      }
    } catch (reason) { if (version === requestVersion.current) handleOperationError(reason) }
    finally { if (version === requestVersion.current) { setBusy(false); setProgress(null) } }
  }

  function cancelOperation() {
    requestVersion.current += 1
    sampleFetch.current?.abort()
    sampleFetch.current = null
    getRuntime().cancel()
    setBusy(false); setProgress(null); setRecordings([]); setRecordingId('')
    setHeight(''); setSensorHeight(''); setCohort(''); setClockTimezone(''); setDayWindows([]); setSelectedDay('all'); setWindowStart('0'); setWindowDuration('60'); setError(''); setFileErrors([])
    setNotice(`The operation was cancelled.${dayResults.length > 0 ? ` Results from ${dayResults.length} completed day${dayResults.length === 1 ? '' : 's'} remain available below.` : ''} Load a recording to start again.`)
  }

  return <div className="app-shell">
    <a className="skip-link" href="#workspace">Skip to analysis workspace</a>
    <header className="app-header">
      <a href={appUrl('')} className="brand" aria-label="mobgap browser lab home"><span className="brand-symbol"><Activity aria-hidden="true" /></span><span>mobgap<span className="brand-divider">/</span><span className="brand-label">Browser lab</span></span></a>
      <Badge variant="outline"><LockKeyhole data-icon="inline-start" />Files stay on your device</Badge>
    </header>
    <main>
      <div className="page-intro">
        <div><p className="eyebrow">Gait analysis, locally</p><h1>From recording to walking parameters.</h1><p className="intro-copy">Run the Mobilise-D pipeline on lower-back sensor data. Choose an example or load MATLAB or CWA files to begin.</p></div>
        <p className="prototype-note"><FlaskConical aria-hidden="true" />Research prototype</p>
      </div>
      <div id="workspace" className="workspace">
        <aside className="controls-panel" aria-label="Analysis setup">
          <section aria-labelledby="data-heading" className="setup-section">
            <div className="section-heading"><span className="step-number">01</span><h2 id="data-heading">Choose a recording</h2></div>
            <div className={cn('file-dropzone', dragging && 'file-dropzone-active')} onDragOver={event => { event.preventDefault(); if (!busy) setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); if (!busy) void loadFiles(Array.from(event.dataTransfer.files)) }}>
              <FileUp className="size-7 text-muted-foreground" aria-hidden="true" />
              <p>Drop sensor files here</p><span>Mobilise-D .mat or AX6/AX3 .cwa</span>
              <Input ref={fileInput} id="matlab-files" type="file" accept=".mat,.cwa" multiple disabled={busy} className="sr-only" aria-label="Choose MATLAB or CWA files" onChange={event => { if (event.target.files) void loadFiles(Array.from(event.target.files)); event.target.value = '' }} />
              <Button variant="outline" disabled={busy} onClick={() => fileInput.current?.click()}>Choose files</Button>
            </div>
            <p className="help-copy">You can select data and participant metadata files together. CWA files are inspected before decoding a selected window. Analysis happens entirely in your browser.</p>
            <div className="example-list"><p className="eyebrow">Or try an example</p>{samples.map(sample => <Button key={sample.id} variant="ghost" className="example-button" disabled={busy} onClick={() => void loadSample(sample)}><span><strong>{sample.label}</strong><small>{sample.description}</small></span><ArrowRight data-icon="inline-end" /></Button>)}{sampleError ? <p className="text-xs text-muted-foreground">Examples are unavailable. You can still load a local file.</p> : null}</div>
            {recordings.length > 0 ? <FieldGroup className="mt-5"><Field><FieldLabel htmlFor="recording">Recording</FieldLabel><Select value={recordingId} onValueChange={id => selectRecording(id)} disabled={busy}><SelectTrigger id="recording" className="w-full"><SelectValue placeholder="Select a recording" /></SelectTrigger><SelectContent><SelectGroup>{recordings.map(value => <SelectItem key={value.id} value={value.id}>{recordingLabel(value)}</SelectItem>)}</SelectGroup></SelectContent></Select></Field>{recording ? <div className="recording-facts"><FileCheck2 className="size-4" aria-hidden="true" /><span>{duration(recording.durationSeconds)} · {recording.samplingRateHz} Hz<br />{recording.samples === null ? 'Sample count available after decoding' : `${recording.samples.toLocaleString()} samples`} · {recording.sensorPosition}</span></div> : null}</FieldGroup> : null}
            {cwa ? <div className="mt-5 space-y-4">
              <div className="rounded-lg border bg-muted/30 p-3 text-xs text-muted-foreground">
                <p className="font-medium text-foreground">CWA recording · metadata inspected</p>
                <p className="mt-1 break-words">{cwa.startTimeRaw} to {cwa.endTimeRaw}</p>
                <p className="mt-1">Sensor clock times; timezone has not been applied.</p>
                <p className="mt-2">{cwa.hasGyroscope ? 'Acceleration and gyroscope channels detected.' : 'Acceleration-only recording. These full pipeline presets require gyroscope data.'}</p>
              </div>
              <FieldSet disabled={busy}>
                <FieldLegend>Split by day</FieldLegend>
                <FieldDescription>Calendar days use the timezone below, including partial first and last days. Days are decoded and processed in sequence.</FieldDescription>
                <FieldGroup>
                  <Field><FieldLabel htmlFor="cwa-mode">Analysis scope</FieldLabel><Select value={cwaMode} onValueChange={value => { setCwaMode(value as 'days' | 'window'); clearResults() }}><SelectTrigger id="cwa-mode" className="w-full"><SelectValue /></SelectTrigger><SelectContent><SelectGroup><SelectItem value="days">Split by day</SelectItem><SelectItem value="window">Short time window</SelectItem></SelectGroup></SelectContent></Select></Field>
                  {cwaMode === 'window' ? <>
                  <Field><FieldLabel htmlFor="window-start">Start offset (seconds)</FieldLabel><Input id="window-start" type="number" min="0" step="1" value={windowStart} onChange={event => { setWindowStart(event.target.value); clearResults() }} /><FieldDescription>Elapsed time from the first recorded sample.</FieldDescription></Field>
                  <Field><FieldLabel htmlFor="window-duration">Duration (seconds)</FieldLabel><Input id="window-duration" type="number" min="0" max="3600" step="1" value={windowDuration} onChange={event => { setWindowDuration(event.target.value); clearResults() }} />{!windowValid ? <FieldDescription>Choose a positive duration up to 3,600 seconds within this recording.</FieldDescription> : null}</Field>
                  </> : null}
                  <Field data-invalid={clockTimezone !== '' && !timezoneValid}><FieldLabel htmlFor="clock-timezone">Sensor synchronization timezone</FieldLabel><Input id="clock-timezone" placeholder="e.g. Europe/Berlin or UTC" value={clockTimezone} aria-invalid={clockTimezone !== '' && !timezoneValid} onChange={event => { setClockTimezone(event.target.value); setDayWindows([]); clearResults() }} /><FieldDescription>Enter the IANA timezone of the computer that last synchronized the sensor. Its UTC offset at that synchronization is held fixed for this recording.</FieldDescription>{clockTimezone !== '' && !timezoneValid ? <FieldDescription>Enter a valid IANA timezone, or UTC if the sensor clock was synchronized in UTC.</FieldDescription> : null}</Field>
                  {cwaMode === 'days' ? <>
                    <Button variant="outline" disabled={!timezoneValid || busy} onClick={() => void prepareDayWindows()}>List calendar days</Button>
                    {dayWindows.length > 0 ? <Field><FieldLabel htmlFor="cwa-day">Days to analyze</FieldLabel><Select value={selectedDay} onValueChange={value => { setSelectedDay(value); clearResults() }}><SelectTrigger id="cwa-day" className="w-full"><SelectValue /></SelectTrigger><SelectContent><SelectGroup><SelectItem value="all">All {dayWindows.length} days, one at a time</SelectItem>{dayWindows.map(day => <SelectItem key={day.index} value={String(day.index)}>{day.label} · {duration(day.durationSeconds)}</SelectItem>)}</SelectGroup></SelectContent></Select><FieldDescription>Each day's results and CSV exports stay separate. A complete day can still require substantial browser memory.</FieldDescription></Field> : null}
                  </> : null}
                </FieldGroup>
              </FieldSet>
              <p className="help-copy">These presets expect a lower-back sensor in the mobgap sensor-axis convention. Confirm placement and axes before interpreting results.</p>
            </div> : null}
          </section>
          <Separator />
          <section aria-labelledby="pipeline-heading" className="setup-section">
            <div className="section-heading"><span className="step-number">02</span><h2 id="pipeline-heading">Configure the pipeline</h2></div>
            <FieldSet disabled={busy}>
              <FieldLegend className="sr-only">Pipeline and participant metadata</FieldLegend>
              <FieldGroup>
                <Field><FieldLabel>Walking preset</FieldLabel><ToggleGroup type="single" variant="outline" value={pipeline} onValueChange={value => { if (value) { setPipeline(value as PipelinePreset); clearResults() } }} disabled={busy} className="w-full"><ToggleGroupItem className="flex-1" value="healthy">Healthy</ToggleGroupItem><ToggleGroupItem className="flex-1" value="impaired">Impaired</ToggleGroupItem></ToggleGroup><FieldDescription>{pipeline === 'healthy' ? 'Recommended for HA, COPD and CHF cohorts.' : 'Recommended for PD, MS and PFF cohorts.'}</FieldDescription></Field>
                <Field><FieldLabel htmlFor="cohort">Participant cohort</FieldLabel><Select value={cohort} onValueChange={value => { setCohort(value); clearResults() }} disabled={busy}><SelectTrigger id="cohort" className="w-full"><SelectValue placeholder="Select a cohort" /></SelectTrigger><SelectContent><SelectGroup>{cohortOptions.map(([value, label]) => <SelectItem key={value} value={value}>{label} ({value})</SelectItem>)}</SelectGroup></SelectContent></Select></Field>
                <FieldGroup className="metadata-grid"><Field data-invalid={height !== '' && !heightValid}><FieldLabel htmlFor="participant-height">Participant height (m)</FieldLabel><Input id="participant-height" type="number" inputMode="decimal" min="0" step="0.01" placeholder="Enter height" value={height} aria-invalid={height !== '' && !heightValid} onChange={event => { setHeight(event.target.value); clearResults() }} /></Field><Field data-invalid={sensorHeight !== '' && !sensorHeightValid}><FieldLabel htmlFor="sensor-height">Sensor height (m)</FieldLabel><Input id="sensor-height" type="number" inputMode="decimal" min="0" step="0.01" placeholder="Enter height" value={sensorHeight} aria-invalid={sensorHeight !== '' && !sensorHeightValid} onChange={event => { setSensorHeight(event.target.value); clearResults() }} />{sensorHeight !== '' && !sensorHeightValid ? <FieldDescription>Enter a positive sensor height no greater than participant height.</FieldDescription> : null}</Field></FieldGroup>
                <p className="help-copy">Measure sensor height from the floor to the lower-back sensor. Both heights are used to calculate stride length and apply parameter thresholds.</p>
                <Field><FieldLabel htmlFor="condition">Recording setting</FieldLabel><Select value={condition} onValueChange={value => { setCondition(value as 'laboratory' | 'free_living'); clearResults() }} disabled={busy}><SelectTrigger id="condition" className="w-full"><SelectValue /></SelectTrigger><SelectContent><SelectGroup><SelectItem value="laboratory">Laboratory</SelectItem><SelectItem value="free_living">Free living</SelectItem></SelectGroup></SelectContent></Select></Field>
              </FieldGroup>
            </FieldSet>
            <Button size="lg" className="mt-5 w-full" disabled={!ready} onClick={() => void runAnalysis()}><Play data-icon="inline-start" />{cwa && cwaMode === 'days' ? 'Analyze selected days' : 'Run analysis'}</Button>
            {!recording && !busy ? <p className="mt-2 text-center text-xs text-muted-foreground">Load a recording to enable analysis.</p> : null}
          </section>
        </aside>
        <section className="results-panel" aria-label="Analysis results" aria-busy={busy}>
          {busy && progress ? <div className="loading-panel" role="status" aria-live="polite"><LoaderCircle className="size-7 animate-spin text-primary" aria-hidden="true" /><h2>{progress.message}</h2><p>The first run downloads and prepares the analysis engine. Keep this tab open.</p>{progress.percent !== undefined ? <Progress className="w-full max-w-xs" value={progress.percent} aria-label="Analysis progress" /> : <div className="indeterminate-track" aria-hidden="true"><span /></div>}<Button variant="ghost" onClick={cancelOperation}>Cancel</Button></div> : null}
          {!busy && notice ? <Alert className="mb-6"><AlertTitle>Cancelled</AlertTitle><AlertDescription>{notice}</AlertDescription></Alert> : null}
          {!busy && (error || fileErrors.length > 0) ? <Alert variant="destructive" className="mb-6"><TriangleAlert /><AlertTitle>{error ? 'Could not complete this step' : 'Some files could not be loaded'}</AlertTitle><AlertDescription>{error ? <p className="break-words whitespace-pre-wrap">{error}</p> : null}{fileErrors.map(message => <p key={message}>{message}</p>)}<p>Check the file and participant details, then try again.</p></AlertDescription></Alert> : null}
          {!busy && dayErrors.length > 0 ? <Alert variant="destructive" className="mb-6"><AlertTitle>Some days could not be analyzed</AlertTitle><AlertDescription>{dayErrors.map(entry => <p key={entry.day.index} className="break-words"><strong>{entry.day.label}:</strong> {entry.message}</p>)}<p>Days are processed separately.{dayResults.length > 0 ? ' Successful results remain available below.' : ''}</p></AlertDescription></Alert> : null}
          {!busy && dayResults.length > 0 ? <Field className="mb-6"><FieldLabel htmlFor="completed-day">Daily results · {dayResults.length} completed</FieldLabel><Select value={resultDay} onValueChange={value => { setResultDay(value); setResult(dayResults.find(entry => String(entry.day.index) === value)?.result ?? null) }}><SelectTrigger id="completed-day" className="w-full"><SelectValue /></SelectTrigger><SelectContent><SelectGroup>{dayResults.map(entry => <SelectItem key={entry.day.index} value={String(entry.day.index)}>{entry.day.label}</SelectItem>)}</SelectGroup></SelectContent></Select></Field> : null}
          {!busy && result ? <ResultsPanel key={result.recordingId + resultDay + result.preset + result.summary.processingSeconds} result={result} recordingLabel={`${resultSourceLabel || result.recordingId}${dayResults.length > 0 ? ` · ${selectedDayResult?.day.label ?? ''}` : ''}`} downloadPrefix={selectedDayResult ? `mobgap-${selectedDayResult.day.label}` : 'mobgap'} /> : null}
          {!busy && !result ? <Empty className="results-empty"><EmptyHeader><EmptyMedia variant="icon"><Activity /></EmptyMedia><EmptyTitle>{recording ? 'Ready when you are' : 'A clearer view of your walking data'}</EmptyTitle><EmptyDescription>{recording ? 'Check the participant details and run the pipeline. Walking bouts, cadence, stride length and speed will appear here.' : 'Choose an example, a MATLAB recording or a CWA file. Your calculated walking parameters and downloadable tables will appear here.'}</EmptyDescription></EmptyHeader></Empty> : null}
        </section>
      </div>
    </main>
    <footer className="app-footer"><span>mobgap · Mobilise-D gait analysis</span><span>Computed locally. Research use.</span></footer>
  </div>
}
