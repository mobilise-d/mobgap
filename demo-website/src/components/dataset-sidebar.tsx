import { useRef, useState } from 'react'
import { ArrowRight, FileUp, LoaderCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Field, FieldDescription, FieldGroup, FieldLabel, FieldLegend, FieldSet } from '@/components/ui/field'
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { cn } from '@/lib/utils'
import { cwaMode } from './dataset-model'
import type { Recording } from '@/lib/contracts'
import type { CwaConfiguration, DatasetFieldErrors, ParticipantConfiguration, Sample } from './dataset-model'

const cohortOptions = [
  ['HA', 'Healthy adults'], ['COPD', 'COPD'], ['CHF', 'Chronic heart failure'],
  ['PD', 'Parkinson’s disease'], ['MS', 'Multiple sclerosis'], ['PFF', 'Proximal femoral fracture'],
] as const

type SidebarProps = {
  file: File | null; infoFile: File | null; samples: Sample[]; sampleError: boolean
  participant: ParticipantConfiguration; cwa: CwaConfiguration; cwaRecording: Recording | null; fieldErrors: DatasetFieldErrors
  busy: boolean; building: boolean; sampleLoading: boolean; canBuild: boolean; hasCwa: boolean; indexBuilt: boolean
  onFiles: (files: File[]) => void; onInfoFile: (file: File) => void; onSample: (sample: Sample) => void
  onParticipant: (value: ParticipantConfiguration) => void; onCwa: (value: CwaConfiguration) => void; onBuild: () => void
}

function ParticipantStep({ participant, infoFile, hasCwa, fieldErrors, busy, onParticipant, onInfoFile }: Pick<SidebarProps, 'participant' | 'infoFile' | 'hasCwa' | 'fieldErrors' | 'busy' | 'onParticipant' | 'onInfoFile'>) {
  const infoInput = useRef<HTMLInputElement>(null)
  const updateParticipant = (value: Partial<ParticipantConfiguration>) => onParticipant({ ...participant, ...value })
  return <section className="setup-section" aria-labelledby="participant-heading">
        <div className="section-heading"><span className="step-number">02</span><h2 id="participant-heading">Participant information</h2></div>
        <FieldSet disabled={busy}>
          <FieldLegend className="sr-only">Participant information</FieldLegend>
          {!hasCwa ? <div className="mb-4 flex flex-col gap-2">
            <Input ref={infoInput} id="participant-info-files" type="file" accept=".mat" disabled={busy} className="sr-only size-px" aria-label="Upload infoForAlgo participant metadata" onChange={event => { if (event.target.files?.[0]) onInfoFile(event.target.files[0]); event.target.value = '' }} />
            <Button variant={participant.mode === 'file' ? 'secondary' : 'outline'} disabled={busy} onClick={() => infoInput.current?.click()}>Upload infoForAlgo</Button>
            <Button variant={participant.mode === 'manual' ? 'secondary' : 'outline'} disabled={busy} onClick={() => updateParticipant({ mode: 'manual' })}>Enter manually</Button>
            {participant.mode === 'file' ? <p className="help-copy">{infoFile?.name}<br />Heights will be read from this recording’s companion file.</p> : <p className="help-copy">Supply a separate participant metadata file, or enter the required heights.</p>}
</div> : <p className="mb-4 help-copy">CWA recordings require manual participant information. No participant measurements are inferred from the sensor file.</p>}
          <FieldGroup>
            {participant.mode === 'manual' ? <>
              <Field data-invalid={!!fieldErrors.height}><FieldLabel htmlFor="participant-height">Participant height (m)</FieldLabel><Input id="participant-height" aria-invalid={!!fieldErrors.height} aria-describedby="participant-height-description" type="number" inputMode="decimal" min="0" step="0.01" placeholder="Enter measured height" value={participant.height} onChange={event => updateParticipant({ height: event.target.value })} /><FieldDescription id="participant-height-description">{fieldErrors.height}</FieldDescription></Field>
              <Field data-invalid={!!fieldErrors.sensorHeight}><FieldLabel htmlFor="sensor-height">Sensor height (m)</FieldLabel><Input id="sensor-height" aria-invalid={!!fieldErrors.sensorHeight} aria-describedby="sensor-height-description" type="number" inputMode="decimal" min="0" step="0.01" placeholder="Enter measured height" value={participant.sensorHeight} onChange={event => updateParticipant({ sensorHeight: event.target.value })} /><FieldDescription id="sensor-height-description">{fieldErrors.sensorHeight ?? 'Floor to the lower-back sensor; no greater than participant height.'}</FieldDescription></Field>
              <p className="help-copy">Use measured values for this participant and lower-back sensor.</p>
            </> : null}
            <Field><FieldLabel htmlFor="cohort">Participant cohort</FieldLabel><Select value={participant.cohort} onValueChange={cohort => updateParticipant({ cohort })} disabled={busy}><SelectTrigger id="cohort" className="w-full"><SelectValue placeholder="Select a cohort" /></SelectTrigger><SelectContent><SelectGroup>{cohortOptions.map(([value, label]) => <SelectItem key={value} value={value}>{label} ({value})</SelectItem>)}</SelectGroup></SelectContent></Select></Field>
            <Field><FieldLabel htmlFor="condition">Recording setting</FieldLabel><Select value={participant.condition} onValueChange={condition => updateParticipant({ condition: condition as ParticipantConfiguration['condition'] })} disabled={busy}><SelectTrigger id="condition" className="w-full"><SelectValue /></SelectTrigger><SelectContent><SelectGroup><SelectItem value="laboratory">Laboratory</SelectItem><SelectItem value="free_living">Free living</SelectItem></SelectGroup></SelectContent></Select></Field>
          </FieldGroup>
        </FieldSet>
      </section>
}

function CwaStep({ cwa, cwaRecording, fieldErrors, busy, onCwa }: Pick<SidebarProps, 'cwa' | 'cwaRecording' | 'fieldErrors' | 'busy' | 'onCwa'>) {
  const updateCwa = (value: Partial<CwaConfiguration>) => onCwa({ ...cwa, ...value })
  return <section className="setup-section" aria-labelledby="cwa-heading">
          <div className="section-heading"><span className="step-number">03</span><h2 id="cwa-heading">CWA dataset index</h2></div>
          <FieldSet disabled={busy}><FieldLegend className="sr-only">CWA processing and timezone</FieldLegend><FieldGroup>
            {cwaRecording ? <Field>
              <FieldLabel htmlFor="cwa-scope">Process recording</FieldLabel>
              <Select value={cwaMode(cwaRecording, cwa)} onValueChange={mode => updateCwa({ mode: mode as 'days' | 'file' })} disabled={busy}>
                <SelectTrigger id="cwa-scope" className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent><SelectGroup><SelectItem value="file">Single file</SelectItem><SelectItem value="days">Split by calendar day</SelectItem></SelectGroup></SelectContent>
              </Select>
              <FieldDescription>Recordings longer than 24 hours default to calendar days; recordings of 24 hours or less default to a single file. Change this choice and rebuild to override.</FieldDescription>
            </Field> : null}
            {!cwaRecording ? <p className="help-copy">Build the dataset to read recording duration. Recordings longer than 24 hours default to calendar days; recordings of 24 hours or less default to a single file. You can then change this choice and rebuild.</p> : null}
            <Field data-invalid={!!fieldErrors.timezone}><FieldLabel htmlFor="clock-timezone">Sensor synchronization timezone</FieldLabel><Input id="clock-timezone" aria-invalid={!!fieldErrors.timezone} aria-describedby="clock-timezone-description" placeholder="e.g. Europe/Berlin or UTC" value={cwa.timezone} onChange={event => updateCwa({ timezone: event.target.value })} /><FieldDescription id="clock-timezone-description">{fieldErrors.timezone ?? 'IANA timezone of the computer that last synchronized the sensor. The UTC offset at synchronization remains fixed for the recording.'}</FieldDescription></Field>
            <p className="help-copy">Full presets require gyroscope channels and a lower-back sensor in the mobgap sensor-axis convention. A complete day can require substantial browser memory.</p>
          </FieldGroup></FieldSet>
        </section>
}

export function DatasetSidebar({ file, infoFile, samples, sampleError, participant, cwa, cwaRecording, fieldErrors, busy, building, sampleLoading, canBuild, hasCwa, indexBuilt, onFiles, onInfoFile, onSample, onParticipant, onCwa, onBuild }: SidebarProps) {
  const fileInput = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  return <aside className="controls-panel" aria-label="Dataset configuration">
    <section className="setup-section" aria-labelledby="files-heading">
      <div className="section-heading"><span className="step-number">01</span><h2 id="files-heading">Recording file</h2></div>
      <div className={cn('file-dropzone', dragging && 'file-dropzone-active')} onDragOver={event => { event.preventDefault(); if (!busy) setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); if (!busy) onFiles(Array.from(event.dataTransfer.files)) }}>
        <FileUp className="size-6 text-muted-foreground" aria-hidden="true" />
        <p>Drop one recording file here</p><span>Mobilise-D .mat or AX6/AX3 .cwa</span>
        <Input ref={fileInput} id="matlab-files" type="file" accept=".mat,.cwa" disabled={busy} className="sr-only size-px" aria-label="Choose recording file" onChange={event => { if (event.target.files) onFiles(Array.from(event.target.files)); event.target.value = '' }} />
        <Button variant="outline" disabled={busy} onClick={() => fileInput.current?.click()}>Choose file</Button>
      </div>
      {file ? <p className="mt-3 break-words text-xs text-muted-foreground" aria-label="Selected recording file">{file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB</p> : null}
      <p className="mt-3 help-copy">Files stay on your device. Nothing is read until you supply participant information and build the dataset.</p>
      <div className="example-list">{sampleLoading ? <p className="help-copy" role="status">Loading example files…</p> : null}<p className="eyebrow">Or use an example</p>{samples.map(sample => <Button key={sample.id} variant="ghost" className="example-button" disabled={busy} onClick={() => onSample(sample)}><span><strong>{sample.label}</strong><small>{sample.description}</small></span><ArrowRight className="shrink-0" data-icon="inline-end" /></Button>)}{sampleError ? <p className="help-copy">Examples are unavailable. Choose local files instead.</p> : null}</div>
    </section>
    {file ? <>
      <Separator />
      <ParticipantStep participant={participant} infoFile={infoFile} fieldErrors={fieldErrors} hasCwa={hasCwa} busy={busy} onParticipant={onParticipant} onInfoFile={onInfoFile} />
      {hasCwa ? <>
        <Separator />
        <CwaStep cwa={cwa} cwaRecording={cwaRecording} fieldErrors={fieldErrors} busy={busy} onCwa={onCwa} />
      </> : null}
      <Separator />
      <section className="setup-section">
        <Button className="w-full" size="lg" disabled={!canBuild || busy} onClick={onBuild}>{building ? <LoaderCircle className="animate-spin" data-icon="inline-start" /> : null}{indexBuilt ? 'Rebuild dataset' : 'Build dataset'}</Button>
        <p className="mt-3 help-copy">Complete participant information first. Building the dataset reveals its index; all rows start selected.</p>
      </section>
    </> : null}
  </aside>
}
