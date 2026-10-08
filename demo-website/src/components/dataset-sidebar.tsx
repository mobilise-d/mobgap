import { useRef, useState } from 'react'
import { ArrowRight, FileUp, LoaderCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Field, FieldDescription, FieldGroup, FieldLabel, FieldLegend, FieldSet } from '@/components/ui/field'
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { cn } from '@/lib/utils'
import type { CwaConfiguration, ParticipantConfiguration, Sample } from './dataset-model'

const cohortOptions = [
  ['HA', 'Healthy adults'], ['COPD', 'COPD'], ['CHF', 'Chronic heart failure'],
  ['PD', 'Parkinson’s disease'], ['MS', 'Multiple sclerosis'], ['PFF', 'Proximal femoral fracture'],
] as const

type SidebarProps = {
  files: File[]; infoFiles: File[]; samples: Sample[]; sampleError: boolean
  participant: ParticipantConfiguration; cwa: CwaConfiguration
  busy: boolean; building: boolean; sampleLoading: boolean; canBuild: boolean; hasCwa: boolean; indexBuilt: boolean
  onFiles: (files: File[]) => void; onInfoFiles: (files: File[]) => void; onSample: (sample: Sample) => void
  onParticipant: (value: ParticipantConfiguration) => void; onCwa: (value: CwaConfiguration) => void; onBuild: () => void
}

function ParticipantStep({ participant, infoFiles, hasCwa, busy, onParticipant, onInfoFiles }: Pick<SidebarProps, 'participant' | 'infoFiles' | 'hasCwa' | 'busy' | 'onParticipant' | 'onInfoFiles'>) {
  const infoInput = useRef<HTMLInputElement>(null)
  const updateParticipant = (value: Partial<ParticipantConfiguration>) => onParticipant({ ...participant, ...value })
  return <section className="setup-section" aria-labelledby="participant-heading">
        <div className="section-heading"><span className="step-number">02</span><h2 id="participant-heading">Participant information</h2></div>
        <FieldSet disabled={busy}>
          <FieldLegend className="sr-only">Participant information</FieldLegend>
          {!hasCwa ? <div className="mb-4 flex flex-col gap-2">
            <Input ref={infoInput} id="participant-info-files" type="file" accept=".mat" multiple disabled={busy} className="sr-only size-px" aria-label="Upload infoForAlgo participant metadata" onChange={event => { if (event.target.files) onInfoFiles(Array.from(event.target.files)); event.target.value = '' }} />
            <Button variant={participant.mode === 'file' ? 'secondary' : 'outline'} disabled={busy} onClick={() => infoInput.current?.click()}>Upload infoForAlgo</Button>
            <Button variant={participant.mode === 'manual' ? 'secondary' : 'outline'} disabled={busy} onClick={() => updateParticipant({ mode: 'manual' })}>Enter manually</Button>
            {participant.mode === 'file' ? <p className="help-copy">{infoFiles.map(file => file.name).join(', ')}<br />Heights will be read separately for each dataset recording.</p> : <p className="help-copy">Supply a separate participant metadata file, or enter the required heights.</p>}
          </div> : <p className="mb-4 help-copy">CWA recordings require manual participant information. No participant measurements are inferred from the sensor file.</p>}
          <FieldGroup>
            {participant.mode === 'manual' ? <>
              <Field><FieldLabel htmlFor="participant-height">Participant height (m)</FieldLabel><Input id="participant-height" type="number" inputMode="decimal" min="0" step="0.01" placeholder="Enter measured height" value={participant.height} onChange={event => updateParticipant({ height: event.target.value })} /></Field>
              <Field><FieldLabel htmlFor="sensor-height">Sensor height (m)</FieldLabel><Input id="sensor-height" type="number" inputMode="decimal" min="0" step="0.01" placeholder="Enter measured height" value={participant.sensorHeight} onChange={event => updateParticipant({ sensorHeight: event.target.value })} /><FieldDescription>Floor to the lower-back sensor; no greater than participant height.</FieldDescription></Field>
              <p className="help-copy">Manual values apply to every selected recording file.</p>
            </> : null}
            <Field><FieldLabel htmlFor="cohort">Participant cohort</FieldLabel><Select value={participant.cohort} onValueChange={cohort => updateParticipant({ cohort })} disabled={busy}><SelectTrigger id="cohort" className="w-full"><SelectValue placeholder="Select a cohort" /></SelectTrigger><SelectContent><SelectGroup>{cohortOptions.map(([value, label]) => <SelectItem key={value} value={value}>{label} ({value})</SelectItem>)}</SelectGroup></SelectContent></Select></Field>
            <Field><FieldLabel htmlFor="condition">Recording setting</FieldLabel><Select value={participant.condition} onValueChange={condition => updateParticipant({ condition: condition as ParticipantConfiguration['condition'] })} disabled={busy}><SelectTrigger id="condition" className="w-full"><SelectValue /></SelectTrigger><SelectContent><SelectGroup><SelectItem value="laboratory">Laboratory</SelectItem><SelectItem value="free_living">Free living</SelectItem></SelectGroup></SelectContent></Select></Field>
          </FieldGroup>
        </FieldSet>
      </section>
}

function CwaStep({ cwa, busy, onCwa }: Pick<SidebarProps, 'cwa' | 'busy' | 'onCwa'>) {
  const updateCwa = (value: Partial<CwaConfiguration>) => onCwa({ ...cwa, ...value })
  return <section className="setup-section" aria-labelledby="cwa-heading">
          <div className="section-heading"><span className="step-number">03</span><h2 id="cwa-heading">CWA dataset index</h2></div>
          <FieldSet disabled={busy}><FieldLegend className="sr-only">CWA windows and timezone</FieldLegend><FieldGroup>
            <Field><FieldLabel htmlFor="cwa-scope">Split recording</FieldLabel><Select value={cwa.scope} onValueChange={scope => updateCwa({ scope: scope as CwaConfiguration['scope'] })} disabled={busy}><SelectTrigger id="cwa-scope" className="w-full"><SelectValue /></SelectTrigger><SelectContent><SelectGroup><SelectItem value="days">By calendar day</SelectItem><SelectItem value="window">Bounded time window</SelectItem></SelectGroup></SelectContent></Select><FieldDescription>Calendar days include partial first and last days. Rows run in sequence.</FieldDescription></Field>
            {cwa.scope === 'window' ? <>
              <Field><FieldLabel htmlFor="window-start">Start offset (seconds)</FieldLabel><Input id="window-start" type="number" min="0" step="1" value={cwa.start} onChange={event => updateCwa({ start: event.target.value })} /></Field>
              <Field><FieldLabel htmlFor="window-duration">Duration (seconds)</FieldLabel><Input id="window-duration" type="number" min="0" max="3600" step="1" value={cwa.duration} onChange={event => updateCwa({ duration: event.target.value })} /><FieldDescription>Positive duration, up to one hour, from the first recorded sample.</FieldDescription></Field>
            </> : null}
            <Field><FieldLabel htmlFor="clock-timezone">Sensor synchronization timezone</FieldLabel><Input id="clock-timezone" placeholder="e.g. Europe/Berlin or UTC" value={cwa.timezone} onChange={event => updateCwa({ timezone: event.target.value })} /><FieldDescription>IANA timezone of the computer that last synchronized the sensor. The UTC offset at synchronization remains fixed for the recording.</FieldDescription></Field>
            <p className="help-copy">Full presets require gyroscope channels and a lower-back sensor in the mobgap sensor-axis convention. A complete day can require substantial browser memory.</p>
          </FieldGroup></FieldSet>
        </section>
}

export function DatasetSidebar({ files, infoFiles, samples, sampleError, participant, cwa, busy, building, sampleLoading, canBuild, hasCwa, indexBuilt, onFiles, onInfoFiles, onSample, onParticipant, onCwa, onBuild }: SidebarProps) {
  const fileInput = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  return <aside className="controls-panel" aria-label="Dataset configuration">
    <section className="setup-section" aria-labelledby="files-heading">
      <div className="section-heading"><span className="step-number">01</span><h2 id="files-heading">Recording files</h2></div>
      <div className={cn('file-dropzone', dragging && 'file-dropzone-active')} onDragOver={event => { event.preventDefault(); if (!busy) setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); if (!busy) onFiles(Array.from(event.dataTransfer.files)) }}>
        <FileUp className="size-6 text-muted-foreground" aria-hidden="true" />
        <p>Drop recording files here</p><span>Mobilise-D .mat or AX6/AX3 .cwa</span>
        <Input ref={fileInput} id="matlab-files" type="file" accept=".mat,.cwa" multiple disabled={busy} className="sr-only size-px" aria-label="Choose recording files" onChange={event => { if (event.target.files) onFiles(Array.from(event.target.files)); event.target.value = '' }} />
        <Button variant="outline" disabled={busy} onClick={() => fileInput.current?.click()}>Choose files</Button>
      </div>
      {files.length > 0 ? <ul className="mt-3 flex flex-col gap-1 text-xs text-muted-foreground" aria-label="Selected recording files">{files.map(file => <li key={file.name} className="break-words">{file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB</li>)}</ul> : null}
      <p className="mt-3 help-copy">Files stay on your device. Nothing is read until you supply participant information and build the dataset.</p>
      <div className="example-list">{sampleLoading ? <p className="help-copy" role="status">Loading example files…</p> : null}<p className="eyebrow">Or use an example</p>{samples.map(sample => <Button key={sample.id} variant="ghost" className="example-button" disabled={busy} onClick={() => onSample(sample)}><span><strong>{sample.label}</strong><small>{sample.description}</small></span><ArrowRight className="shrink-0" data-icon="inline-end" /></Button>)}{sampleError ? <p className="help-copy">Examples are unavailable. Choose local files instead.</p> : null}</div>
    </section>
    {files.length > 0 ? <>
      <Separator />
      <ParticipantStep participant={participant} infoFiles={infoFiles} hasCwa={hasCwa} busy={busy} onParticipant={onParticipant} onInfoFiles={onInfoFiles} />
      {hasCwa ? <>
        <Separator />
        <CwaStep cwa={cwa} busy={busy} onCwa={onCwa} />
      </> : null}
      <Separator />
      <section className="setup-section">
        <Button className="w-full" size="lg" disabled={!canBuild || busy} onClick={onBuild}>{building ? <LoaderCircle className="animate-spin" data-icon="inline-start" /> : null}{indexBuilt ? 'Rebuild dataset' : 'Build dataset'}</Button>
        <p className="mt-3 help-copy">Complete participant information first. Building the dataset reveals its index; all rows start selected.</p>
      </section>
    </> : null}
  </aside>
}
