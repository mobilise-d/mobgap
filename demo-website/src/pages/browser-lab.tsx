import { Link } from '@tanstack/react-router'
import { LoaderCircle, Play, TriangleAlert } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Field, FieldTitle } from '@/components/ui/field'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { ResultsPanel } from '@/components/results-panel'
import { DatasetIndex } from '@/components/dataset-index'
import { DatasetSidebar } from '@/components/dataset-sidebar'
import { cwaMode, rowLabel } from '@/components/dataset-model'
import { useLab } from '@/lib/lab-controller'
import { validateLabSearch } from '@/lib/lab-search'
import type { PipelinePreset } from '@/lib/contracts'

const keepSearch = (previous: Record<string, unknown>) =>
  validateLabSearch(previous)
function PageIntro({
  step,
  title,
  description
}: {
  step: string
  title: string
  description: string
}) {
  return (
    <div className="page-intro">
      <div>
        <p className="eyebrow">{step}</p>
        <h1>{title}</h1>
        <p className="intro-copy">{description}</p>
      </div>
    </div>
  )
}
function OperationMessages() {
  const lab = useLab()
  return (
    <>
      {lab.notice ? (
        <Alert>
          <AlertTitle>Operation stopped</AlertTitle>
          <AlertDescription>{lab.notice}</AlertDescription>
        </Alert>
      ) : null}
      {lab.error || lab.fileErrors.length > 0 ? (
        <Alert variant="destructive">
          <TriangleAlert />
          <AlertTitle>Could not complete this step</AlertTitle>
          <AlertDescription>
            {lab.error ? (
              <p className="break-words whitespace-pre-wrap">{lab.error}</p>
            ) : null}
            {lab.fileErrors.map((message) => (
              <p key={message} className="break-words">
                {message}
              </p>
            ))}
          </AlertDescription>
        </Alert>
      ) : null}
    </>
  )
}
function SessionRequired({ built = false }: { built?: boolean }) {
  const lab = useLab()
  return (
    <div className="flex flex-col items-start gap-4 rounded-xl border bg-card p-6">
      <h2 className="text-lg font-semibold">
        {!lab.file
          ? 'Select your recording again'
          : built
            ? 'Build this dataset first'
            : 'No completed results yet'}
      </h2>
      <p className="text-sm text-muted-foreground">
        {!lab.file
          ? 'Files and results stay in this browser session. A URL cannot restore a local file after a reload or in another tab.'
          : built
            ? 'The index is unavailable for this configuration or dataset link. Return to Upload and build from your selected recording.'
            : 'Run selected dataset rows to compute results. Opening this page never starts an analysis.'}
      </p>
      <Link
        to={!lab.file || built ? '/upload' : '/dataset'}
        search={keepSearch}
        className="text-sm font-medium text-primary underline underline-offset-4"
      >
        {!lab.file || built ? 'Go to Upload' : 'Go to Dataset'}
      </Link>
    </div>
  )
}
export function UploadPage() {
  const lab = useLab()
  return (
    <>
      <PageIntro
        step="01 · Upload and configuration"
        title="Configure your recording."
        description="Choose one recording, supply participant information, then build its dataset index. Files remain on your device."
      />
      <div className="upload-page flex flex-col gap-5">
        <OperationMessages />
        <DatasetSidebar
          file={lab.file}
          infoFile={lab.infoFile}
          samples={lab.samples}
          sampleError={lab.sampleError}
          participant={lab.participant}
          cwa={lab.cwa}
          cwaRecording={lab.cwaRecording}
          fieldErrors={lab.fieldErrors}
          busy={lab.busy}
          building={lab.operation === 'build'}
          sampleLoading={lab.operation === 'sample'}
          canBuild={lab.canBuild}
          hasCwa={lab.hasCwa}
          indexBuilt={lab.rows.length > 0}
          onFiles={lab.stageFiles}
          onInfoFile={lab.stageInfoFile}
          onSample={(sample) => void lab.loadSample(sample)}
          onParticipant={lab.updateParticipant}
          onCwa={lab.updateCwa}
          onBuild={() => void lab.buildDataset()}
        />
        {lab.operation === 'sample' ? (
          <Button
            variant="outline"
            className="self-start"
            onClick={lab.cancelOperation}
          >
            Cancel example loading
          </Button>
        ) : lab.busy ? (
          <Link
            to="/progress"
            search={keepSearch}
            className="text-sm text-primary underline"
          >
            View active operation
          </Link>
        ) : null}
      </div>
    </>
  )
}
function DatasetLoading() {
  const lab = useLab()
  return (
    <section
      className="flex flex-col items-start gap-4 rounded-xl border bg-card p-6"
      aria-live="polite"
    >
      <div className="flex items-start gap-3">
        <LoaderCircle
          className="size-5 shrink-0 animate-spin text-primary"
          aria-hidden="true"
        />
        <p className="break-words text-sm">
          {lab.progress?.message ?? 'Preparing dataset…'}
        </p>
      </div>
      <Button variant="outline" onClick={lab.cancelOperation}>
        Cancel
      </Button>
    </section>
  )
}

export function DatasetPage() {
  const lab = useLab()
  return (
    <>
      <PageIntro
        step="02 · Dataset"
        title="Choose the rows to analyze."
        description="All rows start selected. Select MATLAB trials or CWA calendar days, then choose a walking preset."
      />
      <div className="flex min-w-0 flex-col gap-5">
        <OperationMessages />
        {lab.operation === 'build' ? (
          <DatasetLoading />
        ) : !lab.file ||
          !lab.sessionMatches ||
          !lab.configurationMatches ||
          lab.rows.length === 0 ? (
          <SessionRequired built />
        ) : (
          <section
            className="flex min-w-0 flex-col gap-4 rounded-xl border bg-card p-5"
            aria-labelledby="index-heading"
          >
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2
                  id="index-heading"
                  className="text-lg font-semibold tracking-tight"
                >
                  Dataset index
                </h2>
                <p className="mt-1 break-words text-xs text-muted-foreground">
                  {lab.file.name} · {lab.selected.size} of {lab.rows.length}{' '}
                  rows selected
                </p>
              </div>
              <Badge variant="outline">
                {lab.cwaRecording
                  ? cwaMode(lab.cwaRecording, lab.cwa) === 'days'
                    ? 'Calendar days'
                    : 'Single file'
                  : 'MATLAB trials'}
              </Badge>
            </div>
            <div className="flex flex-wrap items-end justify-between gap-3">
              <Field className="w-full max-w-xs">
                <FieldTitle id="walking-preset-label">
                  Walking preset
                </FieldTitle>
                <ToggleGroup
                  aria-labelledby="walking-preset-label"
                  type="single"
                  variant="outline"
                  value={lab.pipeline}
                  onValueChange={(value) => {
                    if (value) lab.selectPipeline(value as PipelinePreset)
                  }}
                  disabled={lab.busy}
                  className="w-full"
                >
                  <ToggleGroupItem className="flex-1" value="healthy">
                    Healthy
                  </ToggleGroupItem>
                  <ToggleGroupItem className="flex-1" value="impaired">
                    Impaired
                  </ToggleGroupItem>
                  <ToggleGroupItem className="flex-1" value="auto">
                    Auto
                  </ToggleGroupItem>
                </ToggleGroup>
              </Field>
              <Button
                disabled={!lab.canRun}
                onClick={() => void lab.runSelected()}
              >
                <Play data-icon="inline-start" />
                Run selected ({lab.selected.size})
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              {lab.pipeline === 'auto'
                ? 'Auto selects healthy for HA, COPD and CHF; impaired for PD, MS and PFF.'
                : lab.pipeline === 'healthy'
                  ? 'Healthy is recommended for HA, COPD and CHF.'
                  : 'Impaired is recommended for PD, MS and PFF.'}{' '}
              Rows run sequentially.
            </p>
            {!lab.mounted ? (
              <p className="text-xs text-destructive">
                The worker stopped. Return to Upload and rebuild before running
                more rows.
              </p>
            ) : lab.selectedProblems.length > 0 ? (
              <p className="text-xs text-destructive">
                {lab.selectedProblems.length} selected rows cannot run. Check
                their status or deselect them.
              </p>
            ) : null}
            <DatasetIndex
              rows={lab.rows}
              selected={lab.selected}
              outcomes={lab.outcomes}
              disabled={lab.busy || !lab.mounted}
              onToggle={lab.toggleRow}
              onToggleAll={lab.toggleAll}
              onViewResult={lab.viewResult}
            />
            <div className="flex flex-wrap gap-4 text-sm">
              <Link
                to="/upload"
                search={keepSearch}
                className="text-primary underline underline-offset-4"
              >
                Change configuration
              </Link>
              {lab.busy ? (
                <Link
                  to="/progress"
                  search={keepSearch}
                  className="text-primary underline underline-offset-4"
                >
                  View progress
                </Link>
              ) : null}
              {!lab.busy && lab.resultRows.length > 0 ? (
                <Link
                  to="/results"
                  search={keepSearch}
                  className="text-primary underline underline-offset-4"
                >
                  View results
                </Link>
              ) : null}
            </div>
          </section>
        )}
        {lab.warnings.length > 0 ? (
          <Alert>
            <AlertTitle>Dataset notes</AlertTitle>
            <AlertDescription>
              {lab.warnings.map((message) => (
                <p key={message} className="break-words">
                  {message}
                </p>
              ))}
            </AlertDescription>
          </Alert>
        ) : null}
      </div>
    </>
  )
}
export function ProgressPage() {
  const lab = useLab()
  return (
    <>
      <PageIntro
        step="03 · Progress"
        title={lab.busy ? 'Processing locally.' : 'Operation finished.'}
        description="The worker continues while you move between pages. Opening this page does not start or repeat an operation."
      />
      <div className="flex flex-col gap-5">
        <OperationMessages />
        {!lab.file ? (
          <SessionRequired />
        ) : (
          <section className="flex flex-col gap-4 rounded-xl border bg-card p-6">
            {lab.busy ? (
              <>
                <div
                  className="flex items-start gap-3"
                  role="status"
                  aria-live="polite"
                >
                  <LoaderCircle
                    className="size-5 shrink-0 animate-spin text-primary"
                    aria-hidden="true"
                  />
                  <p className="break-words text-sm">
                    {lab.progress?.message ??
                      (lab.operation === 'build'
                        ? 'Preparing dataset…'
                        : lab.operation === 'sample'
                          ? 'Loading example files…'
                          : 'Starting selected rows…')}
                  </p>
                </div>
                {lab.operation === 'run' ? (
                  <Progress
                    value={
                      lab.jobRows.length
                        ? (lab.processedCount / lab.jobRows.length) * 100
                        : 0
                    }
                    aria-label="Selected rows processed"
                  />
                ) : lab.progress?.percent !== undefined ? (
                  <Progress
                    value={lab.progress.percent}
                    aria-label="Operation progress"
                  />
                ) : null}
                <Button
                  variant="outline"
                  className="self-start"
                  onClick={lab.cancelOperation}
                >
                  Cancel
                </Button>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">
                No operation is running. Use Dataset to start a new analysis
                explicitly.
              </p>
            )}
            {lab.operation === 'run' || Object.keys(lab.outcomes).length > 0 ? (
              <>
                <p className="text-xs text-muted-foreground">
                  {lab.processedCount} of {lab.jobRows.length} rows processed.
                  Completed results remain available after cancellation.
                </p>
                <ul className="flex flex-col gap-3">
                  {lab.jobRows.map((row) => (
                    <li
                      key={row.id}
                      className="flex flex-wrap justify-between gap-2 border-t pt-3 text-sm"
                    >
                      <span className="break-words">{rowLabel(row)}</span>
                      {lab.outcomes[row.id]?.result ? (
                        <span className="text-muted-foreground">Complete</span>
                      ) : (
                        <span className="break-words text-muted-foreground">
                          {lab.outcomes[row.id]?.message ??
                            lab.outcomes[row.id]?.status}
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              </>
            ) : null}
            <div className="flex flex-wrap gap-4 text-sm">
              <Link
                to="/dataset"
                search={keepSearch}
                className="text-primary underline underline-offset-4"
              >
                Go to Dataset
              </Link>
              <Link
                to="/upload"
                search={keepSearch}
                className="text-primary underline underline-offset-4"
              >
                Go to Upload
              </Link>
              {!lab.busy && lab.resultRows.length > 0 ? (
                <Link
                  to="/results"
                  search={keepSearch}
                  className="text-primary underline underline-offset-4"
                >
                  View completed results
                </Link>
              ) : null}
            </div>
          </section>
        )}
      </div>
    </>
  )
}
function BatchSummary() {
  const lab = useLab()
  return (
    <section
      className="flex flex-col gap-4 rounded-xl border bg-card p-5"
      aria-label="Batch results summary"
    >
      <h2 className="text-lg font-semibold">
        {lab.processedCount} of {lab.jobRows.length} rows processed
      </h2>
      <p className="text-sm text-muted-foreground">
        {lab.jobRows.filter((row) => lab.outcomes[row.id]?.result).length}{' '}
        completed with results ·{' '}
        {
          lab.jobRows.filter((row) => lab.outcomes[row.id]?.status === 'error')
            .length
        }{' '}
        failed
      </p>
      <ul className="flex flex-col gap-3">
        {lab.jobRows.map((row) => (
          <li
            key={row.id}
            className="flex flex-col gap-1 border-t pt-3 text-sm"
          >
            <span className="font-medium">{rowLabel(row)}</span>
            <span
              className={
                lab.outcomes[row.id]?.status === 'error'
                  ? 'break-words text-destructive'
                  : 'text-muted-foreground'
              }
            >
              {lab.outcomes[row.id]?.result
                ? 'Complete'
                : (lab.outcomes[row.id]?.message ??
                  lab.outcomes[row.id]?.status)}
            </span>
          </li>
        ))}
      </ul>
      {!lab.jobRows.some((row) => lab.outcomes[row.id]?.result) ? (
        <p className="text-sm">
          No selected rows produced results. The row messages above explain what
          failed or was stopped.
        </p>
      ) : null}
      <Link
        to="/dataset"
        search={keepSearch}
        className="self-start text-sm text-primary underline underline-offset-4"
      >
        Back to Dataset
      </Link>
    </section>
  )
}

export function ResultsPage() {
  const lab = useLab()
  if (lab.busy) return <ProgressPage />
  const requested =
    lab.search.result === undefined ? undefined : lab.rows[lab.search.result]
  const row =
    requested &&
    lab.resultRows.includes(requested) &&
    lab.outcomes[requested.id]?.result
      ? requested
      : lab.resultRows.at(-1)
  const result = row ? lab.outcomes[row.id]?.result : undefined
  return (
    <>
      <PageIntro
        step="04 · Results"
        title="Inspect computed results."
        description="Choose a completed row and result table. CSV exports values without display rounding."
      />
      <div className="flex min-w-0 flex-col gap-5">
        <OperationMessages />
        {!lab.file || !lab.sessionMatches ? (
          <SessionRequired built={!!lab.file} />
        ) : (
          <>
            {lab.jobRows.length > 0 ? <BatchSummary /> : <SessionRequired />}
            {row && result ? (
              <section
                className="results-panel"
                aria-label="Selected row results"
              >
                <Field className="mb-6">
                  <FieldTitle id="result-row-label">
                    Computed rows · {lab.resultRows.length}
                  </FieldTitle>
                  <Select
                    value={String(lab.rows.indexOf(row))}
                    onValueChange={(value) =>
                      lab.updateSearch(
                        {
                          result: Number(value),
                          table: undefined,
                          page: undefined
                        },
                        false
                      )
                    }
                  >
                    <SelectTrigger
                      id="result-row"
                      aria-labelledby="result-row-label"
                      className="w-full"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        {lab.resultRows.map((value) => (
                          <SelectItem
                            key={value.id}
                            value={String(lab.rows.indexOf(value))}
                          >
                            {rowLabel(value)}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </Field>
                <ResultsPanel
                  key={`${row.id}:${result.preset}:${result.summary.processingSeconds}:${lab.search.table ?? ''}`}
                  result={result}
                  selectedTable={lab.search.table}
                  page={lab.search.page}
                  onPageChange={(page) => lab.updateSearch({ page }, false)}
                  onTableChange={(table) =>
                    lab.updateSearch({ table, page: undefined }, false)
                  }
                  recordingLabel={rowLabel(row)}
                  downloadPrefix={`mobgap-${row.recording.fileName.replace(/\.(mat|cwa)$/i, '')}-${row.day?.label ?? row.label.replace(/[^a-zA-Z0-9-]+/g, '-')}`}
                />
              </section>
            ) : null}
          </>
        )}
      </div>
    </>
  )
}
