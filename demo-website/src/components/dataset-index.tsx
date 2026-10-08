import { useEffect, useRef } from 'react'
import { CheckCircle2, LoaderCircle, TriangleAlert } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { duration, rowLabel, rowProblem } from './dataset-model'
import type { DatasetRow, RowOutcome } from './dataset-model'

const heightFormat = new Intl.NumberFormat('en', { maximumFractionDigits: 3 })
const heightLabel = (value: number | undefined) => value === undefined ? '—' : heightFormat.format(value)

function SelectionCheckbox({ checked, mixed = false, label, disabled, onChange }: {
  checked: boolean; mixed?: boolean; label: string; disabled: boolean; onChange: () => void
}) {
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => { if (input.current) input.current.indeterminate = mixed }, [mixed])
  return <input ref={input} type="checkbox" checked={checked} aria-label={label} aria-checked={mixed ? 'mixed' : checked} disabled={disabled} onChange={onChange} className="size-4 cursor-pointer accent-primary disabled:cursor-default focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring" />
}

export function DatasetIndex({ rows, selected, outcomes, disabled, onToggle, onToggleAll, onViewResult }: {
  rows: DatasetRow[]
  selected: Set<string>
  outcomes: Record<string, RowOutcome>
  disabled: boolean
  onToggle: (id: string) => void
  onToggleAll: () => void
  onViewResult: (id: string) => void
}) {
  const columns = [...new Set(rows.flatMap(row => Object.keys(row.indexValues)))]
  return <div className="overflow-hidden rounded-lg border">
    <Table aria-label="Dataset index">
      <TableHeader><TableRow>
        <TableHead className="w-10"><SelectionCheckbox checked={selected.size === rows.length} mixed={selected.size > 0 && selected.size < rows.length} label="Select all dataset rows" disabled={disabled} onChange={onToggleAll} /></TableHead>
        <TableHead>File</TableHead>
        {columns.map(column => <TableHead key={column} className="whitespace-nowrap">{column.replaceAll('_', ' ')}</TableHead>)}
        <TableHead className="whitespace-nowrap">Duration / rate</TableHead>
        <TableHead className="whitespace-nowrap">Participant / sensor height</TableHead>
        <TableHead>Status</TableHead>
      </TableRow></TableHeader>
      <TableBody>{rows.map(row => {
        const outcome = outcomes[row.id]
        const problem = rowProblem(row)
        return <TableRow key={row.id} data-state={selected.has(row.id) ? 'selected' : undefined}>
          <TableCell><SelectionCheckbox checked={selected.has(row.id)} label={`Select ${rowLabel(row)}`} disabled={disabled} onChange={() => onToggle(row.id)} /></TableCell>
          <TableCell className="max-w-52"><span className="block break-words text-xs font-medium">{row.recording.fileName}</span><span className="text-[10px] text-muted-foreground">{row.recording.sourceFormat === 'cwa' ? 'CWA' : 'MATLAB'}</span></TableCell>
          {columns.map(column => <TableCell key={column} className="max-w-64 break-words text-xs">{row.indexValues[column] ?? '—'}</TableCell>)}
          <TableCell className="whitespace-nowrap text-xs tabular-nums">{duration(row.durationSeconds)}<br /><span className="text-muted-foreground">{row.recording.samplingRateHz} Hz</span></TableCell>
          <TableCell className="whitespace-nowrap text-xs tabular-nums">{heightLabel(row.recording.metadata.heightM)} / {heightLabel(row.recording.metadata.sensorHeightM)} m</TableCell>
          <TableCell className="min-w-40 max-w-64 text-xs">
            {outcome?.result ? <Button variant="ghost" size="sm" onClick={() => onViewResult(row.id)}><CheckCircle2 data-icon="inline-start" />View results</Button>
              : outcome ? <span className="flex items-start gap-2">{outcome.status === 'running' ? <LoaderCircle className="size-3 shrink-0 animate-spin" aria-hidden="true" /> : outcome.status === 'error' ? <TriangleAlert className="size-3 shrink-0 text-destructive" aria-hidden="true" /> : null}<span className="break-words">{outcome.message ?? outcome.status}</span></span>
              : <span className={problem ? 'text-destructive' : 'text-muted-foreground'}>{problem ?? 'Ready'}</span>}
          </TableCell>
        </TableRow>
      })}</TableBody>
    </Table>
  </div>
}
