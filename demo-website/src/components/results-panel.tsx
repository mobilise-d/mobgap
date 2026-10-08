import { useState } from 'react'
import { ArrowDownToLine, ChevronLeft, ChevronRight, CheckCircle2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Field, FieldLabel } from '@/components/ui/field'
import type { AnalysisResult } from '@/lib/contracts'

type ResultTable = AnalysisResult['tables'][string]
const PAGE_SIZE = 25
const labels: Record<string, string> = {
  walking_bouts: 'Walking bouts', gait_sequences: 'Gait sequences', initial_contacts: 'Initial contacts',
  turns: 'Turns', per_second_parameters: 'Parameters per second', raw_per_stride_parameters: 'Raw stride parameters',
  aggregated_parameters: 'Aggregated parameters', per_stride_parameters: 'Strides',
}
const tableLabel = (name: string) => labels[name]

const numericFormat = new Intl.NumberFormat('en', { maximumFractionDigits: 4 })
const cellText = (value: unknown): string => {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return numericFormat.format(value)
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
function csvCell(value: unknown): string {
  const text = value === null || value === undefined ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value)
  return `"${text.replaceAll('"', '""')}"`
}
function downloadCsv(table: ResultTable, filename: string) {
  const rows = table.rows.map(row => row.map(csvCell).join(','))
  const csv = '\ufeff' + [table.columns.map(csvCell).join(','), ...rows].join('\r\n')
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export function ResultsPanel({ result, recordingLabel, downloadPrefix = 'mobgap' }: { result: AnalysisResult; recordingLabel: string; downloadPrefix?: string }) {
  const tableNames = Object.keys(result.tables)
  const defaultTable = tableNames.find(name => name === 'walking_bouts') ?? tableNames[0] ?? ''
  const [selectedTable, setSelectedTable] = useState(defaultTable)
  const [page, setPage] = useState(0)
  const table = result.tables[selectedTable]
  const totalPages = table ? Math.max(1, Math.ceil(table.rows.length / PAGE_SIZE)) : 1
  const statistics = [
    ['Walking bouts', result.summary.walkingBouts],
    ['Gait sequences', result.summary.gaitSequences],
    ['Initial contacts', result.summary.initialContacts],
    ['Strides', result.summary.strides],
  ] as const

  return <div className="flex flex-col gap-6">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <div className="mb-2 flex items-center gap-2"><CheckCircle2 className="size-4 text-primary" /><Badge variant="secondary">Analysis complete</Badge></div>
        <h2 className="text-xl font-semibold tracking-tight">Your results</h2>
        <p className="mt-1 text-sm text-muted-foreground break-words">{recordingLabel}</p>
      </div>
      <div className="text-right text-xs text-muted-foreground"><p>{result.preset === 'healthy' ? 'Healthy walking' : 'Impaired walking'} preset</p><p className="mt-1 tabular-nums">{result.summary.processingSeconds.toFixed(2)} s computation</p></div>
    </div>
    <dl className="results-stats">
      {statistics.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{numericFormat.format(value)}</dd></div>)}
    </dl>
    {result.warnings.length > 0 ? <Alert><AlertTitle>Notes from this analysis</AlertTitle><AlertDescription><ul className="list-disc pl-4">{result.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul></AlertDescription></Alert> : null}
    <div className="results-table-section">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <Field className="w-full max-w-xs"><FieldLabel htmlFor="result-table">Result table</FieldLabel><Select value={selectedTable} onValueChange={name => { setSelectedTable(name); setPage(0) }}><SelectTrigger id="result-table" className="w-full"><SelectValue /></SelectTrigger><SelectContent><SelectGroup>{tableNames.map(name => <SelectItem key={name} value={name}>{tableLabel(name)}</SelectItem>)}</SelectGroup></SelectContent></Select></Field>
        <Button variant="outline" disabled={!table} onClick={() => table && downloadCsv(table, `${downloadPrefix}-${result.preset}-${selectedTable}.csv`)}><ArrowDownToLine data-icon="inline-start" />Download CSV</Button>
      </div>
      {table ? <>
        <div className="mt-4 rounded-lg border">
          <Table aria-label={tableLabel(selectedTable)}>
            <TableHeader><TableRow>{table.columns.map(column => <TableHead key={column} title={column}>{column.replace(/_+/g, ' ')}</TableHead>)}</TableRow></TableHeader>
            <TableBody>{table.rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map((row, i) => <TableRow key={page * PAGE_SIZE + i}>
              {row.map((value, column) => <TableCell className="max-w-72 tabular-nums" key={column}>{cellText(value)}</TableCell>)}
            </TableRow>)}</TableBody>
          </Table>
          {table.rows.length === 0 ? <p className="p-8 text-center text-sm text-muted-foreground">No rows in this result table.</p> : null}
        </div>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
          <p>{table.rows.length.toLocaleString()} rows · Values rounded for display; CSV exports values without display rounding.</p>
          <div className="flex items-center gap-2"><Button variant="outline" size="icon-sm" aria-label="Previous rows" disabled={page === 0} onClick={() => setPage(p => p - 1)}><ChevronLeft /></Button><span className="tabular-nums">{page + 1} / {totalPages}</span><Button variant="outline" size="icon-sm" aria-label="Next rows" disabled={page + 1 >= totalPages} onClick={() => setPage(p => p + 1)}><ChevronRight /></Button></div>
        </div>
      </> : <p className="text-sm text-muted-foreground">This run did not return result tables.</p>}
    </div>
    <details className="text-xs text-muted-foreground"><summary className="cursor-pointer">Runtime versions</summary><dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2">{Object.entries(result.versions ?? {}).map(([name, version]) => <div key={name} className="flex justify-between gap-4"><dt>{name}</dt><dd className="font-mono">{version}</dd></div>)}</dl></details>
  </div>
}
