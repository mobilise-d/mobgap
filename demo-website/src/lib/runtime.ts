import { transfer } from 'comlink'
import { XeusKernel } from './xeus-kernel'
import type { AnalysisResult, CwaDayWindowsResult, DatasetConfiguration, DayAnalysisEvent, InspectionResult, ProgressHandler, RunDaysOptions, RunPipelineOptions } from './contracts'
export type * from './contracts'

/** A Python exception reported by a live kernel, distinct from a lost worker. */
export class PythonExecutionError extends Error {
  override readonly name = 'PythonExecutionError'
}

const JSON_MARKER = '__MOBGAP_RESULT__'
const STARTUP_TIMEOUT_MS = 5 * 60 * 1000

/** A browser-local Xeus Python worker. No selected file is sent to a server. */
export class MobgapRuntime {
  private kernel?: XeusKernel
  private initialization?: Promise<void>
  private abort?: AbortController
  private pending = new Set<(error: Error) => void>()
  private batch = 0
  private busy = false
  private generation = 0
  private readonly assetRoot = new URL('runtime/', new URL(import.meta.env.BASE_URL, document.baseURI))

  get hasActiveKernel(): boolean { return Boolean(this.kernel && !this.kernel.isDisposed) }

  initialize(onProgress?: ProgressHandler): Promise<void> {
    if (this.initialization) return this.initialization
    const abort = new AbortController()
    this.abort = abort
    let failStartup!: (error: Error) => void
    const failure = new Promise<void>((_, reject) => { failStartup = reject })
    const timer = setTimeout(() => failStartup(new Error('Python startup timed out. Check the runtime assets and connection, then retry.')), STARTUP_TIMEOUT_MS)
    this.initialization = Promise.race([this.start(abort.signal, onProgress, failStartup), failure]).catch((error: unknown) => {
      if (this.abort === abort) this.cancel()
      throw error
    }).finally(() => clearTimeout(timer))
    return this.initialization
  }

  private async start(signal: AbortSignal, progress: ProgressHandler | undefined, failStartup: (error: Error) => void): Promise<void> {
    const generation = this.generation
    progress?.({ stage: 'loading', message: 'Loading Python, NumPy and the WebAssembly runtime…' })
    const response = await fetch(new URL('xeus/mobgap-browser/xpython/kernel.json', this.assetRoot), { signal })
    if (!response.ok) throw new Error('The Python runtime assets are missing. Run the asset setup command.')
    const kernelSpec = { ...await response.json(), name: 'xpython', envName: 'mobgap-browser' }
    signal.throwIfAborted()
    const manifestResponse = await fetch(new URL('worker-manifest.json', this.assetRoot), { signal, cache: 'no-cache' })
    if (!manifestResponse.ok) throw new Error('The Python worker bundle is missing. Run npm run worker:build.')
    const manifest = await manifestResponse.json() as { worker: string }
    signal.throwIfAborted()
    const worker = new Worker(new URL(manifest.worker, this.assetRoot), { name: 'mobgap-python' })
    const workerFailure = (event: Event) => {
      if (signal.aborted) return
      const detail = event.type === 'error' ? (event as ErrorEvent).message : 'A worker message could not be decoded.'
      const error = new Error(`Python worker failed: ${detail || 'Check the runtime assets and retry.'}`)
      failStartup(error)
      for (const reject of this.pending) reject(error)
      this.cancel()
    }
    worker.addEventListener('error', workerFailure)
    worker.addEventListener('messageerror', workerFailure)
    const kernel = new XeusKernel(worker, (text) => progress?.({ stage: 'loading', message: text.trim() }))
    this.kernel = kernel
    await this.cancellable(kernel.remote.initialize({ baseUrl: this.assetRoot.href, kernelId: crypto.randomUUID(), kernelSpec, mountDrive: false, browsingContextId: '' }))
    signal.throwIfAborted()
    progress?.({ stage: 'loading', message: 'Importing mobgap…' })
    const bundleResponse = await fetch(new URL('bootstrap.zip', this.assetRoot), { signal })
    if (!bundleResponse.ok) throw new Error('The mobgap runtime bundle is missing. Run the asset setup command.')
    const bundle = new Uint8Array(await bundleResponse.arrayBuffer())
    this.checkGeneration(generation)
    await this.cancellable(kernel.remote.writeBootstrap(transfer(bundle, [bundle.buffer])))
    signal.throwIfAborted()
    await this.execute(`import pathlib, sys, zipfile, json\nzipfile.ZipFile('/mobgap-app.zip').extractall('/mobgap-app')\npathlib.Path('/mobgap-app.zip').unlink()\nsys.path.insert(0, '/mobgap-app')\nimport mobgap_demo_api as api\nimport pyjs\npyjs.js.eval(pathlib.Path('/mobgap-app/workerfs.js').read_text())\npyjs.js.eval(pathlib.Path('/mobgap-app/bridge.js').read_text())`)
    progress?.({ stage: 'ready', message: 'Python is ready. Files stay in this browser.' })
  }

  private execute(code: string): Promise<string> {
    if (!this.kernel) return Promise.reject(new Error('Python runtime is not initialized.'))
    const generation = this.generation
    const future = this.kernel.requestExecute({ code, store_history: false })
    return new Promise<string>((resolve, reject) => {
      let stdout = ''
      let failure: string | undefined
      let fatalMemoryFailure = false
      const cancelled = (error: Error) => { future.dispose(); reject(error) }
      this.pending.add(cancelled)
      future.onIOPub = (message) => {
        if (message.header.msg_type === 'stream' && message.content.name === 'stdout') stdout += message.content.text ?? ''
        if (message.header.msg_type === 'error') {
          failure = message.content.evalue || message.content.traceback?.join('\n') || 'Python execution failed.'
          // Xeus reports class reprs; other kernels use the exception's name.
          const exceptionType = message.content.ename?.replace(/^<class '([^']+)'>$/, '$1').split('.').at(-1)
          fatalMemoryFailure ||= exceptionType === 'MemoryError' || exceptionType === '_ArrayMemoryError'
        }
      }
      void future.done.then(() => {
        this.pending.delete(cancelled)
        future.dispose()
        if (fatalMemoryFailure) {
          if (generation === this.generation) this.cancel()
          reject(new Error(`${failure || 'Python ran out of memory.'} Select the files again to retry.`))
        } else if (failure !== undefined) reject(new PythonExecutionError(failure))
        else resolve(stdout)
      }, (error: unknown) => {
        this.pending.delete(cancelled)
        future.dispose()
        reject(error)
        if (generation === this.generation) this.cancel()
      })
    })
  }

  private cancellable<T>(operation: Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.pending.add(reject)
      void operation.then((value) => { this.pending.delete(reject); resolve(value) }, (error: unknown) => { this.pending.delete(reject); reject(error) })
    })
  }

  private async call<T>(expression: string, generation = this.generation): Promise<T> {
    this.checkGeneration(generation)
    const output = await this.execute(`print(${JSON.stringify(JSON_MARKER)} + api.call_json(lambda: ${expression}))`)
    const line = output.split('\n').find((line) => line.startsWith(JSON_MARKER))
    if (!line) throw new Error('Python completed without returning a result.')
    const response = JSON.parse(line.slice(JSON_MARKER.length)) as { ok: true; result: T } | { ok: false; error: { message: string; fatal: boolean } }
    if (!response.ok) {
      if (response.error.fatal) {
        this.cancel()
        throw new Error(`${response.error.message || 'Python ran out of memory.'} Select the files again to retry.`)
      }
      throw new PythonExecutionError(response.error.message || 'Python execution failed.')
    }
    return response.result
  }

  async inspectFiles(files: File[], onProgress?: ProgressHandler, configuration?: DatasetConfiguration): Promise<InspectionResult> {
    return this.exclusive(async (generation) => {
      await this.initialize(onProgress)
      this.checkGeneration(generation)
      const names = files.map((file) => file.name.replaceAll(/[\\/]/g, '_'))
      if (new Set(names).size !== files.length) throw new Error('Select files with different filenames in one batch.')
      const folder = `/mobgap/uploads/${++this.batch}`
      onProgress?.({ stage: 'mounting', message: 'Mounting selected files in the local Python worker…' })
      await this.cancellable(this.kernel!.remote.callGlobalReceiver('mobgapWorkerFiles', 'mount', files, folder))
      this.checkGeneration(generation)
      const paths = names.map((name) => `${folder}/${name}`)
      onProgress?.({ stage: 'inspecting', message: 'Inspecting recordings and sensor metadata…' })
      const config = { cohort: configuration?.cohort, participantHeightM: configuration?.heightM, sensorHeightM: configuration?.sensorHeightM, measurementCondition: configuration?.measurementCondition, timezone: configuration?.timezone }
      return this.call<InspectionResult>(`api.inspect_files(${JSON.stringify(paths)}, json.loads(${JSON.stringify(JSON.stringify(config))}))`, generation)
    })
  }

  inspectMat(file: File, onProgress?: ProgressHandler, configuration?: DatasetConfiguration): Promise<InspectionResult> { return this.inspectFiles([file], onProgress, configuration) }

  async getCwaDayWindows(recordingId: string, timezone: string, onProgress?: ProgressHandler): Promise<CwaDayWindowsResult> {
    return this.exclusive(async (generation) => {
      this.requireActiveKernel(generation)
      onProgress?.({ stage: 'inspecting', message: 'Finding calendar days in the selected timezone…' })
      return this.call<CwaDayWindowsResult>(`api.cwa_day_windows(${JSON.stringify(recordingId)}, ${JSON.stringify(timezone)})`, generation)
    })
  }

  /** Consume one Python AX6Dataset iterator, preserving results after each yield. */
  async runDays(options: RunDaysOptions, onDay: (event: DayAnalysisEvent) => void, onProgress?: ProgressHandler): Promise<void> {
    return this.exclusive(async (generation) => {
      this.requireActiveKernel(generation)
      const args = { preset: options.pipeline, participantHeightM: options.heightM, sensorHeightM: options.sensorHeightM, cohort: options.cohort, measurementCondition: options.measurementCondition ?? 'free_living', timezone: options.timezone }
      const batch = await this.call<CwaDayWindowsResult & { totalDays: number }>(`api.start_cwa_day_batch(${JSON.stringify(options.recordingId)}, json.loads(${JSON.stringify(JSON.stringify(args))}), ${JSON.stringify(options.dayIndices)})`, generation)
      let completed = 0
      try {
        while (true) {
          this.checkGeneration(generation)
          const day = batch.windows[completed]
          if (day) onProgress?.({ stage: 'analyzing', message: `Analyzing ${day.label} (${completed + 1}/${batch.totalDays})…` })
          const next = await this.call<{ done: boolean; packet?: DayAnalysisEvent }>('api.next_cwa_day()', generation)
          this.checkGeneration(generation)
          if (next.packet) {
            onDay(next.packet)
            completed++
            if (next.packet.fatal) {
              this.cancel()
              throw new Error(`${next.packet.error ?? 'The worker could not complete this day.'} Select the files again to retry.`)
            }
          }
          if (next.done) break
        }
      } finally {
        if (generation === this.generation) await this.call('api.cancel_cwa_day_batch()', generation)
      }
      onProgress?.({ stage: 'ready', message: `Finished processing ${completed} days.` })
    })
  }

  async runPipeline(options: RunPipelineOptions, onProgress?: ProgressHandler): Promise<AnalysisResult> {
    return this.exclusive(async (generation) => {
      this.requireActiveKernel(generation)
      onProgress?.({ stage: 'analyzing', message: 'Running the pipeline. The first run compiles Numba functions…' })
      const args = { preset: options.pipeline, participantHeightM: options.heightM, sensorHeightM: options.sensorHeightM, cohort: options.cohort, measurementCondition: options.measurementCondition ?? 'laboratory', cwaFile: options.cwaFile, cwaDay: options.cwaDay }
      return this.call<AnalysisResult>(`api.analyze_recording(${JSON.stringify(options.recordingId)}, json.loads(${JSON.stringify(JSON.stringify(args))}))`, generation)
    })
  }

  private requireActiveKernel(generation: number): void {
    this.checkGeneration(generation)
    if (!this.hasActiveKernel) throw new Error('The Python worker is no longer available. Select the files again to retry.')
  }

  private checkGeneration(generation: number): void {
    if (generation !== this.generation) throw new Error('The operation was cancelled. Select the files again to retry.')
  }

  private async exclusive<T>(action: (generation: number) => Promise<T>): Promise<T> {
    if (this.busy) throw new Error('Wait for the current operation or cancel it first.')
    this.busy = true
    const generation = this.generation
    try { return await action(generation) } finally { if (generation === this.generation) this.busy = false }
  }

  /** Cancelling destroys the worker and its loaded recordings; initialize to retry. */
  cancel(): void {
    this.generation++
    this.busy = false
    this.abort?.abort()
    this.abort = undefined
    for (const reject of this.pending) reject(new Error('The operation was cancelled. Select the files again to retry.'))
    this.pending.clear()
    if (this.kernel) this.kernel.dispose()
    this.kernel = undefined
    this.initialization = undefined
  }
}

const runtime = new MobgapRuntime()
export function getRuntime(): MobgapRuntime { return runtime }
