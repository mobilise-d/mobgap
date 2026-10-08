import { releaseProxy, wrap } from 'comlink'
import type { Remote } from 'comlink'
import type { DirectWorkerApi } from './xeus.worker'

export interface KernelMessage {
  header: { msg_type: string }
  parent_header?: { msg_id?: string }
  channel?: string
  content: { name?: string; text?: string; ename?: string; evalue?: string; traceback?: string[]; execution_state?: string; status?: string }
}
export interface ExecuteFuture { onIOPub: ((message: KernelMessage) => void) | null; done: Promise<void>; dispose(): void }

/** The application needs only execute requests, their reply, and the final idle status. */
export class XeusKernel {
  readonly remote: Remote<DirectWorkerApi>
  isDisposed = false
  private session = crypto.randomUUID()
  private requests = new Map<string, { future: ExecuteFuture; resolve(): void; reject(error: unknown): void; replied: boolean; idle: boolean; sawError: boolean }>()

  readonly worker: Worker

  constructor(worker: Worker, onLog?: (text: string) => void) {
    this.worker = worker
    this.remote = wrap<DirectWorkerApi>(worker)
    worker.addEventListener('message', (event: MessageEvent<KernelMessage & { _stream?: { text: string } }>) => {
      const message = event.data
      if (message._stream) { onLog?.(message._stream.text); return }
      if (!message.header) return // Comlink responses use a separate protocol.
      const request = this.requests.get(message.parent_header?.msg_id ?? '')
      if (!request) return
      if (message.channel === 'iopub') {
        request.future.onIOPub?.(message)
        if (message.header.msg_type === 'error') request.sawError = true
        if (message.header.msg_type === 'status' && message.content.execution_state === 'idle') request.idle = true
      }
      if (message.header.msg_type === 'execute_reply') {
        request.replied = true
        if (message.content.status === 'error' && !request.sawError) request.future.onIOPub?.({ ...message, header: { msg_type: 'error' } })
      }
      if (request.replied && request.idle) request.resolve()
    })
  }

  requestExecute(options: { code: string; store_history: false }): ExecuteFuture {
    if (this.isDisposed) throw new Error('The Python worker is no longer available.')
    const id = crypto.randomUUID()
    let resolve!: () => void
    let reject!: (error: unknown) => void
    const done = new Promise<void>((yes, no) => { resolve = yes; reject = no })
    const future: ExecuteFuture = {
      onIOPub: null,
      done,
      dispose: () => { this.requests.delete(id) },
    }
    this.requests.set(id, { future, resolve, reject, replied: false, idle: false, sawError: false })
    const msg = {
      header: { msg_id: id, username: 'mobgap', session: this.session, date: new Date().toISOString(), msg_type: 'execute_request', version: '5.3' },
      parent_header: {}, metadata: {}, channel: 'shell', buffers: [],
      content: { ...options, silent: false, user_expressions: {}, allow_stdin: false, stop_on_error: true },
    }
    void this.remote.processMessage({ msg, parent: msg }).catch(reject)
    return future
  }

  dispose(): void {
    if (this.isDisposed) return
    this.isDisposed = true
    for (const request of this.requests.values()) request.reject(new Error('The Python worker was terminated. Select the files again to retry.'))
    this.requests.clear()
    this.remote[releaseProxy]()
    this.worker.terminate()
  }
}
