/** Bundle the classic worker separately so Vite development needs no module worker. */
import { build } from 'vite'
import { cp, copyFile, mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const output = join(root, 'public/runtime')
const scratch = join(root, 'runtime/build/worker')
const wasmPath = join(root, 'node_modules/@emscripten-forge/untarjs/lib/unpack.wasm')
const fingerprint = (bytes) => createHash('sha256').update(bytes).digest('hex').slice(0, 16)
const unpacker = `unpack-${fingerprint(await readFile(wasmPath))}.wasm`
await build({
  configFile: false, root, publicDir: false,
  plugins: [{
    name: 'untarjs-wasm-url', enforce: 'pre',
    resolveId(id) { if (id === './unpack.wasm' || id.endsWith('/unpack.wasm')) return '\0untarjs-wasm-url' },
    load(id) { if (id === '\0untarjs-wasm-url') return `export default ${JSON.stringify(unpacker)}` },
  }],
  build: {
    outDir: scratch, emptyOutDir: true, sourcemap: false,
    rolldownOptions: { input: join(root, 'src/lib/xeus.worker.ts'), output: { format: 'iife', entryFileNames: 'worker.js' } },
  },
})
const bytes = await readFile(join(scratch, 'worker.js'))
const worker = `worker-${fingerprint(bytes)}.js`
await mkdir(output, { recursive: true })
for (const name of await readdir(output)) {
  if (/^(worker-[a-f0-9]+\.js|unpack-[a-f0-9]+\.wasm)$/.test(name)) await rm(join(output, name))
}
await rename(join(scratch, 'worker.js'), join(output, worker))
await copyFile(wasmPath, join(output, unpacker))
await cp(join(root, 'runtime/licenses/direct-worker'), join(output, 'licenses/direct-worker'), { recursive: true })
await writeFile(join(output, 'worker-manifest.json'), JSON.stringify({ worker, unpacker }) + '\n')
console.log(`Direct worker: ${bytes.length} bytes; unpacker: ${(await readFile(wasmPath)).length} bytes`)
