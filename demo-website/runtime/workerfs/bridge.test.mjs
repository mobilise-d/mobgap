import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const BLOCK = 1024 * 1024;
function environment() {
  const physicalReads = [];
  class BlobHandle {
    constructor(bytes) { this.bytes = bytes; this.size = bytes.length; }
    slice(start, end) { return new BlobHandle(this.bytes.subarray(start, end)); }
    arrayBuffer() { throw new Error('Whole-file reads are forbidden'); }
  }
  class FileHandle extends BlobHandle {
    constructor(bytes, name) { super(bytes); this.name = name; this.lastModifiedDate = new Date(0); }
  }
  class Reader {
    readAsArrayBuffer(blob) {
      physicalReads.push(blob.size);
      return blob.bytes.slice().buffer;
    }
  }
  let mounted;
  const FS = {
    filesystems: {},
    createNode(parent, name, mode) { return { parent, name, mode, id: 1 }; },
    isFile(mode) { return (mode & 0o170000) === 0o100000; },
    mkdirTree() {},
    rmdir() {},
    unmount() { mounted = undefined; },
    mount(backend, opts, path) { mounted = backend.mount({ opts, mountpoint: path }); },
    ErrnoError: class extends Error { constructor(errno) { super(String(errno)); this.errno = errno; } },
  };
  const context = vm.createContext({ Module: { FS, ERRNO_CODES: { ENOENT: 44, EPERM: 63, EIO: 29, EINVAL: 28 } }, File: FileHandle, FileReaderSync: Reader, Date, Uint8Array });
  for (const file of ['workerfs.js', 'bridge.js']) vm.runInContext(readFileSync(new URL(file, import.meta.url), 'utf8'), context);
  function mount(files) { context.mobgapWorkerFiles.mount(files, '/mobgap/uploads/1'); }
  function read(name, position, length) {
    const buffer = new Uint8Array(length);
    const count = context.WORKERFS.stream_ops.read({ node: mounted.contents[name] }, buffer, 0, length, position);
    return { count, bytes: buffer.subarray(0, count) };
  }
  return { context, FS, FileHandle, physicalReads, mount, read, node: (name) => mounted.contents[name], stats: () => context.mobgapWorkerFiles.stats() };
}

test('CWA reads preserve seek ranges, EOF and read-only data with one bounded cache block', () => {
  const e = environment();
  const bytes = Uint8Array.from({ length: BLOCK * 2 + 53 }, (_, i) => i % 251);
  e.mount([new e.FileHandle(bytes, 'recording.cwa')]);
  assert.equal(e.physicalReads.length, 0, 'mount must not read bytes');
  assert.deepEqual([...e.read('recording.cwa', 0, 4).bytes], [0, 1, 2, 3]);
  assert.deepEqual([...e.read('recording.cwa', 10, 5).bytes], [10, 11, 12, 13, 14]);
  assert.equal(e.physicalReads.length, 1, 'nearby reads reuse a block');
  assert.equal(e.physicalReads[0], BLOCK);
  for (const [position, length] of [[BLOCK - 2, 6], [bytes.length - 3, 10], [0, 3], [BLOCK + 6, 100], [BLOCK - 2, BLOCK + 63]]) {
    const actual = e.read('recording.cwa', position, length);
    assert.deepEqual(actual.bytes, bytes.subarray(position, position + length));
    assert.equal(actual.count, Math.min(length, bytes.length - position));
  }
  const calls = e.physicalReads.length;
  assert.equal(e.read('recording.cwa', bytes.length, 3).count, 0);
  assert.equal(e.read('recording.cwa', bytes.length + 50, 3).count, 0);
  assert.equal(e.physicalReads.length, calls, 'EOF must not fetch another block');
  const stream = { node: e.node('recording.cwa'), position: 20 };
  assert.equal(e.context.WORKERFS.stream_ops.llseek(stream, 5, 1), 25);
  assert.equal(e.context.WORKERFS.stream_ops.llseek(stream, -3, 2), bytes.length - 3);
  assert.throws(() => e.context.WORKERFS.stream_ops.llseek(stream, -1, 0));
  assert.throws(() => e.context.WORKERFS.stream_ops.write(stream, new Uint8Array([1]), 0, 1, 0));
  const offsetBuffer = new Uint8Array(15).fill(99);
  const offsetCount = e.context.WORKERFS.stream_ops.read(stream, offsetBuffer, 5, 4, 100);
  assert.equal(offsetCount, 4);
  assert.deepEqual([...offsetBuffer.slice(0, 5)], [99, 99, 99, 99, 99]);
  assert.deepEqual([...offsetBuffer.slice(5, 9)], [100, 101, 102, 103]);
  const stats = e.stats();
  assert.equal(stats.physicalBytesRead, e.physicalReads.reduce((sum, size) => sum + size, 0));
  assert.equal(stats.physicalReadCalls, e.physicalReads.length);
  assert.equal(stats.logicalBytesRead, BLOCK + 180);
  assert.ok(stats.cacheResidentBytes <= BLOCK);
  assert.ok(e.physicalReads.every((size) => size <= BLOCK));
});

test('remount and alternating CWA files never reuse another file’s bytes', () => {
  const e = environment();
  e.mount([new e.FileHandle(new Uint8Array(BLOCK + 5).fill(7), 'one.cwa')]);
  assert.deepEqual([...e.read('one.cwa', 0, 3).bytes], [7, 7, 7]);
  e.mount([new e.FileHandle(new Uint8Array(BLOCK + 5).fill(8), 'one.cwa'), new e.FileHandle(new Uint8Array(BLOCK + 5).fill(9), 'two.cwa')]);
  assert.equal(e.stats().physicalReadCalls, 0);
  for (const [name, value] of [['one.cwa', 8], ['two.cwa', 9], ['one.cwa', 8]]) assert.deepEqual([...e.read(name, 1, 2).bytes], [value, value]);
  assert.equal(e.stats().physicalReadCalls, 3);
  assert.ok(e.stats().cacheResidentBytes <= BLOCK);
});

test('other formats retain exact requested-range reads without read-ahead', () => {
  const e = environment();
  e.mount([new e.FileHandle(new Uint8Array(BLOCK * 2).fill(42), 'large.bin')]);
  assert.deepEqual([...e.read('large.bin', BLOCK + 8, 4).bytes], [42, 42, 42, 42]);
  assert.deepEqual(e.physicalReads, [4]);
  assert.equal(e.stats().cacheResidentBytes, 0);
});
