/** Browser File handles enter this worker through Xeus's callGlobalReceiver RPC. */
(() => {
  const { FS } = globalThis.Module;
  const backend = globalThis.WORKERFS;
  const CACHE_BYTES = 1024 * 1024;
  let mountedPath;
  let cache;
  let stats;
  const read = backend.stream_ops.read;
  function counters() {
    return { bytesRead: 0, readCalls: 0, maxReadBytes: 0, logicalBytesRead: 0, logicalReadCalls: 0, maxLogicalReadBytes: 0, physicalBytesRead: 0, physicalReadCalls: 0, maxPhysicalReadBytes: 0 };
  }
  function accountPhysical(file, bytes) {
    for (const counter of [stats, file]) {
      counter.physicalBytesRead += bytes;
      counter.physicalReadCalls += 1;
      counter.maxPhysicalReadBytes = Math.max(counter.maxPhysicalReadBytes, bytes);
    }
  }
  backend.stream_ops.read = function (stream, buffer, offset, length, position) {
    const node = stream.node;
    const file = stats.files[node.name];
    let count;
    if (!node.name.toLowerCase().endsWith('.cwa')) {
      count = read(stream, buffer, offset, length, position);
      if (count) accountPhysical(file, count);
    } else {
      count = Math.max(0, Math.min(length, node.size - position));
      let copied = 0;
      while (copied < count) {
        const absolute = position + copied;
        const start = Math.floor(absolute / CACHE_BYTES) * CACHE_BYTES;
        if (!cache || cache.file !== node.contents || cache.start !== start) {
          const bytes = new Uint8Array(backend.reader.readAsArrayBuffer(node.contents.slice(start, start + CACHE_BYTES)));
          accountPhysical(file, bytes.byteLength);
          cache = { file: node.contents, start, bytes };
        }
        const within = absolute - start;
        const take = Math.min(count - copied, cache.bytes.length - within);
        buffer.set(cache.bytes.subarray(within, within + take), offset + copied);
        copied += take;
      }
    }
    for (const counter of [stats, file]) {
      counter.logicalBytesRead += count;
      counter.logicalReadCalls += 1;
      counter.maxLogicalReadBytes = Math.max(counter.maxLogicalReadBytes, count);
      // Keep the original probe names as aliases for bytes returned to Python.
      counter.bytesRead = counter.logicalBytesRead;
      counter.readCalls = counter.logicalReadCalls;
      counter.maxReadBytes = counter.maxLogicalReadBytes;
    }
    stats.cacheResidentBytes = cache?.bytes.byteLength ?? 0;
    return count;
  };
  globalThis.mobgapWorkerFiles = {
    mount(files, path) {
      if (mountedPath) {
        FS.unmount(mountedPath);
        FS.rmdir(mountedPath);
        mountedPath = undefined;
      }
      cache = undefined;
      FS.mkdirTree(path);
      stats = { backend: 'WORKERFS', mountedBytes: 0, ...counters(), cacheCapacityBytes: CACHE_BYTES, cacheResidentBytes: 0, files: Object.create(null) };
      const blobs = files.map((file) => {
        const name = file.name.replaceAll(/[\\/]/g, '_');
        stats.mountedBytes += file.size;
        stats.files[name] = { size: file.size, type: file instanceof File ? 'File' : 'Blob', ...counters() };
        return { name, data: file };
      });
      // WORKERFS retains Blob handles. FileReaderSync reads slices only when Python asks.
      FS.mount(backend, { blobs }, path);
      mountedPath = path;
    },
    stats() { return stats; },
  };
})();
