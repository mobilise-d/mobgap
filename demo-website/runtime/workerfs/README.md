# WORKERFS backend for the Xeus demo

`workerfs.js` adapts the official Emscripten 4.0.9 backend, preserving its file,
seek, directory and read-only operations. `libworkerfs.upstream.js` is the exact
original; `provenance.json` records its URL and SHA-256. The original is an
Emscripten build library and cannot be executed directly because it contains
build macros. The adapter resolves those macros against the kernel's exported
`Module.FS` and `Module.ERRNO_CODES`, then exposes `globalThis.WORKERFS`.

Evaluate the adapter inside the initialized Xeus worker. Mount selected browser
Files with `Module.FS.mount(WORKERFS, {files}, mountPath)` after creating the
mount directory. Pass File objects through the existing Xeus
`callGlobalReceiver` bridge using structured clone, without putting them in a
transfer list. Use distinct mount paths for files with duplicate names.

Unmount and release global references when replacing the input or cancelling.
This removes the complete base64/upload-buffer/MEMFS copy at ingress. Reads
still copy requested File slices into the Wasm buffer, and SciPy's MATLAB
loader still allocates decoded arrays; this is not a zero-copy pipeline.

The backend is copyright 2015 The Emscripten Authors, SPDX MIT. Its header is
retained, and `LICENSE.emscripten` contains the complete upstream license text.
Native mobgap behavior and APIs are unchanged. Browser integration and actual
MATLAB/pipeline checks belong to the runtime integration owner.
