# Browser prototype scope and acceptance

Build a standalone page in `demo-website` using React, TanStack Router and shadcn/ui. Users select Mobilise-D MATLAB or AX6 CWA recordings, supply participant metadata, run full mobgap presets locally in a browser, and inspect or export actual results. The runtime uses genuine Xeus Python/Numba and compiled PyWavelets, Python xxhash and cwa_reader_rs extensions. The reader artifact comes from cwa_reader_rs PR #7 with pinned build/source provenance.

Selected browser File handles reach the worker through structured clone and mount on the official Emscripten 4.0.9 WORKERFS backend. Do not copy whole selected files into ArrayBuffer/base64/MEMFS or persistent browser storage. CWA reads use a bounded shared 1 MiB read-ahead cache to amortize tiny packet reads. Logical reader requests and physical browser reads have separate counters. MATLAB retains its existing eager decoding behavior.

For multiday CWA files, offer “Split by day” and explicitly request the sensor clock synchronization timezone. Use the public AX6Dataset with split_by_local_days, iterate one dataset in Python and apply a pipeline to each day item. Include partial first/last days and local daylight-saving boundaries. Process days sequentially, return separate per-day tables and CSV exports, and preserve completed results after an ordinary day error or cancellation. Fatal memory/worker errors stop and reset the runtime. Do not substitute hourly chunks for whole-day pipeline semantics. An optional bounded manual window is a separate mode.

Preserve mobgap's existing public interfaces and numerical behavior. Daily WASM runs require sliding windows constructed at their final hop without an oversized intermediate view. AX6 window caching must use tpcp 3.2.0 or newer, whose hybrid_cache releases the previous entry before loading a replacement by default. Remove the prototype-local cache workaround and retain the release-before-load, repeat-window and optional disk-cache regression checks.

The generic Mobilise-D pipeline and both full presets expose `retain_intermediate_results=True`. False removes the stored datapoint, executed detector, GS iterator, stride selection, WBA and aggregation instances after execution, including failures, and clears retained instances from a previous run before starting. All successful output tables, including raw tables, remain unchanged. The browser uses False. This does not clear dataset caches, caller-owned references or exception tracebacks, nor promise a lower processing peak. The universal cohort-dispatch wrapper is outside this parameter change. Verify table parity for both presets with/without aggregation, weak-reference input lifetime, reuse/toggling and failure cleanup.

## Acceptance checks

- Original and bundled MATLAB examples, participant metadata, malformed files and missing metadata.
- Both full presets produce all eight native-equivalent result tables in the actual Xeus browser worker.
- Real multiday CWA input, raw/resampled reader parity and full daily pipeline parity; explicitly distinguish benchmark metadata from actual participant measurements.
- One Python AX6Dataset loop, individual/all-day selection, per-day progress/results/exports, failure handling and cancellation/retry.
- WORKERFS seek/EOF/read-only correctness on a 512 MiB file; no whole-file File.arrayBuffer calls; cache boundary/remount tests.
- Native/browser timings and memory evidence with clear measurement limits. Do not commit private recordings or participant result files.
- Desktop/narrow layouts, subpath deployment, worker failure/retry, TypeScript/Vite build, Python tests, core regression snapshots and React Doctor.
- Reproducible local runtime assets, pinned artifacts/licenses and static deployment instructions; generated environments remain outside Git.
- Reviewable commits, resolved asynchronous reviews, final feature_ready panel and updated draft PR.

Parent owns browser integration, verification, commits, review and PR delivery. Agents own their assigned source slices. Broad browser/mobile certification and MATLAB v7.3/HDF5 support remain outside this prototype.

COMPACTION CONTINUITY: Re-read implement-code-change and the task-defining artifacts before continuing after compaction or session restoration.
