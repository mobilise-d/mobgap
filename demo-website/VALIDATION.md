# Prototype validation

The integration was exercised in the collaborative Chromium browser against the production Vite build served as static files. The frontend used its actual Xeus kernel and the compiled local PyWavelets and Python xxhash packages. No numerical adapters or mocked result tables were used.

## Numerical results

| Input and preset | Samples | Gait sequences | Initial contacts | Walking bouts | Strides |
| --- | ---: | ---: | ---: | ---: | ---: |
| Healthy MATLAB example, healthy preset | 13,759 | 6 | 60 | 6 | 47 |
| Original MS `data.mat` with `infoForAlgo.mat`, impaired preset | 22,728 | 5 | 98 | 5 | 84 |

All eight result tables from each browser run matched a native run of the same Python adapter against the original repository MATLAB recording, with relative tolerance 1e-8 and absolute tolerance 1e-9. Table columns, row structure and nonnumeric values also matched. The recordings were TimeMeasure1 / Test11 / Trial1. Heights and cohorts were the actual example metadata, not defaults inferred from file names.

The healthy example's walking-bout CSV exported one header and six computed data rows. CSV uses the exported JSON values without the table display rounding; Python serializes floats with pandas double_precision=15, so exact float64 round trips are not promised.

These MATLAB runs cover the selected short examples and both full presets. Real multiday CWA measurements are documented below; broad browser compatibility remains untested. Cold runtime preparation and JIT compilation are excluded from earlier warm algorithm benchmarks; the application reports the duration of the particular pipeline call it just ran.

## Interface and build checks

- Original multi-recording file selection populated measured participant and sensor heights from its companion metadata, while still requiring an explicit cohort.
- A malformed MAT file displayed its error and left analysis disabled.
- Cancellation during initialization and during a pipeline call cleared the loaded recording and removed the runtime frame. Loading again and running the healthy preset succeeded.
- Desktop 1280px and narrow 390px layouts were inspected; the narrow page uses one column without horizontal page overflow.
- The Python adapter's 22 focused tests, Ruff checks and TypeScript/Vite production build passed.
- React Doctor's changed-source scan reported no security or performance findings. Remaining warnings concern standard shadcn variant exports, its generated field error-list key, and the orchestration component's control-flow complexity. Generated third-party runtime bundles are outside this source scan.
- A separate clean runtime build using the locked packages and freshly bundled repository source completed successfully.

## Reproducing checks

Run the native adapter tests from the repository root with mobgap and pytest installed:

```sh
python -m pytest demo-website/python/test_mobgap_demo_api.py
```

Run the frontend type check and production build from `demo-website`:

```sh
npm run build
```

For browser verification, prepare runtime assets, build and preview as described in README.md. Try both examples, then select an original repository `data.mat` with its companion `infoForAlgo.mat`. Select Test11 / Trial1, choose the appropriate cohort and preset, run analysis and export the walking-bout table. Also check an invalid MAT file, missing metadata, cancellation and retry.

## WORKERFS verification

Both bundled MATLAB examples were rerun through WORKERFS with `File.prototype.arrayBuffer` replaced by a throwing function. There were zero calls to that function, and all eight result tables for each preset still matched native with the tolerances above. The healthy run took 3.43 s including first-use compilation; the subsequent impaired run took 0.59 s. These are individual measurements, not controlled speed comparisons.

A disk-backed synthetic 512 MiB browser File mounted with zero data reads. Seeking to its beginning, middle and end read 80 bytes total, with a largest request of 32 bytes; an EOF read returned zero bytes. The combined bytes matched the native SHA256 `b721d9e84e8bb9b14fcdcd62e4d3bdf1557a58343935bb748ed68cee66df23f5`. The test created and then deleted its own temporary OPFS fixture. This fixture creation was test setup; the application does not copy selected files into OPFS.

## Real multiday CWA verification

The integrated reader is the genuine CPython side module from cwa_reader_rs PR #7. Its compiled sources match revision `f731c97801c9b48208582b26aab7b9021a1e891f`; artifact hashes and build provenance are checked in under `runtime/reader`. Two existing local six-axis recordings were exercised: 436,792,320 bytes spanning roughly four days, and 194,510,848 bytes spanning roughly 1.78 days. Recordings and participant result files are not committed. The benchmark supplied height 1.7 m, sensor height 0.9 m, Europe/Berlin clock timezone and HA/MS cohort solely to compare implementations; these are not assertions about the recorded participants.

Raw and nominal-rate resampled 60-second windows near the beginning, middle and end of both recordings matched native hashes for every timestamp and float32 value. Metadata also matched. The files stayed mounted as browser File objects with whole-file `File.arrayBuffer` disabled. Automation staged its private fixtures into temporary local OPFS files using a streamed loopback transfer to obtain disk-backed File objects; this is test setup, not the application's file selection path.

The uncached 437 MB metadata scan was still running when stopped after 242 seconds. With the bounded 1 MiB CWA read cache, one scan took 0.85 seconds: 1,706,219 logical requests became 417 physical browser reads. It still read the complete 436,792,320-byte recording sequentially; the cache changes call granularity, not the reader's scan algorithm. Native metadata scanning took about 0.56 seconds. A later-file short read also scans timing packets before decoding its selected samples.

The five local-calendar segments of the four-day recording all completed through one Python AX6Dataset loop. All eight tables and non-timing summaries matched native at the same tolerances as MATLAB. The WASM linear-memory capacity grew from 346,554,368 to 1,678,049,280 bytes and stayed at that capacity for subsequent days. This is the WASM heap's capacity/high-water mark, not a measurement of live arrays or browser process RSS. Native peak RSS after the cache correction was 1,251,556 KiB. These measurements are not directly interchangeable.

The initial successful runs used two native-compatible corrections. The window helper previously formed a stride-one view of more than 10 GB before subsampling, exceeding 32-bit NumPy's representable array size. It now directly forms the requested read-only windows. The AX6 cache previously retained the old day while loading the new one. The initial local early-eviction workaround has since been replaced by tpcp 3.2.0’s default hybrid_cache behavior. A weak-reference test verifies release before the next Rust read, while repeated-window and disk-cache tests verify reuse.

Daily errors are returned as JSON without leaving array-bearing tracebacks in IPython history. Memory errors stop/reset the worker and preserve completed JSON results. The initial failing batch exercised this path and retained its two completed days. Broad browser/mobile certification and arbitrary recording sizes remain untested.

Subpath deployment was tested at `/mobgap/`: route, home link, sample manifest, MATLAB file and runtime assets all resolved under that prefix, and the healthy sample loaded in the actual kernel. An injected startup worker ErrorEvent terminated the worker and removed its iframe; loading again and running the healthy pipeline succeeded. This was an event-injection test, not a simulated network outage.

The more active recording's two calendar days also passed both presets. All eight tables matched native across 122,512 numeric table values for the healthy run and 109,725 for the impaired run. WASM capacity remained 1,678,049,280 bytes across both batches. The healthy days reported 21.71 and 4.94 seconds of pipeline computation; impaired days reported 17.46 and 5.14 seconds. These calls include day decoding and pipeline execution, but exclude runtime startup, prior metadata/day-list scans and React table rendering. The first healthy run of the earlier file also included first-use compilation.

Final native reruns used the same package versions, dataset loop and cache behavior. For the active recording:

| Preset | Day | Native computation | Browser computation | Browser/native |
| --- | --- | ---: | ---: | ---: |
| Healthy | First | 10.92 s | 21.71 s | 1.99× |
| Healthy | Second | 2.37 s | 4.94 s | 2.09× |
| Impaired | First | 10.47 s | 17.46 s | 1.67× |
| Impaired | Second | 2.70 s | 5.14 s | 1.90× |

Native peak RSS was 1.17 GiB for healthy and 1.08 GiB for impaired. The browser retained a 1.56 GiB WASM memory capacity after earlier batches. Timings are individual local runs, not a statistically controlled benchmark or a browser-wide performance guarantee. Imports, runtime downloads and file selection are excluded from this table. Every numeric table value in these active-recording runs was exactly equal, stronger than the stated comparison tolerance.

Final focused verification: 122 native tests covering the adapter, AX6 dataset, window utility and GsdIluz snapshots; three WORKERFS/cache tests; Ruff formatting/checks; TypeScript/Vite build. React Doctor remains 78/100 with the same five existing source warnings and no added diagnostics. The calendar-day result selector was exercised, and the first active impaired day exported 160 walking-bout rows with a date-specific CSV filename.

Cancellation during a real five-day batch preserved the first completed day, its filename/timezone caption and CSV action, and removed the worker iframe. Loading the healthy MATLAB example again and running it succeeded (6 walking bouts, 60 initial contacts, 47 strides). Whole-file `File.arrayBuffer` calls remained zero. Both private OPFS test fixtures were deleted after verification.

A worker ErrorEvent injected during calendar-day listing cleared the stale recording/day list and participant fields, removed the iframe, and requested file reselection. Selecting the public CWA example again created a new kernel and listed its day successfully. Switching from CWA to the healthy MATLAB example restored Laboratory settings and its measured heights.

After the final UI simplification, the active recording completed both healthy days again with zero native differences across all eight tables. Selecting the first day displayed its matching caption and exported 214 walking-bout rows as `mobgap-2025-12-05-healthy-walking_bouts.csv`. Cancelling a second batch after its first day retained that day and removed the worker. Whole-file reads remained zero, and the temporary OPFS fixture was deleted again. This fresh-worker rerun took 35.23 and 5.59 seconds; first-use compilation and concurrent system load illustrate why the earlier timings are individual measurements, not a fixed speed ratio.

The shared Python JSON boundary now classifies chained memory failures across inspection, planning, single-window/MAT analysis and serialization, clears exception frames, and requests worker reset. Six focused error-boundary cases extend the adapter suite to 22 tests. In actual Xeus, an injected pipeline MemoryError reset the worker while preserving earlier healthy result JSON; stale-ID operations created no replacement iframe. An idle worker ErrorEvent behaved the same way, and file reselection followed by the complete healthy preset succeeded. An empty-message Python exception was rejected without treating it as success. These are controlled fault injections, not deliberate physical memory exhaustion. A fresh locked runtime build also completed using the corrected virtualenv executable discovery on Linux; Windows execution remains untested.

A separate actual-Xeus fault during response printing, outside the JSON helper, verified the fallback exception-type handling. Xeus reports class representations such as `<class 'MemoryError'>`; the bridge normalizes this and resets the worker while preserving earlier JSON results. `ValueError('MemoryError text only')` remains an ordinary error with a live kernel, confirming classification uses the exception type rather than message contents.

## Adoption of tpcp 3.2

The project now requires tpcp >=3.2.0 and the browser wheel lock pins 3.2.0. AX6 again calls the upstream `hybrid_cache(self.memory, 1)` directly; the module-local cache, key hashing and lock were removed. Existing weak-reference, repeat-read and disk-cache tests pass against the upstream default `evict_before_load=True`. Broader dependency verification passed 417 tests with 12 skips across datasets, full pipelines, evaluation utilities, windowing, GsdIluz and the browser adapter.

The actual Xeus kernel reported tpcp 3.2.0 and processed both days of the active real CWA recording. All eight tables and non-timing summaries matched a fresh native tpcp 3.2 run (122,532 numeric comparisons, zero differences). Browser calls took 31.19/5.94 s; native calls took 13.64/3.16 s, again individual measurements including first-use effects. Browser WASM capacity was 1,627,455,488 bytes after the batch; native peak RSS was 1,225,408 KiB. Whole-file File.arrayBuffer calls remained zero. Temporary OPFS input was deleted after verification. The earlier timing tables describe the preceding tpcp 3.1/local-cache implementation.


## Updated reader, dataset UI and Auto dispatch

The reader is now PR #7 revision `84891bbb87239eb2a9d822ad7073f3936185d999`, with verified CI payload provenance and a distinct local conda build identity. Earlier read-ahead-cache measurements above are historical. The current bridge has no JavaScript read-ahead cache, and AX6 exposes no timing report or sampling-warning option.

On the 194,510,848-byte two-day recording, building the index and inspecting the first second required 5,242 bytes in 11 WORKERFS reads. The Xeus reader signature includes `batch_packets=256, overlap_packets=1`; the packaged side-module SHA256 matches `b54ca209c6953230267e31d8adacf3aadcb5559a1e89860ec7d6117177cea2c3`.

Fresh full-day runs with the updated reader matched native results across all eight tables and non-timing summaries: 122,532 numeric values for Healthy and 109,745 for Auto with cohort MS, which resolved to Impaired. Both comparisons had zero differences at `atol=1e-9, rtol=1e-8`.

| Preset | Day | Native processing seconds | Browser processing seconds |
| --- | --- | ---: | ---: |
| Healthy | First | 13.43 | 30.38 |
| Healthy | Second | 2.59 | 5.28 |
| Auto → Impaired | First | 12.79 | 22.57 |
| Auto → Impaired | Second | 4.24 | 5.44 |

These are single desktop measurements, with first-use compilation and process/cache effects. Native peak RSS was 1,227,316 KiB for Healthy and 1,133,124 KiB for Impaired. Benchmark participant heights and cohort are comparison inputs, not assertions about the recorded person.

The revised UI was checked under a subpath: staging separate MATLAB data/info files creates neither a kernel nor center content; Build dataset shows the actual index with all rows selected; deselecting all disables Run; configuration edits clear the index. Healthy and Auto→Impaired sample runs produce the expected 6/5 bouts and 60/98 contacts. At 390 px width, result/index tables scroll internally without page overflow. The native adapter has 28 passing tests, including Auto parity for HA/MS and malformed separate metadata. Core dataset/pipeline/gait-sequence/window tests pass 432 with 12 skips. WORKERFS has 3 passing tests; TypeScript/Vite, Python Ruff and formatting pass. React Doctor reports 5 existing warnings plus 3 deliberate sequential-await warnings for the exclusive worker.
