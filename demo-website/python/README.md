# MATLAB and CWA files in the browser demo

`mobgap_demo_api.py` passes uploaded worker-filesystem paths to file-backed tpcp datasets. MATLAB uses `UploadedMatlabDataset`, a small `GenericMobilisedDataset` subclass that inherits its signal loading and adds upload metadata paths/overrides. CWA inspection and analysis use `AX6Dataset`. It keeps the loader's acceleration conversion and sensor frame; the full selected pipeline handles body-frame conversion. Cohort is explicitly selected. Unknown uploads have no invented participant heights. A companion `infoForAlgo.mat`, or an embedded `infoForAlgo` variable, supplies measured heights in centimetres, converted once to metres.

External companion metadata is not assigned when another uploaded file is unreadable: that file could belong to the participant described by the companion. The valid recordings remain available with a warning to enter heights manually. Embedded metadata remains paired with its own file.

The runtime calls `inspect_files(paths)` and then `analyze_recording(id, options)`. The latter requires `preset`, `cohort`, `sensorHeightM` and `participantHeightM`, with actual companion heights available as defaults. `measurementCondition` is laboratory unless explicitly selected. The TypeScript runtime maps UI `pipeline` to `preset` and UI `heightM` to `participantHeightM`. Inspection metadata uses `heightM` and `sensorHeightM`.

Inspection lists every available trial, prioritizing Test11 for sample selection, and returns file-specific errors. It reports MATLAB v7.3/HDF5 as unsupported by the existing SciPy loader and points to the [MATLAB -v7 save option](https://www.mathworks.com/help/matlab/ref/save.html). Analysis returns complete result tables with scalar JSON cells, counts, elapsed processing time, warnings and actual installed package versions. Python rule objects are excluded from display tables.

`prepare_samples.py` losslessly extracts TimeMeasure1/Test11/Trial1 from the repository's public HA/001 and MS/001 MAT examples. It retains the LowerBack data and original acquisition metadata, embeds participant metadata, and writes `public/samples/healthy.mat`, `impaired.mat` and `manifest.json`. The manifest records source paths, hashes and known metadata. Samples are 674 KB and 1.2 MB. They contain 13,759 and 22,728 samples at 100 Hz. The script does not synthesize sensor values.

Run focused checks with a compatible native environment:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p no:cacheprovider -c /dev/null demo-website/python/test_mobgap_demo_api.py -q
ruff check demo-website/python
```

Tests cover original multi-trial MAT files, full healthy results from an independently run native MAT pipeline, required metadata, user sensor-height overrides, unsupported/corrupt uploads, ambiguous companion metadata and exact DataFrame equality between derived samples and originals. Use `export_native_results.py --output /tmp/mobgap-demo-native` to regenerate full JSON evidence from the original MAT files. Parent-owned browser checks compare both full presets against that evidence.

`file_access_probe.py` provides native and browser-worker read checks without loading a large test file into a byte array:

```sh
python demo-website/python/file_access_probe.py /tmp/mobgap-workerfs-probe.bin --create-sparse-mib 512
python demo-website/python/file_access_probe.py example_data/data/lab/HA/001/data.mat --mat
```

The first command creates a sparse 512 MiB file and reads only 80 bytes at three positions. A browser can upload the file and call `seek_read_probe(mounted_path)` to compare offsets, bytes and the probe hash with native results. In the worker, `mobgapWorkerFiles.stats()` reports actual FileReaderSync slice reads, independently of the Python byte count. The second command counts SciPy reads and measures decoded arrays and retained LowerBack DataFrames.

WORKERFS removes the full upload copy into the worker's filesystem. The MATLAB dataset still uses an eager decoder to discover its trial index. Its standard one-file hybrid cache retains the decoded trials; the adapter registry stores only dataset paths/index selections and descriptions, not sensor DataFrames. Embedded participant metadata also uses a separate MATLAB decode. Neither WORKERFS nor the seek proof bounds SciPy or pipeline memory.

For the original HA/001 and MS/001 files, native measurements were:

| File | File bytes | Decoded MATLAB array bytes | Retained DataFrame bytes, all three trials |
| --- | ---: | ---: | ---: |
| HA/001/data.mat | 1,026,637 | 1,525,816 | 900,480 |
| MS/001/data.mat | 1,602,284 | 2,417,216 | 1,416,408 |

These counts are not peak memory measurements: Python object overhead and pipeline intermediates are additional. `whosmat` read 131,235 bytes for either file; full `loadmat` read the whole file plus 27 bytes of repeated header reads, using reads of at most 131,072 bytes.

For CWA, `inspect_files` constructs an `AX6Dataset` and reads its header and a window of at most one second to inspect channels. The provisional UTC interpretation is used only for inspection; analysis still requires the user’s synchronization timezone. The recording's `samples` is `null` until a selected day/window is decoded. Heights and cohort are supplied manually. Both full presets require actual gyroscope channels; acceleration-only AX3 files can be inspected but cannot run those presets.

`cwa_day_windows(id, timezone)` enumerates local calendar days using the public `AX6Dataset` and `split_by_local_days`. The first and last days may be partial; daylight-saving calendar days use their actual elapsed duration. For a batch, `start_cwa_day_batch(id, options, day_indices)` creates one lazy `AX6Dataset` and a Python generator that loops over its day datapoints, applying a fresh pipeline to each. Options include `timezone` alongside the preset and participant measurements. The browser advances `next_cwa_day()` and receives `{done, packet?: {day, result?, error?, fatal?}}`; process any packet before checking `done`, since a fatal memory error includes its final packet. `cancel_cwa_day_batch()` closes the generator between days. Successful result JSON survives later day failures. No hour subdivision changes the day pipeline's semantics.

Both presets run with `retain_intermediate_results=False`, releasing executed internal objects and the pipeline’s dataset reference before table serialization. All eight exported tables remain available. This does not clear the loader/dataset cache or reduce temporary allocations within an algorithm.

The generator releases pipeline and temporary frame references, clears exception-chain tracebacks, and collects garbage before yielding. Failed pipelines therefore do not escape into IPython's error history holding large arrays. Memory errors terminate the batch and request a worker reset; ordinary day errors allow later days to continue. The original single-day `analyze_recording(id, {…, cwaDay: {index, timezone}})` remains available. An optional `cwaWindow: {startSeconds, durationSeconds, timezone}` selects a manual window of at most 3,600 seconds; that limit does not apply to calendar days. CWA analysis defaults to `free_living`.

The supplied timezone is the IANA timezone of the computer that last synchronized the sensor. `AX6Dataset` derives the fixed sensor clock offset from the last configuration timestamp, then resamples at the nominal header rate, converts acceleration from g to m/s², and renames gyro axes to mobgap's sensor columns. This preserves its existing clock-synchronization assumption. CWA analysis assumes the sensor was on the lower back in the expected frame; the filename does not prove placement, orientation or clinical cohort.

The current Rust reader scans every 512-byte data packet for `read_metadata` and every seconds-cut plan, even for a short late-file window. It decodes only the selected sample range plus interpolation context, but allocates a timing vector of about 24 bytes per valid packet when resolving a seconds cut. `AX6Dataset` also reads and caches a sampling report. WORKERFS avoids a complete filesystem upload copy; it does not eliminate these scans, decoded day frames or pipeline buffers. The dataset uses tpcp 3.2's default early-eviction hybrid_cache; its one-entry memory cache releases its previous window before a cache miss begins decoding the next. Repeated access to the same window reuses the frame, and optional joblib disk caching remains available. Explicit caller references can still retain an earlier frame; the batch keeps result JSON rather than raw day frames.

`cwa_window_probe.py` is executable natively and its `probe_file(path, windows)` function runs in the browser worker. It hashes every microsecond timestamp and every float32 sensor value in little-endian order, reports DataFrame bytes, and exports no sensor rows. `--verify-context` additionally requires exact equality between each short read and the matching slice of a slightly larger window, for both raw and cubic-resampled output. Its UTC offset of zero is a numerical parity convention, not an inferred synchronization timezone.

`cwa_pipeline_probe.py` advances the same Python dataset-loop generator for actual selected calendar days. Supply the real clock timezone and measured participant metadata; no private recordings or probe outputs belong in the repository:

```sh
python demo-website/python/cwa_window_probe.py /local/recording.cwa --start 0 --start 86400 --duration 60 --verify-context --output /tmp/cwa-windows.json
python demo-website/python/cwa_pipeline_probe.py /local/recording.cwa --timezone Europe/Berlin --preset healthy --participant-height-m 1.7 --sensor-height-m 0.9 --cohort HA --output /tmp/cwa-days.json
```

The example heights above illustrate CLI arguments and are not defaults for uploaded recordings. Linux CLI runs report process peak RSS; browser memory measurements come from runtime instrumentation. Native probes used PR 7 commit `f731c97801c9b48208582b26aab7b9021a1e891f`, reader 0.4.0, NumPy 2.4.6 and pandas 3.0.6.
