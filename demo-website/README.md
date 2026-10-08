# mobgap browser demo

A standalone React page using TanStack Router and shadcn/ui. Select a Mobilise-D MATLAB or AX6 CWA recording, choose a full mobgap pipeline, and inspect or export the computed results. Analysis runs locally in a Xeus-Python browser worker with genuine Numba, PyWavelets and xxhash extensions.

## Development

From this directory, with Node.js 24 and Python 3.13 for the pinned runtime build tools:

```sh
npm ci
npm run runtime:prepare
npm run dev
```

The first preparation downloads build tools and the pinned Python environment. On Linux x86-64 the script installs a checksum-verified micromamba into its local build directory. On another platform supply micromamba 2.9.0 explicitly:

```sh
npm run runtime:prepare -- --micromamba /path/to/micromamba
```

Build and serve the deployable static application:

```sh
npm run build
npm run preview
```

Deploy the complete `dist/` directory. Generated runtime assets and build environments are ignored by Git; the custom binary packages, build provenance, dependency locks and setup script live under `runtime/` and `scripts/`. See [runtime documentation](runtime/README.md) for details.

After changing the Python adapter or mobgap source, refresh its bundle without rebuilding the full environment:

```sh
npm run runtime:prepare -- --bundle-only
```

Then rebuild the frontend for a production preview. The bundled source includes a compatible sliding-window fix for 32-bit NumPy: it constructs only requested window hops, avoiding an oversized intermediate view on day-long recordings.

## Supported input

The prototype targets the Mobilise-D MATLAB structure used by this repository's example data. It is not a general MATLAB variable explorer. Try the healthy or impaired example, or choose `example_data/data/lab/HA/001/data.mat` from the repository. You may select its `infoForAlgo.mat` companion at the same time. Select a recording from the file and supply participant metadata when it is missing. Cohort is an explicit choice, not inferred from a filename.

For multiday CWA recordings, select **Split by day**, enter the timezone of the computer that synchronized the sensor, and load the calendar-day list. Choose all days or one day, supply cohort and measured participant/sensor heights, then start analysis. Days run sequentially, including partial first and last days. Select a completed day to inspect or export its tables. A failed day is reported separately; cancellation stops the worker and preserves completed daily results.

Calendar boundaries use the selected local timezone. The sensor clock uses the fixed offset at its last synchronization, including when a recording crosses a daylight-saving transition. The demo reuses mobgap's `AX6Dataset` and `split_by_local_days` for this behavior. CWA pipeline analysis assumes the device was worn on the lower back in the expected sensor orientation. Both full presets require gyroscope channels; acceleration-only AX3 recordings can be inspected but cannot run these presets.

## Runtime architecture

The React page controls a same-origin Xeus kernel through a hidden JupyterLite frame. The visible application has no notebook interface. This reuses the kernel loader from the verified WASM experiment. It is a prototype integration, not a new standalone Xeus JavaScript SDK.

Selected browser `File` handles are structured-cloned to the kernel worker and mounted read-only through Emscripten WORKERFS. Reads use `File.slice()` and `FileReaderSync`, so mounting does not copy the entire input into WASM memory or persistent browser storage. MATLAB input is parsed with mobgap's existing loader. They are not uploaded to an analysis server. The site serves static application and runtime assets. Results return to the page for display and download. The worker stays alive between analyses; cancelling resets the worker and requires loading the file again. CWA batches use one `AX6Dataset` with `split_by_local_days`; a Python generator iterates the dataset and runs a fresh pipeline on each item, returning each day's result to React. WORKERFS still copies each requested read into WASM memory. MATLAB decoding eagerly materializes the file contents and retains its sensor trials; removing the input copy does not remove those allocations.

## Scope

This prototype targets desktop browsers. Daily CWA analysis can require substantial memory even though the input file itself is not copied into WASM memory. Runtime download, imports and first compilation take longer than a warm analysis. See [validation results](VALIDATION.md) for the real multiday recordings exercised and measured limits. Mobile memory limits and broad browser support need separate validation. MATLAB v7.3/HDF5 input is outside the existing SciPy-based loader's support.

A production deployment should serve the generated frontend and runtime together over HTTPS, retain the runtime directory structure, and use an SPA fallback only where it does not replace runtime assets. No Python server is needed after the build.
