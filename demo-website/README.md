# mobgap WASM

A standalone React application using TanStack Router and shadcn/ui. Stage one Mobilise-D MATLAB or AX6 CWA recording, supply participant information, build its dataset index, and run a full mobgap pipeline on selected rows. Analysis runs locally in a Xeus-Python browser worker with genuine Numba, PyWavelets and xxhash extensions.

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

Then rebuild the frontend for a production preview. The bundled source includes a compatible sliding-window fix for 32-bit NumPy: it constructs only requested window hops, avoiding an oversized intermediate view on day-long recordings. AX6 uses tpcp 3.2.0’s built-in early cache eviction; no local cache workaround is needed. Both browser presets set `retain_intermediate_results=False` so executed pipeline internals are released before exporting the result tables.

## Supported input

The prototype targets the Mobilise-D MATLAB structure used by this repository's example data. It is not a general MATLAB variable explorer. Try the healthy or impaired example, or choose `example_data/data/lab/HA/001/data.mat` from the repository. Choose the recording file first, then upload its separate `infoForAlgo.mat` companion in Participant information or choose Enter manually and supply both measured heights. Cohort is an explicit choice, not inferred from a filename. The bundled examples provide separate recording and participant files. Nothing is inspected until you choose Build dataset; the Dataset page then shows the actual trial index with all rows selected. Choose Healthy, Impaired or Auto, then Run selected. Auto is the initial choice and uses the Universal pipeline to select Healthy for HA/COPD/CHF or Impaired for PD/MS/PFF from the supplied cohort. Bundled examples retain their explicit preset hints.

For CWA recordings, supply cohort and measured participant/sensor heights, then enter the timezone of the computer that synchronized the sensor. Choose Build dataset to read metadata and construct the index. Recordings longer than 24 hours default to **Split by calendar day**; recordings of 24 hours or less default to **Single file**. The Upload page then lets you override this choice and rebuild; an explicit choice is preserved. Calendar-day rows include partial first and last days. A single-file row processes the entire recording. There is no bounded-window option.

All rows start selected; choose the rows to run, then Run selected. Days run sequentially through one Python dataset loop. Progress shows the complete selected batch and changes to Results only after every row completes or fails. Select a completed row there to inspect or export its tables. A failed row is reported separately; cancellation stops the worker and preserves completed results. Changing dataset configuration clears the old index and requires rebuilding.

Calendar boundaries use the selected local timezone. The sensor clock uses the fixed offset at its last synchronization, including when a recording crosses a daylight-saving transition. The demo reuses mobgap's `AX6Dataset` and `split_by_local_days` for this behavior. CWA pipeline analysis assumes the device was worn on the lower back in the expected sensor orientation. Both full presets require gyroscope channels; acceleration-only AX3 recordings can be inspected but cannot run these presets.

## Runtime architecture

The React application controls a same-origin Xeus kernel through a hidden JupyterLite frame. The visible application has no notebook interface. This reuses the kernel loader from the verified WASM experiment. It is a prototype integration, not a new standalone Xeus JavaScript SDK.

Selected browser `File` handles are structured-cloned to the kernel worker and mounted read-only through Emscripten WORKERFS. Reads use `File.slice()` and `FileReaderSync`, so mounting does not copy the entire input into WASM memory or persistent browser storage. Both formats are loaded through file-backed tpcp datasets: `GenericMobilisedDataset` with upload metadata overrides for MATLAB, and `AX6Dataset` for CWA. They are not uploaded to an analysis server. The site serves static application and runtime assets. Results return to the page for display and download. The worker stays alive between analyses; cancelling resets the worker and requires rebuilding the dataset from the retained selected file handles. CWA batches use one `AX6Dataset` with `split_by_local_days`; a Python generator iterates the dataset and runs a fresh pipeline on each item, returning each day's result to React. WORKERFS still copies each requested read into WASM memory. The MATLAB dataset’s one-file cache retains eagerly decoded sensor trials; removing the input copy does not remove those allocations.

## Scope

This prototype targets desktop browsers. Daily CWA analysis can require substantial memory even though the input file itself is not copied into WASM memory. Runtime download, imports and first compilation take longer than a warm analysis. See [validation results](VALIDATION.md) for the real multiday recordings exercised and measured limits. Mobile memory limits and broad browser support need separate validation. MATLAB v7.3/HDF5 input is outside the existing SciPy-based loader's support.

A production deployment should serve the generated frontend and runtime together over HTTPS, retain the runtime directory structure, and use an SPA fallback only where it does not replace runtime assets. No Python server is needed after the build.

The upload workflow accepts exactly one recording file, with an optional separate infoForAlgo file for MATLAB. Multiple dataset rows come from trials within that MATLAB file or calendar days within that CWA file. Multiple recording-file drops are rejected; there is no multi-recording mode.

## Pages and static hosting

The workflow uses `/upload`, `/dataset`, `/progress` and `/results`; `/` redirects to Upload. Upload supplies files and participant configuration, Dataset selects trials/days, Progress tracks the entire selected batch, and Results shows its outcomes and tables. Ordinary row errors continue; cancellation stays on Progress with completed results retained. Back/Forward restores URL-backed controls without repeating analysis. File handles, participant measurements and results stay in memory: reloading or opening a direct link in another tab requires selecting the recording again.

For a subpath deployment, build with `npx vite build --base=/mobgap/`. Serve the complete output under that path and configure an SPA fallback for page routes. Runtime and generated asset requests must retain normal missing-file responses, rather than receiving `index.html`. For example, an nginx deployment can use:

```nginx
location /mobgap/runtime/ { try_files $uri =404; }
location /mobgap/assets/ { try_files $uri =404; }
location /mobgap/ { try_files $uri $uri/ /mobgap/index.html; }
```

`npm run preview` supplies the page fallback for local checks. No analysis server is required.

TanStack Query is the canonical in-memory cache for recording/metadata File handles, the dataset index and CWA description, and per-row outcomes/result payloads. A stable QueryClientProvider wraps the router. Manually supplied resources use skipToken, infinite garbage-collection/stale times and structuralSharing=false; they do not refetch or persist to browser storage. Staging a new recording replaces the old resource session, releasing its handles and computed data. The controller retains form drafts and active-job/cancellation state, and URL search retains view settings and selection.
