# Browser lab

register: product

Researchers and clinicians inspect gait recordings on a laptop in a bright office or lab, comparing a known dataset against a local analysis. The interface should feel precise, calm and candid about what has run. Files remain in the browser; the app requires no account, upload or backend. Its central task is constructing a file-backed dataset from Mobilise-D MATLAB or AX6 CWA recordings and complete participant metadata. The left sidebar stages one local recording File handle and configures metadata, recording setting, timezone and calendar-day or single-file processing. Nothing is inspected or decoded before the user chooses Build dataset. MATLAB needs a separate infoForAlgo file or explicit manual heights; CWA always needs manual heights and a synchronization timezone. Cohort is explicit. The center stays empty until construction succeeds, then shows the actual dataset index with all rows selected by default. Users choose trials or days and run a full mobgap preset for selected rows. Each CWA file is processed through one Python AX6Dataset loop; MATLAB trials run sequentially. Per-row progress, results, errors and exports remain available after cancellation. Configuration changes invalidate the old dataset index and require rebuilding.

Use a light neutral surface, one restrained indigo accent for active controls, accessible labels and tabular numeric data. Avoid medical marketing imagery, decorative dashboards, invented result values and technical deployment details inside the main workflow. Loading and failed runs must state what the app is doing and what the user can do next. A prototype's limitations should be visible without becoming a checklist.

Walking presets are Healthy, Impaired and Auto. Auto is the initial choice and uses the Universal pipeline to select Healthy for HA/COPD/CHF or Impaired for PD/MS/PFF from the explicitly supplied cohort. Bundled examples retain their explicit preset hints.

CWA offers only Single file and Split by calendar day. Metadata determines the initial choice: recordings longer than 24 hours use calendar days, recordings of 24 hours or less use the entire file. Users can override that choice after metadata inspection and rebuild without losing their explicit selection.

The upload workflow accepts exactly one recording file, with an optional separate infoForAlgo file for MATLAB. Multiple dataset rows come from trials within that MATLAB file or calendar days within that CWA file. Multiple recording-file drops are rejected; there is no multi-recording mode.
