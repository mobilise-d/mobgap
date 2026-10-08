# Interface decisions

The official Mobilise-D header and four-step navigation connect separate Upload, Dataset, Progress and Results pages. Upload stages one local recording and complete participant configuration. Dataset contains the actual trial/day index, selection checkboxes and walking preset; it has no persistent upload sidebar. Progress shows the frozen batch row list, aggregate completed-or-failed count and cancellation. Results contains batch outcomes, completed-row selection, all output tables and CSV exports.

Files, metadata and results remain in one in-memory controller across route transitions. URLs describe pipeline, CWA split/timezone, selected row indexes and result/table views. They cannot restore local files after a reload. Absent row selection means all, while an explicit empty selection means none. Back/Forward changes the view and controls without starting work. Configuration changes invalidate the index and require rebuilding.

Only an explicit Run starts the selected rows. The running batch keeps its original row list even if URL selection changes. Ordinary row errors continue and count toward total progress. No partial-results link or table is shown while active; after all selected rows are terminal the app switches to Results, including when every row failed. Cancellation stays on Progress and retains completed results.

System sans text, a fixed type scale and tabular numbers support reading. Slate neutrals, restrained indigo selected states, thin borders and consistent shadcn components provide hierarchy. Running states show real runtime messages and indeterminate activity, never fabricated percentages. Computed results include all output tables and CSV export without display rounding.

Walking presets are Healthy, Impaired and Auto. Auto is the initial choice and uses the Universal pipeline to select Healthy for HA/COPD/CHF or Impaired for PD/MS/PFF from the explicitly supplied cohort. Bundled examples retain their explicit preset hints.

CWA offers only Single file and Split by calendar day. Metadata determines the initial choice: recordings longer than 24 hours use calendar days, recordings of 24 hours or less use the entire file. Users can override that choice after metadata inspection and rebuild without losing their explicit selection.

The upload workflow accepts exactly one recording file, with an optional separate infoForAlgo file for MATLAB. Multiple dataset rows come from trials within that MATLAB file or calendar days within that CWA file. Multiple recording-file drops are rejected; there is no multi-recording mode.

TanStack Query is the canonical in-memory cache for recording/metadata File handles, the dataset index and CWA description, and per-row outcomes/result payloads. A stable QueryClientProvider wraps the router. Manually supplied resources use skipToken, infinite garbage-collection/stale times and structuralSharing=false; they do not refetch or persist to browser storage. Staging a new recording replaces the old resource session, releasing its handles and computed data. The controller retains form drafts and active-job/cancellation state, and URL search retains view settings and selection.
