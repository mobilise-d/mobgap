# Interface decisions

A narrow header introduces mobgap and indicates local browser processing. A two-column workspace places dataset configuration in the left sidebar and the dataset index and results in the larger center area. It becomes a single column on small screens. The center stays empty until the user supplies participant information and chooses Build dataset. Local recording files are staged first; MATLAB then requires separate infoForAlgo metadata or explicit manual heights, while CWA always uses manual information. Cohort, recording setting and CWA timezone are explicit controls.

The constructed index shows actual MATLAB trials or CWA calendar days, with all rows selected initially. Accessible row and all-row checkboxes, the walking preset and Run selected controls sit together above the results. Rows run sequentially; CWA days from each file share one Python dataset loop. Per-row status, errors and completed results remain visible after cancellation. Changing dataset configuration clears the old index and requires rebuilding.

System sans text, a fixed type scale and tabular numbers support reading. Slate neutrals, restrained indigo selected states, thin borders and consistent shadcn components provide hierarchy. Running states show real runtime messages and indeterminate activity, never fabricated percentages. Computed results include all output tables and CSV export without display rounding.

Walking presets are Healthy, Impaired and Auto. Auto is the initial choice and uses the Universal pipeline to select Healthy for HA/COPD/CHF or Impaired for PD/MS/PFF from the explicitly supplied cohort. Bundled examples retain their explicit preset hints.
